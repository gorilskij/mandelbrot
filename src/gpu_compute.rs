//! Fullscreen GPU perturbation backend.
//!
//! Computes the entire Mandelbrot frame in a single compute dispatch (plus a
//! small number of glitch-correction passes). Compared to the old per-tile
//! approach this eliminates thousands of round-trips and lets the GPU stay
//! busy for the full frame duration.
//!
//! Algorithm:
//!   1. Pick the screen-centre pixel as the reference point; compute its
//!      orbit at arbitrary precision on the CPU.
//!   2. Upload the orbit once and dispatch one thread per pixel.  Each thread
//!      runs the delta-perturbation recurrence, marking pixels as "glitched"
//!      when the delta grows too large relative to the true orbit value.
//!   3. Collect glitched pixels, pick the one whose delta lasted the longest
//!      before blowing up (best proxy for a point close to the set), compute
//!      its orbit on the CPU, and re-dispatch only the glitched pixels against
//!      that new reference.  Repeat until no glitches remain (or after a fixed
//!      number of passes).

use crate::rendering::{CoordinatesBox, Orbit, Pf, calculate_orbit, val_to_color};
use bytemuck::{Pod, Zeroable};
use dashu::float::FBig;
use num::Complex;
use std::num::NonZeroUsize;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// WGSL shader
// ---------------------------------------------------------------------------

const SHADER_SRC: &str = r#"
struct Uniforms {
    pixel_count : u32,
    orbit_len   : u32,
    is_full     : u32,
    dispatch_w  : u32,  // gx * 64; stride to reconstruct flat idx from 2D gid
}

@group(0) @binding(0) var<uniform>             uniforms     : Uniforms;
@group(0) @binding(1) var<storage, read>       pixel_deltas : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       orbit_data   : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output       : array<u32>;

// Bit 31 set → glitched; bits 0-30 → iteration count when glitch was detected.
const GLITCH_BIT : u32  = 0x80000000u;
// Glitch threshold: |δ|² > ε² · |X+δ|²  with ε = 1e-3.
const EPSILON_SQ : f32  = 1e-6;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.y * uniforms.dispatch_w + gid.x;
    if idx >= uniforms.pixel_count { return; }

    let d0    = pixel_deltas[idx];
    var delta = d0;

    for (var i = 0u; i < uniforms.orbit_len; i++) {
        let c = orbit_data[i];
        let x = c + delta;

        // Escape check.
        if dot(x, x) > 4.0 {
            output[idx] = i + 1u;
            return;
        }

        // Glitch check: perturbation approximation has broken down.
        if dot(delta, delta) > EPSILON_SQ * dot(x, x) {
            output[idx] = GLITCH_BIT | (i + 1u);
            return;
        }

        // δ_{n+1} = 2 X_n δ_n + δ_n² + δ_0
        let re = 2.0 * c.x * delta.x - 2.0 * c.y * delta.y
               + delta.x * delta.x    - delta.y * delta.y
               + d0.x;
        let im = 2.0 * c.x * delta.y + 2.0 * c.y * delta.x
               + 2.0 * delta.x * delta.y
               + d0.y;
        delta = vec2<f32>(re, im);
    }

    // Orbit exhausted.
    if uniforms.is_full != 0u {
        output[idx] = 0u;                               // in the set
    } else {
        output[idx] = GLITCH_BIT | uniforms.orbit_len;  // reference escaped — treat as glitch
    }
}
"#;

// ---------------------------------------------------------------------------
// CPU-side types
// ---------------------------------------------------------------------------

const GLITCH_BIT: u32 = 0x80000000;
const MAX_GLITCH_PASSES: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    pixel_count: u32,
    orbit_len:   u32,
    is_full:     u32,
    dispatch_w:  u32,
}

// ---------------------------------------------------------------------------
// GpuCompute
// ---------------------------------------------------------------------------

pub struct GpuCompute {
    device:   wgpu::Device,
    queue:    wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl:      wgpu::BindGroupLayout,
}

impl GpuCompute {
    pub fn new() -> Self {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .expect("no GPU adapter");
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("failed to create GPU device");

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu_compute"),
                source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_compute_bgl"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_compute_pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu_compute"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            GpuCompute { device, queue, pipeline, bgl }
        })
    }

    /// Compute a full frame. Returns one `0x00RRGGBB` color per pixel,
    /// row-major (row 0 = top of screen).
    pub fn compute_frame(
        &self,
        coords: &CoordinatesBox,
        width: u32,
        height: u32,
        iterations: usize,
    ) -> Vec<u32> {
        if width == 0 || height == 0 {
            return Vec::new();
        }

        let center_px = width / 2;
        let center_py = height / 2;
        let view = coords.view.inner;

        // Reference orbit anchored at screen centre (high precision).
        let center_coord = screen_to_mandelbrot(coords, center_px, center_py);
        let (_, center_orbit) = calculate_orbit(center_coord.clone(), iterations);

        // Initial dispatch: all pixels relative to centre.
        let n = (width * height) as usize;
        let deltas = all_deltas_from(n, width, center_px, center_py, view);
        let mut results = self.dispatch(&deltas, &center_orbit);

        // Glitch-correction passes.
        for _ in 0..MAX_GLITCH_PASSES {
            // Collect glitches and find the one with the longest pre-detection path.
            let mut best_flat = 0usize;
            let mut best_itr = 0u32;
            let mut glitch_indices: Vec<usize> = Vec::new();

            for (i, &r) in results.iter().enumerate() {
                if r & GLITCH_BIT != 0 {
                    let itr = r & !GLITCH_BIT;
                    if itr > best_itr {
                        best_itr = itr;
                        best_flat = i;
                    }
                    glitch_indices.push(i);
                }
            }

            if glitch_indices.is_empty() {
                break;
            }

            // New reference: glitch pixel with the longest path.
            let ref_px = (best_flat % width as usize) as u32;
            let ref_py = (best_flat / width as usize) as u32;
            let ref_coord =
                pixel_coord(&center_coord, center_px, center_py, ref_px, ref_py, view);
            let (_, ref_orbit) = calculate_orbit(ref_coord, iterations);

            // Deltas for glitched pixels relative to the new reference.
            let new_deltas: Vec<[f32; 2]> = glitch_indices
                .iter()
                .map(|&i| {
                    let px = (i % width as usize) as u32;
                    let py = (i / width as usize) as u32;
                    pixel_delta(px, py, ref_px, ref_py, view)
                })
                .collect();

            let new_results = self.dispatch(&new_deltas, &ref_orbit);

            // Scatter results back to their original positions.
            for (j, &i) in glitch_indices.iter().enumerate() {
                results[i] = new_results[j];
            }
        }

        // Convert iteration counts to colors; residual glitches → black.
        results
            .iter()
            .map(|&r| {
                if r & GLITCH_BIT != 0 {
                    0
                } else {
                    val_to_color(NonZeroUsize::new(r as usize))
                }
            })
            .collect()
    }

    /// Upload `deltas` and `orbit`, dispatch, and read back raw results.
    /// Returns one u32 per delta: escape iteration, 0 (in set), or
    /// GLITCH_BIT | iteration.
    fn dispatch(&self, deltas: &[[f32; 2]], orbit: &Orbit<Pf>) -> Vec<u32> {
        let n = deltas.len() as u32;
        if n == 0 {
            return Vec::new();
        }

        let device = &self.device;
        let queue  = &self.queue;

        let orbit_data: Vec<[f32; 2]> =
            orbit.orbit.iter().map(|c| [c.re as f32, c.im as f32]).collect();

        // Dispatch in 2D to stay within the 65535 workgroup-per-dimension limit.
        let groups = (n + 63) / 64;
        let gx = groups.min(65535);
        let gy = (groups + 65534) / 65535;
        let dispatch_w = gx * 64;

        let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: bytemuck::bytes_of(&Uniforms {
                pixel_count: n,
                orbit_len:   orbit_data.len() as u32,
                is_full:     orbit.is_full as u32,
                dispatch_w,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let deltas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: bytemuck::cast_slice(deltas),
            usage:    wgpu::BufferUsages::STORAGE,
        });
        let orbit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: bytemuck::cast_slice(&orbit_data),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let output_size = n as u64 * 4;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              None,
            size:               output_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              None,
            size:               output_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: deltas_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: orbit_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output_buf.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        queue.submit([enc.finish()]);

        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let data = slice.get_mapped_range();
        let results: Vec<u32> = data
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        drop(data);
        staging_buf.unmap();

        results
    }
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

/// Mandelbrot coordinate for screen pixel `(px, py)`.
fn screen_to_mandelbrot(coords: &CoordinatesBox, px: u32, py: u32) -> Complex<FBig> {
    let view = coords.view.inner;
    Complex {
        re: &coords.origin.x + &FBig::try_from(px as f64 * view).unwrap(),
        im: &coords.origin.y + &FBig::try_from(py as f64 * view).unwrap(),
    }
}

/// Mandelbrot coordinate for `(px, py)` expressed as a delta from the
/// already-known `center_coord` at `(center_px, center_py)`. All arithmetic
/// is in f64 (exact, since both points are in the same view).
fn pixel_coord(
    center_coord: &Complex<FBig>,
    center_px: u32,
    center_py: u32,
    px: u32,
    py: u32,
    view: f64,
) -> Complex<FBig> {
    let dx = (px as f64 - center_px as f64) * view;
    let dy = (py as f64 - center_py as f64) * view;
    Complex {
        re: &center_coord.re + &FBig::try_from(dx).unwrap(),
        im: &center_coord.im + &FBig::try_from(dy).unwrap(),
    }
}

/// f32 delta for pixel `(px, py)` from reference `(ref_px, ref_py)`.
fn pixel_delta(px: u32, py: u32, ref_px: u32, ref_py: u32, view: f64) -> [f32; 2] {
    [
        ((px as f64 - ref_px as f64) * view) as f32,
        ((py as f64 - ref_py as f64) * view) as f32,
    ]
}

/// Build deltas for all `n` pixels in row-major order relative to
/// `(ref_px, ref_py)`.
fn all_deltas_from(n: usize, width: u32, ref_px: u32, ref_py: u32, view: f64) -> Vec<[f32; 2]> {
    (0..n)
        .map(|i| {
            let px = (i % width as usize) as u32;
            let py = (i / width as usize) as u32;
            pixel_delta(px, py, ref_px, ref_py, view)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// wgpu helper
// ---------------------------------------------------------------------------

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}
