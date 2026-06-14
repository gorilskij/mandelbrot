//! GPU perturbation backend (wgpu compute).
//!
//! The reference orbits (projected to Pf = f32) are packed into GPU storage
//! buffers. A compute shader runs the delta iteration per pixel, writing raw
//! iteration counts (0 = black/non-divergent, u32::MAX = glitch sentinel).
//! Glitched pixels are resolved on the CPU by computing their own high-
//! precision orbit and storing the result directly.

use super::{Perturbator, RefList, RefOrbit};
use crate::rendering::{Pf, calculate_orbit, check_orbit, val_to_color};
use crate::tiles::store::{
    GROUP_POW, GROUP_TILES, PASS_STRIDES, TILE_SIZE, Tile, floor_div_pow2, pixel_to_coord,
    units_per_pixel, working_precision,
};
use bytemuck::{Pod, Zeroable};
use dashu::integer::IBig;
use num::Complex;
use std::num::NonZeroUsize;
use std::sync::Arc;
use waker_interrupter::MultiInterrupter;
use wgpu::util::DeviceExt;

const SHADER_SRC: &str = r#"
struct Uniforms {
    pixel_count: u32,
    ref_count: u32,
    _pad0: u32,
    _pad1: u32,
}

// 32 bytes, all f32/u32 (align 4), array stride = 32
struct RefMeta {
    delta_corr_re: f32,
    delta_corr_im: f32,
    is_full: u32,
    orbit_len: u32,
    orbit_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>             uniforms:     Uniforms;
@group(0) @binding(1) var<storage, read>       pixel_deltas: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       ref_metas:    array<RefMeta>;
@group(0) @binding(3) var<storage, read>       orbit_data:   array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> output:       array<u32>;

const GLITCH: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.pixel_count { return; }

    let delta_0 = pixel_deltas[idx];

    for (var ri = 0u; ri < uniforms.ref_count; ri++) {
        let rm = ref_metas[ri];
        let corr = vec2<f32>(rm.delta_corr_re, rm.delta_corr_im);
        var delta = delta_0 - corr;
        let d0 = delta;

        for (var i = 0u; i < rm.orbit_len; i++) {
            let c = orbit_data[rm.orbit_offset + i];
            let x = c + delta;
            if dot(x, x) > 4.0 {
                output[idx] = i + 1u;
                return;
            }
            // delta_{n+1} = 2*c_n*delta_n + delta_n^2 + delta_0
            let re = 2.0 * c.x * delta.x - 2.0 * c.y * delta.y
                   + delta.x * delta.x - delta.y * delta.y
                   + d0.x;
            let im = 2.0 * c.x * delta.y + 2.0 * c.y * delta.x
                   + 2.0 * delta.x * delta.y
                   + d0.y;
            delta = vec2<f32>(re, im);
        }

        if rm.is_full != 0u {
            output[idx] = 0u;
            return;
        }
        // glitched against this reference — try next
    }

    output[idx] = GLITCH;
}
"#;

/// GPU-side uniform block. Must match `Uniforms` in the shader exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    pixel_count: u32,
    ref_count: u32,
    _pad: [u32; 2],
}

/// GPU-side reference metadata. Must match `RefMeta` in the shader exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RefMetaGpu {
    delta_corr_re: f32,
    delta_corr_im: f32,
    is_full: u32,
    orbit_len: u32,
    orbit_offset: u32,
    _pad: [u32; 3],
}

pub struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuState {
    pub fn new() -> Self {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .expect("no GPU adapter found");

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("failed to create GPU device");

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("perturbation"),
                source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("perturbation_bgl"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });

            let pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("perturbation_pl"),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });

            let pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("perturbation"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            GpuState { device, queue, pipeline, bgl }
        })
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// wgpu perturbation backend.
pub struct Gpu(pub Arc<GpuState>);

impl Perturbator for Gpu {
    fn render_tile_pass(
        &self,
        tile: &Tile,
        refs: &RefList,
        anchor_px: (i64, i64),
        pass: u8,
        iterations: usize,
        int: &MultiInterrupter,
    ) -> bool {
        render_tile_pass_gpu(&self.0, tile, refs, anchor_px, pass, iterations, int)
    }
}

fn render_tile_pass_gpu(
    state: &GpuState,
    tile: &Tile,
    refs: &RefList,
    anchor_px: (i64, i64),
    pass: u8,
    iterations: usize,
    int: &MultiInterrupter,
) -> bool {
    let stride = PASS_STRIDES[pass as usize];
    let coarser_stride = (pass > 0).then(|| PASS_STRIDES[pass as usize - 1]);

    let gx = floor_div_pow2(&tile.key.x, GROUP_POW);
    let gy = floor_div_pow2(&tile.key.y, GROUP_POW);
    let local_x = i64::try_from(&(&tile.key.x - &gx * IBig::from(GROUP_TILES as u64)))
        .expect("tile-in-group offset fits i64");
    let local_y = i64::try_from(&(&tile.key.y - &gy * IBig::from(GROUP_TILES as u64)))
        .expect("tile-in-group offset fits i64");
    let depth = tile.key.depth;
    let upp = units_per_pixel(depth) as Pf;
    let anchor_dx = local_x * TILE_SIZE as i64 - anchor_px.0;
    let anchor_dy = local_y * TILE_SIZE as i64 - anchor_px.1;
    let tile_size_ibig = IBig::from(TILE_SIZE as u64);
    let tile_px0_x = &tile.key.x * &tile_size_ibig;
    let tile_px0_y = &tile.key.y * &tile_size_ibig;
    let prec = working_precision(depth);

    // collect pixels to compute this pass
    let mut pixel_indices: Vec<u32> = Vec::new();
    let mut pixel_deltas: Vec<[Pf; 2]> = Vec::new();
    for r in (0..TILE_SIZE).step_by(stride) {
        for c in (0..TILE_SIZE).step_by(stride) {
            if let Some(cs) = coarser_stride {
                if r % cs == 0 && c % cs == 0 {
                    continue;
                }
            }
            let idx = r * TILE_SIZE + c;
            if tile.load(idx).get().is_some() {
                continue;
            }
            pixel_indices.push(idx as u32);
            pixel_deltas.push([
                (anchor_dx + c as i64) as Pf * upp,
                (anchor_dy + r as i64) as Pf * upp,
            ]);
        }
    }

    if pixel_indices.is_empty() {
        tile.finish_pass(pass + 1);
        return true;
    }

    if int.interrupted() {
        return false;
    }

    // use only the most recent reference — the front of the list
    let front = refs.iter().next().expect("ref list is never empty");
    let ref_meta = RefMetaGpu {
        delta_corr_re: front.delta_corr.re as f32,
        delta_corr_im: front.delta_corr.im as f32,
        is_full: front.orbit.is_full as u32,
        orbit_len: front.orbit.orbit.len() as u32,
        orbit_offset: 0,
        _pad: [0; 3],
    };
    let orbit_data: Vec<[f32; 2]> = front.orbit.orbit.iter().map(|c| [c.re as f32, c.im as f32]).collect();
    let pixel_deltas_f32: Vec<[f32; 2]> = pixel_deltas.iter().map(|d| [d[0] as f32, d[1] as f32]).collect();

    let counts = gpu_dispatch(state, &pixel_deltas_f32, &[ref_meta], &orbit_data);

    for (i, &tile_idx) in pixel_indices.iter().enumerate() {
        let count = counts[i];
        if count == u32::MAX {
            // glitch: compute true orbit at high precision
            let r = tile_idx as usize / TILE_SIZE;
            let c = tile_idx as usize % TILE_SIZE;
            let x_0 = Complex {
                re: pixel_to_coord(
                    &tile_px0_x + IBig::from(c as u64),
                    depth,
                    prec,
                ),
                im: pixel_to_coord(
                    &tile_px0_y + IBig::from(r as u64),
                    depth,
                    prec,
                ),
            };
            let (_, new_orbit) = calculate_orbit(x_0, iterations);
            let val = check_orbit(&new_orbit).unwrap();
            tile.store(tile_idx as usize, val_to_color(val).into());
            refs.push_front(RefOrbit {
                delta_corr: Complex { re: pixel_deltas[i][0], im: pixel_deltas[i][1] },
                orbit: new_orbit,
            });
        } else {
            let val = NonZeroUsize::new(count as usize);
            tile.store(tile_idx as usize, val_to_color(val).into());
        }
    }

    tile.finish_pass(pass + 1);
    true
}

/// Dispatch the compute shader; returns raw iteration counts per pixel.
/// 0 = non-divergent, u32::MAX = glitch, other = escape iteration index.
fn gpu_dispatch(
    state: &GpuState,
    pixel_deltas: &[[f32; 2]],
    ref_metas: &[RefMetaGpu],
    orbit_data: &[[f32; 2]],
) -> Vec<u32> {
    let n = pixel_deltas.len() as u32;
    let device = &state.device;
    let queue = &state.queue;

    let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&Uniforms {
            pixel_count: n,
            ref_count: ref_metas.len() as u32,
            _pad: [0; 2],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let deltas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(pixel_deltas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let metas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(ref_metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let orbit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(orbit_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = (n as u64) * 4;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &state.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: deltas_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: metas_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: orbit_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output_buf.as_entire_binding() },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&state.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n + 63) / 64, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
    queue.submit([encoder.finish()]);

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
