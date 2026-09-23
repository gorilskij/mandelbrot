//! GPU perturbation backend.
//!
//! For each progressive pass all visible tiles that still need it are batched
//! into a single GPU compute dispatch.  Pixels that the perturbation
//! approximation cannot resolve (glitches) are re-dispatched in subsequent
//! passes with a new high-precision reference orbit, up to MAX_GLITCH_PASSES
//! times.  Residual glitches are then resolved exactly on the CPU (in
//! parallel, interruptibly); any left unresolved are not stored.
//!
//! Coordinate math
//! ---------------
//! The initial reference point is the screen centre.  Its Mandelbrot
//! coordinate is computed at arbitrary precision on the CPU.  For every other
//! pixel, the delta from the reference is:
//!
//!   δx = (col * units_per_pixel(depth) + offset_x)  [f32]
//!
//! where `col = tile_i * TILE_SIZE + c` is the pixel's integer position in
//! the depth-level grid (relative to the tile-grid origin x0), and
//!
//!   offset_x = (sx0 − width/2) * view
//!
//! is a constant per pass that shifts from "grid pixels" to "Mandelbrot units
//! centred on the screen".  The computation is done in f64 and then cast to
//! f32.  No large-number cancellation occurs because both operands are small
//! (a few thousand pixels, a small sub-pixel offset), so f32 precision is
//! adequate for the initial delta.

use super::{PassBatchCtx, Perturbator, TileItem};
use crate::rendering::{calculate_orbit, check_orbit, val_to_color};
use crate::tiles::store::{
    PASS_STRIDES, TILE_SIZE, pixel_to_coord, units_per_pixel, upp_log2, working_precision,
};
use bytemuck::{Pod, Zeroable};
use dashu::float::FBig;
use dashu::integer::IBig;
use num::Complex;
use rayon::prelude::*;
use std::num::NonZeroUsize;
use std::sync::Arc;
use waker_interrupter::MultiInterrupter;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// WGSL shaders (sources live in shaders/)
// ---------------------------------------------------------------------------

/// Plain-f32 perturbation, used at shallow/moderate zoom.
const SHADER_SRC: &str = include_str!("shaders/perturbation.wgsl");

/// floatexp helpers, prepended to the deep-zoom shader.
const FE_HELPERS: &str = include_str!("shaders/floatexp.wgsl");
/// Deep-zoom perturbation; iterates the delta in floatexp.
const SHADER_FE_BODY: &str = include_str!("shaders/perturbation_floatexp.wgsl");

// ---------------------------------------------------------------------------
// CPU-side types
// ---------------------------------------------------------------------------

const GLITCH_BIT: u32 = 0x80000000;
const MAX_GLITCH_PASSES: usize = 8;

/// Use the floatexp pipeline once units-per-pixel drops below 2^FE_THRESHOLD.
/// The f32 seed starts breaking down near 2^-126; -100 leaves a safe margin
/// while keeping the fast f32 path for all shallower zooms.
pub(crate) const FE_THRESHOLD: i64 = -100;

/// Sentinel exponent for a zero floatexp value (matches floatexp.wgsl).
const FE_ZERO_EXP: i64 = -2_000_000_000;

/// Pack a complex offset given in units of `2^base_exp` into a shared-exponent
/// floatexp seed `[mantissa.x, mantissa.y, exponent_as_f32, 0]`.  Working in
/// these units keeps `dx`/`dy` at O(pixel-count) so the f64 math never
/// underflows, however deep the zoom; the depth scale rides in `base_exp`.
fn pack_fe(dx: f64, dy: f64, base_exp: i64) -> [f32; 4] {
    let a = dx.abs().max(dy.abs());
    if a == 0.0 {
        return [0.0, 0.0, FE_ZERO_EXP as f32, 0.0];
    }
    let e_local = a.log2().floor() as i64;
    let s = (-(e_local as f64)).exp2(); // a * s in [1, 2)
    [(dx * s) as f32, (dy * s) as f32, (base_exp + e_local) as f32, 0.0]
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    pixel_count: u32,
    orbit_len:   u32,
    is_full:     u32,
    dispatch_w:  u32,
}

// ---------------------------------------------------------------------------
// GpuState — owns the wgpu device / pipeline
// ---------------------------------------------------------------------------

pub struct GpuState {
    device:      wgpu::Device,
    queue:       wgpu::Queue,
    /// Plain-f32 perturbation pipeline (shallow zoom).
    pipeline:    wgpu::ComputePipeline,
    /// floatexp perturbation pipeline (deep zoom).
    pipeline_fe: wgpu::ComputePipeline,
    bgl:         wgpu::BindGroupLayout,
    /// `max_storage_buffer_binding_size`; dispatches are chunked to stay under
    /// it (the delta buffer is the largest binding).
    max_binding: usize,
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

            let max_binding = device.limits().max_storage_buffer_binding_size as usize;

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label:   Some("perturbation_bgl"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("perturbation_pl"),
                bind_group_layouts:   &[Some(&bgl)],
                immediate_size:       0,
            });

            let make_pipeline = |label: &str, src: String| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label:  Some(label),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label:               Some(label),
                    layout:              Some(&layout),
                    module:              &module,
                    entry_point:         Some("main"),
                    compilation_options: Default::default(),
                    cache:               None,
                })
            };

            let pipeline    = make_pipeline("perturbation", SHADER_SRC.to_string());
            let pipeline_fe = make_pipeline(
                "perturbation_fe",
                format!("{FE_HELPERS}\n{SHADER_FE_BODY}"),
            );

            GpuState { device, queue, pipeline, pipeline_fe, bgl, max_binding }
        })
    }

    /// Plain-f32 dispatch: one `[f32; 2]` delta per pixel.
    fn dispatch_f32(&self, deltas: &[[f32; 2]], orbit_data: &[[f32; 2]], is_full: bool) -> Vec<u32> {
        self.dispatch(&self.pipeline, bytemuck::cast_slice(deltas), deltas.len(), 8, orbit_data, is_full)
    }

    /// floatexp dispatch: one `[f32; 4]` seed per pixel (mantissa.xy, exp, _).
    fn dispatch_fe(&self, deltas: &[[f32; 4]], orbit_data: &[[f32; 2]], is_full: bool) -> Vec<u32> {
        self.dispatch(&self.pipeline_fe, bytemuck::cast_slice(deltas), deltas.len(), 16, orbit_data, is_full)
    }

    /// Upload deltas + orbit, dispatch `pipeline`, and read back raw results.
    /// Each result: `0` = in-set, `n` = escaped at iteration n,
    /// `GLITCH_BIT | n` = glitched at iteration n.
    ///
    /// `delta_bytes` holds `n` deltas of `elem` bytes each.  Large batches are
    /// split into chunks so no storage buffer exceeds the device's
    /// `max_storage_buffer_binding_size`.
    fn dispatch(
        &self,
        pipeline:    &wgpu::ComputePipeline,
        delta_bytes: &[u8],
        n:           usize,
        elem:        usize,
        orbit_data:  &[[f32; 2]],
        is_full:     bool,
    ) -> Vec<u32> {
        if n == 0 { return Vec::new(); }
        let max_pixels = ((self.max_binding / elem) * 9 / 10).max(1);
        if n <= max_pixels {
            return self.dispatch_chunk(pipeline, delta_bytes, n, orbit_data, is_full);
        }
        let mut out = Vec::with_capacity(n);
        for chunk in delta_bytes.chunks(max_pixels * elem) {
            out.extend(self.dispatch_chunk(pipeline, chunk, chunk.len() / elem, orbit_data, is_full));
        }
        out
    }

    /// One dispatch over a chunk small enough to fit the binding-size limit.
    fn dispatch_chunk(
        &self,
        pipeline:    &wgpu::ComputePipeline,
        delta_bytes: &[u8],
        n_pixels:    usize,
        orbit_data:  &[[f32; 2]],
        is_full:     bool,
    ) -> Vec<u32> {
        let n = n_pixels as u32;
        if n == 0 { return Vec::new(); }

        let device = &self.device;
        let queue  = &self.queue;

        let groups     = n.div_ceil(64);
        let gx         = groups.min(65535);
        let gy         = groups.div_ceil(65535);
        let dispatch_w = gx * 64;

        let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: bytemuck::bytes_of(&Uniforms {
                pixel_count: n,
                orbit_len:   orbit_data.len() as u32,
                is_full:     is_full as u32,
                dispatch_w,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let deltas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: delta_bytes,
            usage:    wgpu::BufferUsages::STORAGE,
        });
        let orbit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: bytemuck::cast_slice(orbit_data),
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
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        let t_submit = std::time::Instant::now(); // DIAG
        queue.submit([enc.finish()]);
        log::info!("[diag gpu] dispatch chunk {n_pixels} px submitted, waiting..."); // DIAG

        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        log::info!(
            "[diag gpu] dispatch chunk {n_pixels} px ({}x{} groups): gpu wait {:.1} ms",
            gx, gy, t_submit.elapsed().as_secs_f64() * 1e3,
        );

        let data = slice.get_mapped_range().expect("staging buffer mapped");
        let results: Vec<u32> = data
            .as_chunks::<4>().0.iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        drop(data);
        staging_buf.unmap();
        results
    }
}

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

/// DIAG: one-line summary of raw GPU results (in-set / glitched / escape range
/// / number of distinct values / share of the most common value).
fn raw_stats(raw: &[u32]) -> String {
    let mut counts = std::collections::HashMap::<u32, usize>::new();
    let (mut in_set, mut glitched) = (0usize, 0usize);
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    for &r in raw {
        *counts.entry(r).or_default() += 1;
        if r & GLITCH_BIT != 0 { glitched += 1; }
        else if r == 0 { in_set += 1; }
        else { lo = lo.min(r); hi = hi.max(r); }
    }
    let (top_val, top_n) = counts.iter().max_by_key(|(_, n)| **n).map(|(v, n)| (*v, *n)).unwrap_or((0, 0));
    format!(
        "n={} in_set={} glitched={} escaped=[{}..{}] distinct={} top={:#x}@{:.1}%",
        raw.len(), in_set, glitched,
        if lo == u32::MAX { 0 } else { lo }, hi,
        counts.len(), top_val, 100.0 * top_n as f64 / raw.len().max(1) as f64,
    )
}

// ---------------------------------------------------------------------------
// Gpu — the public backend struct
// ---------------------------------------------------------------------------

pub struct Gpu(pub Arc<GpuState>);

// Maps one flat result index back to the tile + pixel that produced it.
struct PixelRef {
    tile_idx:  usize,  // index into the `tiles` slice
    pixel_idx: usize,  // flat tile index: r * TILE_SIZE + c
    col:       i64,    // depth-grid col  = tile_i * TILE_SIZE + c  (relative to x0)
    row:       i64,    // depth-grid row  = tile_j * TILE_SIZE + r
}

impl Perturbator for Gpu {
    fn render_pass_batch(
        &self,
        ctx:   &PassBatchCtx,
        tiles: &[TileItem],
        pass:  u8,
        int:   &MultiInterrupter,
    ) {
        if tiles.is_empty() || int.interrupted() { return; }

        let state   = &*self.0;
        let t_pass  = std::time::Instant::now(); // DIAG
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3; // DIAG
        let stride  = PASS_STRIDES[pass as usize];
        let coarser = (pass > 0).then(|| PASS_STRIDES[pass as usize - 1]);
        let scale   = units_per_pixel(ctx.depth); // Mandelbrot units per depth pixel
        let view    = ctx.coords.view.inner;

        // Beyond this depth the f32 seed underflows, so iterate the delta in
        // floatexp instead. The depth scale `2^upp` then rides in the floatexp
        // exponent and offsets are computed in pixel units (always O(1e4), so
        // the f64 math never underflows however deep the zoom).
        let upp    = upp_log2(ctx.depth);
        let use_fe = upp < FE_THRESHOLD;

        // f32 path: absolute Mandelbrot-unit offsets from the screen centre.
        let offset_x = (ctx.sx0 - ctx.width  as f64 / 2.0) * view;
        let offset_y = (ctx.sy0 - ctx.height as f64 / 2.0) * view;
        // floatexp path: same offsets but in units of `2^upp` (pixel units).
        // ratio = view / scale ≈ O(1), formed in log space to avoid over/underflow.
        let ratio    = (view.log2() - upp as f64).exp2();
        let off_x_px = (ctx.sx0 - ctx.width  as f64 / 2.0) * ratio;
        let off_y_px = (ctx.sy0 - ctx.height as f64 / 2.0) * ratio;

        // ----------------------------------------------------------------
        // Collect all pixels that need computing this pass.
        // ----------------------------------------------------------------
        let mut pixel_refs:  Vec<PixelRef>  = Vec::new();
        let mut deltas_f32:  Vec<[f32; 2]>  = Vec::new();
        let mut deltas_fe:   Vec<[f32; 4]>  = Vec::new();

        for (ti, item) in tiles.iter().enumerate() {
            let tile   = &item.tile;
            let tile_i = i64::try_from(&(&tile.key.x - &ctx.x0))
                .expect("tile grid offset fits i64");
            let tile_j = i64::try_from(&(&tile.key.y - &ctx.y0))
                .expect("tile grid offset fits i64");

            for r in (0..TILE_SIZE).step_by(stride) {
                for c in (0..TILE_SIZE).step_by(stride) {
                    if let Some(cs) = coarser
                        && r % cs == 0 && c % cs == 0 { continue; } // done by coarser pass
                    let pixel_idx = r * TILE_SIZE + c;
                    if tile.load(pixel_idx).get().is_some() { continue; } // already computed

                    let col = tile_i * TILE_SIZE as i64 + c as i64;
                    let row = tile_j * TILE_SIZE as i64 + r as i64;
                    if use_fe {
                        deltas_fe.push(pack_fe(
                            col as f64 + off_x_px,
                            row as f64 + off_y_px,
                            upp,
                        ));
                    } else {
                        deltas_f32.push([
                            (col as f64 * scale + offset_x) as f32,
                            (row as f64 * scale + offset_y) as f32,
                        ]);
                    }
                    pixel_refs.push(PixelRef { tile_idx: ti, pixel_idx, col, row });
                }
            }
        }

        // All pixels already computed (e.g. retrying after an interrupt).
        if pixel_refs.is_empty() {
            for item in tiles { item.tile.finish_pass(pass + 1); }
            return;
        }

        // ----------------------------------------------------------------
        // Initial dispatch against the screen-centre reference orbit.
        // ----------------------------------------------------------------
        let t_collect = ms(t_pass); // DIAG
        let t_ref = std::time::Instant::now(); // DIAG
        let ref_x = &ctx.coords.origin.x
            + &FBig::try_from(ctx.width  as f64 / 2.0 * view).unwrap();
        let ref_y = &ctx.coords.origin.y
            + &FBig::try_from(ctx.height as f64 / 2.0 * view).unwrap();
        let (_, ref_orbit) = calculate_orbit(
            Complex { re: ref_x, im: ref_y },
            ctx.iterations,
        );
        let orbit_data: Vec<[f32; 2]> = ref_orbit.orbit.iter()
            .map(|c| [c.re as f32, c.im as f32])
            .collect();

        let t_ref = ms(t_ref); // DIAG
        let t_disp = std::time::Instant::now(); // DIAG
        let mut raw = if use_fe {
            state.dispatch_fe(&deltas_fe, &orbit_data, ref_orbit.is_full)
        } else {
            state.dispatch_f32(&deltas_f32, &orbit_data, ref_orbit.is_full)
        };

        let t_disp = ms(t_disp); // DIAG
        log::info!(
            "[diag gpu] pass {pass} timing: collect {} px {t_collect:.1} ms, ref orbit {t_ref:.1} ms, \
             initial dispatch {t_disp:.1} ms",
            pixel_refs.len(),
        );
        // DIAG
        {
            let last = pixel_refs.len() - 1;
            let seeds = if use_fe {
                format!("seed[0]={:?} seed[last]={:?}", deltas_fe[0], deltas_fe[last])
            } else {
                format!("seed[0]={:?} seed[last]={:?}", deltas_f32[0], deltas_f32[last])
            };
            log::info!(
                "[diag gpu] pass {pass} depth {} upp 2^{upp} view {view:.4e} ratio {ratio:.6} pipe {} | \
                 ref orbit len {} full {} | off_px ({off_x_px:.3}, {off_y_px:.3}) offset ({offset_x:.4e}, {offset_y:.4e}) | {seeds}",
                ctx.depth, if use_fe { "fe" } else { "f32" },
                orbit_data.len(), ref_orbit.is_full,
            );
            log::info!("[diag gpu] pass {pass} initial: {}", raw_stats(&raw));
        }

        // ----------------------------------------------------------------
        // Glitch-correction passes.
        // ----------------------------------------------------------------
        for round in 0..MAX_GLITCH_PASSES {
            if int.interrupted() { break; }

            let mut glitch_indices: Vec<usize> = Vec::new();
            let mut best_flat = 0usize;
            let mut best_itr  = 0u32;

            for (i, &r) in raw.iter().enumerate() {
                if r & GLITCH_BIT != 0 {
                    let itr = r & !GLITCH_BIT;
                    if itr > best_itr { best_itr = itr; best_flat = i; }
                    glitch_indices.push(i);
                }
            }
            if glitch_indices.is_empty() { break; }

            // New reference: the glitch pixel with the longest path before
            // detection (best proxy for a point close to the set boundary).
            let pr       = &pixel_refs[best_flat];
            let ref_tile = &tiles[pr.tile_idx].tile;
            let ref_c    = pr.pixel_idx % TILE_SIZE;
            let ref_r    = pr.pixel_idx / TILE_SIZE;
            let ts       = IBig::from(TILE_SIZE as u64);
            let prec     = working_precision(ctx.depth);
            let t_gref = std::time::Instant::now(); // DIAG
            let (_, new_orbit) = calculate_orbit(
                Complex {
                    re: pixel_to_coord(&ref_tile.key.x * &ts + IBig::from(ref_c as u64), ctx.depth, prec),
                    im: pixel_to_coord(&ref_tile.key.y * &ts + IBig::from(ref_r as u64), ctx.depth, prec),
                },
                ctx.iterations,
            );
            let new_orbit_data: Vec<[f32; 2]> = new_orbit.orbit.iter()
                .map(|c| [c.re as f32, c.im as f32])
                .collect();

            // Deltas of glitched pixels from the new reference (integer pixel
            // differences — exact, and tiny enough for f64 at any depth).
            let ref_col = pr.col;
            let ref_row = pr.row;
            let t_gref = ms(t_gref); // DIAG
            let t_gdisp = std::time::Instant::now(); // DIAG
            let new_raw = if use_fe {
                let nd: Vec<[f32; 4]> = glitch_indices.iter().map(|&gi| {
                    let pr = &pixel_refs[gi];
                    pack_fe((pr.col - ref_col) as f64, (pr.row - ref_row) as f64, upp)
                }).collect();
                state.dispatch_fe(&nd, &new_orbit_data, new_orbit.is_full)
            } else {
                let nd: Vec<[f32; 2]> = glitch_indices.iter().map(|&gi| {
                    let pr = &pixel_refs[gi];
                    [
                        ((pr.col - ref_col) as f64 * scale) as f32,
                        ((pr.row - ref_row) as f64 * scale) as f32,
                    ]
                }).collect();
                state.dispatch_f32(&nd, &new_orbit_data, new_orbit.is_full)
            };
            for (j, &gi) in glitch_indices.iter().enumerate() {
                raw[gi] = new_raw[j];
            }
            log::info!(
                "[diag gpu] pass {pass} glitch round {round}: {} glitched, new ref orbit len {} full {} \
                 (orbit {t_gref:.1} ms, dispatch {:.1} ms) -> {}",
                glitch_indices.len(), new_orbit_data.len(), new_orbit.is_full, ms(t_gdisp),
                raw_stats(&new_raw),
            );
        }

        // ----------------------------------------------------------------
        // Resolve any residual glitches exactly on the CPU.
        //
        // Each residual pixel gets its own exact orbit, so this can be slow
        // (thousands of residuals at ~1 ms each when every reference escapes
        // early).  It runs across the rayon pool and checks the interrupt per
        // pixel, so a stale view is abandoned promptly.  Pixels left
        // unresolved keep GLITCH_BIT and are not stored below, so the next
        // generation retries them; the ones that did resolve are exact and
        // safe to cache.
        // ----------------------------------------------------------------
        if !int.interrupted() {
            let residual: Vec<usize> = raw.iter().enumerate()
                .filter(|&(_, &r)| r & GLITCH_BIT != 0)
                .map(|(i, _)| i)
                .collect();
            if !residual.is_empty() {
                log::info!("[diag gpu] pass {pass}: resolving {} residual glitches on CPU", residual.len());
            }
            let ts   = IBig::from(TILE_SIZE as u64);
            let prec = working_precision(ctx.depth);
            let t_resid = std::time::Instant::now(); // DIAG
            let resolved: Vec<(usize, u32)> = residual.par_iter()
                .filter_map(|&i| {
                    if int.interrupted() { return None; }
                    let pr   = &pixel_refs[i];
                    let tile = &tiles[pr.tile_idx].tile;
                    let c    = pr.pixel_idx % TILE_SIZE;
                    let rr   = pr.pixel_idx / TILE_SIZE;
                    let coord = Complex {
                        re: pixel_to_coord(&tile.key.x * &ts + IBig::from(c as u64),  ctx.depth, prec),
                        im: pixel_to_coord(&tile.key.y * &ts + IBig::from(rr as u64), ctx.depth, prec),
                    };
                    let (_, orbit) = calculate_orbit(coord, ctx.iterations);
                    Some((i, check_orbit(&orbit).unwrap().map_or(0, |n| n.get() as u32)))
                })
                .collect();
            if !residual.is_empty() {
                log::info!(
                    "[diag gpu] pass {pass}: residual CPU resolve {}/{} in {:.1} ms",
                    resolved.len(), residual.len(), ms(t_resid),
                );
            }
            for (i, r) in resolved { raw[i] = r; }
        }
        log::info!(
            "[diag gpu] pass {pass} final{}: {}",
            if int.interrupted() { " (interrupted)" } else { "" },
            raw_stats(&raw),
        );

        // ----------------------------------------------------------------
        // Scatter results into tile pixel stores.  A pixel still carrying
        // GLITCH_BIT here means we were interrupted before resolving it; leave
        // it unstored so the next generation retries it (never cache a guess).
        // ----------------------------------------------------------------
        for (i, pr) in pixel_refs.iter().enumerate() {
            let r = raw[i];
            if r & GLITCH_BIT != 0 { continue; }
            let color = val_to_color(NonZeroUsize::new(r as usize));
            tiles[pr.tile_idx].tile.store(pr.pixel_idx, color.into());
        }

        // Only advance pass counters if we weren't interrupted mid-glitch-loop.
        if !int.interrupted() {
            for item in tiles {
                item.tile.finish_pass(pass + 1);
            }
        }
        log::info!(
            "[diag gpu] pass {pass} total {:.1} ms{}",
            ms(t_pass), if int.interrupted() { " (interrupted)" } else { "" },
        );
    }
}
