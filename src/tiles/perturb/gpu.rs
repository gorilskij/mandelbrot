//! GPU perturbation backend.
//!
//! For each progressive pass the pixels of all visible tiles that still need
//! it are collected in cursor order and dispatched in chunks sized to take
//! about TARGET_CHUNK_MS each, with an interrupt check between chunks
//! (in-flight GPU work cannot be cancelled).  After each chunk its results
//! are stored and every tile whose pixels are all done finishes the pass, so
//! the image fills in outward from the cursor.
//!
//! Pixels that outlive the reference orbit (glitches) are re-dispatched
//! against a better reference, up to MAX_GLITCH_PASSES times per chunk.
//! Residual glitches are then resolved exactly on the CPU (in parallel,
//! interruptibly); any left unresolved are not stored, and their tiles stay
//! unfinished so the next generation retries them.
//!
//! References
//! ----------
//! Glitches only happen when a pixel outlives the reference orbit, so the
//! reference is chosen to outlive everything: a hyperbolic-component nucleus
//! near the view (see `nucleus.rs`), whose orbit never escapes.  It is cached
//! in `GpuState` and reused across passes and nearby views.  When no nucleus
//! is found from the view centre, the glitch rounds pick the longest-lived of
//! a sample of glitched pixels (exact orbits, in parallel) and seed the
//! nucleus search from it instead; improvements are cached for later passes.
//!
//! Coordinate math
//! ---------------
//! A reference `c` need not lie on the pixel grid.  Its position in pixel
//! units relative to the tile-grid origin (`ref_px`) is computed exactly in
//! FBig and rounded to f64 once; each pixel's offset is then
//! `(col - ref_px.x, row - ref_px.y)`, O(1e4) pixels, exact enough in f64 at
//! any depth.  The f32 pipeline multiplies by `2^upp`; the floatexp pipeline
//! carries `2^upp` in the exponent (`pack_fe`).

use super::nucleus::{MAX_REF_DIST, find_nucleus};
use super::{PassBatchCtx, Perturbator, TileItem};
use crate::rendering::{calculate_orbit, check_orbit, val_to_color};
use crate::tiles::store::{TILE_SIZE, pass_pixels, pixel_to_coord, upp_log2, working_precision};
use bytemuck::{Pod, Zeroable};
use dashu::float::FBig;
use dashu::integer::IBig;
use num::Complex;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::Ordering;
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

/// Glitched pixels whose exact orbits are computed (in parallel) to pick the
/// next reference in a glitch round.
const GLITCH_CANDIDATES: usize = 32;

/// Target GPU time per dispatch.  In-flight work cannot be cancelled, so this
/// bounds how long an interrupt waits; each dispatch also costs ~0.8 ms of
/// fixed overhead, so much smaller chunks would waste GPU time.
const TARGET_CHUNK_MS: f64 = 30.0;

/// Floor on chunk size, so a bad cost estimate can never produce tiny,
/// overhead-dominated dispatches.
const MIN_CHUNK_PX: usize = 65_536;

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
// Reference points
// ---------------------------------------------------------------------------

/// A perturbation reference: an exact point and its orbit as uploaded.
struct Reference {
    c:          Complex<FBig>,
    orbit:      Vec<[f32; 2]>,
    is_full:    bool,
    iterations: usize,
    /// Period, when `c` is a nucleus found by Newton.
    period:     Option<usize>,
}

impl Reference {
    fn new(c: Complex<FBig>, iterations: usize, period: Option<usize>) -> Self {
        let (_, orbit) = calculate_orbit(c.clone(), iterations);
        Self {
            orbit: orbit.orbit.iter().map(|z| [z.re as f32, z.im as f32]).collect(),
            is_full: orbit.is_full,
            c,
            iterations,
            period,
        }
    }

    /// A longer orbit covers more pixels; a full one covers all of them.
    fn beats(&self, other: &Reference) -> bool {
        self.orbit.len() > other.orbit.len()
    }

    fn describe(&self) -> String {
        let kind = match self.period {
            Some(p) => format!("nucleus p={p}"),
            None    => "point".to_string(),
        };
        format!("{kind} len {} full {}", self.orbit.len(), self.is_full)
    }
}

/// The view's centre and half-diagonal, at the precision reference search
/// needs for this depth.
struct ViewGeom {
    center: Complex<FBig>,
    radius: FBig,
    prec:   usize,
    upp:    i64,
}

impl ViewGeom {
    fn new(ctx: &PassBatchCtx) -> Self {
        let view = ctx.coords.view.inner;
        let (w, h) = (ctx.width as f64, ctx.height as f64);
        let prec = working_precision(ctx.depth);
        let half = |o: &FBig, px: f64| {
            (o + &FBig::try_from(px / 2.0 * view).unwrap()).with_precision(prec).value()
        };
        Self {
            center: Complex { re: half(&ctx.coords.origin.x, w), im: half(&ctx.coords.origin.y, h) },
            radius: FBig::try_from(view * w.hypot(h) / 2.0).unwrap(),
            prec,
            upp:    upp_log2(ctx.depth),
        }
    }

    /// Whether `c` is close enough to the view to serve as its reference.
    fn near(&self, c: &Complex<FBig>) -> bool {
        let dx = &c.re - &self.center.re;
        let dy = &c.im - &self.center.im;
        let max = &self.radius * FBig::from(MAX_REF_DIST);
        &dx * &dx + &dy * &dy <= &max * &max
    }
}

/// Position of `c` in the pass's pixel grid (units of `2^upp`, relative to
/// the tile-grid origin `x0, y0`).  Exact in FBig; only the O(1e4) result is
/// rounded to f64.
fn ref_px(c: &Complex<FBig>, ctx: &PassBatchCtx) -> (f64, f64) {
    let shift = -upp_log2(ctx.depth) as isize;
    let ts    = IBig::from(TILE_SIZE as u64);
    let px = |v: &FBig, grid0: &IBig| {
        ((v.clone() << shift) - FBig::from_parts(grid0 * &ts, 0)).to_f64().value()
    };
    (px(&c.re, &ctx.x0), px(&c.im, &ctx.y0))
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
    /// Best reference found so far, reused across passes and nearby views.
    reference:   Mutex<Option<Arc<Reference>>>,
    /// (centre, radius, iterations) of the last view whose centre-seeded
    /// nucleus search failed, so the passes of one view search only once.
    failed_search: Mutex<Option<(Complex<FBig>, FBig, usize)>>,
    /// Measured cost of the last chunk, in ms per pixel; sizes the next one.
    ms_per_px:   Mutex<f64>,
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

            GpuState {
                device, queue, pipeline, pipeline_fe, bgl, max_binding,
                reference: Mutex::new(None),
                failed_search: Mutex::new(None),
                ms_per_px: Mutex::new(1e-4),
            }
        })
    }

    /// Reference for a pass: a cached one if it is still near and full,
    /// otherwise a nucleus near the view centre, otherwise the longer of the
    /// centre's orbit and the cached one.  `None` if interrupted.
    fn initial_reference(
        &self,
        ctx: &PassBatchCtx,
        g:   &ViewGeom,
        int: &MultiInterrupter,
    ) -> Option<Arc<Reference>> {
        let iters  = ctx.iterations;
        let cached = self.reference.lock().clone().filter(|r| g.near(&r.c));
        if let Some(r) = &cached {
            if r.iterations == iters && r.is_full { return Some(r.clone()); }
            // A nucleus stays a nucleus when only the iteration count changed.
            if r.period.is_some() && r.iterations != iters {
                let r = Arc::new(Reference::new(r.c.clone(), iters, r.period));
                if r.is_full { return Some(self.remember(r, g)); }
            }
        }

        let key = (g.center.clone(), g.radius.clone(), iters);
        if self.failed_search.lock().as_ref() != Some(&key) {
            let t = std::time::Instant::now(); // DIAG
            match find_nucleus(&g.center, &g.radius, g.upp, iters, g.prec, &|| int.interrupted()) {
                Err(_) => return None,
                Ok(Some(n)) => {
                    let r = Arc::new(Reference::new(n.c, iters, Some(n.period)));
                    log::info!(
                        "[diag gpu] reference: {} from view centre in {:.1} ms",
                        r.describe(), t.elapsed().as_secs_f64() * 1e3,
                    );
                    if r.is_full { return Some(self.remember(r, g)); }
                }
                Ok(None) => {
                    log::info!(
                        "[diag gpu] reference: no nucleus from view centre ({:.1} ms)",
                        t.elapsed().as_secs_f64() * 1e3,
                    );
                    *self.failed_search.lock() = Some(key);
                }
            }
        }

        let centre = Arc::new(Reference::new(g.center.clone(), iters, None));
        let best = match cached {
            Some(r) if r.iterations == iters && r.beats(&centre) => r,
            _ => centre,
        };
        Some(self.remember(best, g))
    }

    /// Pick a new reference among glitched pixels: exact orbits for
    /// `candidates` in parallel, keep the longest, and if it still escapes
    /// look for a nucleus seeded from it.  `None` if interrupted.
    fn glitch_reference(
        &self,
        ctx:        &PassBatchCtx,
        g:          &ViewGeom,
        candidates: Vec<Complex<FBig>>,
        int:        &MultiInterrupter,
    ) -> Option<Arc<Reference>> {
        let iters = ctx.iterations;
        let mut best = Arc::new(
            candidates.into_par_iter()
                .map(|c| Reference::new(c, iters, None))
                .max_by_key(|r| r.orbit.len())?,
        );
        if int.interrupted() { return None; }
        if !best.is_full {
            match find_nucleus(&best.c, &g.radius, g.upp, iters, g.prec, &|| int.interrupted()) {
                Err(_) => return None,
                Ok(Some(n)) if g.near(&n.c) => {
                    let r = Arc::new(Reference::new(n.c, iters, Some(n.period)));
                    if r.beats(&best) { best = r; }
                }
                Ok(_) => {}
            }
        }
        Some(self.remember(best, g))
    }

    /// Cache `r` unless the cached reference still applies to this view and
    /// is strictly better.
    fn remember(&self, r: Arc<Reference>, g: &ViewGeom) -> Arc<Reference> {
        let mut cache = self.reference.lock();
        let keep = cache.as_ref().is_some_and(|c| {
            c.iterations == r.iterations && g.near(&c.c) && c.beats(&r)
        });
        if !keep { *cache = Some(r.clone()); }
        r
    }

    /// Pixels for the next chunk, from the last measured cost.
    fn chunk_len(&self) -> usize {
        ((TARGET_CHUNK_MS / *self.ms_per_px.lock()) as usize).max(MIN_CHUNK_PX)
    }

    /// Record a chunk's measured cost.  Tail chunks much smaller than the
    /// floor are dominated by fixed overhead and would skew the estimate.
    fn record_chunk(&self, n_px: usize, ms: f64) {
        if n_px >= MIN_CHUNK_PX / 4 {
            *self.ms_per_px.lock() = (ms / n_px as f64).max(1e-7);
        }
    }

    /// Dispatch pixels given as offsets from `r` in pixel units.
    fn dispatch_offsets(&self, offs: &[(f64, f64)], upp: i64, r: &Reference) -> Vec<u32> {
        if upp < FE_THRESHOLD {
            let d: Vec<[f32; 4]> = offs.iter().map(|&(x, y)| pack_fe(x, y, upp)).collect();
            self.dispatch_fe(&d, &r.orbit, r.is_full)
        } else {
            // upp >= FE_THRESHOLD, so 2^upp is a normal f64 and f32.
            let scale = (upp as f64).exp2();
            let d: Vec<[f32; 2]> = offs.iter()
                .map(|&(x, y)| [(x * scale) as f32, (y * scale) as f32])
                .collect();
            self.dispatch_f32(&d, &r.orbit, r.is_full)
        }
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
        let view    = ctx.coords.view.inner;

        // Beyond this depth the f32 seed underflows, so iterate the delta in
        // floatexp instead (see `dispatch_offsets`).  Offsets are always in
        // pixel units, O(1e4), so the f64 math never underflows.
        let upp    = upp_log2(ctx.depth);
        let use_fe = upp < FE_THRESHOLD;

        // ----------------------------------------------------------------
        // Collect all pixels that need computing this pass, in tile order.
        // Tiles arrive sorted by distance from the cursor, so each tile's
        // pixels are contiguous and chunks fill in outward from the cursor.
        // ----------------------------------------------------------------
        let mut pixel_refs: Vec<PixelRef> = Vec::new();

        for (ti, item) in tiles.iter().enumerate() {
            let tile   = &item.tile;
            let tile_i = i64::try_from(&(&tile.key.x - &ctx.x0))
                .expect("tile grid offset fits i64");
            let tile_j = i64::try_from(&(&tile.key.y - &ctx.y0))
                .expect("tile grid offset fits i64");

            for (r, c) in pass_pixels(pass) {
                let pixel_idx = r * TILE_SIZE + c;
                if tile.load(pixel_idx).get().is_some() { continue; } // already computed

                let col = tile_i * TILE_SIZE as i64 + c as i64;
                let row = tile_j * TILE_SIZE as i64 + r as i64;
                pixel_refs.push(PixelRef { tile_idx: ti, pixel_idx, col, row });
            }
        }

        // A tile finishes this pass as soon as every pixel it needed has been
        // stored — possibly mid-pass, so progress shows chunk by chunk.  A
        // pixel left glitched is never stored, so its tile stays unfinished
        // and the next generation retries it (never cache a guess).  Tiles
        // that have not finished the previous pass cannot finish this one.
        let mut remaining = vec![0usize; tiles.len()];
        for pr in &pixel_refs { remaining[pr.tile_idx] += 1; }
        let finish = |ti: usize| {
            let tile = &tiles[ti].tile;
            if tile.passes_done() >= pass { tile.finish_pass(pass + 1); }
        };
        for (ti, &n) in remaining.iter().enumerate() {
            if n == 0 { finish(ti); }
        }
        if pixel_refs.is_empty() {
            ctx.progress.fetch_add(1, Ordering::Release);
            return;
        }

        // Offsets of the given pixels from reference `r`, in pixel units.
        let offsets = |indices: &mut dyn Iterator<Item = usize>, r: &Reference| -> Vec<(f64, f64)> {
            let (rx, ry) = ref_px(&r.c, ctx);
            indices.map(|i| {
                let pr = &pixel_refs[i];
                (pr.col as f64 - rx, pr.row as f64 - ry)
            }).collect()
        };
        // Exact coordinate of a pixel, for CPU-side orbits.
        let ts = IBig::from(TILE_SIZE as u64);
        let geom = ViewGeom::new(ctx);
        let pixel_coord = |i: usize| {
            let pr   = &pixel_refs[i];
            let tile = &tiles[pr.tile_idx].tile;
            let c    = pr.pixel_idx % TILE_SIZE;
            let r    = pr.pixel_idx / TILE_SIZE;
            Complex {
                re: pixel_to_coord(&tile.key.x * &ts + IBig::from(c as u64), ctx.depth, geom.prec),
                im: pixel_to_coord(&tile.key.y * &ts + IBig::from(r as u64), ctx.depth, geom.prec),
            }
        };

        // ----------------------------------------------------------------
        // Reference for the view: ideally a nucleus, whose orbit never
        // escapes, so nothing glitches.
        // ----------------------------------------------------------------
        let t_ref = std::time::Instant::now(); // DIAG
        let Some(mut reference) = state.initial_reference(ctx, &geom, int) else { return };
        // DIAG
        {
            let (rx, ry) = ref_px(&reference.c, ctx);
            log::info!(
                "[diag gpu] pass {pass} depth {} upp 2^{upp} view {view:.4e} pipe {} | {} px | \
                 ref {} at px ({rx:.3}, {ry:.3}) in {:.1} ms",
                ctx.depth, if use_fe { "fe" } else { "f32" }, pixel_refs.len(),
                reference.describe(), ms(t_ref),
            );
        }

        // ----------------------------------------------------------------
        // Chunks: each dispatch is sized to take about TARGET_CHUNK_MS, so the
        // interrupt is honoured within that time (in-flight GPU work cannot
        // be cancelled) and results reach the screen as they arrive.
        // ----------------------------------------------------------------
        let mut pos    = 0;
        let mut chunks = 0usize; // DIAG
        while pos < pixel_refs.len() {
            if int.interrupted() { break; }
            let len   = state.chunk_len().min(pixel_refs.len() - pos);
            let chunk = pos..pos + len;

            let t_disp = std::time::Instant::now();
            let offs   = offsets(&mut chunk.clone(), &reference);
            let mut raw = state.dispatch_offsets(&offs, upp, &reference);
            let disp_ms = ms(t_disp);
            state.record_chunk(len, disp_ms);
            log::info!(
                "[diag gpu] pass {pass} chunk {chunks}: {len} px at {pos} in {disp_ms:.1} ms -> {}",
                raw_stats(&raw),
            );

            // Glitch rounds: a pixel is glitched only when it outlived the
            // reference's (escaping) orbit.  Re-dispatch those against a
            // longer-lived reference chosen among them.
            for round in 0..MAX_GLITCH_PASSES {
                if int.interrupted() { break; }
                let glitched: Vec<usize> = raw.iter().enumerate()
                    .filter(|&(_, &r)| r & GLITCH_BIT != 0)
                    .map(|(j, _)| j)
                    .collect();
                if glitched.is_empty() { break; }

                // Candidates spread evenly over the glitched pixels.
                let step = glitched.len().div_ceil(GLITCH_CANDIDATES);
                let candidates = glitched.iter().step_by(step).map(|&j| pixel_coord(pos + j)).collect();

                let t_gref = std::time::Instant::now(); // DIAG
                let Some(new_ref) = state.glitch_reference(ctx, &geom, candidates, int) else { break };
                let t_gref = ms(t_gref); // DIAG

                let t_gdisp = std::time::Instant::now(); // DIAG
                let offs    = offsets(&mut glitched.iter().map(|&j| pos + j), &new_ref);
                let new_raw = state.dispatch_offsets(&offs, upp, &new_ref);
                for (k, &j) in glitched.iter().enumerate() {
                    raw[j] = new_raw[k];
                }
                log::info!(
                    "[diag gpu] pass {pass} chunk {chunks} glitch round {round}: {} glitched, new ref {} \
                     (reference {t_gref:.1} ms, dispatch {:.1} ms) -> {}",
                    glitched.len(), new_ref.describe(), ms(t_gdisp), raw_stats(&new_raw),
                );
                // Later chunks start from the better reference.
                if new_ref.beats(&reference) { reference = new_ref; }
            }

            // Residual glitches: resolve exactly on the CPU, in parallel and
            // interruptibly.  Unresolved ones keep GLITCH_BIT and are not
            // stored; the resolved ones are exact and safe to cache.
            if !int.interrupted() {
                let residual: Vec<usize> = raw.iter().enumerate()
                    .filter(|&(_, &r)| r & GLITCH_BIT != 0)
                    .map(|(j, _)| j)
                    .collect();
                if !residual.is_empty() {
                    let t_resid = std::time::Instant::now(); // DIAG
                    let resolved: Vec<(usize, u32)> = residual.par_iter()
                        .filter_map(|&j| {
                            if int.interrupted() { return None; }
                            let (_, orbit) = calculate_orbit(pixel_coord(pos + j), ctx.iterations);
                            Some((j, check_orbit(&orbit).unwrap().map_or(0, |n| n.get() as u32)))
                        })
                        .collect();
                    log::info!(
                        "[diag gpu] pass {pass} chunk {chunks}: residual CPU resolve {}/{} in {:.1} ms",
                        resolved.len(), residual.len(), ms(t_resid),
                    );
                    for (j, r) in resolved { raw[j] = r; }
                }
            }

            // Store results, finish completed tiles, and tell the compositor.
            for (j, &r) in raw.iter().enumerate() {
                if r & GLITCH_BIT != 0 { continue; }
                let pr = &pixel_refs[pos + j];
                let color = val_to_color(NonZeroUsize::new(r as usize));
                tiles[pr.tile_idx].tile.store(pr.pixel_idx, color.into());
                remaining[pr.tile_idx] -= 1;
                if remaining[pr.tile_idx] == 0 { finish(pr.tile_idx); }
            }
            ctx.progress.fetch_add(1, Ordering::Release);

            pos    += len;
            chunks += 1;
        }

        log::info!(
            "[diag gpu] pass {pass} total {:.1} ms, {chunks} chunks, {pos}/{} px{}",
            ms(t_pass), pixel_refs.len(), if int.interrupted() { " (interrupted)" } else { "" },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WGSL is only compiled at runtime (when the GPU backend starts), so
    /// check here that both perturbation shaders parse and validate.
    #[test]
    fn shaders_validate() {
        for (name, src) in [
            ("perturbation", SHADER_SRC.to_string()),
            ("perturbation_floatexp", format!("{FE_HELPERS}\n{SHADER_FE_BODY}")),
        ] {
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::default())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
    use crate::rendering::{CoordinatesBox, View};
    use crate::support::Point;
    use crate::tiles::perturb::nucleus::newton_nucleus;
    use crate::tiles::store::tile_index;

    /// End-to-end check of the reference path the GPU uses, run on the CPU:
    /// in a deep view beside a minibrot, where the view centre itself
    /// escapes, the nucleus found from the view centre must give a full
    /// orbit, `ref_px` offsets must match exact pixel coordinates, and
    /// perturbation against it must reproduce exact per-pixel escape counts
    /// with no glitches.
    ///
    /// The perturbation here mirrors the shaders (with rebasing) but runs in
    /// f64, so that it tests the reference, not the working float: in f32
    /// (what the GPU uses) ~12% of these long-lived near-boundary pixels get
    /// a different count (~30% without rebasing) — see
    /// `diag_reference_precision`.
    #[test]
    fn nucleus_reference_matches_exact_orbits() {
        let depth = 40; // 2^-45 units per pixel: the minibrot is sub-pixel
        let prec  = working_precision(depth);
        let iters = 3000;
        let fp = |x: f64| FBig::try_from(x).unwrap().with_precision(prec).value();

        // A period-998 nucleus near the seahorse-valley point, to full precision.
        let seed = Complex { re: fp(-0.743_643_887_037_151), im: fp(0.131_825_904_205_330) };
        let rough = find_nucleus(&seed, &fp(1e-12), -50, 10_000, prec, &|| false).unwrap().unwrap();
        let tol2  = FBig::ONE << (2 * (upp_log2(depth) - 60)) as isize;
        let nuc   = newton_nucleus(&rough.c, rough.period, prec, &tol2, &fp(1.0), &|| false).unwrap().unwrap();

        // A 256x256 view (1 screen px = 1 depth px) whose centre is ~70 px
        // from the nucleus, so the reference is off-centre and off-grid.
        let (w, h) = (256usize, 256usize);
        let upp    = upp_log2(depth);
        let view   = (upp as f64).exp2();
        let origin = Point::new(
            &nuc.re - &(FBig::from(128 - 50) << upp as isize),
            &nuc.im - &(FBig::from(128 + 48) << upp as isize),
        );
        let ctx = PassBatchCtx {
            coords: CoordinatesBox { origin: origin.clone(), view: View::new(view) },
            depth,
            x0: tile_index(&origin.x, depth),
            y0: tile_index(&origin.y, depth),
            width: w,
            height: h,
            iterations: iters,
            progress: Default::default(),
        };
        let g = ViewGeom::new(&ctx);
        let n = find_nucleus(&g.center, &g.radius, g.upp, iters, g.prec, &|| false)
            .unwrap()
            .expect("nucleus from view centre");
        let r = Reference::new(n.c, iters, Some(n.period));
        assert!(r.is_full, "{}", r.describe());
        let centre = Reference::new(g.center.clone(), iters, None);
        eprintln!("reference {}; view centre alone: {}", r.describe(), centre.describe());
        // The case that used to glitch: the centre's own orbit escapes.
        assert!(!centre.is_full);

        // Sample pixels on a grid across the view.
        let (rx, ry) = ref_px(&r.c, &ctx);
        let ts = IBig::from(TILE_SIZE as u64);
        let (gx0, gy0) = (&ctx.x0 * &ts, &ctx.y0 * &ts);
        let pix: Vec<(i64, i64)> = (0..16).flat_map(|j| (0..16).map(move |i| (i * 16 + 3, j * 16 + 5))).collect();
        let (ref_orbit, _) = calculate_orbit(r.c.clone(), iters);

        let mismatches: Vec<_> = pix.par_iter().filter_map(|&(col, row)| {
            let exact = Complex {
                re: pixel_to_coord(&gx0 + IBig::from(col), depth, prec),
                im: pixel_to_coord(&gy0 + IBig::from(row), depth, prec),
            };
            // ref_px offset vs exact offset, in pixels.
            let (dx, dy) = (col as f64 - rx, row as f64 - ry);
            let ex = ((&exact.re - &r.c.re) << -upp as isize).to_f64().value();
            let ey = ((&exact.im - &r.c.im) << -upp as isize).to_f64().value();
            assert!((dx - ex).abs() < 1e-6 && (dy - ey).abs() < 1e-6, "offset ({dx},{dy}) vs ({ex},{ey})");

            let scale = (upp as f64).exp2();
            let got   = perturb_rebase::<f64>(&ref_orbit.orbit, (dx * scale, dy * scale), ref_orbit.is_full).0
                .expect("no glitch with a full reference");
            let (_, own) = calculate_orbit(exact, iters);
            let want  = check_orbit(&own).unwrap().map(|n| n.get());
            (got != want).then_some(((col, row), got, want))
        }).collect();

        eprintln!("{} / {} sampled pixels differ: {:?}", mismatches.len(), pix.len(), &mismatches[..mismatches.len().min(5)]);
        assert!(mismatches.len() <= pix.len() / 100, "too many mismatches");
    }

    /// Diagnostic: per-pixel perturbation in f32/f64 against a given orbit,
    /// counting Pauldelbrot's precision-loss condition |z|² < 1e-6·|X|².
    fn perturb<F: num::Float>(orbit: &[Complex<FBig>], d0: (f64, f64), full: bool) -> (Result<Option<usize>, ()>, bool) {
        perturb_x::<F>(orbit, d0, full, false)
    }

    /// Perturbation with rebasing (Zhuoran): when |z| < |δ|, restart against
    /// the start of the reference orbit.  In the X_0 = C convention the next
    /// value is z² + c = X_0 + (z² + δ₀), so δ ← z² + δ₀ at index 0.
    fn perturb_rebase<F: num::Float>(orbit: &[Complex<FBig>], d0: (f64, f64), full: bool) -> (Result<Option<usize>, ()>, usize) {
        let x: Vec<Complex<F>> = orbit.iter().map(|c| Complex {
            re: F::from(c.re.to_f64().value()).unwrap(), im: F::from(c.im.to_f64().value()).unwrap(),
        }).collect();
        let d0 = Complex { re: F::from(d0.0).unwrap(), im: F::from(d0.1).unwrap() };
        let two = F::from(2.0).unwrap();
        let (mut d, mut m, mut rebases) = (d0, 0usize, 0usize);
        let n = orbit.len() - 1; // iterations: orbit holds X_0..=X_n when full
        for i in 0..orbit.len() {
            let z = x[m] + d;
            if z.norm_sqr() > F::from(4.0).unwrap() { return (Ok(Some(i + 1)), rebases); }
            if i == n { break; }
            if z.norm_sqr() < d.norm_sqr() {
                d = z * z + d0;
                m = 0;
                rebases += 1;
            } else {
                d = x[m] * d * two + d * d + d0;
                m += 1;
                if m >= x.len() { return (if full { Ok(None) } else { Err(()) }, rebases); }
            }
        }
        (if full { Ok(None) } else { Err(()) }, rebases)
    }

    /// As `perturb`, optionally rounding the reference orbit to f32 first.
    fn perturb_x<F: num::Float>(orbit: &[Complex<FBig>], d0: (f64, f64), full: bool, x_f32: bool) -> (Result<Option<usize>, ()>, bool) {
        let rnd = |v: f64| if x_f32 { v as f32 as f64 } else { v };
        let x: Vec<Complex<F>> = orbit.iter().map(|c| Complex {
            re: F::from(rnd(c.re.to_f64().value())).unwrap(), im: F::from(rnd(c.im.to_f64().value())).unwrap(),
        }).collect();
        let d0 = Complex { re: F::from(d0.0).unwrap(), im: F::from(d0.1).unwrap() };
        let mut d = d0;
        let mut pauldel = false;
        let two = F::from(2.0).unwrap();
        for (i, &xn) in x.iter().enumerate() {
            let z = xn + d;
            let zn = z.norm_sqr();
            if zn > F::from(4.0).unwrap() { return (Ok(Some(i + 1)), pauldel); }
            if zn < F::from(1e-6).unwrap() * xn.norm_sqr() { pauldel = true; }
            d = xn * d * two + d * d + d0;
        }
        (if full { Ok(None) } else { Err(()) }, pauldel)
    }

    /// Measurement, not a pass/fail test: escape-count accuracy of
    /// perturbation against the nucleus vs the (escaping) view-centre
    /// reference, in f32 / f64, with and without rebasing.  Run with
    /// `--ignored --nocapture`.  Measured 2026-09-23 (nucleus, 256 px):
    /// f32 76 wrong, f64 1, f32+rebase 30, f64+rebase 0.
    #[test]
    #[ignore]
    fn diag_reference_precision() {
        let depth = 40;
        let prec  = working_precision(depth);
        let iters = 3000;
        let fp = |x: f64| FBig::try_from(x).unwrap().with_precision(prec).value();
        let seed = Complex { re: fp(-0.743_643_887_037_151), im: fp(0.131_825_904_205_330) };
        let rough = find_nucleus(&seed, &fp(1e-12), -50, 10_000, prec, &|| false).unwrap().unwrap();
        let upp  = upp_log2(depth);
        let tol2 = FBig::ONE << (2 * (upp - 60)) as isize;
        let nuc  = newton_nucleus(&rough.c, rough.period, prec, &tol2, &fp(1.0), &|| false).unwrap().unwrap();
        let origin = Point::new(
            &nuc.re - &(FBig::from(128 - 50) << upp as isize),
            &nuc.im - &(FBig::from(128 + 48) << upp as isize),
        );
        let ctx = PassBatchCtx {
            coords: CoordinatesBox { origin: origin.clone(), view: View::new((upp as f64).exp2()) },
            depth, x0: tile_index(&origin.x, depth), y0: tile_index(&origin.y, depth),
            width: 256, height: 256, iterations: iters, progress: Default::default(),
        };
        let g = ViewGeom::new(&ctx);
        let ts = IBig::from(TILE_SIZE as u64);
        let (gx0, gy0) = (&ctx.x0 * &ts, &ctx.y0 * &ts);
        let pix: Vec<(i64, i64)> = (0..16).flat_map(|j| (0..16).map(move |i| (i * 16 + 3, j * 16 + 5))).collect();
        let exact: Vec<Option<usize>> = pix.par_iter().map(|&(col, row)| {
            let c = Complex {
                re: pixel_to_coord(&gx0 + IBig::from(col), depth, prec),
                im: pixel_to_coord(&gy0 + IBig::from(row), depth, prec),
            };
            check_orbit(&calculate_orbit(c, iters).1).unwrap().map(|n| n.get())
        }).collect();

        for (name, refc) in [("nucleus", nuc.clone()), ("centre", g.center.clone())] {
            let (orbit, _) = calculate_orbit(refc.clone(), iters);
            let (rx, ry) = ref_px(&refc, &ctx);
            let scale = (upp as f64).exp2();
            for fname in ["f32", "f64", "f32 rebase", "f64 rebase"] {
                let (mut wrong, mut glitched, mut pd, mut wrong_pd) = (0, 0, 0, 0);
                for (k, &(col, row)) in pix.iter().enumerate() {
                    let d0 = ((col as f64 - rx) * scale, (row as f64 - ry) * scale);
                    let (res, p) = match fname {
                        "f32" => perturb::<f32>(&orbit.orbit, d0, orbit.is_full),
                        "f64" => perturb::<f64>(&orbit.orbit, d0, orbit.is_full),
                        "f32 rebase" => (perturb_rebase::<f32>(&orbit.orbit, d0, orbit.is_full).0, false),
                        "f64 rebase" => (perturb_rebase::<f64>(&orbit.orbit, d0, orbit.is_full).0, false),
                        _     => perturb_x::<f64>(&orbit.orbit, d0, orbit.is_full, true),
                    };
                    if p { pd += 1; }
                    match res {
                        Err(()) => glitched += 1,
                        Ok(v) if v != exact[k] => { wrong += 1; if p { wrong_pd += 1; } }
                        Ok(_) => {}
                    }
                }
                eprintln!("{name:8} {fname}: orbit len {} | glitched {glitched} | wrong {wrong} (of which pauldelbrot {wrong_pd}) | pauldelbrot-flagged {pd}",
                    orbit.orbit.len());
            }
        }
    }
}
