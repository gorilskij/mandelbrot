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

use super::bla::BlaTable;
use super::nucleus::{MAX_REF_DIST, find_nucleus};
use super::{PassBatchCtx, Perturbator, TileItem};
use crate::rendering::{calculate_orbit, check_orbit};
use crate::tiles::store::{TILE_SIZE, pass_pixels, pixel_to_coord, upp_log2, working_precision};
use bytemuck::{Pod, Zeroable};
use dashu::float::FBig;
use dashu::integer::IBig;
use num::Complex;
use parking_lot::Mutex;
use rayon::prelude::*;
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
/// Shared by both shaders: BLA table lookup/apply and interior detection.
const COMMON: &str = include_str!("shaders/perturb_common.wgsl");

/// Full source of a perturbation shader: floatexp helpers, the shared code,
/// then `body`.
fn shader_source(body: &str) -> String {
    format!("{FE_HELPERS}\n{COMMON}\n{body}")
}

/// Use BLA (see bla.rs). Off = plain iteration, for A/B tests.
const USE_BLA: bool = true;

// ---------------------------------------------------------------------------
// CPU-side types
// ---------------------------------------------------------------------------

const GLITCH_BIT: u32 = 0x80000000;
const MAX_GLITCH_PASSES: usize = 8;

/// Glitched pixels whose exact orbits are computed (in parallel) to pick the
/// next reference in a glitch round.
const GLITCH_CANDIDATES: usize = 32;

/// Target GPU time per dispatch.  In-flight work cannot be cancelled, so this
/// bounds how long an interrupt waits.  It must also stay well below the
/// point where macOS kills a long-running command buffer (see NOT_RUN): at
/// 2^-305 with 32768 iterations a 65k-pixel dispatch (~400 ms) was killed at
/// random, 16k-pixel ones (~135 ms) never were.
const TARGET_CHUNK_MS: f64 = 30.0;

/// Floor on chunk size: each dispatch costs ~0.8 ms of fixed overhead, so
/// fewer pixels would be overhead-dominated for cheap pixels (expensive
/// pixels near a deep minibrot cost ~0.06 ms each, so even this is ~60 ms).
const MIN_CHUNK_PX: usize = 1024;

/// First chunk of every pass: a small probe that measures the cost of the
/// pixels nearest the cursor (usually the most expensive) before sizing the
/// rest; the previous pass ended far from the cursor, where pixels are cheap.
const PROBE_CHUNK_PX: usize = 2048;

/// Output-buffer fill value the shaders never write (they write 0, n, or
/// GLITCH_BIT | n with n <= orbit_len).  Still present after a dispatch
/// means the GPU did not run it (macOS kills command buffers that run too
/// long, and drops some following ones while it recovers; wgpu reports
/// neither).  Such pixels are retried in smaller dispatches, and if that
/// fails too they are left uncomputed.
const NOT_RUN: u32 = u32::MAX;

/// Smallest dispatch a NOT_RUN chunk is split into when retrying.
const MIN_RETRY_PX: usize = 256;

/// Interior detection (with a nucleus reference of period p): every window of
/// `p·ceil(INTERIOR_MIN_WINDOW/p)` iterations, compare log2|dz/dz₀|² with its
/// value a window earlier. Inside a component the cycle's multiplier |λ| < 1,
/// so it keeps shrinking; just outside, |λ| ≥ 1. A pixel whose derivative
/// shrank by at least INTERIOR_Q per period over INTERIOR_WINDOWS consecutive
/// windows is declared in the set. Windows much shorter than ~64 iterations
/// let escaping pixels pass (brief contractions near 0); measured with
/// `diag_interior_detection`: 0 false positives over 6 views at these values.
const INTERIOR_MIN_WINDOW: usize = 128;
const INTERIOR_WINDOWS:    u32   = 2;
const INTERIOR_Q:          f64   = 0.9;

/// How far (in view half-diagonals) a cached full reference (a nucleus) may
/// be from the view and still be reused, so zooming does not search again
/// (1-4 s at 2^-320) until the nucleus is this far out. Measured
/// (`diag_far_reference`, 2^-314..2^-330, 262144 iterations): a nucleus
/// 3500 radii away was as accurate as at 13 (vs exact: 122 vs 138 of 600
/// off by a few iterations, 3 vs 5 by >50) and as fast. The limit is f32:
/// pixel offsets have a 24-bit mantissa, so positions are quantized to
/// ~distance·2^-24; 1024 radii (~2^21 px) is ~0.1 px, while at 56000 radii
/// (2^26 px, 4 px) 26 of 600 were wrong vs 0 with the view's own nucleus.
/// (Views further out scored 0 only because they were nearly uniform.)
const MAX_REF_REUSE_DIST: u32 = 1024;

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
    /// Interior-detection window in iterations; 0 disables it.
    interior_window:      u32,
    /// log2 of the squared contraction a window must show.
    interior_contraction: f32,
    /// Consecutive contracting windows needed.
    interior_windows:     u32,
    /// BLA table: number of levels (0 disables BLA) and log2 of level 0's size.
    bla_levels:           u32,
    bla_log2_size:        u32,
    _pad:                 [u32; 3],
}

// ---------------------------------------------------------------------------
// Reference points
// ---------------------------------------------------------------------------

/// A perturbation reference: an exact point and its orbit as uploaded.
struct Reference {
    c:          Complex<FBig>,
    orbit:      Vec<[f32; 2]>,
    /// The orbit as floatexp seeds (`pack_fe`), for the deep pipeline: a
    /// deep orbit comes far closer to 0 than f32 reaches (a nucleus of
    /// period p passes within ~2^-149 and ~2^-271 of 0 at the periods of
    /// its parents), and flushing those values to 0 there drops the pixel's
    /// linear term, so pixels shadow the reference and never escape.
    orbit_fe:   Vec<[f32; 4]>,
    /// The same orbit in f64, for building BLA tables (f64's range keeps
    /// the tiny values near the orbit's zeros that f32 flushes to 0).
    orbit64:    Vec<Complex<f64>>,
    is_full:    bool,
    iterations: usize,
    /// Period, when `c` is a nucleus found by Newton.
    period:     Option<usize>,
}

impl Reference {
    fn new(c: Complex<FBig>, iterations: usize, period: Option<usize>) -> Self {
        let (exact, orbit) = calculate_orbit(c.clone(), iterations);
        let orbit64: Vec<Complex<f64>> = exact.orbit.iter()
            .map(|z| Complex { re: z.re.to_f64().value(), im: z.im.to_f64().value() })
            .collect();
        Self {
            orbit: orbit.orbit.iter().map(|z| [z.re as f32, z.im as f32]).collect(),
            orbit_fe: orbit64.iter().map(|z| pack_fe(z.re, z.im, 0)).collect(),
            orbit64,
            is_full: orbit.is_full,
            c,
            iterations,
            period,
        }
    }

    /// (window, log2 contraction threshold) for interior detection; only a
    /// nucleus has a known period (window 0 disables detection).
    fn interior(&self) -> (u32, f32) {
        match self.period {
            Some(p) if p > 0 && self.is_full => {
                let periods = INTERIOR_MIN_WINDOW.div_ceil(p);
                ((p * periods) as u32, (2.0 * INTERIOR_Q.log2() * periods as f64) as f32)
            }
            _ => (0, 0.0),
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

    /// Whether a search from the view (centre, radius) would say about this
    /// view what it said about its own: our centre lies within it and the
    /// radii differ by less than 2×.
    fn covered_by(&self, (center, radius, _): &(Complex<FBig>, FBig, usize)) -> bool {
        let dx = &self.center.re - &center.re;
        let dy = &self.center.im - &center.im;
        &dx * &dx + &dy * &dy <= radius * radius
            && &self.radius * FBig::from(2) > *radius
            && &self.radius < &(radius * FBig::from(2))
    }

    /// Whether `c` is close enough to the view to serve as its reference.
    fn near(&self, c: &Complex<FBig>) -> bool {
        self.within(c, MAX_REF_DIST)
    }

    /// Whether the cached reference `r` may be reused for this view: a full
    /// one (a nucleus) much further away than a search would look.
    fn reusable(&self, r: &Reference) -> bool {
        self.within(&r.c, if r.is_full { MAX_REF_REUSE_DIST } else { MAX_REF_DIST })
    }

    fn within(&self, c: &Complex<FBig>, radii: u32) -> bool {
        let dx = &c.re - &self.center.re;
        let dy = &c.im - &self.center.im;
        let max = &self.radius * FBig::from(radii);
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

/// A reference ready for dispatching: its orbit and BLA table uploaded once
/// (shared by every chunk of a pass), plus the uniform values.
struct GpuRef {
    orbit_buf:     wgpu::Buffer,
    bla_buf:       wgpu::Buffer,
    orbit_len:     u32,
    is_full:       bool,
    interior:      (u32, f32),
    bla_levels:    u32,
    bla_log2_size: u32,
}

/// log2 of the largest |δ₀| among pixel offsets `offs` (pixel units at
/// 2^upp), with half a bit of margin; bounds the BLA table's radii.
fn log2_dc(offs: &[(f64, f64)], upp: i64) -> f64 {
    let max = offs.iter().map(|&(x, y)| x.hypot(y)).fold(0.0, f64::max);
    max.max(1.0).log2() + 0.5 + upp as f64
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
    /// nucleus search failed, so the passes of one view, and nearby views
    /// (`ViewGeom::covered_by`), search only once.
    failed_search: Mutex<Option<(Complex<FBig>, FBig, usize)>>,
    /// The same for searches seeded from glitched pixels (each costs as
    /// much as a centre search; seen: ~1 s per glitch round, all failing).
    failed_glitch_search: Mutex<Option<(Complex<FBig>, FBig, usize)>>,
    /// Nucleus search running in the background (see `SearchJob`).
    search:      Mutex<Option<SearchJob>>,
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
                    bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
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

            let pipeline    = make_pipeline("perturbation", shader_source(SHADER_SRC));
            let pipeline_fe = make_pipeline("perturbation_fe", shader_source(SHADER_FE_BODY));

            GpuState {
                device, queue, pipeline, pipeline_fe, bgl, max_binding,
                reference: Mutex::new(None),
                failed_search: Mutex::new(None),
                failed_glitch_search: Mutex::new(None),
                search: Mutex::new(None),
                ms_per_px: Mutex::new(1e-4),
            }
        })
    }

    /// Reference for a pass, without waiting for a nucleus search: a cached
    /// full one within `MAX_REF_REUSE_DIST`, else a background search's
    /// result for this view, else (starting that search) a provisional one:
    /// the longer of the view centre's orbit and a cached one within reach.
    /// Not a cached nucleus further out: pixel offsets have a 24-bit
    /// mantissa, so positions are quantized to ~distance·2^-24 (4 px at 2^26
    /// px, seen as 26 of 600 wrong vs 0). The flag says whether a search is
    /// still running for this view (`search_result` tells when it is done).
    /// `None` if interrupted.
    fn initial_reference(
        &self,
        ctx: &PassBatchCtx,
        g:   &ViewGeom,
        int: &MultiInterrupter,
    ) -> Option<(Arc<Reference>, bool)> {
        let iters  = ctx.iterations;
        let mut cached = self.reference.lock().clone();
        if let Some(r) = &cached {
            if r.iterations == iters && r.is_full && g.reusable(r) { return Some((r.clone(), false)); }
            // A nucleus stays a nucleus when only the iteration count changed.
            if r.period.is_some() && r.iterations != iters && g.reusable(r) {
                let r = Arc::new(Reference::new(r.c.clone(), iters, r.period));
                if int.interrupted() { return None; }
                if r.is_full { return Some((self.remember(r, g), false)); }
            }
        }
        cached = cached.filter(|r| r.iterations == iters);

        match self.search_result(g, iters) {
            Some(Some(r)) => return Some((self.remember(r, g), false)),
            Some(None) => { // failed: remember, fall back below
                *self.failed_search.lock() = Some((g.center.clone(), g.radius.clone(), iters));
            }
            None => {
                let failed = self.failed_search.lock().as_ref().is_some_and(|f| f.2 == iters && g.covered_by(f));
                if !failed { self.start_search(g, iters); }
            }
        }
        let pending = self.search_result(g, iters).is_none() && self.search_covers(g, iters);

        let centre = Arc::new(Reference::new(g.center.clone(), iters, None));
        if int.interrupted() { return None; }
        let best = match cached {
            Some(r) if r.beats(&centre) && g.reusable(&r) => r,
            _ => centre,
        };
        Some((self.remember(best, g), pending))
    }

    /// Whether the background search (running or done) is for this view.
    fn search_covers(&self, g: &ViewGeom, iters: usize) -> bool {
        self.search.lock().as_ref().is_some_and(|j| j.key.2 == iters && g.covered_by(&j.key))
    }

    /// The background search's outcome for this view: `None` if there is
    /// none or it is still running, `Some(None)` if it found no full nucleus.
    fn search_result(&self, g: &ViewGeom, iters: usize) -> Option<Option<Arc<Reference>>> {
        let job = self.search.lock();
        let job = job.as_ref().filter(|j| j.key.2 == iters && g.covered_by(&j.key))?;
        job.result.lock().clone()
    }

    /// Start a nucleus search for this view in a background thread, unless
    /// one for it is already running or done; cancels one for another view.
    fn start_search(&self, g: &ViewGeom, iters: usize) {
        let mut job = self.search.lock();
        if job.as_ref().is_some_and(|j| j.key.2 == iters && g.covered_by(&j.key)) { return; }
        if let Some(old) = job.take() { old.cancel.store(true, Ordering::Relaxed); }
        let key = (g.center.clone(), g.radius.clone(), iters);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let result = Arc::new(Mutex::new(None));
        let (center, radius, upp, prec) = (g.center.clone(), g.radius.clone(), g.upp, g.prec);
        let (c2, r2) = (cancel.clone(), result.clone());
        std::thread::spawn(move || {
            let t = std::time::Instant::now(); // DIAG
            let cancelled = || c2.load(Ordering::Relaxed);
            let found = match find_nucleus(&center, &radius, upp, iters, prec, &cancelled) {
                Err(_) => return,
                Ok(n) => n.map(|n| Arc::new(Reference::new(n.c, iters, Some(n.period)))),
            };
            if cancelled() { return; }
            log::info!(
                "[diag gpu] background search: {} in {:.1} ms",
                found.as_ref().map_or("no nucleus".to_string(), |r| r.describe()),
                t.elapsed().as_secs_f64() * 1e3,
            );
            *r2.lock() = Some(found.filter(|r| r.is_full));
        });
        log::info!("[diag gpu] background search started");
        *job = Some(SearchJob { key, cancel, result });
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
        let failed = self.failed_glitch_search.lock().as_ref().is_some_and(|f| f.2 == iters && g.covered_by(f));
        if !best.is_full && !failed {
            let found = match find_nucleus(&best.c, &g.radius, g.upp, iters, g.prec, &|| int.interrupted()) {
                Err(_) => return None,
                Ok(Some(n)) if g.near(&n.c) => {
                    let r = Arc::new(Reference::new(n.c, iters, Some(n.period)));
                    let full = r.is_full;
                    if r.beats(&best) { best = r; }
                    full
                }
                Ok(_) => false,
            };
            if !found {
                *self.failed_glitch_search.lock() = Some((g.center.clone(), g.radius.clone(), iters));
            }
        }
        Some(self.remember(best, g))
    }

    /// Cache `r` unless the cached reference still applies to this view and
    /// is strictly better.
    fn remember(&self, r: Arc<Reference>, g: &ViewGeom) -> Arc<Reference> {
        let mut cache = self.reference.lock();
        let keep = cache.as_ref().is_some_and(|c| {
            c.iterations == r.iterations && g.reusable(c) && c.beats(&r)
        });
        if !keep { *cache = Some(r.clone()); }
        r
    }

    /// Upload `r` for dispatching pixels with |δ₀| ≤ 2^log2_dc at depth
    /// `upp` (which picks the pipeline, and so the orbit's format).
    fn prepare(&self, r: &Reference, log2_dc: f64, upp: i64) -> GpuRef {
        self.prepare_with(r, log2_dc, upp, USE_BLA)
    }

    fn prepare_with(&self, r: &Reference, log2_dc: f64, upp: i64, bla: bool) -> GpuRef {
        let table = if bla { BlaTable::build(&r.orbit64, log2_dc) } else { BlaTable::disabled() };
        let buf = |bytes: &[u8]| self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytes, usage: wgpu::BufferUsages::STORAGE,
        });
        GpuRef {
            orbit_buf:     if upp < FE_THRESHOLD {
                buf(bytemuck::cast_slice(&r.orbit_fe))
            } else {
                buf(bytemuck::cast_slice(&r.orbit))
            },
            bla_buf:       buf(bytemuck::cast_slice(&table.entries)),
            orbit_len:     r.orbit.len() as u32,
            is_full:       r.is_full,
            interior:      r.interior(),
            bla_levels:    table.levels,
            bla_log2_size: table.log2_size,
        }
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
    fn dispatch_offsets(&self, offs: &[(f64, f64)], upp: i64, r: &GpuRef) -> Vec<u32> {
        let d = Deltas::pack(offs, upp);
        self.dispatch(self.pipeline_for(&d), &d.bytes, d.n, d.elem(), r)
    }

    fn pipeline_for(&self, d: &Deltas) -> &wgpu::ComputePipeline {
        if d.fe { &self.pipeline_fe } else { &self.pipeline }
    }

    /// Submit packed seeds against `r` without waiting for them, so the CPU
    /// can post-process the previous chunk meanwhile; `finish` reads them
    /// back.
    fn submit_deltas(&self, d: Deltas, r: Arc<GpuRef>) -> PendingChunk {
        let max_pixels = ((self.max_binding / d.elem()) * 9 / 10).max(1);
        let parts = d.bytes.chunks(max_pixels * d.elem())
            .filter_map(|b| self.submit_once(self.pipeline_for(&d), b, b.len() / d.elem(), &r))
            .collect();
        PendingChunk { deltas: d, gref: r, parts, t_submit: std::time::Instant::now() }
    }

    /// Wait for a submitted chunk and read it back, retrying pixels the GPU
    /// did not run (see `dispatch_chunk`).
    fn finish(&self, p: PendingChunk) -> Vec<u32> {
        let (d, elem) = (&p.deltas, p.deltas.elem());
        let mut out = Vec::with_capacity(d.n);
        let mut at = 0;
        for part in p.parts {
            let n = part.n;
            let bytes = &d.bytes[at * elem..(at + n) * elem];
            let got = self.wait(part);
            out.extend(self.retry_not_run(self.pipeline_for(d), bytes, n, &p.gref, got));
            at += n;
        }
        out
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
        r:           &GpuRef,
    ) -> Vec<u32> {
        if n == 0 { return Vec::new(); }
        let max_pixels = ((self.max_binding / elem) * 9 / 10).max(1);
        if n <= max_pixels {
            return self.dispatch_chunk(pipeline, delta_bytes, n, r);
        }
        let mut out = Vec::with_capacity(n);
        for chunk in delta_bytes.chunks(max_pixels * elem) {
            out.extend(self.dispatch_chunk(pipeline, chunk, chunk.len() / elem, r));
        }
        out
    }

    /// One dispatch over a chunk small enough to fit the binding-size limit,
    /// retrying pixels the GPU did not run (NOT_RUN) in halves down to
    /// MIN_RETRY_PX.  Pixels still not run are returned as NOT_RUN.
    fn dispatch_chunk(
        &self,
        pipeline:    &wgpu::ComputePipeline,
        delta_bytes: &[u8],
        n_pixels:    usize,
        r:           &GpuRef,
    ) -> Vec<u32> {
        let out = self.dispatch_once(pipeline, delta_bytes, n_pixels, r);
        self.retry_not_run(pipeline, delta_bytes, n_pixels, r, out)
    }

    /// Given the results `out` of a dispatch, redo the pixels it did not run.
    fn retry_not_run(
        &self,
        pipeline:    &wgpu::ComputePipeline,
        delta_bytes: &[u8],
        n_pixels:    usize,
        r:           &GpuRef,
        out:         Vec<u32>,
    ) -> Vec<u32> {
        let missing = out.iter().filter(|&&v| v == NOT_RUN).count();
        if missing == 0 {
            return out;
        }
        log::warn!("GPU did not run {missing}/{n_pixels} px of a dispatch; retrying in halves");
        if n_pixels <= MIN_RETRY_PX {
            let retry = self.dispatch_once(pipeline, delta_bytes, n_pixels, r);
            return out.into_iter().zip(retry).map(|(a, b)| if a == NOT_RUN { b } else { a }).collect();
        }
        let elem = delta_bytes.len() / n_pixels;
        let half = n_pixels / 2;
        let mut merged = self.dispatch_chunk(pipeline, &delta_bytes[..half * elem], half, r);
        merged.extend(self.dispatch_chunk(pipeline, &delta_bytes[half * elem..], n_pixels - half, r));
        out.into_iter().zip(merged).map(|(a, b)| if a == NOT_RUN { b } else { a }).collect()
    }

    /// A single dispatch; pixels the GPU did not run come back as NOT_RUN.
    fn dispatch_once(
        &self,
        pipeline:    &wgpu::ComputePipeline,
        delta_bytes: &[u8],
        n_pixels:    usize,
        r:           &GpuRef,
    ) -> Vec<u32> {
        self.submit_once(pipeline, delta_bytes, n_pixels, r).map_or_else(Vec::new, |f| self.wait(f))
    }

    /// Submit a single dispatch (`None` for no pixels); `wait` reads it back.
    fn submit_once(
        &self,
        pipeline:    &wgpu::ComputePipeline,
        delta_bytes: &[u8],
        n_pixels:    usize,
        r:           &GpuRef,
    ) -> Option<InFlight> {
        let n = n_pixels as u32;
        let (interior_window, interior_contraction) = r.interior;
        if n == 0 { return None; }

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
                orbit_len:   r.orbit_len,
                is_full:     r.is_full as u32,
                dispatch_w,
                interior_window,
                interior_contraction,
                interior_windows: INTERIOR_WINDOWS,
                bla_levels:    r.bla_levels,
                bla_log2_size: r.bla_log2_size,
                _pad: [0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let deltas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: delta_bytes,
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let output_size = n as u64 * 4;
        // Pre-filled with NOT_RUN so a dispatch the GPU dropped is detectable.
        let output_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: &vec![0xFFu8; output_size as usize],
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
                wgpu::BindGroupEntry { binding: 2, resource: r.orbit_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: r.bla_buf.as_entire_binding() },
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
        let index = queue.submit([enc.finish()]);
        staging_buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        Some(InFlight { staging: staging_buf, index, n: n_pixels, t_submit })
    }

    /// Wait for one submitted dispatch (only that one: later submissions
    /// keep the GPU busy meanwhile) and read back its results.
    fn wait(&self, f: InFlight) -> Vec<u32> {
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(f.index), timeout: None }).unwrap();
        log::info!( // DIAG
            "[diag gpu] dispatch {} px done {:.1} ms after submit", f.n, f.t_submit.elapsed().as_secs_f64() * 1e3,
        );
        let staging_buf = f.staging;
        let slice = staging_buf.slice(..);
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

/// Per-pixel seeds packed for one of the two pipelines: `[f32; 2]` deltas
/// (plain f32) or `[f32; 4]` floatexp seeds (mantissa.xy, exp, _).
struct Deltas {
    bytes: Vec<u8>,
    n:     usize,
    fe:    bool,
}

impl Deltas {
    /// Pack offsets in pixel units at depth `upp`.
    fn pack(offs: &[(f64, f64)], upp: i64) -> Self {
        let fe = upp < FE_THRESHOLD;
        let bytes = if fe {
            let d: Vec<[f32; 4]> = offs.iter().map(|&(x, y)| pack_fe(x, y, upp)).collect();
            bytemuck::cast_slice(&d).to_vec()
        } else {
            // upp >= FE_THRESHOLD, so 2^upp is a normal f64 and f32.
            let scale = (upp as f64).exp2();
            let d: Vec<[f32; 2]> = offs.iter()
                .map(|&(x, y)| [(x * scale) as f32, (y * scale) as f32])
                .collect();
            bytemuck::cast_slice(&d).to_vec()
        };
        Self { bytes, n: offs.len(), fe }
    }

    fn elem(&self) -> usize { if self.fe { 16 } else { 8 } }
}

/// A nucleus search for one view (centre, radius, iterations) running in a
/// background thread, so rendering can start with a provisional reference.
/// `result`: `None` while running, then `Some(None)` (no full nucleus) or
/// `Some(Some(r))`.
struct SearchJob {
    key:    (Complex<FBig>, FBig, usize),
    cancel: Arc<std::sync::atomic::AtomicBool>,
    result: Arc<Mutex<Option<Option<Arc<Reference>>>>>,
}

/// One submitted dispatch, not yet read back.
struct InFlight {
    staging:  wgpu::Buffer,
    index:    wgpu::SubmissionIndex,
    n:        usize,
    t_submit: std::time::Instant, // DIAG
}

/// A chunk submitted with `submit_offsets`: its seeds and reference are
/// kept for retrying pixels the GPU did not run.
struct PendingChunk {
    deltas:   Deltas,
    gref:     Arc<GpuRef>,
    parts:    Vec<InFlight>,
    t_submit: std::time::Instant,
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
        let Some((mut reference, mut searching)) = state.initial_reference(ctx, &geom, int) else { return };
        // Bound on |δ₀| over the whole pass, for the BLA table's radii.
        let pass_dc = |r: &Reference| {
            let (rx, ry) = ref_px(&r.c, ctx);
            let max = pixel_refs.iter()
                .map(|pr| (pr.col as f64 - rx).hypot(pr.row as f64 - ry))
                .fold(0.0, f64::max);
            log2_dc(&[(max, 0.0)], upp)
        };
        let t_prep = std::time::Instant::now(); // DIAG
        let mut gref = Arc::new(state.prepare(&reference, pass_dc(&reference), upp));
        log::info!("[diag gpu] pass {pass}: reference uploaded (BLA {} levels) in {:.1} ms", gref.bla_levels, ms(t_prep));
        // DIAG
        {
            let (rx, ry) = ref_px(&reference.c, ctx);
            log::info!(
                "[diag gpu] pass {pass} depth {} upp 2^{upp} view {view:.4e} pipe {} | {} px | \
                 ref {} at px ({rx:.3}, {ry:.3}){} in {:.1} ms",
                ctx.depth, if use_fe { "fe" } else { "f32" }, pixel_refs.len(),
                reference.describe(), if searching { " (provisional)" } else { "" }, ms(t_ref),
            );
        }

        // Dispatch pixels `ids` against `r`, in chunks of ~TARGET_CHUNK_MS
        // (glitched or deferred pixels can be many: keep dispatches short).
        // Pixels left when interrupted come back as NOT_RUN.
        let dispatch_ids = |ids: &[usize], r: &Reference, gr: &GpuRef| -> Vec<u32> {
            let mut out = Vec::with_capacity(ids.len());
            for part in ids.chunks(state.chunk_len()) {
                if int.interrupted() { out.resize(ids.len(), NOT_RUN); break; }
                let t = std::time::Instant::now();
                out.extend(state.dispatch_offsets(&offsets(&mut part.iter().copied(), r), upp, gr));
                state.record_chunk(part.len(), ms(t));
            }
            out
        };

        // Glitch rounds, residual CPU resolve and storing for pixels `ids`
        // (indices into `pixel_refs`) with GPU results `raw`.
        let resolve_and_store = |ids: &[usize], raw: &mut [u32], reference: &mut Arc<Reference>,
                                 gref: &mut Arc<GpuRef>, remaining: &mut [usize], label: &str| {
            // Glitch rounds: a pixel is glitched only when it outlived the
            // reference's (escaping) orbit.  Re-dispatch those against a
            // longer-lived reference chosen among them.
            for round in 0..MAX_GLITCH_PASSES {
                if int.interrupted() { break; }
                let glitched: Vec<usize> = raw.iter().enumerate()
                    .filter(|&(_, &r)| r != NOT_RUN && r & GLITCH_BIT != 0)
                    .map(|(j, _)| j)
                    .collect();
                if glitched.is_empty() { break; }

                // Candidates spread evenly over the glitched pixels.
                let step = glitched.len().div_ceil(GLITCH_CANDIDATES);
                let candidates = glitched.iter().step_by(step).map(|&j| pixel_coord(ids[j])).collect();

                let t_gref = std::time::Instant::now(); // DIAG
                let Some(new_ref) = state.glitch_reference(ctx, &geom, candidates, int) else { break };
                let t_gref = ms(t_gref); // DIAG

                let t_gdisp = std::time::Instant::now(); // DIAG
                let g_ids: Vec<usize> = glitched.iter().map(|&j| ids[j]).collect();
                let offs = offsets(&mut g_ids.iter().copied(), &new_ref);
                let new_gref = state.prepare(&new_ref, log2_dc(&offs, upp), upp);
                let new_raw = dispatch_ids(&g_ids, &new_ref, &new_gref);
                for (k, &j) in glitched.iter().enumerate() {
                    raw[j] = new_raw[k];
                }
                log::info!(
                    "[diag gpu] pass {pass} {label} glitch round {round}: {} glitched, new ref {} \
                     (reference {t_gref:.1} ms, dispatch {:.1} ms) -> {}",
                    glitched.len(), new_ref.describe(), ms(t_gdisp), raw_stats(&new_raw),
                );
                // Later chunks start from the better reference.
                if new_ref.beats(reference) {
                    *reference = new_ref;
                    *gref = Arc::new(state.prepare(reference, pass_dc(reference), upp));
                }
            }

            // Residual glitches: resolve exactly on the CPU, in parallel and
            // interruptibly.  Unresolved ones keep GLITCH_BIT and are not
            // stored; the resolved ones are exact and safe to cache.
            if !int.interrupted() {
                let residual: Vec<usize> = raw.iter().enumerate()
                    .filter(|&(_, &r)| r != NOT_RUN && r & GLITCH_BIT != 0)
                    .map(|(j, _)| j)
                    .collect();
                if !residual.is_empty() {
                    let t_resid = std::time::Instant::now(); // DIAG
                    let resolved: Vec<(usize, u32)> = residual.par_iter()
                        .filter_map(|&j| {
                            if int.interrupted() { return None; }
                            let (_, orbit) = calculate_orbit(pixel_coord(ids[j]), ctx.iterations);
                            Some((j, check_orbit(&orbit).unwrap().map_or(0, |n| n.get() as u32)))
                        })
                        .collect();
                    log::info!(
                        "[diag gpu] pass {pass} {label}: residual CPU resolve {}/{} in {:.1} ms",
                        resolved.len(), residual.len(), ms(t_resid),
                    );
                    for (j, r) in resolved { raw[j] = r; }
                }
            }

            // Store results, finish completed tiles, and tell the compositor.
            for (j, &r) in raw.iter().enumerate() {
                // Never cache a guess: glitched or not run by the GPU.
                if r == NOT_RUN || r & GLITCH_BIT != 0 { continue; }
                let pr = &pixel_refs[ids[j]];
                tiles[pr.tile_idx].tile.store(pr.pixel_idx, r.into()); // escape iteration, 0 = in set
                remaining[pr.tile_idx] -= 1;
                if remaining[pr.tile_idx] == 0 { finish(pr.tile_idx); }
            }
            ctx.progress.fetch_add(1, Ordering::Release);
        };

        // ----------------------------------------------------------------
        // Chunks: each dispatch is sized to take about TARGET_CHUNK_MS, so the
        // interrupt is honoured within that time (in-flight GPU work cannot
        // be cancelled) and results reach the screen as they arrive.
        //
        // One chunk on the GPU at a time (two queued ones ran concurrently,
        // each ~2x slower per pixel, and could not be timed), but the CPU
        // work overlaps it: the next chunk's seeds are packed while waiting,
        // and it is submitted as soon as this one is read back, before this
        // one's glitch rounds, CPU resolve and storing (~30% of a pass with
        // the GPU idle before). Its size comes from the chunk before this
        // one, except right after the probe.
        //
        // While a nucleus search runs in the background and the reference is
        // a provisional escaping point, glitched pixels are deferred rather
        // than sent through glitch rounds (seconds of CPU orbits and
        // searches): once the nucleus arrives, later chunks use it and the
        // deferred pixels are redone against it at the end of the pass.
        // ----------------------------------------------------------------
        let mut chunks = 0usize; // DIAG
        let pack = |pos: usize, first: bool, r: &Reference| {
            let len = if first { PROBE_CHUNK_PX } else { state.chunk_len() }.min(pixel_refs.len() - pos);
            (pos, len, Deltas::pack(&offsets(&mut (pos..pos + len), r), upp))
        };
        let submit = |(pos, len, d): (usize, usize, Deltas), gref: &Arc<GpuRef>| {
            (pos, len, state.submit_deltas(d, gref.clone()))
        };
        let iters = ctx.iterations;
        // Take the background search's result once it is in.
        let poll_search = |searching: &mut bool, reference: &mut Arc<Reference>, gref: &mut Arc<GpuRef>| {
            if !*searching { return; }
            let Some(found) = state.search_result(&geom, iters) else {
                // Replaced by a search for another view: nothing to wait for.
                if !state.search_covers(&geom, iters) { *searching = false; }
                return;
            };
            *searching = false;
            if let Some(r) = found.filter(|r| !reference.is_full || r.beats(reference)) {
                log::info!("[diag gpu] pass {pass}: switching to {} from the background search", r.describe());
                *reference = r;
                *gref = Arc::new(state.prepare(reference, pass_dc(reference), upp));
            }
        };
        let mut deferred: Vec<usize> = Vec::new();
        let mut current = (!int.interrupted()).then(|| submit(pack(0, true, &reference), &gref));
        let mut pos = 0;
        while let Some((at, len, pending)) = current.take() {
            let more = at + len < pixel_refs.len();
            let packed = (more && chunks > 0 && !int.interrupted()).then(|| pack(at + len, false, &reference));
            let t_wait = std::time::Instant::now();
            let t_submit = pending.t_submit;
            let mut raw = state.finish(pending);
            let done = std::time::Instant::now();
            // Only a wait that blocked tells when the GPU finished.
            if (done - t_wait).as_secs_f64() > 1e-4 {
                state.record_chunk(len, (done - t_submit).as_secs_f64() * 1e3);
            }
            let old_ref = reference.clone();
            poll_search(&mut searching, &mut reference, &mut gref);
            if more && !int.interrupted() {
                // Repack if the reference just changed.
                let packed = packed.filter(|_| Arc::ptr_eq(&old_ref, &reference));
                current = Some(submit(packed.unwrap_or_else(|| pack(at + len, false, &reference)), &gref));
            }
            pos = at;
            log::info!(
                "[diag gpu] pass {pass} chunk {chunks}: {len} px at {pos}, gpu {:.1} ms -> {}",
                (done - t_submit).as_secs_f64() * 1e3, raw_stats(&raw),
            );

            let ids: Vec<usize> = (at..at + len).collect();
            if searching && !old_ref.is_full {
                // Defer glitched pixels until the nucleus arrives.
                for (j, r) in raw.iter_mut().enumerate() {
                    if *r != NOT_RUN && *r & GLITCH_BIT != 0 { deferred.push(ids[j]); *r = NOT_RUN; }
                }
            }
            resolve_and_store(&ids, &mut raw, &mut reference, &mut gref, &mut remaining, &format!("chunk {chunks}"));

            pos    += len;
            chunks += 1;
        }

        // Deferred pixels: wait for the search (interruptibly), then redo
        // them against its nucleus; if it found none, glitch rounds as usual.
        if !deferred.is_empty() && !int.interrupted() {
            let t_wait = std::time::Instant::now(); // DIAG
            while searching && !int.interrupted() {
                poll_search(&mut searching, &mut reference, &mut gref);
                if searching { std::thread::sleep(std::time::Duration::from_millis(5)); }
            }
            if int.interrupted() { return; }
            let mut raw = if reference.is_full {
                dispatch_ids(&deferred, &reference, &gref)
            } else {
                vec![GLITCH_BIT; deferred.len()] // no nucleus: glitch rounds
            };
            log::info!(
                "[diag gpu] pass {pass}: {} deferred px, waited {:.1} ms for the search, ref {} -> {}",
                deferred.len(), ms(t_wait), reference.describe(), raw_stats(&raw),
            );
            resolve_and_store(&deferred, &mut raw, &mut reference, &mut gref, &mut remaining, "deferred");
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
            ("perturbation", shader_source(SHADER_SRC)),
            ("perturbation_floatexp", shader_source(SHADER_FE_BODY)),
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

    /// CPU mirror of the shader loop with rebasing and cycle-multiplier
    /// interior detection: every `period` iterations compare log2|dz/dz0|²
    /// with its value a cycle earlier; `k` consecutive drops below
    /// `log2_q2` declare the pixel interior.  Returns (result, iterations).
    fn perturb_interior<F: num::Float>(
        orbit: &[Complex<FBig>], d0: (f64, f64), full: bool, period: usize, k: u32, log2_q2: f64,
    ) -> (Result<Option<usize>, ()>, usize) {
        let x: Vec<Complex<F>> = orbit.iter().map(|c| Complex {
            re: F::from(c.re.to_f64().value()).unwrap(), im: F::from(c.im.to_f64().value()).unwrap(),
        }).collect();
        let d0 = Complex { re: F::from(d0.0).unwrap(), im: F::from(d0.1).unwrap() };
        let two = F::from(2.0).unwrap();
        let (mut d, mut m) = (d0, 0usize);
        let (mut ld, mut prev, mut streak) = (F::zero(), None::<F>, 0u32);
        let lq = F::from(log2_q2).unwrap();
        let n_max = orbit.len() - 1;
        for n in 0..orbit.len() {
            let z = x[m] + d;
            let z2 = z.norm_sqr();
            if z2 > F::from(4.0).unwrap() { return (Ok(Some(n + 1)), n + 1); }
            if n == n_max { break; }
            ld = ld + two + z2.log2(); // |dz_{n+1}|² = 4|z_n|²|dz_n|²
            if period > 0 && (n + 1) % period == 0 {
                streak = match prev { Some(p) if ld - p < lq => streak + 1, _ => 0 };
                prev = Some(ld);
                if streak >= k { return (Ok(None), n + 1); }
            }
            if z2 < d.norm_sqr() {
                d = z * z + d0;
                m = 0;
            } else {
                d = x[m] * d * two + d * d + d0;
                m += 1;
            }
        }
        (if full { Ok(None) } else { Err(()) }, orbit.len())
    }

    struct InteriorCase { name: &'static str, depth: i64, iters: usize, centre: Option<(f64, f64)>, off: (i64, i64) }

    struct InteriorResult { period: usize, in_set: usize, detected: usize, false_pos: usize, work: f64 }

    /// Run the shader's interior detection (CPU mirror, f32, parameters from
    /// `Reference::interior`) on a `side`² grid of a 256 px view, against
    /// exact orbits. `None` if no nucleus is found (detection is then off).
    fn interior_case(case: &InteriorCase, side: i64) -> Option<InteriorResult> {
        let depth = case.depth;
        let prec  = working_precision(depth);
        let upp   = upp_log2(depth);
        let fp = |x: f64| FBig::try_from(x).unwrap().with_precision(prec).value();
        let origin = match case.centre {
            Some((cx, cy)) => Point::new(
                &fp(cx) - &(FBig::from(128) << upp as isize),
                &fp(cy) - &(FBig::from(128) << upp as isize),
            ),
            None => {
                // Beside the period-998 minibrot near the seahorse valley.
                let seed = Complex { re: fp(-0.743_643_887_037_151), im: fp(0.131_825_904_205_330) };
                let rough = find_nucleus(&seed, &fp(1e-12), -50, 10_000, prec, &|| false).unwrap().unwrap();
                let tol2 = FBig::ONE << (2 * (upp - 60)) as isize;
                let nuc = newton_nucleus(&rough.c, rough.period, prec, &tol2, &fp(1.0), &|| false).unwrap().unwrap();
                Point::new(
                    &nuc.re - &(FBig::from(128 + case.off.0) << upp as isize),
                    &nuc.im - &(FBig::from(128 + case.off.1) << upp as isize),
                )
            }
        };
        let ctx = PassBatchCtx {
            coords: CoordinatesBox { origin: origin.clone(), view: View::new((upp as f64).exp2()) },
            depth, x0: tile_index(&origin.x, depth), y0: tile_index(&origin.y, depth),
            width: 256, height: 256, iterations: case.iters, progress: Default::default(),
        };
        let g = ViewGeom::new(&ctx);
        let n = find_nucleus(&g.center, &g.radius, g.upp, case.iters, g.prec, &|| false).unwrap()?;
        let r = Reference::new(n.c.clone(), case.iters, Some(n.period));
        let (window, contraction) = r.interior();
        let (orbit, _) = calculate_orbit(n.c.clone(), case.iters);
        let (rx, ry) = ref_px(&n.c, &ctx);
        let scale = (upp as f64).exp2();
        let ts = IBig::from(TILE_SIZE as u64);
        let (gx0, gy0) = (&ctx.x0 * &ts, &ctx.y0 * &ts);
        let step = 256 / side;
        let pix: Vec<(i64, i64)> = (0..side).flat_map(|j| (0..side).map(move |i| (i * step + 3, j * step + 5))).collect();
        let exact: Vec<Option<usize>> = pix.par_iter().map(|&(col, row)| {
            let c = Complex {
                re: pixel_to_coord(&gx0 + IBig::from(col), depth, prec),
                im: pixel_to_coord(&gy0 + IBig::from(row), depth, prec),
            };
            check_orbit(&calculate_orbit(c, case.iters).1).unwrap().map(|v| v.get())
        }).collect();
        let mut out = InteriorResult {
            period: n.period, in_set: exact.iter().filter(|e| e.is_none()).count(),
            detected: 0, false_pos: 0, work: 0.0,
        };
        let (mut run, mut base) = (0usize, 0usize);
        for &(col, row) in &pix {
            let d0 = ((col as f64 - rx) * scale, (row as f64 - ry) * scale);
            let (res, used) = perturb_interior::<f32>(&orbit.orbit, d0, orbit.is_full, window as usize, INTERIOR_WINDOWS, contraction as f64);
            let (res0, used0) = perturb_interior::<f32>(&orbit.orbit, d0, orbit.is_full, 0, 0, 0.0);
            run += used;
            base += used0;
            if matches!(res, Ok(None)) && used < used0 {
                out.detected += 1;
                // Detection changed the answer: without it, this pixel escapes.
                if matches!(res0, Ok(Some(_))) { out.false_pos += 1; }
            }
        }
        out.work = run as f64 / base as f64;
        Some(out)
    }

    /// Interior detection must never mark an escaping pixel as in the set,
    /// and must save real work where there is interior.
    #[test]
    fn interior_detection_is_safe_and_useful() {
        let cases = [
            (InteriorCase { name: "shallow period-3 bulb", depth: 8, iters: 2000, centre: Some((-0.122, 0.745)), off: (0, 0) }, 0.3),
            (InteriorCase { name: "shallow cardioid", depth: 6, iters: 2000, centre: Some((-0.1, 0.65)), off: (0, 0) }, 0.5),
            (InteriorCase { name: "deep minibrot inside", depth: 55, iters: 6000, centre: None, off: (-50, 48) }, 0.65),
            (InteriorCase { name: "deep beside minibrot", depth: 40, iters: 4000, centre: None, off: (-50, 48) }, 1.01),
        ];
        for (case, max_work) in &cases {
            let r = interior_case(case, 16).expect("nucleus");
            eprintln!("{}: p={} in-set {} detected {} false+ {} work {:.1}%",
                case.name, r.period, r.in_set, r.detected, r.false_pos, 100.0 * r.work);
            assert_eq!(r.false_pos, 0, "{}", case.name);
            assert!(r.work <= *max_work, "{}: work {:.2}", case.name, r.work);
        }
    }

    /// Broader measurement (more views, more pixels). Run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn diag_interior_detection() {
        let cases = [
            InteriorCase { name: "shallow cardioid", depth: 6, iters: 4000, centre: Some((-0.1, 0.65)), off: (0, 0) },
            InteriorCase { name: "shallow bulb edge", depth: 9, iters: 4000, centre: Some((-0.75, 0.05)), off: (0, 0) },
            InteriorCase { name: "shallow seahorse", depth: 12, iters: 4000, centre: Some((-0.7436, 0.1318)), off: (0, 0) },
            InteriorCase { name: "shallow period-3 bulb", depth: 8, iters: 4000, centre: Some((-0.122, 0.745)), off: (0, 0) },
            InteriorCase { name: "shallow cardioid cusp", depth: 10, iters: 4000, centre: Some((0.25, 0.0)), off: (0, 0) },
            InteriorCase { name: "deep minibrot inside", depth: 55, iters: 8000, centre: None, off: (-50, 48) },
            InteriorCase { name: "deep beside minibrot", depth: 40, iters: 8000, centre: None, off: (-50, 48) },
        ];
        for case in &cases {
            match interior_case(case, 32) {
                None => eprintln!("{}: no nucleus -> detection off (safe)", case.name),
                Some(r) => eprintln!("{} (depth {}, {} iters): p={} in-set {}/1024 detected {} false+ {} work {:.1}%",
                    case.name, case.depth, case.iters, r.period, r.in_set, r.detected, r.false_pos, 100.0 * r.work),
            }
        }
    }

    /// A reported view (Cmd-C string `x,y|view`) sampled on an `iw`x`ih`
    /// grid over a `w`x`h` window: exact escape iterations, plus what's
    /// needed to run the GPU on the same pixels.
    struct ViewSample {
        ctx:   PassBatchCtx,
        geom:  ViewGeom,
        /// depth-grid pixel (col, row) relative to x0, y0, per sample
        pix:   Vec<(i64, i64)>,
        exact: Vec<u32>,
    }

    fn sample_view(clip: &str, iters: usize, (w, h): (usize, usize), (iw, ih): (usize, usize)) -> ViewSample {
        let coords: CoordinatesBox = clip.parse().unwrap();
        let view  = coords.view.inner;
        let depth = crate::tiles::store::depth_for_view(view);
        let prec  = working_precision(depth);
        let ctx = PassBatchCtx {
            coords: coords.clone(), depth,
            x0: tile_index(&coords.origin.x, depth), y0: tile_index(&coords.origin.y, depth),
            width: w, height: h, iterations: iters, progress: Default::default(),
        };
        let geom = ViewGeom::new(&ctx);
        let ts = IBig::from(TILE_SIZE as u64);
        let (gx0, gy0) = (&ctx.x0 * &ts, &ctx.y0 * &ts);
        let ratio = view / (geom.upp as f64).exp2();
        let o00 = crate::tiles::store::TileKey { depth, x: ctx.x0.clone(), y: ctx.y0.clone() }.origin();
        let sx0 = (&o00.x - &coords.origin.x).to_f64().value() / view;
        let sy0 = (&o00.y - &coords.origin.y).to_f64().value() / view;
        let pix: Vec<(i64, i64)> = (0..ih).flat_map(|j| (0..iw).map(move |i| (i, j)))
            .map(|(i, j)| {
                let sx = (i as f64 + 0.5) * w as f64 / iw as f64;
                let sy = (j as f64 + 0.5) * h as f64 / ih as f64;
                (((sx - sx0) * ratio).round() as i64, ((sy - sy0) * ratio).round() as i64)
            }).collect();
        let exact = pix.par_iter().map(|&(col, row)| {
            let c = Complex {
                re: pixel_to_coord(&gx0 + IBig::from(col), depth, prec),
                im: pixel_to_coord(&gy0 + IBig::from(row), depth, prec),
            };
            check_orbit(&calculate_orbit(c, iters).1).unwrap().map_or(0, |v| v.get() as u32)
        }).collect();
        ViewSample { ctx, geom, pix, exact }
    }

    /// Run the real shader on the sample with the view's own nucleus.
    fn gpu_on_sample(v: &ViewSample, gpu: &GpuState) -> (Reference, Vec<u32>) {
        gpu_on_sample_with(v, gpu, USE_BLA)
    }

    fn gpu_on_sample_with(v: &ViewSample, gpu: &GpuState, bla: bool) -> (Reference, Vec<u32>) {
        let g = &v.geom;
        let n = find_nucleus(&g.center, &g.radius, g.upp, v.ctx.iterations, g.prec, &|| false).unwrap().expect("nucleus");
        let r = Reference::new(n.c.clone(), v.ctx.iterations, Some(n.period));
        let (rx, ry) = ref_px(&r.c, &v.ctx);
        let offs: Vec<(f64, f64)> = v.pix.iter().map(|&(col, row)| (col as f64 - rx, row as f64 - ry)).collect();
        let gr = gpu.prepare_with(&r, log2_dc(&offs, g.upp), g.upp, bla);
        let out = gpu.dispatch_offsets(&offs, g.upp, &gr);
        (r, out)
    }

    /// (wrong, off by >50 iterations, black) of `got` against the exact values.
    fn score(got: &[u32], exact: &[u32]) -> (usize, usize, usize) {
        (
            got.iter().zip(exact).filter(|(a, b)| a != b).count(),
            got.iter().zip(exact).filter(|(a, b)| a.abs_diff(**b) > 50).count(),
            got.iter().filter(|&&v| v == 0).count(),
        )
    }

    fn dump_rgb(name: &str, vals: &[u32]) {
        if let Ok(out) = std::env::var("DIAG_OUT") {
            let mut palette = vec![];
            let rgb: Vec<u8> = vals.iter().flat_map(|&v| {
                let c = crate::gpu_compositor::color_of_for_tests(v, &mut palette);
                [(c >> 16) as u8, (c >> 8) as u8, c as u8]
            }).collect();
            std::fs::write(format!("{out}/{name}.rgb"), rgb).unwrap();
        }
    }

    /// Regression for the artifact view reported on 2026-09-23: black
    /// octagon and smeared streaks at 2^-220, 32768 iterations. The floatexp
    /// shader switched to f32 at δ ≈ 2^-100; at the nucleus orbit's zero
    /// points δ ← δ² + δ₀ then underflowed to exactly 0 and pixels followed
    /// the reference forever (before the fix: 8461 of 15000 wrong, 515
    /// black; after: 1184, all f32 rounding, 0 black). Real shader on this
    /// machine's GPU vs exact orbits; needs a GPU and ~15 s, so `--ignored`.
    /// With DIAG_OUT=dir, writes both renders as raw RGB (150x100).
    #[test]
    #[ignore]
    fn artifact_view_2026_09_23_matches_exact() {
        let clip = "0.36268061816918528044899172250760567988128488705999553580413024078293361870845608332075349484941,-0.64268799384608729642124986015130477562370908052480481690338843577039749867381572196329048271251|6.226537747227718e-67";
        let v = sample_view(clip, 32768, (3000, 2000), (150, 100));
        let (_, got) = gpu_on_sample(&v, &GpuState::new());
        let (wrong, off50, black) = score(&got, &v.exact);
        eprintln!("GPU vs exact over {} px: wrong {wrong}, off by >50 {off50}, black {black}", got.len());
        dump_rgb("a1_exact", &v.exact);
        dump_rgb("a1_gpu", &got);
        assert_eq!(black, 0, "pixels lost their delta and followed the reference");
        assert!(wrong * 10 < got.len(), "{wrong} wrong: more than f32 rounding");
        assert!(off50 * 100 < got.len(), "{off50} pixels off by >50 iterations");
    }

    /// `sample_view`, with the exact values cached in $DIAG_OUT/<name>.exact.
    fn sample_view_cached(name: &str, clip: &str, iters: usize, win: (usize, usize), img: (usize, usize)) -> ViewSample {
        let path = std::env::var("DIAG_OUT").ok().map(|d| format!("{d}/{name}.exact"));
        if let Some(bytes) = path.as_ref().and_then(|p| std::fs::read(p).ok()) {
            let mut v = sample_view_geometry(clip, iters, win, img);
            v.exact = bytemuck::cast_slice(&bytes).to_vec();
            return v;
        }
        let v = sample_view(clip, iters, win, img);
        if let Some(p) = path { std::fs::write(p, bytemuck::cast_slice(&v.exact)).unwrap(); }
        v
    }

    /// `sample_view` without the exact orbits.
    fn sample_view_geometry(clip: &str, iters: usize, win: (usize, usize), img: (usize, usize)) -> ViewSample {
        let mut v = sample_view(clip, 1, win, img);
        v.ctx.iterations = iters;
        v.exact.clear();
        v
    }

    /// Run whole generations (tiles, passes, chunks, caches) with the real
    /// GPU backend, then read the tiles back at the sample points. Returns
    /// the values per generation; `None` = pixel not stored.
    fn run_pipeline(v: &ViewSample, backend: &Gpu, store: &crate::tiles::store::TileStore, generations: usize) -> Vec<Vec<Option<u32>>> {
        use crate::tiles::render::{GroupCache, run_generation};
        let mut out = vec![];
        for generation in 0..generations {
            if generation > 0 { store.clear(); } // what Space does to the tiles
            let group_cache = parking_lot::Mutex::new(GroupCache::new());
            let (tx, rx) = waker_interrupter::channel::<()>();
            tx.send(());
            let mut tx = Some(tx);
            rx.run_multithreaded(None, None, |(), int| {
                run_generation(store, &group_cache, v.ctx.width, v.ctx.height, &v.ctx.coords,
                    v.ctx.iterations, None, int, backend);
                if let Some(tx) = tx.take() { tx.terminate(); }
            });
            let ts = TILE_SIZE as i64;
            out.push(v.pix.iter().map(|&(col, row)| {
                let key = crate::tiles::store::TileKey {
                    depth: v.ctx.depth,
                    x: &v.ctx.x0 + IBig::from(col.div_euclid(ts)),
                    y: &v.ctx.y0 + IBig::from(row.div_euclid(ts)),
                };
                store.get(&key)?.load((row.rem_euclid(ts) * ts + col.rem_euclid(ts)) as usize).get()
            }).collect());
        }
        out
    }

    /// Regression for the second view reported on 2026-09-23 (2^-305, 32768
    /// iterations): black and partly black tiles, different on every reset.
    /// Cause: 65k-pixel minimum chunks near a deep minibrot made dispatches
    /// of several hundred ms, which macOS killed at random (and dropped
    /// following ones), leaving the zero-initialised output = "in set".
    /// Whole pipeline (tiles, passes, chunks, caches) with the real GPU
    /// backend, fresh and after a Space-style reset; every generation must
    /// match exact orbits up to f32 rounding (before: 1070, 5169, 6979 black
    /// of 15000; after: 109, the exact count, every time) and agree with each
    /// other up to a few pixels of f32 rounding. ~2 min per
    /// generation without BLA, ~1.5 s with it (DIAG_GENS, default 2); exact
    /// values cached in $DIAG_OUT (computing them takes ~1 min).
    #[test]
    #[ignore]
    fn view_2026_09_23b_pipeline_matches_exact() {
        let clip = "0.362680618169185280448991722507605679881284887059995535804130241675861432076423165340038916307547696001473666663642181909,-0.642687993846087296421249860151304775623709080524804816903388435265396126015520705061576074177687940915500836211594418175|1.7963169535192306e-92";
        let _ = env_logger::builder().is_test(true).try_init();
        let v = sample_view_cached("b", clip, 32768, (3000, 2000), (150, 100));
        let backend = Gpu(Arc::new(GpuState::new()));
        let store = crate::tiles::store::TileStore::new(crate::tiles::store::MEMORY_BUDGET_BYTES);
        let t = std::time::Instant::now();
        let n_gens = std::env::var("DIAG_GENS").ok().and_then(|g| g.parse().ok()).unwrap_or(2);
        let gens = run_pipeline(&v, &backend, &store, n_gens);
        eprintln!("{n_gens} generation(s) in {:?}", t.elapsed());
        for (g, vals) in gens.iter().enumerate() {
            let missing = vals.iter().filter(|x| x.is_none()).count();
            let got: Vec<u32> = vals.iter().map(|x| x.unwrap_or(u32::MAX)).collect();
            let (wrong, off50, black) = score(&got, &v.exact);
            let exact_black = v.exact.iter().filter(|&&x| x == 0).count();
            eprintln!("generation {g}: missing {missing}, wrong {wrong}, off by >50 {off50}, black {black} (exact black {exact_black})");
            dump_rgb(&format!("b_pipe{g}"), &got.iter().map(|&x| if x == u32::MAX { 0 } else { x }).collect::<Vec<_>>());
            assert_eq!(missing, 0, "generation {g}");
            assert_eq!(black, exact_black, "generation {g}: in-set pixels that escape");
            assert!(off50 * 100 < got.len(), "generation {g}: {off50} off by >50");
        }
        // The first generation starts on a provisional reference (the view
        // centre) while the nucleus search runs in the background, so a few
        // pixels round differently in f32 than after a reset (seen: 1 of
        // 15000). The bug this guards against changed thousands.
        for w in gens.windows(2) {
            let differ = w[0].iter().zip(&w[1]).filter(|(a, b)| a != b).count();
            eprintln!("generations differ in {differ} px");
            assert!(differ * 1000 < v.pix.len(), "generations differ in {differ} px");
        }
    }

    /// Same input dispatched repeatedly must give identical output, also as
    /// one oversized dispatch (~400 ms of work) that macOS may kill: the
    /// NOT_RUN sentinel + retry in `dispatch_chunk` must recover it. Needs a
    /// GPU (~10 s).
    #[test]
    #[ignore]
    fn dispatch_is_deterministic_even_when_killed() {
        let clip = "0.362680618169185280448991722507605679881284887059995535804130241675861432076423165340038916307547696001473666663642181909,-0.642687993846087296421249860151304775623709080524804816903388435265396126015520705061576074177687940915500836211594418175|1.7963169535192306e-92";
        let v = sample_view_geometry(clip, 32768, (3000, 2000), (1, 1));
        let g = &v.geom;
        let n = find_nucleus(&g.center, &g.radius, g.upp, 32768, g.prec, &|| false).unwrap().unwrap();
        let r = Reference::new(n.c.clone(), 32768, Some(n.period));
        let (rx, ry) = ref_px(&r.c, &v.ctx);
        // 256x256 block of depth-grid pixels around the reference.
        let offs: Vec<(f64, f64)> = (0..256).flat_map(|j| (0..256).map(move |i| (i as f64 - 128.0 + 0.3, j as f64 - 128.0 + 0.7)))
            .map(|(dx, dy)| (dx * 3.0, dy * 3.0)).collect();
        let _ = (rx, ry);
        let gpu = GpuState::new();
        // BLA off: with it this block takes ~20 ms, too short to be killed.
        let r = gpu.prepare_with(&r, log2_dc(&offs, g.upp), g.upp, false);
        let mut first: Option<Vec<u32>> = None;
        for run in 0..6 {
            let t = std::time::Instant::now();
            let out = gpu.dispatch_offsets(&offs, g.upp, &r);
            let zeros = out.iter().filter(|&&x| x == 0).count();
            let diff = first.as_ref().map_or(0, |f| f.iter().zip(&out).filter(|(a, b)| a != b).count());
            eprintln!("run {run}: {:?}, zeros {zeros}, differs from run 0 in {diff} px", t.elapsed());
            assert_eq!(diff, 0, "run {run}");
            first.get_or_insert(out);
        }
        // Same pixels, split into small dispatches.
        for size in [1024usize, 4096, 16384] {
            for run in 0..2 {
                let t = std::time::Instant::now();
                let mut out = vec![];
                let mut worst = std::time::Duration::ZERO;
                for chunk in offs.chunks(size) {
                    let tc = std::time::Instant::now();
                    out.extend(gpu.dispatch_offsets(chunk, g.upp, &r));
                    worst = worst.max(tc.elapsed());
                }
                let zeros = out.iter().filter(|&&x| x == 0).count();
                let diff = first.as_ref().map_or(0, |f| f.iter().zip(&out).filter(|(a, b)| a != b).count());
                eprintln!("chunks of {size}, run {run}: total {:?}, slowest dispatch {worst:?}, zeros {zeros}, differs from first full run in {diff}",
                    t.elapsed());
                assert_eq!(diff, 0, "chunks of {size}");
            }
        }
    }

    /// Measured 2026-09-23: with the fallback every threshold from 2^-60 to
    /// 2^-110 is equally accurate (109 black, ~110 off by >50), and 2^-60 is
    /// fastest (0.96 s vs 1.23 s at 2^-110 on the heavy block).
    ///
    /// A/B the floatexp -> f32 switch threshold (with the tiny-δ fallback):
    /// accuracy on the second reported view's sample vs exact, and GPU time
    /// on a heavy 256x256 block beside the minibrot. Needs the cached exact
    /// values ($DIAG_OUT/b.exact from diag_view_2026_09_23b_*).
    #[test]
    #[ignore]
    fn diag_switch_threshold() {
        let clip = "0.362680618169185280448991722507605679881284887059995535804130241675861432076423165340038916307547696001473666663642181909,-0.642687993846087296421249860151304775623709080524804816903388435265396126015520705061576074177687940915500836211594418175|1.7963169535192306e-92";
        let v = sample_view_cached("b", clip, 32768, (3000, 2000), (150, 100));
        let g = &v.geom;
        let n = find_nucleus(&g.center, &g.radius, g.upp, 32768, g.prec, &|| false).unwrap().unwrap();
        let r = Reference::new(n.c.clone(), 32768, Some(n.period));
        let (rx, ry) = ref_px(&r.c, &v.ctx);
        let sample: Vec<[f32; 4]> = v.pix.iter().map(|&(c, w)| pack_fe(c as f64 - rx, w as f64 - ry, g.upp)).collect();
        let block: Vec<[f32; 4]> = (0..256).flat_map(|j| (0..256).map(move |i| (i, j)))
            .map(|(i, j)| pack_fe((i as f64 - 128.3) * 3.0, (j as f64 - 127.6) * 3.0, g.upp)).collect();
        let gpu = GpuState::new();
        let r = gpu.prepare_with(&r, v.geom.upp as f64 + 12.0, v.geom.upp, false); // as measured, before BLA
        let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[Some(&gpu.bgl)], immediate_size: 0,
        });
        let body = SHADER_FE_BODY;
        for (sw, min_exp) in [(-60, -62), (-80, -82), (-90, -92), (-100, -102), (-110, -112)] {
            let src = body
                .replace("const F32_SWITCH_EXP : i32 = -60;", &format!("const F32_SWITCH_EXP : i32 = {sw};"))
                .replace("const F32_MIN_DELTA : f32 = 2.168404344971009e-19;", &format!("const F32_MIN_DELTA : f32 = {:e};", (min_exp as f64).exp2()));
            assert!(body.contains("const F32_SWITCH_EXP : i32 = -60;") && body.contains("const F32_MIN_DELTA : f32 = 2.168404344971009e-19;"));
            let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None, source: wgpu::ShaderSource::Wgsl(shader_source(&src).into()),
            });
            let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None, layout: Some(&layout), module: &module, entry_point: Some("main"),
                compilation_options: Default::default(), cache: None,
            });
            let run = |d: &[[f32; 4]]| -> Vec<u32> {
                d.chunks(4096).flat_map(|c| gpu.dispatch(&pipeline, bytemuck::cast_slice(c), c.len(), 16, &r)).collect()
            };
            let got = run(&sample);
            let (wrong, off50, black) = score(&got, &v.exact);
            let _ = run(&block);
            let t = std::time::Instant::now();
            let _ = run(&block);
            eprintln!("switch 2^{sw}, fallback below 2^{min_exp}: wrong {wrong}, off>50 {off50}, black {black} (exact 109) | heavy block {:?}",
                t.elapsed());
        }
    }

    /// A/B BLA off vs on with the real shaders: accuracy on each view's
    /// sample against exact orbits, and GPU time on a heavy 256x256 block
    /// beside the reference. Exact values cached in $DIAG_OUT (name.exact).
    ///
    /// Measured 2026-09-23 (wrong / off by >50 of 15000; heavy block):
    ///   2^-305 view: 5946/112, 1050 ms  ->  611/10, 35 ms
    ///   2^-220 view: 1184/23,   144 ms  ->  352/7,  15 ms
    ///   seahorse (2^-17), bulb (2^-13): identical results, ~5-10% slower
    ///   (BLA rarely applies there; the search back-off keeps it cheap).
    #[test]
    #[ignore]
    fn diag_bla_ab() {
        let views = [
            ("seahorse", "-0.755044091796875,0.12417060546875|7.62939453125e-06", 4096),
            ("bulb", "-0.30510546875,0.6229296875|0.0001220703125", 4096),
            ("a", "0.36268061816918528044899172250760567988128488705999553580413024078293361870845608332075349484941,-0.64268799384608729642124986015130477562370908052480481690338843577039749867381572196329048271251|6.226537747227718e-67", 32768),
            ("b", "0.362680618169185280448991722507605679881284887059995535804130241675861432076423165340038916307547696001473666663642181909,-0.642687993846087296421249860151304775623709080524804816903388435265396126015520705061576074177687940915500836211594418175|1.7963169535192306e-92", 32768),
        ];
        let gpu = GpuState::new();
        for (name, clip, iters) in views {
            let v = sample_view_cached(name, clip, iters, (3000, 2000), (150, 100));
            let exact_black = v.exact.iter().filter(|&&x| x == 0).count();
            let g = &v.geom;
            let n = find_nucleus(&g.center, &g.radius, g.upp, iters, g.prec, &|| false).unwrap().expect("nucleus");
            let r = Reference::new(n.c.clone(), iters, Some(n.period));
            let block: Vec<(f64, f64)> = (0..256).flat_map(|j| (0..256).map(move |i| ((i as f64 - 128.3) * 3.0, (j as f64 - 127.6) * 3.0))).collect();
            for bla in [false, true] {
                let (_, got) = gpu_on_sample_with(&v, &gpu, bla);
                let (wrong, off50, black) = score(&got, &v.exact);
                let gr = gpu.prepare_with(&r, log2_dc(&block, g.upp), g.upp, bla);
                let run = || -> Vec<u32> { block.chunks(4096).flat_map(|c| gpu.dispatch_offsets(c, g.upp, &gr)).collect() };
                let _ = run();
                let t = std::time::Instant::now();
                let _ = run();
                eprintln!("{name:9} (p={:4}, depth {:3}) BLA {:3}: wrong {wrong:5} off>50 {off50:4} black {black:4} (exact {exact_black:4}) | heavy block {:?}",
                    n.period, v.ctx.depth, if bla { "on" } else { "off" }, t.elapsed());
                dump_rgb(&format!("ab_{name}_{}", if bla { "on" } else { "off" }), &got);
            }
        }
    }

    /// BLA must not change results when the reference is an escaping point
    /// (the fallback when no nucleus is found). Written for the uniform
    /// 2^-314 view of 2026-09-23 while suspecting BLA jumps through the
    /// reference's escape; that was not it (see view_2026_09_23c), but the
    /// check stays. Real GPU, BLA on vs off with the same escaping
    /// reference; needs a GPU.
    #[test]
    #[ignore]
    fn bla_with_escaping_reference_matches_plain() {
        let clip = "0.362680618169185280448991722507605679881284887059995535804130241675861432076423165340038916307547696001473666663642181909,-0.642687993846087296421249860151304775623709080524804816903388435265396126015520705061576074177687940915500836211594418175|1.7963169535192306e-92";
        let iters = 262144;
        let v = sample_view_geometry(clip, iters, (3000, 2000), (150, 100));
        let g = &v.geom;
        // An escaping point in the view: the top-left sample pixel.
        let ts = IBig::from(TILE_SIZE as u64);
        let (col, row) = v.pix[0];
        let c = Complex {
            re: pixel_to_coord(&v.ctx.x0 * &ts + IBig::from(col), v.ctx.depth, g.prec),
            im: pixel_to_coord(&v.ctx.y0 * &ts + IBig::from(row), v.ctx.depth, g.prec),
        };
        let r = Reference::new(c, iters, None);
        assert!(!r.is_full, "need an escaping reference");
        eprintln!("reference {}", r.describe());
        let (rx, ry) = ref_px(&r.c, &v.ctx);
        let offs: Vec<(f64, f64)> = v.pix.iter().map(|&(c, w)| (c as f64 - rx, w as f64 - ry)).collect();
        let gpu = GpuState::new();
        let run = |bla: bool| {
            let gr = gpu.prepare_with(&r, log2_dc(&offs, g.upp), g.upp, bla);
            offs.chunks(1024).flat_map(|c| gpu.dispatch_offsets(c, g.upp, &gr)).collect::<Vec<u32>>()
        };
        let (plain, bla) = (run(false), run(true));
        let glitch = |v: &[u32]| v.iter().filter(|&&x| x & GLITCH_BIT != 0).count();
        let mode = |v: &[u32]| {
            let mut h = std::collections::HashMap::new();
            for &x in v { *h.entry(x).or_insert(0usize) += 1; }
            h.into_iter().max_by_key(|&(_, n)| n).unwrap()
        };
        let differ = plain.iter().zip(&bla).filter(|(a, b)| a != b).count();
        let far = plain.iter().zip(&bla)
            .filter(|&(&a, &b)| (a & GLITCH_BIT != b & GLITCH_BIT) || (a & !GLITCH_BIT).abs_diff(b & !GLITCH_BIT) > 50)
            .count();
        eprintln!("plain: glitched {}, most common {:?}", glitch(&plain), mode(&plain));
        eprintln!("BLA:   glitched {}, most common {:?}", glitch(&bla), mode(&bla));
        eprintln!("differ {differ}, glitch status differs or off by >50: {far} (of {})", plain.len());
        assert!(far * 100 < plain.len(), "BLA changed {far} results");
    }

    /// Regression for the view copied 2026-09-23 (2^-313.9, 262144
    /// iterations): every GPU pixel came out in set (exact: none). The
    /// nucleus orbit (p=33760) passes within 2^-149 / 2^-271 of 0 every
    /// 2110 / 4220 steps; uploaded as f32 those values flushed to 0, which
    /// dropped the 2·X·δ term while δ was smaller still, so pixels shadowed
    /// the reference. An f64 mirror with the orbit rounded to f32 showed the
    /// same. Fixed by giving the deep shader a floatexp orbit (before: 600
    /// of 600 black; after: 139 wrong by f32 rounding, 0 off by >50, 0
    /// black). Needs a GPU; exact values (~10 s) cached in $DIAG_OUT.
    #[test]
    #[ignore]
    fn view_2026_09_23c_matches_exact() {
        let clip = "0.3626806181691852804489917225076056798812848870599955358041302416758614320764231653400389334599052087605123550187040027890282,-0.6426879938460872964212498601513047756237090805248048169033884352653961260155207050615760571472782180516346029887216862939575|3.290623677226253e-95";
        let v = sample_view_cached("c30", clip, 262144, (3000, 2000), (30, 20));
        let (r, got) = gpu_on_sample(&v, &GpuState::new());
        let (wrong, off50, black) = score(&got, &v.exact);
        let exact_black = v.exact.iter().filter(|&&x| x == 0).count();
        eprintln!("reference {}: wrong {wrong}, off by >50 {off50}, black {black} (exact {exact_black}) of {}", r.describe(), got.len());
        dump_rgb("c_exact", &v.exact);
        dump_rgb("c_gpu", &got);
        assert_eq!(black, exact_black, "pixels shadowed the reference");
        assert!(off50 * 100 < got.len(), "{off50} pixels off by >50 iterations");
    }

    /// Measurement: nucleus search below the 2^-314 view, zooming about its
    /// centre. Times the search, checks whether the nucleus orbit is full at
    /// the working precision, and at 2·log2|dz_p/dc| + 64 bits.
    #[test]
    #[ignore]
    fn diag_deep_nucleus_search() {
        let base = "0.3626806181691852804489917225076056798812848870599955358041302416758614320764231653400389334599052087605123550187040027890282,-0.6426879938460872964212498601513047756237090805248048169033884352653961260155207050615760571472782180516346029887216862939575|3.290623677226253e-95";
        let iters = 262144;
        let (w, h) = (3000usize, 2000usize);
        let b = sample_view_geometry(base, iters, (w, h), (1, 1));
        let levels: Vec<f64> = std::env::var("DIAG_LEVELS").ok()
            .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
            .unwrap_or(vec![-316.0, -320.0, -325.0, -330.0]);
        for l in levels {
            let view = l.exp2();
            let mut coords: CoordinatesBox = base.parse().unwrap();
            let prec = working_precision(crate::tiles::store::depth_for_view(view)) + 64;
            let off = |px: usize| FBig::try_from(px as f64 / 2.0 * view).unwrap();
            coords.origin.x = (&b.geom.center.re - &off(w)).with_precision(prec).value();
            coords.origin.y = (&b.geom.center.im - &off(h)).with_precision(prec).value();
            coords.view.inner = view;
            let clip = format!("{coords}");
            let v = sample_view_geometry(&clip, iters, (w, h), (1, 1));
            let g = &v.geom;
            if std::env::var("DIAG_PIPE").is_ok() {
                let _ = env_logger::builder().is_test(true).try_init();
                let backend = Gpu(Arc::new(GpuState::new()));
                let store = crate::tiles::store::TileStore::new(crate::tiles::store::MEMORY_BUDGET_BYTES);
                let t = std::time::Instant::now();
                run_pipeline(&v, &backend, &store, 1);
                eprintln!("2^{l}: one generation in {:?}", t.elapsed());
                continue;
            }
            for mult in [1u32, 4, 16] {
                let r = &g.radius * FBig::from(mult);
                let t = std::time::Instant::now();
                let p = crate::tiles::perturb::nucleus::ball_period(&g.center, &r, iters, g.prec, &|| false).unwrap();
                let t_ball = t.elapsed();
                let t = std::time::Instant::now();
                let res = p.map(|p| {
                    let tol = FBig::ONE << (g.upp - 40) as isize;
                    let md = &g.radius * FBig::from(MAX_REF_DIST);
                    newton_nucleus(&g.center, p, g.prec, &(&tol * &tol), &(&md * &md), &|| false).unwrap().is_some()
                });
                eprintln!("    x{mult}: period {p:?} ({t_ball:?}), newton ok {res:?} ({:?})", t.elapsed());
            }
            let t = std::time::Instant::now();
            let n = find_nucleus(&g.center, &g.radius, g.upp, iters, g.prec, &|| false).unwrap();
            let t_search = t.elapsed();
            let Some(n) = n else { eprintln!("2^{l}: no nucleus ({t_search:?})"); continue };
            // log2 |dz_p/dc| at the nucleus (z_0 = 0 convention).
            let (mut z, mut dz) = (Complex { re: FBig::ZERO, im: FBig::ZERO }, Complex { re: FBig::ZERO, im: FBig::ZERO });
            let c = &n.c;
            for _ in 0..n.period {
                let zdz = &z * &dz;
                dz = Complex { re: (zdz.re << 1) + FBig::ONE, im: zdz.im << 1 };
                z = &z * &z + c;
            }
            let l_dz = (&dz.re * &dz.re + &dz.im * &dz.im).to_f64().value().log2() / 2.0;
            let t = std::time::Instant::now();
            let r = Reference::new(n.c.clone(), iters, Some(n.period));
            eprintln!("2^{l}: p={} in {t_search:?}, prec {}, log2|dz_p| {l_dz:.0}, orbit {} ({:?})",
                n.period, g.prec, r.describe(), t.elapsed());
        }
    }

    /// A pixel dispatched against its own (escaping) orbit, δ₀ = 0 exactly,
    /// must escape with it, in both pipelines. Needs a GPU.
    #[test]
    #[ignore]
    fn zero_delta_escapes_with_reference() {
        let clip = "0.3626806181691852804489917225076056798812848870599955358041302416758614320764231653400389334599052087605123550187040027890282,-0.6426879938460872964212498601513047756237090805248048169033884352653961260155207050615760571472782180516346029887216862939575|3.290623677226253e-95";
        let v = sample_view_geometry(clip, 262144, (3000, 2000), (1, 1));
        let g = &v.geom;
        let r = Reference::new(g.center.clone(), 262144, None);
        let exact = check_orbit(&calculate_orbit(g.center.clone(), 262144).1).unwrap().map_or(0, |n| n.get() as u32);
        let (rx, ry) = ref_px(&r.c, &v.ctx);
        let gpu = GpuState::new();
        let offs = [(0.0, 0.0), (1e-3, 0.0)];
        let gr = gpu.prepare_with(&r, log2_dc(&offs, g.upp), g.upp, false);
        eprintln!("BLA off: {:x?}", gpu.dispatch_offsets(&offs, g.upp, &gr));
        let gr = gpu.prepare(&r, log2_dc(&offs, g.upp), g.upp);
        let got = gpu.dispatch_offsets(&offs, g.upp, &gr);
        eprintln!("reference {} at ({rx:.3}, {ry:.3}): exact {exact}, got {:x?}", r.describe(), got);
        assert_eq!(got[0], exact);
    }

    /// Measurement: accuracy and speed with a far-away reference. The 2^-314
    /// view's nucleus is reused at views zoomed in about its centre (the
    /// reference ends up hundreds to ~10^6 view radii away), against exact
    /// orbits and against each view's own nucleus. Exact values cached in
    /// $DIAG_OUT (~10 s per view).
    #[test]
    #[ignore]
    fn diag_far_reference() {
        let base = "0.3626806181691852804489917225076056798812848870599955358041302416758614320764231653400389334599052087605123550187040027890282,-0.6426879938460872964212498601513047756237090805248048169033884352653961260155207050615760571472782180516346029887216862939575|3.290623677226253e-95";
        let iters = 262144;
        let (w, h) = (3000usize, 2000usize);
        let b = sample_view_geometry(base, iters, (w, h), (1, 1));
        let g0 = &b.geom;
        let n0 = find_nucleus(&g0.center, &g0.radius, g0.upp, iters, g0.prec, &|| false).unwrap().expect("nucleus");
        let far = Reference::new(n0.c.clone(), iters, Some(n0.period));
        let gpu = GpuState::new();
        let levels: Vec<f64> = std::env::var("DIAG_LEVELS").ok()
            .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
            .unwrap_or(vec![-314.0, -318.0, -322.0, -326.0, -330.0]);
        // DIAG_PIPE: whole generations at each level in turn with one
        // backend (a zoom), timing each and scoring the stored pixels.
        let pipe = std::env::var("DIAG_PIPE").is_ok().then(|| {
            let _ = env_logger::builder().is_test(true).try_init();
            (Gpu(Arc::new(GpuState::new())), crate::tiles::store::TileStore::new(crate::tiles::store::MEMORY_BUDGET_BYTES))
        });
        for l in levels {
            let view = l.exp2();
            let mut coords: CoordinatesBox = base.parse().unwrap();
            let prec = working_precision(crate::tiles::store::depth_for_view(view)) + 64;
            let off = |px: usize| FBig::try_from(px as f64 / 2.0 * view).unwrap();
            coords.origin.x = (&g0.center.re - &off(w)).with_precision(prec).value();
            coords.origin.y = (&g0.center.im - &off(h)).with_precision(prec).value();
            coords.view.inner = view;
            let clip = format!("{coords}");
            let v = sample_view_cached(&format!("far{}", -l as i64), &clip, iters, (w, h), (30, 20));
            let g = &v.geom;
            if let Some((backend, store)) = &pipe {
                // DIAG_FRESH: a new backend per level, like pasting the view.
                let fresh = std::env::var("DIAG_FRESH").is_ok().then(|| {
                    (Gpu(Arc::new(GpuState::new())), crate::tiles::store::TileStore::new(crate::tiles::store::MEMORY_BUDGET_BYTES))
                });
                let (backend, store) = fresh.as_ref().map_or((backend, store), |(b, s)| (b, s));
                let t = std::time::Instant::now();
                let vals = run_pipeline(&v, backend, store, 1).remove(0);
                let got: Vec<u32> = vals.iter().map(|x| x.unwrap_or(u32::MAX)).collect();
                let (wrong, off50, black) = score(&got, &v.exact);
                eprintln!("2^{l} pipeline: {:?}, missing {}, wrong {wrong} off>50 {off50} black {black}",
                    t.elapsed(), vals.iter().filter(|x| x.is_none()).count());
                continue;
            }
            let own = find_nucleus(&g.center, &g.radius, g.upp, iters, g.prec, &|| false).unwrap()
                .map(|n| Reference::new(n.c, iters, Some(n.period)));
            for (name, r) in [("far", Some(&far)), ("own", own.as_ref())] {
                let Some(r) = r else { eprintln!("2^{l} {name}: none"); continue };
                let (rx, ry) = ref_px(&r.c, &v.ctx);
                let offs: Vec<(f64, f64)> = v.pix.iter().map(|&(c, rw)| (c as f64 - rx, rw as f64 - ry)).collect();
                let radii = rx.hypot(ry) / (w as f64).hypot(h as f64) * 2.0;
                let gr = gpu.prepare(r, log2_dc(&offs, g.upp), g.upp);
                let t = std::time::Instant::now();
                let got: Vec<u32> = offs.chunks(256).flat_map(|c| gpu.dispatch_offsets(c, g.upp, &gr)).collect();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let (wrong, off50, black) = score(&got, &v.exact);
                let exact_black = v.exact.iter().filter(|&&x| x == 0).count();
                let glitched = got.iter().filter(|&&x| x & GLITCH_BIT != 0).count();
                eprintln!("2^{l} {name} ({}, {radii:.0} radii): wrong {wrong} off>50 {off50} black {black} (exact {exact_black}) glitched {glitched}, {ms:.0} ms, BLA {} levels",
                    r.describe(), gr.bla_levels);
            }
        }
    }
}
