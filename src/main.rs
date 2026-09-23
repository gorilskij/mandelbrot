// wgpu's Send/Sync auto-trait chain (Global -> Hub -> Registry -> ...) is
// deeper than the default 128; see rust-lang/rust#159228.
#![recursion_limit = "256"]

mod drawing;
mod gpu_compositor;
mod rendering;
mod support;
mod tiles;

use std::borrow::Cow;
use std::fmt::Display;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arboard::Clipboard;
use dashu::float::FBig;
use log::{info, trace, warn};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::drawing::Drawer;
use crate::gpu_compositor::{GpuCompositor, RenderThread};
use crate::rendering::*;
use crate::support::{Length, Point};
use crate::tiles::perturb::{Perturbator, Toggle, gpu::{Gpu, GpuState}};

/// DIAG: ⌃⌥⇧⌘ for the held modifiers.
fn mod_symbols(m: ModifiersState) -> String {
    [(m.control_key(), "⌃"), (m.alt_key(), "⌥"), (m.shift_key(), "⇧"), (m.super_key(), "⌘")]
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, s)| *s)
        .collect()
}

struct ClipBoardData<'a> {
    coords: Cow<'a, CoordinatesBox>,
    iterations: usize,
}

impl Display for ClipBoardData<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.coords, self.iterations)
    }
}

impl FromStr for ClipBoardData<'_> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (coords, iterations) = s
            .split_at_checked(s.chars().position(|c| c == '/').ok_or(())?)
            .ok_or(())?;
        let iterations = &iterations[1..];
        Ok(Self {
            coords: Cow::Owned(coords.parse()?),
            iterations: iterations.parse().map_err(drop)?,
        })
    }
}

/// Priority-ordered: higher variant wins when merging two pending updates.
#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum UpdateKind {
    #[default]
    No,
    AroundCursor,
    AroundCenter,
    Reset,
}

struct App {
    window: Option<Arc<Window>>,
    render_thread: Option<RenderThread>,
    drawer: Option<Drawer>,
    backend: Arc<dyn Perturbator + Send + Sync>,
    use_gpu: Arc<std::sync::atomic::AtomicBool>,

    // Physical pixel dimensions of the window.
    phys_width: u32,
    phys_height: u32,

    // Mandelbrot state.
    coords: CoordinatesBox,
    iterations: usize,

    // Input state.
    cursor_pos: Option<(f64, f64)>,   // physical pixels
    mouse_left_down: bool,
    /// (start_origin_in_units, start_cursor_physical_pixels)
    dragging: Option<(Point<FBig, Units>, (f64, f64))>,
    got_scroll: bool,    // did we receive scroll events since last about_to_wait?
    scrolling: bool,     // were we scrolling last about_to_wait?
    modifiers: ModifiersState,

    // Queued update from key events; processed in about_to_wait.
    pending_update: UpdateKind,

    /// DIAG: last key event and what it did, shown in the title.
    key_debug: String,

    // (left, top, right, bottom) in Mandelbrot units — the initial view rectangle.
    // Nothing outside this rect is ever allowed to be visible.
    bounds: Option<(f64, f64, f64, f64)>,
}

impl App {
    fn new() -> Self {
        let precision = 100;
        let (toggle, use_gpu) = Toggle::new(Gpu(Arc::new(GpuState::new())));
        let exact = |f: f64| FBig::try_from(f).unwrap().with_precision(precision).value();
        Self {
            window: None,
            render_thread: None,
            drawer: None,
            backend: Arc::new(toggle),
            use_gpu,
            phys_width: 0,
            phys_height: 0,
            coords: CoordinatesBox {
                // Temporary origin; recentered on (-0.5, 0) in resumed() once
                // the physical window size (and thus HiDPI scale) is known.
                origin: Point::new(exact(-0.5), exact(0.0)),
                view: View::new(1.0 / 600.0),
            },
            iterations: 2048,
            cursor_pos: None,
            mouse_left_down: false,
            dragging: None,
            got_scroll: false,
            scrolling: false,
            modifiers: ModifiersState::empty(),
            pending_update: UpdateKind::No,
            key_debug: String::new(),
            bounds: None,
        }
    }

    fn resize(&mut self, phys_width: u32, phys_height: u32) {
        if phys_width == 0 || phys_height == 0 {
            return;
        }
        self.phys_width = phys_width;
        self.phys_height = phys_height;

        if let Some(rt) = &self.render_thread {
            rt.resize(phys_width, phys_height);
        }
        if let Some(drawer) = &mut self.drawer {
            drawer.resize(phys_width as usize, phys_height as usize, self.iterations);
        }
    }

    fn bump_precision(&mut self) {
        let needed_bits = (96.0 - self.coords.view.inner.log2().min(0.0)) as usize;
        if self.coords.origin.x.precision() < needed_bits
            || self.coords.origin.y.precision() < needed_bits
        {
            self.coords.origin = self.coords.origin.cast(|f| {
                if f.precision() < needed_bits {
                    f.clone().with_precision(needed_bits).value()
                } else {
                    f.clone()
                }
            });
        }
    }

    fn apply_scroll_zoom(&mut self, delta_y: f64) {
        let multiplier = 1.0 + (delta_y / 100.0).clamp(-0.2, 0.2);
        let old_view = self.coords.view.inner;
        // Clamp new_view against the zoom-out limit *before* computing the
        // origin shift so the cursor is anchored to the view that will actually
        // be applied.  If we clamped only after, the origin would be computed
        // for a view that's then discarded, sending the image in the wrong
        // direction regardless of cursor position.
        let new_view = {
            let raw = old_view / multiplier;
            if let Some((bl, bt, br, bb)) = self.bounds {
                if self.phys_width > 0 && self.phys_height > 0 {
                    let max_view = f64::min(
                        (br - bl) / self.phys_width as f64,
                        (bb - bt) / self.phys_height as f64,
                    );
                    raw.min(max_view)
                } else {
                    raw
                }
            } else {
                raw
            }
        };

        if let Some((cx, cy)) = self.cursor_pos {
            let exact = |f: f64| FBig::try_from(f).unwrap();
            self.coords.origin.x =
                &(&self.coords.origin.x + &exact(cx * old_view)) - &exact(cx * new_view);
            self.coords.origin.y =
                &(&self.coords.origin.y + &exact(cy * old_view)) - &exact(cy * new_view);
        }
        self.coords.view = View::new(new_view);
        self.clamp_to_bounds(); // still needed to clamp origin within bounds
    }

    /// Clamp view scale and origin so the visible rect stays inside the initial
    /// bounding rectangle. Edges that hit the boundary stay fixed; the opposite
    /// side continues until the full initial view is restored.
    fn clamp_to_bounds(&mut self) {
        let Some((bl, bt, br, bb)) = self.bounds else { return };
        if self.phys_width == 0 || self.phys_height == 0 { return }
        let bw = br - bl;
        let bh = bb - bt;
        let pw = self.phys_width as f64;
        let ph = self.phys_height as f64;

        // Zoom-out limit: visible area must not exceed the bounding rect.
        let max_view = f64::min(bw / pw, bh / ph);
        if self.coords.view.inner > max_view {
            self.coords.view = View::new(max_view);
        }

        let view = self.coords.view.inner;
        let vw = pw * view;
        let vh = ph * view;

        let cur_x = self.coords.origin.x.to_f64().value();
        let new_x = cur_x.clamp(bl, br - vw);
        if new_x != cur_x {
            self.coords.origin.x = FBig::try_from(new_x).unwrap();
        }

        let cur_y = self.coords.origin.y.to_f64().value();
        let new_y = cur_y.clamp(bt, bb - vh);
        if new_y != cur_y {
            self.coords.origin.y = FBig::try_from(new_y).unwrap();
        }
    }

    /// Publish the current view to the render thread, which redraws it on its
    /// next loop. Replaces the old `window.request_redraw()` path now that the
    /// main thread no longer renders.
    fn publish_view(&self) {
        if let Some(rt) = &self.render_thread {
            rt.set_view(self.coords.clone(), self.phys_width, self.phys_height);
        }
    }

    /// DIAG: show zoom / depth / pipeline in the window title so the numbers
    /// at which rendering breaks can be read off directly.
    fn update_title(&self) {
        let Some(window) = &self.window else { return };
        let view  = self.coords.view.inner;
        let depth = crate::tiles::store::depth_for_view(view);
        let upp   = crate::tiles::store::upp_log2(depth);
        let gpu   = self.use_gpu.load(std::sync::atomic::Ordering::Relaxed);
        let pipe  = if !gpu { "cpu" }
            else if upp < crate::tiles::perturb::gpu::FE_THRESHOLD { "fe" }
            else { "f32" };
        let (cx, cy) = self.center_coord_f64();
        window.set_title(&format!(
            "Mandelbrot | {} | view {:.4e} (2^{:.2}) | depth {} upp 2^{} | iters {} | centre {:.17}, {:.17} \
             | mods [{}] | key {}",
            pipe, view, view.log2(), depth, upp, self.iterations, cx, cy,
            mod_symbols(self.modifiers), self.key_debug,
        ));
    }

    /// DIAG: screen-centre coordinate, rounded to f64 (for display only).
    fn center_coord_f64(&self) -> (f64, f64) {
        let view = self.coords.view.inner;
        let off = |px: u32| FBig::try_from(px as f64 / 2.0 * view).unwrap();
        (
            (&self.coords.origin.x + &off(self.phys_width)).to_f64().value(),
            (&self.coords.origin.y + &off(self.phys_height)).to_f64().value(),
        )
    }

    /// DIAG: record a key event for the title and the log.
    fn note_key(&mut self, e: &winit::event::KeyEvent) {
        let key = match e.physical_key {
            PhysicalKey::Code(c) => format!("{c:?}"),
            other => format!("{other:?}"),
        };
        let dir = if e.state == ElementState::Pressed { "↓" } else { "↑" };
        let rep = if e.repeat { " (repeat)" } else { "" };
        // The layout's character too: shortcuts match on it, not on `key`.
        let ch = match &e.logical_key {
            Key::Character(c) => format!(" {c:?}"),
            _ => String::new(),
        };
        self.key_debug = format!("{}{key}{ch} {dir}{rep}", mod_symbols(self.modifiers));
        info!("[diag key] {} logical {:?}", self.key_debug, e.logical_key);
    }

    /// DIAG: append what the app did with the last key press.
    fn note_action(&mut self, action: &str) {
        self.key_debug.push_str(" → ");
        self.key_debug.push_str(action);
        info!("[diag key] → {action}");
    }

    fn cursor_usize(&self) -> Option<Point<usize, Pixels>> {
        self.cursor_pos
            .map(|(x, y)| Point::new(x as usize, y as usize))
    }

    fn center(&self) -> Point<usize, Pixels> {
        Point::new(self.phys_width as usize / 2, self.phys_height as usize / 2)
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Mandelbrot")
                        .with_inner_size(LogicalSize::new(1500u32, 1000u32))
                        .with_resizable(true),
                )
                .unwrap(),
        );

        let PhysicalSize { width, height } = window.inner_size();

        // Center the view on (-0.5, 0) using the actual physical size.
        let view = self.coords.view.inner;
        let exact = |f: f64| FBig::try_from(f).unwrap();
        self.coords.origin = Point::new(
            exact(-0.5 - (width as f64 / 2.0) * view),
            exact(0.0  - (height as f64 / 2.0) * view),
        );

        // Record the initial bounding rectangle. Zoom-out and pan are clamped
        // so this rect is always the maximum visible area.
        let ox = self.coords.origin.x.to_f64().value();
        let oy = self.coords.origin.y.to_f64().value();
        self.bounds = Some((ox, oy, ox + width as f64 * view, oy + height as f64 * view));

        let compositor = GpuCompositor::new(window.clone());
        let drawer = Drawer::new(
            width as usize,
            height as usize,
            self.coords.clone(),
            self.iterations,
            self.backend.clone(),
        );

        // Hand the compositor and a store handle to the render thread; it owns
        // the surface and draws independently of the main event loop from here.
        let render_thread = RenderThread::spawn(
            compositor,
            drawer.store().clone(),
            self.coords.clone(),
            width,
            height,
        );

        self.window = Some(window);
        self.render_thread = Some(render_thread);
        self.drawer = Some(drawer);

        self.resize(width, height);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let WindowEvent::KeyboardInput { event: ref e, .. } = event {
            self.note_key(e);
        }
        match event {
            WindowEvent::CloseRequested => {
                // Stop the render thread first so it releases the surface
                // before we tear down the compute side and exit.
                if let Some(mut rt) = self.render_thread.take() {
                    rt.stop();
                }
                if let Some(drawer) = self.drawer.take() {
                    drawer.stop();
                }
                event_loop.exit();
            }

            WindowEvent::Resized(PhysicalSize { width, height }) => {
                self.resize(width, height);
                self.clamp_to_bounds();
                self.publish_view();
            }

            // Rendering is driven by the render thread, not RedrawRequested.
            WindowEvent::RedrawRequested => {}

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = Some((position.x, position.y));

                // Update drag coords immediately on every cursor move.
                if self.mouse_left_down
                    && let Some((start_origin, start_cursor)) = self.dragging.clone()
                {
                    let dx = position.x - start_cursor.0;
                    let dy = position.y - start_cursor.1;
                    let view = self.coords.view.inner;
                    let exact = |f: f64| FBig::try_from(f).unwrap();
                    self.coords.origin = Point::new(
                        &start_origin.x - &exact(dx * view),
                        &start_origin.y - &exact(dy * view),
                    );
                    self.clamp_to_bounds();
                    self.pending_update = self.pending_update.max(UpdateKind::AroundCursor);
                    self.publish_view();
                }
            }

            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => match state {
                ElementState::Pressed => {
                    self.mouse_left_down = true;
                    if let Some(cursor) = self.cursor_pos {
                        trace!("start dragging");
                        self.dragging = Some((self.coords.origin.clone(), cursor));
                    }
                }
                ElementState::Released => {
                    self.mouse_left_down = false;
                    if self.dragging.take().is_some() {
                        trace!("stop dragging");
                    }
                }
            },

            WindowEvent::MouseWheel { delta, .. } => {
                // Only zoom when not dragging (matches old behaviour).
                if !self.mouse_left_down {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y as f64 * 2.5,
                        MouseScrollDelta::PixelDelta(pos) => pos.y / 4.0,
                    };
                    self.apply_scroll_zoom(y);
                    self.bump_precision();
                    // Show the zoomed view immediately (interpolated from
                    // existing tiles); schedule the compute refresh for the
                    // about_to_wait batch so we don't thrash it per event.
                    self.pending_update = self.pending_update.max(UpdateKind::AroundCursor);
                    self.publish_view();
                    self.scrolling = true;
                    self.got_scroll = true;
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
                info!("[diag key] modifiers [{}]", mod_symbols(self.modifiers));
            }

            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        logical_key,
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                let ctrl = self.modifiers.control_key() || self.modifiers.super_key();
                // Letter shortcuts follow the character the keyboard layout
                // produces (Dvorak, AZERTY, ...), not the physical key position.
                // With Cmd held, winit on macOS reports the unmodified
                // character, so Cmd-C is Character("c") on any layout.
                let letter = match &logical_key {
                    Key::Character(s) => s.to_lowercase(),
                    _ => String::new(),
                };
                match code {
                    KeyCode::Space => {
                        info!("reset (spacebar)");
                        self.pending_update = UpdateKind::Reset;
                        self.note_action("reset");
                    }
                    KeyCode::Escape => {
                        let now_gpu = !self.use_gpu.load(std::sync::atomic::Ordering::Relaxed);
                        self.use_gpu.store(now_gpu, std::sync::atomic::Ordering::Relaxed);
                        info!("backend: {}", if now_gpu { "GPU" } else { "CPU" });
                        self.pending_update = UpdateKind::Reset;
                        self.note_action(if now_gpu { "backend GPU" } else { "backend CPU" });
                    }
                    KeyCode::ArrowUp => {
                        self.iterations *= 2;
                        info!("iterations: {}", self.iterations);
                        self.pending_update =
                            self.pending_update.max(UpdateKind::AroundCursor);
                        self.note_action("iterations ×2");
                    }
                    KeyCode::ArrowDown if self.iterations > 1 => {
                        self.iterations /= 2;
                        info!("iterations: {}", self.iterations);
                        self.pending_update =
                            self.pending_update.max(UpdateKind::AroundCursor);
                        self.note_action("iterations ÷2");
                    }
                    _ if ctrl && letter == "c" => {
                        let data = ClipBoardData {
                            coords: Cow::Borrowed(&self.coords),
                            iterations: self.iterations,
                        };
                        let s = data.to_string();
                        info!("COPIED: {s}");
                        match Clipboard::new().and_then(|mut cb| cb.set_text(s)) {
                            Ok(()) => self.note_action("copied"),
                            Err(e) => {
                                warn!("clipboard copy failed: {e}");
                                self.note_action("copy FAILED (clipboard error)");
                            }
                        }
                    }
                    _ if ctrl && letter == "v" => {
                        let contents = Clipboard::new()
                            .and_then(|mut cb| cb.get_text())
                            .unwrap_or_default();
                        if let Ok(new_data) = contents.parse::<ClipBoardData>() {
                            self.coords = new_data.coords.into_owned();
                            self.iterations = new_data.iterations;
                            info!("PASTED: {}", contents.trim());
                            self.pending_update =
                                self.pending_update.max(UpdateKind::AroundCenter);
                            self.note_action("pasted");
                        } else {
                            warn!(
                                "PASTE FAILED: invalid clipboard contents: {contents:?}"
                            );
                            self.note_action("paste FAILED (not coordinates)");
                        }
                    }
                    _ => self.note_action("no action"),
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Detect scroll-end: got_scroll is set by each MouseWheel event.
        // When it's false for a whole about_to_wait cycle, the gesture ended.
        if std::mem::take(&mut self.got_scroll) {
            // Scroll in progress — zoom already applied per event, log zoom level.
            let w = Length::<_, Pixels>::new(self.phys_width as f64) * self.coords.view;
            info!("zoom level: 10^{}", -w.inner.log10());
        } else if self.scrolling {
            self.scrolling = false;
            // Final update once the scroll gesture ends.
            self.pending_update = self.pending_update.max(UpdateKind::AroundCursor);
        }

        self.bump_precision();

        // Apply pending drawer update.
        let update = std::mem::take(&mut self.pending_update);
        let cursor = self.cursor_usize();
        let center = self.center();
        let coords = self.coords.clone();
        let iters = self.iterations;
        if let Some(drawer) = &mut self.drawer {
            match update {
                UpdateKind::No => {}
                UpdateKind::AroundCursor => {
                    drawer.update(coords, iters, cursor.as_ref());
                }
                UpdateKind::AroundCenter => {
                    drawer.update(coords, iters, Some(&center));
                }
                UpdateKind::Reset => {
                    drawer.reset(coords, iters, Some(&center));
                }
            }
        }
        self.update_title();
        if update != UpdateKind::No {
            // Push the (possibly changed) view so the render thread repaints.
            self.publish_view();
        }

        // The render thread owns continuous redraw, so the main thread is now
        // purely event-driven. The one exception is scroll-end detection: it
        // relies on a follow-up about_to_wait after the gesture stops, so while
        // a scroll is in flight we schedule a short wakeup; otherwise we sleep
        // until the next OS event.
        if self.scrolling {
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(150),
            ));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

/// DIAG: writes every log line to both stderr and `gpu.log` (flushed per
/// line, so the file is complete even if the app hangs and gets killed).
struct Tee(std::fs::File);

impl std::io::Write for Tee {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let _ = std::io::stderr().write_all(buf);
        self.0.write_all(buf)?;
        self.0.flush()?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { self.0.flush() }
}

fn main() {
    // DIAG: default to `info` (RUST_LOG still overrides) and tee into gpu.log.
    let mut logger = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
    match std::fs::File::create("gpu.log") {
        Ok(file) => { logger.target(env_logger::Target::Pipe(Box::new(Tee(file)))); }
        Err(e)   => eprintln!("could not create gpu.log: {e}"),
    }
    logger.init();
    info!("logging to {}", std::env::current_dir().map(|d| d.join("gpu.log").display().to_string()).unwrap_or_default());
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
