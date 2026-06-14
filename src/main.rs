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
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::drawing::Drawer;
use crate::gpu_compositor::{GpuCompositor, RenderThread};
use crate::rendering::*;
use crate::support::{Length, Point};
use crate::tiles::perturb::{Perturbator, Toggle, gpu::{Gpu, GpuState}};

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
        let new_view = old_view / multiplier;

        if let Some((cx, cy)) = self.cursor_pos {
            let exact = |f: f64| FBig::try_from(f).unwrap();
            self.coords.origin.x =
                &(&self.coords.origin.x + &exact(cx * old_view)) - &exact(cx * new_view);
            self.coords.origin.y =
                &(&self.coords.origin.y + &exact(cy * old_view)) - &exact(cy * new_view);
        }
        self.coords.view = View::new(new_view);
    }

    /// Publish the current view to the render thread, which redraws it on its
    /// next loop. Replaces the old `window.request_redraw()` path now that the
    /// main thread no longer renders.
    fn publish_view(&self) {
        if let Some(rt) = &self.render_thread {
            rt.set_view(self.coords.clone(), self.phys_width, self.phys_height);
        }
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
            }

            // Rendering is driven by the render thread, not RedrawRequested.
            WindowEvent::RedrawRequested => {}

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = Some((position.x, position.y));

                // Update drag coords immediately on every cursor move.
                if self.mouse_left_down {
                    if let Some((start_origin, start_cursor)) = self.dragging.clone() {
                        let dx = position.x - start_cursor.0;
                        let dy = position.y - start_cursor.1;
                        let view = self.coords.view.inner;
                        let exact = |f: f64| FBig::try_from(f).unwrap();
                        self.coords.origin = Point::new(
                            &start_origin.x - &exact(dx * view),
                            &start_origin.y - &exact(dy * view),
                        );
                        self.pending_update = self.pending_update.max(UpdateKind::AroundCursor);
                        self.publish_view();
                    }
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
            }

            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                let ctrl = self.modifiers.control_key() || self.modifiers.super_key();
                match code {
                    KeyCode::Space => {
                        info!("reset (spacebar)");
                        self.pending_update = UpdateKind::Reset;
                    }
                    KeyCode::Escape => {
                        let now_gpu = !self.use_gpu.load(std::sync::atomic::Ordering::Relaxed);
                        self.use_gpu.store(now_gpu, std::sync::atomic::Ordering::Relaxed);
                        info!("backend: {}", if now_gpu { "GPU" } else { "CPU" });
                        self.pending_update = UpdateKind::Reset;
                    }
                    KeyCode::ArrowUp => {
                        self.iterations *= 2;
                        info!("iterations: {}", self.iterations);
                        self.pending_update =
                            self.pending_update.max(UpdateKind::AroundCursor);
                    }
                    KeyCode::ArrowDown if self.iterations > 1 => {
                        self.iterations /= 2;
                        info!("iterations: {}", self.iterations);
                        self.pending_update =
                            self.pending_update.max(UpdateKind::AroundCursor);
                    }
                    KeyCode::KeyC if ctrl => {
                        let data = ClipBoardData {
                            coords: Cow::Borrowed(&self.coords),
                            iterations: self.iterations,
                        };
                        let s = data.to_string();
                        info!("COPIED: {s}");
                        match Clipboard::new().and_then(|mut cb| cb.set_text(s)) {
                            Ok(()) => {}
                            Err(e) => warn!("clipboard copy failed: {e}"),
                        }
                    }
                    KeyCode::KeyV if ctrl => {
                        let contents = Clipboard::new()
                            .and_then(|mut cb| cb.get_text())
                            .unwrap_or_default();
                        if let Ok(new_data) = contents.parse::<ClipBoardData>() {
                            self.coords = new_data.coords.into_owned();
                            self.iterations = new_data.iterations;
                            info!("PASTED: {}", contents.trim());
                            self.pending_update =
                                self.pending_update.max(UpdateKind::AroundCenter);
                        } else {
                            warn!(
                                "PASTE FAILED: invalid clipboard contents: {contents:?}"
                            );
                        }
                    }
                    _ => {}
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

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
