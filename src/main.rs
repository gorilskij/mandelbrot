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

use clipboard::{ClipboardContext, ClipboardProvider};
use dashu::float::FBig;
use log::{info, trace, warn};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::drawing::Drawer;
use crate::gpu_compositor::GpuCompositor;
use crate::rendering::*;
use crate::support::{Length, Point, ToFBig};
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
    compositor: Option<GpuCompositor>,
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
        Self {
            window: None,
            compositor: None,
            drawer: None,
            backend: Arc::new(toggle),
            use_gpu,
            phys_width: 0,
            phys_height: 0,
            coords: CoordinatesBox {
                origin: Point::new(
                    (-2.5_f64).to_fbig_with_precision(precision),
                    (-1.0_f64).to_fbig_with_precision(precision),
                ),
                view: View::new(1.0 / 300.0),
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

        if let Some(compositor) = &mut self.compositor {
            compositor.resize(phys_width, phys_height);
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

    fn call_drawer_update(&mut self) {
        let coords = self.coords.clone();
        let iters = self.iterations;
        let cursor = self.cursor_usize();
        if let Some(drawer) = &mut self.drawer {
            drawer.update(coords, iters, cursor.as_ref());
        }
    }

    fn do_request_redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
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

        let compositor = GpuCompositor::new(window.clone());
        let drawer = Drawer::new(
            width as usize,
            height as usize,
            self.coords.clone(),
            self.iterations,
            self.backend.clone(),
        );

        self.window = Some(window);
        self.compositor = Some(compositor);
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
                if let Some(drawer) = self.drawer.take() {
                    drawer.stop().unwrap();
                }
                event_loop.exit();
            }

            WindowEvent::Resized(PhysicalSize { width, height }) => {
                self.resize(width, height);
            }

            WindowEvent::RedrawRequested => {
                let (Some(compositor), Some(drawer)) =
                    (&mut self.compositor, &self.drawer)
                else {
                    return;
                };
                compositor.render(
                    drawer.store(),
                    &self.coords,
                    self.phys_width,
                    self.phys_height,
                );
            }

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
                        self.call_drawer_update();
                        self.do_request_redraw();
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
                    self.call_drawer_update();
                    self.do_request_redraw();
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
                        match ClipboardProvider::new()
                            .and_then(|mut ctx: ClipboardContext| ctx.set_contents(s))
                        {
                            Ok(()) => {}
                            Err(e) => warn!("clipboard copy failed: {e}"),
                        }
                    }
                    KeyCode::KeyV if ctrl => {
                        let mut ctx: ClipboardContext =
                            ClipboardProvider::new().unwrap();
                        let contents = ctx.get_contents().unwrap_or_default();
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

        // Pace to ~60 fps and request a redraw every frame so rendering
        // progress is shown continuously.
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(16),
        ));
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
