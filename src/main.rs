mod rendering;

use log::trace;
use minifb::{MouseButton, MouseMode, Window, WindowOptions};
use parking_lot::{Condvar, Mutex};
use rendering::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{mem, thread};

struct SubBuffer {
    buffer: Box<[u32]>,
    origin: Point<Units>,
    view: View,
}

impl SubBuffer {
    fn new(width: usize, height: usize, origin: Point<Units>, view: View) -> Self {
        Self {
            buffer: vec![0; width * height].into_boxed_slice(),
            origin,
            view,
        }
    }
}

#[derive(Clone)]
struct Buffer {
    base: Arc<Mutex<SubBuffer>>,
    zoomed: Arc<Mutex<SubBuffer>>,
}

impl Buffer {
    fn new(width: usize, height: usize, origin: Point<Units>, view: View) -> Self {
        Self {
            base: Arc::new(Mutex::new(SubBuffer::new(width, height, origin, view))),
            zoomed: Arc::new(Mutex::new(SubBuffer::new(width, height, origin, view))),
        }
    }
}

#[derive(Debug)]
enum RedrawThreadState {
    Run { view_ratio: f64 },
    Wait,
    Terminate,
}

fn main() {
    env_logger::init();

    let width = 1000;
    let height = 600;

    let mut window = Window::new("Mandelbrot", width, height, WindowOptions::default()).unwrap();

    window.set_target_fps(60);

    // these (origin, view) are always in sync with (zoomed.origin, zoomed.view)
    let mut origin = Point::<Units>::new(-2.5, -1.0);
    // range (1) / pixel
    let mut view = View::new(1.0 / 300.0);

    let buffer = Buffer::new(width, height, origin, view);

    let wake_redraw_thread = Arc::new((Mutex::new(RedrawThreadState::Wait), Condvar::new()));
    let redraw_thread_missed_update = Arc::new(Mutex::new(false));

    let redraw_thread = {
        let buffer = buffer.clone();
        let wake_redraw_thread = wake_redraw_thread.clone();
        let redraw_thread_missed_update = redraw_thread_missed_update.clone();

        thread::spawn(move || {
            let mut tmp_buffer = vec![0; width * height].into_boxed_slice();
            let (lock, cvar) = &*wake_redraw_thread;

            let mut lock = lock.lock();
            let mut last_update = Instant::now();
            loop {
                trace!("re: wait");

                cvar.wait_for(&mut lock, Duration::from_millis(200));

                trace!("re: woken {:?}", *lock);

                match mem::replace(&mut *lock, RedrawThreadState::Wait) {
                    RedrawThreadState::Run { view_ratio } => {
                        if view_ratio.ln().abs() < 1.1_f64.ln()
                            && last_update.elapsed().as_millis() < 500
                        {
                            continue;
                        }
                    }
                    RedrawThreadState::Wait => {
                        let missed_update = &mut *redraw_thread_missed_update.lock();
                        if *missed_update {
                            *missed_update = false
                        } else {
                            continue;
                        }
                    }
                    RedrawThreadState::Terminate => break,
                };

                trace!("re: redrawing");

                let (origin, view) = {
                    let zoomed = buffer.zoomed.lock();
                    (zoomed.origin, zoomed.view)
                };

                trace!("re: rendering");
                render(&mut tmp_buffer, width, height, origin, view);
                trace!("re: done rendering");

                {
                    let base = &mut buffer.base.lock();
                    trace!("re: base acquired");

                    base.buffer.copy_from_slice(&tmp_buffer);
                    base.origin = origin;
                    base.view = view;

                    let zoomed = &mut buffer.zoomed.lock();

                    let zoomed_origin = zoomed.origin;
                    let zoomed_view = zoomed.view;

                    trace!("re: sample zooming");
                    sample_zoomed(
                        &base.buffer,
                        &mut zoomed.buffer,
                        //
                        width,
                        height,
                        //
                        base.origin,
                        base.view,
                        //
                        zoomed_origin,
                        zoomed_view,
                    );
                    trace!("re: done sample zooming");
                }

                last_update = Instant::now();
            }
        })
    };

    let mut first_time = true;
    let mut cached = None;
    let mut dragging = None;

    while window.is_open() {
        let mut zoomed = false;
        let mut dragged = false;

        if let Some((mouse_x, mouse_y)) = window.get_mouse_pos(MouseMode::Discard) {
            let left_mouse_down = window.get_mouse_down(MouseButton::Left);

            if left_mouse_down {
                if dragging.is_none() {
                    trace!("start dragging");
                }

                if let Some((last_x, last_y)) = dragging {
                    dragged = true;

                    let drag =
                        Point::<Pixels>::new((mouse_x - last_x) as f64, (mouse_y - last_y) as f64);
                    origin -= (drag * view).to_vector();
                    {
                        let zoomed = &mut buffer.zoomed.lock();
                        zoomed.origin = origin;
                    }
                }
                dragging = Some((mouse_x, mouse_y));
            } else {
                if dragging.is_some() {
                    trace!("stop dragging");
                    // perform an update on release
                    dragged = true;
                }

                dragging = None;

                // scroll wheel is ignored while dragging
                if let Some((_, scroll_y)) = window.get_scroll_wheel() {
                    zoomed = true;

                    let cursor_rel = Point::<Pixels>::new(mouse_x as f64, mouse_y as f64);
                    let cursor_abs = origin + (cursor_rel * view).to_vector();
                    let multiplier = 1.0 + (scroll_y as f64 / 100.0).clamp(-0.2, 0.2);

                    origin = cursor_abs + (origin - cursor_abs) / multiplier;
                    view = View::new(view.0 / multiplier);

                    {
                        let zoomed = &mut buffer.zoomed.lock();
                        zoomed.origin = origin;
                        zoomed.view = view;
                    }
                }
            }
        }

        let cache_key = (origin, view);

        if cached != Some(cache_key) {
            if first_time {
                {
                    let base = &mut buffer.base.lock();
                    let zoomed = &mut buffer.zoomed.lock();

                    render(&mut base.buffer, width, height, origin, view);
                    zoomed.buffer.copy_from_slice(&base.buffer);
                    base.origin = origin;
                    base.view = view;
                }

                first_time = false;
            } else if dragged || zoomed {
                let view_ratio = {
                    let base = &mut buffer.base.lock();
                    let zoomed = &mut buffer.zoomed.lock();

                    sample_zoomed(
                        &base.buffer,
                        &mut zoomed.buffer,
                        //
                        width,
                        height,
                        //
                        base.origin,
                        base.view,
                        //
                        origin,
                        view,
                    );

                    base.view.0 / zoomed.view.0
                };

                if let Some(mut lock) = wake_redraw_thread.0.try_lock() {
                    *lock = RedrawThreadState::Run { view_ratio };
                    wake_redraw_thread.1.notify_all();
                } else {
                    *redraw_thread_missed_update.lock() = true;
                }
            }
            cached = Some(cache_key);
        }

        {
            let zoomed = &buffer.zoomed.lock();
            window
                .update_with_buffer(&zoomed.buffer, width, height)
                .unwrap();
        }
    }

    *wake_redraw_thread.0.lock() = RedrawThreadState::Terminate;
    wake_redraw_thread.1.notify_all();
    redraw_thread.join().unwrap();
}
