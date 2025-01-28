mod rendering;

use minifb::{Key, KeyRepeat, MouseMode, Window, WindowOptions};
use parking_lot::Mutex;
use rayon::prelude::*;
use rendering::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

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

fn main() {
    let width = 1000;
    let height = 600;

    let mut window = Window::new("Mandelbrot", width, height, WindowOptions::default()).unwrap();

    window.set_target_fps(60);

    let mut origin = Point::<Units>::new(-2.5, -1.0);

    // range (1) / pixel
    let mut view = View::new(1.0 / 300.0);

    let buffer = Buffer::new(width, height, origin, view);

    let mut cached = None;

    let mut first_time = true;

    let redraw_thread_running = Arc::new(AtomicBool::new(true));

    let redraw_thread = {
        let buffer = buffer.clone();
        let redraw_thread_running_cloned = redraw_thread_running.clone();
        thread::spawn(move || {
            let mut tmp_buffer = vec![0; width * height].into_boxed_slice();

            while redraw_thread_running_cloned.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(2));

                println!("redrawing");

                let (origin, view) = {
                    let zoomed = buffer.zoomed.lock();
                    (zoomed.origin, zoomed.view)
                };

                render(&mut tmp_buffer, width, height, origin, view);

                {
                    let base = &mut buffer.base.lock();

                    base.buffer.copy_from_slice(&tmp_buffer);
                    base.origin = origin;
                    base.view = view;

                    let zoomed = &mut buffer.zoomed.lock();

                    let zoomed_origin = zoomed.origin;
                    let zoomed_view = zoomed.view;

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
                }
            }
        })
    };

    while window.is_open() {
        let mut zoomed = false;

        if let Some((mouse_x, mouse_y)) = window.get_mouse_pos(MouseMode::Discard) {
            if let Some((_, scroll_y)) = window.get_scroll_wheel() {
                zoomed = true;

                // [0, 1] in the reference frame given by the window
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

        let cache_key = (origin, view);
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            {
                let base = &mut buffer.base.lock();
                let zoomed = &mut buffer.zoomed.lock();

                render(&mut base.buffer, width, height, origin, view);
                zoomed.buffer.copy_from_slice(&base.buffer);
                base.origin = origin;
                base.view = view;
            }

            cached = Some(cache_key);
        } else {
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
                } else if zoomed {
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
                }
                cached = Some(cache_key);
            }
        }

        {
            let zoomed = &buffer.zoomed.lock();
            window
                .update_with_buffer(&zoomed.buffer, width, height)
                .unwrap();
        }
    }

    redraw_thread_running.store(false, Ordering::Relaxed);
    redraw_thread.join().unwrap();
}
