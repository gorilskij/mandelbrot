mod draw_thread;
mod rendering;

use log::trace;
use minifb::{MouseButton, MouseMode, Window, WindowOptions};
use rendering::*;

fn main() {
    env_logger::init();

    let width = 1000;
    let height = 600;

    let mut window = Window::new("Mandelbrot", width, height, WindowOptions::default()).unwrap();

    window.set_target_fps(60);

    let mut coords = CoordinatesBox {
        // these (origin, view) are always in sync with (zoomed.origin, zoomed.view)
        origin: Point::new(-2.5, -1.0),
        // range (1) / pixel
        view: View::new(1.0 / 300.0),
    };

    let draw_thread = draw_thread::spawn(width, height, coords);

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
                    coords.origin -= (drag * coords.view).to_vector();
                    {
                        let zoomed = &mut draw_thread.buffer.zoomed.lock();
                        zoomed.coords.origin = coords.origin;
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
                    let cursor_abs = coords.origin + (cursor_rel * coords.view).to_vector();
                    let multiplier = 1.0 + (scroll_y as f64 / 100.0).clamp(-0.2, 0.2);

                    coords.origin = cursor_abs + (coords.origin - cursor_abs) / multiplier;
                    coords.view = View::new(coords.view.0 / multiplier);

                    {
                        let zoomed = &mut draw_thread.buffer.zoomed.lock();
                        zoomed.coords = coords;
                    }
                }
            }
        }

        let cache_key = coords;

        if cached != Some(cache_key) {
            if first_time {
                {
                    let base = &mut draw_thread.buffer.base.lock();
                    let zoomed = &mut draw_thread.buffer.zoomed.lock();

                    render(&mut base.buffer, width, height, coords);
                    zoomed.buffer.copy_from_slice(&base.buffer);
                    base.coords = coords;
                }

                first_time = false;
            } else if dragged || zoomed {
                {
                    let base = &mut draw_thread.buffer.base.lock();
                    let zoomed = &mut draw_thread.buffer.zoomed.lock();

                    sample_zoomed(
                        &base.buffer,
                        &mut zoomed.buffer,
                        //
                        width,
                        height,
                        //
                        base.coords,
                        //
                        coords,
                    );
                }

                draw_thread.notify_run();
            }
            cached = Some(cache_key);
        }

        {
            let zoomed = &draw_thread.buffer.zoomed.lock();
            window
                .update_with_buffer(&zoomed.buffer, width, height)
                .unwrap();
        }
    }

    draw_thread.notify_terminate();
    draw_thread.join().unwrap();
}
