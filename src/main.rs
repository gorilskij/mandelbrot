mod drawing;
mod rendering;

use crate::drawing::Drawer;
use log::{info, trace};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
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

    let mut iterations = 2048;

    let mut drawer = Drawer::new(width, height, coords, iterations);

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
                }
                dragging = Some((mouse_x, mouse_y));
            } else {
                if dragging.is_some() {
                    trace!("stop dragging");
                    // perform an update on release
                    dragged = true;
                }

                dragging = None;

                // scroll wheel is only read outside of dragging
                if let Some((_, scroll_y)) = window.get_scroll_wheel() {
                    zoomed = true;

                    let cursor_rel = Point::<Pixels>::new(mouse_x as f64, mouse_y as f64);
                    let cursor_abs = coords.origin + (cursor_rel * coords.view).to_vector();
                    let multiplier = 1.0 + (scroll_y as f64 / 100.0).clamp(-0.2, 0.2);

                    coords.origin = cursor_abs + (coords.origin - cursor_abs) / multiplier;
                    coords.view = View::new(coords.view.0 / multiplier);
                }
            }
        }

        if dragged || zoomed {
            trace!("send update to drawer");
            drawer.update(coords, iterations);
            trace!("done sending update to drawer");
        } else if window.is_key_pressed(Key::Up, KeyRepeat::No) {
            iterations *= 2;
            info!("iterations: {iterations}");
            drawer.update(coords, iterations);
        } else if window.is_key_pressed(Key::Down, KeyRepeat::No) {
            if iterations > 1 {
                iterations /= 2;
                info!("iterations: {iterations}");
                drawer.update(coords, iterations);
            }
        }

        drawer.update_display_buf();
        if drawer.try_cache_buf() {
            trace!("updated cache buffer");
        }

        window
            .update_with_buffer(drawer.display_buf(), width, height)
            .unwrap();
    }

    drawer.stop().unwrap();
}
