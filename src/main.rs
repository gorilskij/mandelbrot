mod drawing;
mod rendering;

use crate::drawing::Drawer;
use euclid::{Point2D, Scale};
use log::{info, trace};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use num_bigfloat::BigFloat;
use rendering::*;

fn main() {
    env_logger::init();

    // let width = 1000;
    // let height = 600;
    let width = 1500;
    let height = 1000;

    let mut window = Window::new("Mandelbrot", width, height, WindowOptions::default()).unwrap();

    window.set_target_fps(60);

    let mut coords = CoordinatesBox {
        // these (origin, view) are always in sync with (zoomed.origin, zoomed.view)
        origin: Point2D::new(BigFloat::from(-2.5), BigFloat::from(-1.0)),
        // range (1) / pixel
        view: View::new(1.0 / 300.0),
    };

    let mut iterations = 2048;

    let mut drawer = Drawer::new(width, height, coords, iterations);

    let mut dragging = None::<Point2D<BigFloat, Pixels>>;

    while window.is_open() {
        let mut zoomed = false;
        let mut dragged = false;

        // mouse position in pixels with the top-left corner of the window as the origin
        let cursor_rel = window
            .get_mouse_pos(MouseMode::Discard)
            .map(|(x, y)| Point2D::<_, Pixels>::new(BigFloat::from(x), BigFloat::from(y)));

        if let Some(cursor_rel) = cursor_rel {
            let view_bf = Scale::<_, Pixels, Units>::new(BigFloat::from(coords.view.get()));

            if window.get_mouse_down(MouseButton::Left) {
                if dragging.is_none() {
                    trace!("start dragging");
                }

                if let Some(last) = dragging {
                    dragged = true;

                    let drag: Point2D<_, Pixels> = (cursor_rel - last).to_point();
                    coords.origin -= (drag * view_bf).to_vector();
                }
                dragging = Some(cursor_rel);
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

                    let cursor_abs = coords.origin + (cursor_rel * view_bf).to_vector();
                    let multiplier = 1.0 + (scroll_y as f64 / 100.0).clamp(-0.2, 0.2);

                    coords.origin =
                        cursor_abs + (coords.origin - cursor_abs) / BigFloat::from(multiplier);
                    coords.view = View::new(coords.view.0 / multiplier);
                }
            }
        }

        // cast to usize
        let cursor_rel = cursor_rel
            .map(|p| Point2D::<_, Pixels>::new(p.x.to_f64() as usize, p.y.to_f64() as usize));

        if dragged || zoomed {
            trace!("send update to drawer");
            drawer.update(coords, iterations, cursor_rel);
            trace!("done sending update to drawer");
        } else if window.is_key_pressed(Key::Up, KeyRepeat::No) {
            iterations *= 2;
            info!("iterations: {iterations}");
            drawer.update(coords, iterations, cursor_rel);
        } else if window.is_key_pressed(Key::Down, KeyRepeat::No) && iterations > 1 {
            iterations /= 2;
            info!("iterations: {iterations}");
            drawer.update(coords, iterations, cursor_rel);
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
