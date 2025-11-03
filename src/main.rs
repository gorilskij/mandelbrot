#![feature(assert_matches)]

mod drawing;
mod rendering;
mod support;

use crate::{
    drawing::Drawer,
    support::{Length, Point},
};
use dashu::float::FBig;
use log::{info, trace};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use rendering::*;
use support::ToFBig;

fn main() {
    env_logger::init();

    // let width = 1000;
    // let height = 600;
    let width = 1500;
    let height = 1000;

    let mut window = Window::new("Mandelbrot", width, height, WindowOptions::default()).unwrap();

    window.set_target_fps(60);

    let precision = 100;
    let mut coords = CoordinatesBox {
        // these (origin, view) are always in sync with (zoomed.origin, zoomed.view)
        origin: Point::new(
            -2.5.to_fbig_with_precision(precision),
            -1.0.to_fbig_with_precision(precision),
        ),
        // range (1) / pixel
        view: View::new(1.0 / 300.0),
    };

    let mut iterations = 2048;

    let mut drawer = Drawer::new(width, height, coords.clone(), iterations);

    let mut dragging = None::<Point<FBig, Pixels>>;

    while window.is_open() {
        let mut zoomed = false;
        let mut dragged = false;

        // mouse position in pixels with the top-left corner of the window as the origin
        let cursor_rel = window
            .get_mouse_pos(MouseMode::Discard)
            .map(|(x, y)| Point::<_, Pixels>::new(x.to_fbig(), y.to_fbig()));

        if let Some(cursor_rel) = &cursor_rel {
            let view_fbig = coords.view.cast(move |f| f.to_fbig());

            if window.get_mouse_down(MouseButton::Left) {
                if dragging.is_none() {
                    trace!("start dragging");
                }

                if let Some(last) = dragging {
                    dragged = true;

                    let drag = cursor_rel - &last;
                    coords.origin -= &(&drag * &view_fbig);
                    // coords.origin = (
                    // &coords.origin.0 - &drag_px.0 * &view_bf_px2un,
                    // &coords.origin.1 - &drag_px.1 * &view_bf_px2un,
                    // );
                }
                dragging = Some(cursor_rel.clone());
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

                    let cursor_abs = &coords.origin + &(cursor_rel * &view_fbig);
                    trace!(
                        "zoomed at {:?} (abs)",
                        cursor_abs.cast(|f| f.to_f64().value())
                    );
                    let multiplier = 1.0 + (scroll_y as f64 / 100.0).clamp(-0.2, 0.2);

                    coords.origin =
                        &cursor_abs + &(&(&coords.origin - &cursor_abs) / &multiplier.to_fbig());
                    coords.view = View::new(coords.view.inner / multiplier);

                    // the zoom level is just the width of the screen in units
                    let width = Length::<_, Pixels>::new(width as f64) * coords.view;
                    info!("zoom level: 10^{}", -width.inner.log10());
                }
            }
        }

        'drawer_update: {
            if dragged || zoomed {
                trace!("send update to drawer");
            } else if window.is_key_pressed(Key::Up, KeyRepeat::No) {
                iterations *= 2;
                info!("iterations: {iterations}");
            } else if window.is_key_pressed(Key::Down, KeyRepeat::No) && iterations > 1 {
                iterations /= 2;
                info!("iterations: {iterations}");
            }
            // for debug
            // else if window.is_key_pressed(Key::Left, KeyRepeat::No) {
            //     // set vertical center to 0
            //     coords.origin.y = -(height as f64 / 2.0 * coords.view.inner).to_fbig()
            // } else if window.is_key_pressed(Key::Right, KeyRepeat::No) {
            //     {
            //         let width = Length::<_, Pixels>::new(width as f64) * coords.view;
            //         println!(
            //             "enter new zoom level (exponent), current: 10^{}",
            //             -width.inner.log10()
            //         );
            //     }

            //     if let Ok(Some(ans)) = read_line_stdin()
            //         && let Ok(ans) = ans.parse::<i32>()
            //     {
            //         let new_view = Scale::new(10_f64.powi(-ans) / width as f64);

            //         // adjust origin
            //         let center_window =
            //             Point::<_, Pixels>::new(width, height).cast(|l| *l as f64) / 2.0;
            //         let center_abs =
            //             &coords.origin + &(center_window * coords.view).cast(|f| f.to_fbig());
            //         let new_origin =
            //             &center_abs - &(center_window * new_view).cast(|f| f.to_fbig());

            //         coords.origin = new_origin;
            //         coords.view = new_view;
            //     }
            // }
            else {
                break 'drawer_update;
            }

            let cursor_rel = cursor_rel.map(|p| p.cast(|f| f.to_f64().value() as usize));
            drawer.update(coords.clone(), iterations, cursor_rel.as_ref());
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
