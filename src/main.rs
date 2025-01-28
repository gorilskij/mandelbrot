mod rendering;

use minifb::{Key, KeyRepeat, MouseMode, Window, WindowOptions};
use rayon::prelude::*;
use rendering::*;

struct Buffer {
    base: Box<[u32]>,
    zoomed: Box<[u32]>,
    //
    base_origin: Point<Units>,
    base_view: View,
}

impl Buffer {
    fn new(width: usize, height: usize, origin: Point<Units>, view: View) -> Self {
        Self {
            base: vec![0; width * height].into_boxed_slice(),
            zoomed: vec![0; width * height].into_boxed_slice(),
            base_origin: origin,
            base_view: view,
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

    let mut buffer = Buffer::new(width, height, origin, view);

    let mut cached = None;

    let mut first_time = true;

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
            }
        }

        let cache_key = (origin, view);
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            render(&mut buffer.base, width, height, origin, view);
            buffer.zoomed.copy_from_slice(&buffer.base);
            buffer.base_origin = origin;
            buffer.base_view = view;

            cached = Some(cache_key);
        } else {
            if cached != Some(cache_key) {
                if first_time {
                    render(&mut buffer.base, width, height, origin, view);
                    buffer.zoomed.copy_from_slice(&buffer.base);
                    buffer.base_origin = origin;
                    buffer.base_view = view;

                    first_time = false;
                } else if zoomed {
                    sample_zoomed(
                        &buffer.base,
                        &mut buffer.zoomed,
                        //
                        width,
                        height,
                        //
                        buffer.base_origin,
                        buffer.base_view,
                        //
                        origin,
                        view,
                    );
                }
                cached = Some(cache_key);
            }
        }

        window
            .update_with_buffer(&buffer.zoomed, width, height)
            .unwrap();
    }
}
