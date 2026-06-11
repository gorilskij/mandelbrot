mod drawing;
mod rendering;
mod support;
mod tiles;

use std::{borrow::Cow, fmt::Display, str::FromStr};

use crate::{
    drawing::Drawer,
    support::{Length, Point},
};
use clipboard::{ClipboardContext, ClipboardProvider};
use dashu::float::FBig;
use log::{info, trace, warn};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use rendering::*;
use support::ToFBig;

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
        let iterations = &iterations[1..]; // remove /
        Ok(Self {
            coords: Cow::Owned(coords.parse()?),
            iterations: iterations.parse().map_err(drop)?,
        })
    }
}

fn main() {
    env_logger::init();

    // let width = 1000;
    // let height = 600;
    let mut width = 1500;
    let mut height = 1000;

    let mut window = Window::new(
        "Mandelbrot",
        width,
        height,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .unwrap();

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

    let mut dragging = None; //None::<Point<FBig, Pixels>>;
    let mut scrolling = false;

    while window.is_open() {
        let mut zoomed = false;
        let mut dragged = false;

        // follow window resizes: resize buffers and re-render at the new size
        let (w, h) = window.get_size();
        if (w, h) != (width, height) && w > 0 && h > 0 {
            width = w;
            height = h;
            drawer.resize(width, height, iterations);
        }

        // mouse position in pixels with the top-left corner of the window as the origin
        let cursor_rel = window
            .get_mouse_pos(MouseMode::Discard)
            .map(|(x, y)| Point::<_, Pixels>::new(x, y));
        let cursor_rel_fbig = cursor_rel.map(|p| p.cast(|f| -> FBig { FBig::try_from(*f).unwrap() }));

        if let Some(cursor_rel) = &cursor_rel_fbig {
            let view_fbig = coords.view.cast(move |f| -> FBig { FBig::try_from(*f).unwrap() });

            if window.get_mouse_down(MouseButton::Left) {
                match &dragging {
                    None => {
                        trace!("start dragging");
                        dragging = Some((coords.origin.clone(), cursor_rel.clone()));
                    }

                    Some((start_origin, start_cursor_rel)) => {
                        coords.origin =
                            start_origin - &(&(cursor_rel - start_cursor_rel) * &view_fbig);
                    }
                }
            } else {
                if dragging.is_some() {
                    trace!("stop dragging");
                    // perform an update on release
                    dragging = None;
                    dragged = true;
                }

                // scroll wheel is only read outside of dragging
                if let Some((_, scroll_y)) = window.get_scroll_wheel() {
                    scrolling = true;

                    let multiplier = 1.0 + (scroll_y as f64 / 100.0).clamp(-0.2, 0.2);
                    let old_view = coords.view.inner;
                    let new_view = old_view / multiplier;

                    // Lock the zoom onto the point under the cursor: hold
                    // `origin + cursor*view` exactly invariant. The cursor
                    // offset is converted f64 -> FBig *exactly* (mantissa
                    // decode, not a lossy decimal round-trip) and there's no
                    // division, so repeated zoom in/out doesn't drift —
                    // new_origin = (origin + cursor*old_view) - cursor*new_view
                    let cx = cursor_rel.x.to_f64().value();
                    let cy = cursor_rel.y.to_f64().value();
                    let exact = |f: f64| -> FBig { FBig::try_from(f).unwrap() };
                    coords.origin.x =
                        &(&coords.origin.x + &exact(cx * old_view)) - &exact(cx * new_view);
                    coords.origin.y =
                        &(&coords.origin.y + &exact(cy * old_view)) - &exact(cy * new_view);
                    coords.view = View::new(new_view);

                    trace!("zoomed (cursor-locked)");

                    // the zoom level is just the width of the screen in units
                    let width = Length::<_, Pixels>::new(width as f64) * coords.view;
                    info!("zoom level: 10^{}", -width.inner.log10());
                } else if scrolling == true {
                    // only update when we stop scrolling
                    scrolling = false;
                    zoomed = true;
                }
            }
        }

        enum UpdateDrawer {
            AroundCursor,
            AroundCenter,
            Reset,
            No,
        }

        let update_drawer = {
            if dragged || zoomed || dragging.is_some() || scrolling {
                // render continuously during the gesture: the tile cache means
                // only newly-revealed tiles are computed, and the renderer is
                // interruptible (per-tile progress is kept), so panning/zooming
                // fills in live instead of only after the gesture ends
                UpdateDrawer::AroundCursor
            } else if window.is_key_pressed(Key::Space, KeyRepeat::No) {
                // TEST: dump everything and re-render with fresh random refs
                info!("reset (spacebar)");
                UpdateDrawer::Reset
            } else if window.is_key_pressed(Key::Up, KeyRepeat::No) {
                iterations *= 2;
                info!("iterations: {iterations}");
                UpdateDrawer::AroundCursor
            } else if window.is_key_pressed(Key::Down, KeyRepeat::No) && iterations > 1 {
                iterations /= 2;
                info!("iterations: {iterations}");
                UpdateDrawer::AroundCursor
            } else if (window.is_key_down(Key::LeftCtrl)
                || window.is_key_down(Key::RightCtrl)
                || window.is_key_down(Key::LeftSuper)
                || window.is_key_down(Key::RightSuper))
                && window.is_key_pressed(Key::C, KeyRepeat::No)
            {
                let data = ClipBoardData {
                    coords: Cow::Borrowed(&coords),
                    iterations,
                };
                let s = data.to_string();
                // always print to the terminal, so coords are recoverable even
                // if the clipboard backend fails
                info!("COPIED: {s}");
                match ClipboardProvider::new()
                    .and_then(|mut ctx: ClipboardContext| ctx.set_contents(s.clone()))
                {
                    Ok(()) => {}
                    Err(e) => warn!("clipboard copy failed: {e}"),
                }
                UpdateDrawer::No
            } else if (window.is_key_down(Key::LeftCtrl)
                || window.is_key_down(Key::RightCtrl)
                || window.is_key_down(Key::LeftSuper)
                || window.is_key_down(Key::RightSuper))
                && window.is_key_pressed(Key::V, KeyRepeat::No)
            {
                let mut ctx: ClipboardContext = ClipboardProvider::new().unwrap();
                let contents = ctx.get_contents().unwrap_or_default();
                if let Ok(new_data) = contents.parse::<ClipBoardData>() {
                    coords = new_data.coords.into_owned();
                    iterations = new_data.iterations;
                    // print the raw pasted string (the parsed origin has
                    // unlimited precision until the top-up below bounds it, so
                    // re-Displaying it via to_decimal would panic)
                    info!("PASTED: {}", contents.trim());
                    UpdateDrawer::AroundCenter
                } else {
                    warn!("PASTE FAILED: invalid clipboard contents: {contents:?}");
                    UpdateDrawer::No
                }
            }
            // for debug
            // else if window.is_key_pressed(Key::Left, KeyRepeat::No) {
            //     // set vertical center to 0
            //     coords.origin.y = -(height as f64 / 2.0 * coords.view.inner).to_fbig();
            //     UpdateDrawer::AroundCenter
            // }
            //
            // else if window.is_key_pressed(Key::Right, KeyRepeat::No) {
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
            //
            else {
                UpdateDrawer::No
            }
        };

        // keep the origin's precision comfortably above what the current
        // zoom depth needs (tile indexing at depth d needs ~d bits to
        // resolve the tile grid, plus a margin)
        let needed_bits = (96.0 - coords.view.inner.log2().min(0.0)) as usize;
        if coords.origin.x.precision() < needed_bits
            || coords.origin.y.precision() < needed_bits
        {
            // raise (never lower) each component's precision
            coords.origin = coords.origin.cast(|f| {
                if f.precision() < needed_bits {
                    f.clone().with_precision(needed_bits).value()
                } else {
                    f.clone()
                }
            });
        }

        match update_drawer {
            UpdateDrawer::AroundCursor => {
                let cursor_rel_usize = cursor_rel.map(|p| p.cast(|f| *f as usize));
                drawer.update(coords.clone(), iterations, cursor_rel_usize.as_ref());
            }
            UpdateDrawer::AroundCenter => {
                let center = Point::new(width / 2, height / 2);
                drawer.update(coords.clone(), iterations, Some(&center));
            }
            UpdateDrawer::Reset => {
                let center = Point::new(width / 2, height / 2);
                drawer.reset(coords.clone(), iterations, Some(&center));
            }
            UpdateDrawer::No => {}
        };

        drawer.update_display_buf();

        window
            .update_with_buffer(drawer.display_buf(), width, height)
            .unwrap();
    }

    drawer.stop().unwrap();
}
