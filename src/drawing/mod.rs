pub mod maybe_pixel;
mod renderer_thread;

use crate::rendering::{CoordinatesBox, Pixels, sample_zoomed};
use euclid::Point2D;
use log::trace;
use std::thread;

pub struct Drawer {
    width: usize,
    height: usize,
    //
    /// This variable indicates whether cache_buf is fresh (i.e. it
    /// consists of the completed rendering of the current view) or
    /// stale (i.e. it's the rendering of a previous view that is
    /// currently used for a zoomed preview while the current view is
    /// being rendered)
    cache_fresh: bool,
    cache_coords: CoordinatesBox,
    zoomed_coords: CoordinatesBox,
    //
    /// Cached version of the renderer buffer, this gets updated when
    /// the renderer finishes rendering a given view. This buffer has
    /// all pixels perfectly aligned and is never directly transformed.
    cache_buf: Box<[u32]>,
    /// Stretched version of cache_buf, this serves as a preview for
    /// the new view currently being rendered (which is displayed over
    /// it as it is generated). This buffer is always a fresh
    /// transformation of cache_buf. When shrinking, the edges are
    /// filled in.
    zoomed_buf: Box<[u32]>,
    /// When generating, this buffer is composed of the zoom_buf
    /// overlayed with the pixels that have been generated in the
    /// renderer buffer. This is the buffer that gets passed along to
    /// the UI renderer.
    display_buf: Box<[u32]>,
    /// The renderer contains its own buffer which it fills in with
    /// pixels as it runs the mandelbrot calculation. Every time there's
    /// a change in view, this buffer gets cleared and computation restarts
    renderer: renderer_thread::Handle,
}

impl Drawer {
    pub fn new(width: usize, height: usize, coords: CoordinatesBox, iterations: usize) -> Self {
        let mut this = Self {
            width,
            height,
            //
            cache_fresh: true,
            cache_coords: coords,
            zoomed_coords: coords,
            //
            cache_buf: vec![0; width * height].into_boxed_slice(),
            zoomed_buf: vec![0; width * height].into_boxed_slice(),
            display_buf: vec![0; width * height].into_boxed_slice(),
            renderer: renderer_thread::spawn(width, height),
        };

        // initial render
        this.update(coords, iterations, None);
        this.update_display_buf();
        this.force_cache_buf();

        this
    }

    /// Pass new view and iterations parameters to the renderer and start
    /// a rendering run (cancel any ongoing rendering). Also update the
    /// zoom buffer (fast) as a preview.
    pub fn update(
        &mut self,
        new_zoomed_coords: CoordinatesBox,
        iterations: usize,
        cursor_rel: Option<Point2D<usize, Pixels>>,
    ) {
        // invalidate cache
        self.cache_fresh = false;

        // relaunch renderer thread
        self.renderer
            .update(new_zoomed_coords, iterations, cursor_rel);

        trace!("drawer: sent update to renderer");

        // update zoomed buffer
        sample_zoomed(
            &self.cache_buf,
            &mut self.zoomed_buf,
            //
            self.width,
            self.height,
            //
            self.cache_coords,
            new_zoomed_coords,
        );
        self.zoomed_coords = new_zoomed_coords;

        trace!("drawer: updated zoomed buffer");

        self.update_display_buf();
    }

    pub fn update_display_buf(&mut self) {
        // if the cache is fresh, the display buf is static
        if self.cache_fresh {
            return;
        }

        // TODO: benchmark
        // self.display_buf.copy_from_slice(&self.zoomed_buf);
        let render_buf = self.renderer.concurrent_view();

        let mut rendered = 0;
        for i in 0..self.display_buf.len() {
            let rendered_pixel = unsafe { render_buf.add(i).read() };
            self.display_buf[i] = match rendered_pixel.get() {
                Some(pixel) => {
                    rendered += 1;
                    pixel
                }
                None => self.zoomed_buf[i],
            }
        }

        trace!("rendered {} / {} pixels", rendered, self.display_buf.len());
    }

    /// If the renderer is done rendering the current view, cache it in
    /// the cache_buf and display_buf, otherwise keep the old cache_buf.
    /// Return true if the cache was updated
    pub fn try_cache_buf(&mut self) -> bool {
        // ignore repeated calls if cache was successful
        if self.cache_fresh {
            return false;
        }

        if let Some(lock) = self.renderer.lock_if_done() {
            self.cache_buf.copy_from_slice(&lock);
            self.cache_coords = self.zoomed_coords;
            self.display_buf.copy_from_slice(&lock);
            self.cache_fresh = true;
            return true;
        }

        false
    }

    /// Wait for the renderer to finish rendering the current view, then
    /// immediately lock it and cache it to cache_buf and display_buf
    pub fn force_cache_buf(&mut self) {
        // ignore repeated calls
        if self.cache_fresh {
            return;
        }

        let lock = self.renderer.lock_when_done();
        self.cache_buf.copy_from_slice(&lock);
        self.cache_coords = self.zoomed_coords;
        self.display_buf.copy_from_slice(&lock);
        self.cache_fresh = true;
    }

    pub fn display_buf(&self) -> &[u32] {
        &self.display_buf
    }

    pub fn stop(self) -> thread::Result<()> {
        self.renderer.terminate_and_join()
    }
}
