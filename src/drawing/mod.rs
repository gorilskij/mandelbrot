pub mod maybe_pixel;
mod renderer_thread;

use crate::{
    drawing::maybe_pixel::MaybePixel,
    rendering::{CoordinatesBox, Pixels, sample_zoomed},
    support::Point,
};
use log::trace;
use std::{iter, mem, thread};

type Buf<T> = Box<[T]>;

struct PartialBuf {
    coords: CoordinatesBox,
    buf: Buf<MaybePixel>,
    zoomed: Buf<MaybePixel>,
}

pub struct Drawer {
    width: usize,
    height: usize,
    coords: CoordinatesBox,
    partial_bufs: Vec<PartialBuf>,
    /// When generating, this buffer is composed of the zoom_buf
    /// overlayed with the pixels that have been generated in the
    /// renderer buffer. This is the buffer that gets passed along to
    /// the UI renderer.
    display_buf: Buf<u32>,
    display_buf_fresh: bool,
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
            coords: coords.clone(),
            //
            partial_bufs: vec![],
            display_buf: vec![0; width * height].into_boxed_slice(),
            display_buf_fresh: false,
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
        cursor_rel: Option<&Point<usize, Pixels>>,
    ) {
        self.display_buf_fresh = false;

        // relaunch renderer thread
        self.renderer
            .update(&new_zoomed_coords, iterations, cursor_rel);

        trace!("drawer: sent update to renderer");

        if self.partial_bufs.is_empty() {
            // SAFETY: MaybePixel is repr(transparent)
            let buf =
                unsafe { mem::transmute::<Buf<u32>, Buf<MaybePixel>>(self.display_buf.clone()) };
            let mut zoomed = vec![MaybePixel::none(); buf.len()].into_boxed_slice();
            sample_zoomed(
                &buf,
                &mut zoomed,
                self.width,
                self.height,
                &self.coords,
                &new_zoomed_coords,
            );
            self.partial_bufs.push(PartialBuf {
                coords: self.coords.clone(),
                buf,
                zoomed,
            });
        } else {
            // update all partial bufs with the new coordinates
            for pb in &mut self.partial_bufs {
                sample_zoomed(
                    &pb.buf,
                    &mut pb.zoomed,
                    self.width,
                    self.height,
                    &pb.coords,
                    &new_zoomed_coords,
                );
            }

            // add a new partial buf
            let buf = self.renderer.cloned_buffer();
            let mut zoomed = vec![MaybePixel::none(); buf.len()].into_boxed_slice();
            sample_zoomed(
                &buf,
                &mut zoomed,
                self.width,
                self.height,
                &self.coords,
                &new_zoomed_coords,
            );
            self.partial_bufs.insert(
                0,
                PartialBuf {
                    coords: self.coords.clone(),
                    buf,
                    zoomed,
                },
            );
        }

        self.coords = new_zoomed_coords;

        // TODO: I think this is not needed here
        self.update_display_buf();
    }

    pub fn update_display_buf(&mut self) {
        if self.display_buf_fresh {
            // no zooming or moving has happened since the last update
            return;
        }

        let render_buf = self.renderer.concurrent_view();

        for i in 0..self.display_buf.len() {
            // SAFETY: this won't interfere with generation,
            //   if a corrupted value is read, it will just be
            //   overwritten on the next iteration, no big deal
            self.display_buf[i] = iter::once(unsafe { render_buf.add(i).read() })
                .chain(self.partial_bufs.iter().map(|pb| pb.zoomed[i]))
                .find_map(|mp| mp.get())
                .unwrap_or(0);
        }
    }

    /// If the renderer is done rendering the current view, cache it.
    /// Return true if the cache was updated
    pub fn try_cache_buf(&mut self) -> bool {
        // ignore repeated calls if cache was successful
        if self.display_buf_fresh {
            return false;
        }

        if let Some(lock) = self.renderer.lock_if_done() {
            self.display_buf.copy_from_slice(&lock);
            self.partial_bufs.clear();
            self.display_buf_fresh = true;
            return true;
        }

        false
    }

    /// Wait for the renderer to finish rendering the current view, then
    /// immediately lock it and cache it to cache_buf and display_buf
    pub fn force_cache_buf(&mut self) {
        // ignore repeated calls
        if self.display_buf_fresh {
            return;
        }

        let lock = self.renderer.lock_when_done();
        self.display_buf.copy_from_slice(&lock);
        self.partial_bufs.clear();
        self.display_buf_fresh = true;
    }

    pub fn display_buf(&self) -> &[u32] {
        &self.display_buf
    }

    pub fn stop(self) -> thread::Result<()> {
        self.renderer.terminate_and_join()
    }
}
