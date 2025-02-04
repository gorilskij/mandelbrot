pub mod maybe_pixel;
mod renderer_thread;

use crate::rendering::{CoordinatesBox, sample_zoomed};
use log::trace;
use std::thread;

pub struct Drawer {
    width: usize,
    height: usize,
    //
    base_coords: CoordinatesBox,
    zoomed_coords: CoordinatesBox,
    //
    base_buf: Box<[u32]>,
    zoomed_buf: Box<[u32]>,
    display_buf: Box<[u32]>,
    renderer: renderer_thread::Handle,
}

impl Drawer {
    pub fn new(width: usize, height: usize, coords: CoordinatesBox) -> Self {
        let mut this = Self {
            width,
            height,
            //
            base_coords: coords,
            zoomed_coords: coords,
            //
            base_buf: vec![0; width * height].into_boxed_slice(),
            zoomed_buf: vec![0; width * height].into_boxed_slice(),
            display_buf: vec![0; width * height].into_boxed_slice(),
            renderer: renderer_thread::spawn(width, height),
        };

        // force redraw
        this.new_coords(coords);
        this.update_display_buf();
        this
    }

    pub fn new_coords(&mut self, new_zoomed_coords: CoordinatesBox) {
        // relaunch renderer thread
        self.renderer.update(new_zoomed_coords);

        trace!("drawer: sent update to renderer");

        // update zoomed buffer
        sample_zoomed(
            &self.base_buf,
            &mut self.zoomed_buf,
            //
            self.width,
            self.height,
            //
            self.base_coords,
            new_zoomed_coords,
        );
        self.zoomed_coords = new_zoomed_coords;

        trace!("drawer: updated zoomed buffer");
    }

    pub fn update_display_buf(&mut self) {
        // TODO: benchmark
        // self.display_buf.copy_from_slice(&self.zoomed_buf);
        let render_buf = self.renderer.concurrent_view();

        for i in 0..self.display_buf.len() {
            let rendered_pixel = unsafe { render_buf.add(i).read() };
            self.display_buf[i] = match rendered_pixel.get() {
                Some(pixel) => pixel,
                None => self.zoomed_buf[i],
            }
        }
    }

    // returns true if the base buffer was updated
    pub fn try_replace_base_buf(&mut self) -> bool {
        if self.base_coords == self.zoomed_coords {
            return false;
        }

        if let Some(lock) = self.renderer.lock_if_done() {
            self.base_buf.copy_from_slice(&lock);
            self.base_coords = self.zoomed_coords;
            return true;
        }

        false
    }

    pub fn display_buf(&self) -> &[u32] {
        &self.display_buf
    }

    pub fn stop(self) -> thread::Result<()> {
        self.renderer.terminate_and_join()
    }
}
