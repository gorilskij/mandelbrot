mod renderer_thread;
pub mod maybe_pixel;

use crate::rendering::{CoordinatesBox, sample_zoomed};
use std::{thread};
use itertools::izip;
use log::trace;

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

    fn update_display_buf(&mut self) {
        // TODO: benchmark
        // self.display_buf.copy_from_slice(&self.zoomed_buf);
        let render_buf = self.renderer.get_partial_buffer();
        for (zoomed, render, display) in izip!(self.zoomed_buf.iter(), render_buf.iter(), self.display_buf.iter_mut()) {
            *display = match render.get() {
                Some(pixel) => pixel,
                None => *zoomed,
            }
        }
    }

    // returns whether the buffer was updated
    pub fn try_update_display_buf(&mut self) -> bool {
        self.renderer.get_partial_buffer()
    }

    pub fn display_buf(&self) -> &[u32] {
        &self.display_buf
    }

    pub fn stop(self) -> thread::Result<()> {
        self.renderer.terminate_and_join()
    }
}
