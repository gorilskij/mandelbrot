pub mod maybe_pixel;
mod renderer_thread;

use crate::rendering::{CoordinatesBox, sample_zoomed};
use log::trace;
use num::Complex;
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
    pub fn new(
        width: usize,
        height: usize,
        coords: CoordinatesBox,
        z: Complex<f64>,
        iterations: usize,
    ) -> Self {
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

        // initial render
        this.update(coords, z, iterations);
        this.update_display_buf();
        this.force_replace_base_buf();

        this
    }

    pub fn update(
        &mut self,
        new_zoomed_coords: CoordinatesBox,
        z: Complex<f64>,
        iterations: usize,
    ) {
        // relaunch renderer thread
        self.renderer.update(new_zoomed_coords, z, iterations);

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

        self.update_display_buf();
    }

    pub fn update_display_buf(&mut self) {
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

    // returns true if the base buffer was updated
    pub fn try_replace_base_buf(&mut self) -> bool {
        if let Some(lock) = self.renderer.lock_if_done() {
            self.base_buf.copy_from_slice(&lock);
            self.base_coords = self.zoomed_coords;
            return true;
        }

        false
    }

    // blocking
    pub fn force_replace_base_buf(&mut self) {
        let lock = self.renderer.lock_when_done();
        self.base_buf.copy_from_slice(&lock);
        self.base_coords = self.zoomed_coords;
    }

    pub fn display_buf(&self) -> &[u32] {
        &self.display_buf
    }

    pub fn stop(self) -> thread::Result<()> {
        self.renderer.terminate_and_join()
    }
}
