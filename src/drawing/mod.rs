pub mod maybe_pixel;
mod renderer_thread;

use crate::{
    drawing::maybe_pixel::MaybePixel,
    rendering::{CoordinatesBox, Pixels, sample_zoomed},
    support::Point,
};
use delegate::delegate;
use itertools::Itertools;
use log::trace;
use std::{iter, mem, thread};

type Buf<T> = Box<[T]>;

fn new_buf<T: Default + Copy>(len: usize) -> Buf<T> {
    vec![T::default(); len].into_boxed_slice()
}

#[derive(Clone)]
struct ZoomedBuf {
    coords: CoordinatesBox,
    buf: Buf<MaybePixel>,
    zoomed: Buf<MaybePixel>,
    hits: usize,
}

impl ZoomedBuf {
    fn new(len: usize, coords: CoordinatesBox) -> Self {
        Self {
            coords,
            buf: new_buf(len),
            zoomed: new_buf(len),
            hits: 0,
        }
    }

    fn from_buf(src: Box<[MaybePixel]>, coords: CoordinatesBox) -> Self {
        Self {
            coords,
            buf: src.clone(),
            zoomed: src,
            hits: 0,
        }
    }

    fn copy_from(&mut self, src: &[u32], coords: CoordinatesBox) {
        // SAFETY: MaybePixel is repr(transparent)
        let src = unsafe { mem::transmute::<&[u32], &[MaybePixel]>(src) };
        self.buf.copy_from_slice(src);
        self.zoomed.copy_from_slice(src);
        self.coords = coords;
    }
}

struct DisplayBuf(Buf<u32>);

impl DisplayBuf {
    fn new(len: usize) -> Self {
        Self(new_buf(len))
    }

    fn set(&mut self, idx: usize, val: u32) {
        self.0[idx] = val
    }

    delegate! {
        to self.0 {
            fn len(&self) -> usize;
            fn copy_from_slice(&mut self, src: &[u32]);
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum DisplayBufState {
    // Moving around, the renderer isn't running
    Zooming,
    // The renderer is running
    Rendering,
    // The renderer has finished, the buffer is up to date
    Fresh,
}

const MAX_NUM_PARTIAL_BUFS: usize = 5;
pub struct Drawer {
    width: usize,
    height: usize,
    // TODO: document
    current_coords: CoordinatesBox,
    //
    cached_buf: ZoomedBuf,
    partial_bufs: Vec<ZoomedBuf>,
    display_buf: DisplayBuf,
    display_buf_state: DisplayBufState,
    //
    renderer: renderer_thread::Handle,
}

impl Drawer {
    pub fn new(width: usize, height: usize, coords: CoordinatesBox, iterations: usize) -> Self {
        let mut this = Self {
            width,
            height,
            //
            current_coords: coords.clone(),
            //
            cached_buf: ZoomedBuf::new(width * height, coords.clone()),
            partial_bufs: vec![],
            display_buf: DisplayBuf::new(width * height),
            display_buf_state: DisplayBufState::Zooming,
            //
            renderer: renderer_thread::spawn(width, height),
        };

        // initial render
        this.update(coords, iterations, None);
        println!("{:?}", this.display_buf_state);
        this.update_display_buf();
        this.force_cache_buf();

        this
    }

    fn resample(&mut self, new_coords: CoordinatesBox) {
        println!("resample");

        // resample all zoomed bufs
        for pb in self
            .partial_bufs
            .iter_mut()
            .chain(iter::once(&mut self.cached_buf))
        {
            sample_zoomed(
                &pb.buf,
                &mut pb.zoomed,
                self.width,
                self.height,
                &pb.coords,
                &new_coords,
            );
        }

        self.current_coords = new_coords;
    }

    // update bufs without launching a redraw
    pub fn soft_update(&mut self, new_coords: CoordinatesBox) {
        self.display_buf_state = DisplayBufState::Zooming;
        self.resample(new_coords);
    }

    /// Pass new view and iterations parameters to the renderer and start
    /// a rendering run (cancel any ongoing rendering). Also update the
    /// zoom buffer (fast) as a preview.
    pub fn update(
        &mut self,
        new_coords: CoordinatesBox,
        iterations: usize,
        cursor_rel: Option<&Point<usize, Pixels>>,
    ) {
        // relaunch renderer thread
        self.display_buf_state = DisplayBufState::Rendering;
        self.renderer.update(&new_coords, iterations, cursor_rel);

        trace!("drawer: sent update to renderer");
        println!("drawer: sent update to renderer");

        // remove the least used partial buf if the list is full
        if self.partial_bufs.len() >= MAX_NUM_PARTIAL_BUFS {
            println!(
                "{:?}",
                self.partial_bufs.iter().map(|pb| pb.hits).collect_vec()
            );

            // vec.iter().enumerate() is a DoubleEndedIterator because
            // std::slice::Iter is ExactSizeIterator and DoubleEndedIterator
            let i = self
                .partial_bufs
                .iter()
                .enumerate()
                .rev()
                .min_by_key(|(_, pb)| pb.hits)
                .unwrap()
                .0;
            println!("remove {i}");
            self.partial_bufs.remove(i);
        }

        // add a new partial buf
        self.partial_bufs.push(ZoomedBuf::from_buf(
            self.renderer.cloned_buffer(),
            self.current_coords.clone(),
        ));

        self.resample(new_coords);

        println!(">> {}", self.partial_bufs.len());
        // }
    }

    // TODO: merge with update_display_buf
    // without the renderer buf
    // pub fn update_display_buf_soft(&mut self) {
    //     if self.display_buf_state == DisplayBufState::Fresh {
    //         // no zooming or moving has happened since the last update
    //         return;
    //     }

    //     println!("soft display update ({})", self.partial_bufs.len());

    //     for i in 0..self.display_buf.len() {
    //         // SAFETY: this won't interfere with generation,
    //         //   if a corrupted value is read, it will just be
    //         //   overwritten on the next iteration, no big deal
    //         self.display_buf.set(
    //             i,
    //             self.partial_bufs
    //                 .iter()
    //                 .chain(iter::once(&self.cached_buf))
    //                 .map(|pb| pb.zoomed[i])
    //                 .find_map(|mp| mp.get())
    //                 .unwrap_or_default(),
    //         );
    //     }
    // }

    pub fn update_display_buf(&mut self) {
        if self.display_buf_state == DisplayBufState::Fresh {
            // no zooming or moving has happened since the last update
            return;
        }

        let render_buf = self.renderer.concurrent_view();

        self.partial_bufs.iter_mut().for_each(|pb| pb.hits = 0);

        match self.display_buf_state {
            DisplayBufState::Rendering => {
                for i in 0..self.display_buf.len() {
                    self.display_buf.set(
                        i,
                        // SAFETY: this won't interfere with generation,
                        //   if a corrupted value is read, it will just be
                        //   overwritten on the next iteration, no big deal
                        iter::once(unsafe { render_buf.add(i).read() })
                            .chain(self.partial_bufs.iter().rev().map(|pb| pb.zoomed[i]))
                            .chain(iter::once(self.cached_buf.zoomed[i]))
                            .find_map(|mp| mp.get())
                            // .map(|p| {
                            //     // TODO: reimplement or remove hits
                            //     // if let Some(j) = j {
                            //     // self.partial_bufs[j].hits += 1;
                            //     // }
                            //     p
                            // })
                            .unwrap_or_default(),
                    );
                }
            }
            DisplayBufState::Zooming => {
                for i in 0..self.display_buf.len() {
                    self.display_buf.set(
                        i,
                        self.partial_bufs
                            .iter()
                            .rev()
                            .map(|pb| pb.zoomed[i])
                            .chain(iter::once(self.cached_buf.zoomed[i]))
                            .find_map(|mp| mp.get())
                            .unwrap_or_default(),
                    );
                }
            }
            DisplayBufState::Fresh => {}
        }
    }

    /// If the renderer is done rendering the current view, cache it.
    /// Return true if the cache was updated
    pub fn try_cache_buf(&mut self) -> bool {
        if self.display_buf_state != DisplayBufState::Rendering {
            return false;
        }

        if let Some(lock) = self.renderer.lock_if_done() {
            self.display_buf.copy_from_slice(&lock);
            self.cached_buf
                .copy_from(&lock, self.current_coords.clone());
            self.partial_bufs.clear();
            self.display_buf_state = DisplayBufState::Fresh;

            println!("successful caching");

            return true;
        }

        false
    }

    /// Wait for the renderer to finish rendering the current view, then
    /// immediately lock it and cache it to cache_buf and display_buf
    pub fn force_cache_buf(&mut self) {
        if self.display_buf_state != DisplayBufState::Rendering {
            return;
        }

        let lock = self.renderer.lock_when_done();
        self.display_buf.copy_from_slice(&lock);
        self.cached_buf
            .copy_from(&lock, self.current_coords.clone());
        self.partial_bufs.clear();
        self.display_buf_state = DisplayBufState::Fresh;

        println!("forced caching");
    }

    pub fn display_buf(&self) -> &[u32] {
        &self.display_buf.0
        // unsafe { mem::transmute::<&[_], &[_]>(&self.cached_buf.zoomed) }
    }

    pub fn stop(self) -> thread::Result<()> {
        self.renderer.terminate_and_join()
    }
}
