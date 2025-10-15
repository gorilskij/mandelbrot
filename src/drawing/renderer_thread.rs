use crate::drawing::maybe_pixel::MaybePixel;
use crate::drawing::renderer_thread::render_buffer::RenderBuffer;
use crate::rendering::{CoordinatesBox, Pixels, render};
use delegate::delegate;
use euclid::Point2D;
use parking_lot::{Mutex, MutexGuard};
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use std::time::Duration;
use std::{mem, thread};
use waker_interrupter as wi;

pub mod render_buffer {
    use super::*;

    pub type Done = Arc<Mutex<bool>>;
    pub type Buffer = Arc<Mutex<Box<[MaybePixel]>>>;

    pub struct RenderBuffer {
        // when the `done` Mutex is held, buffer can be updated but not cleared
        done: Done,
        buffer: Buffer,
        concurrent_view: *const MaybePixel,
    }

    impl RenderBuffer {
        pub fn new(width: usize, height: usize) -> (Self, Done, Buffer) {
            let done = Arc::new(Mutex::new(false));

            let buf = vec![MaybePixel::none(); width * height].into_boxed_slice();
            let concurrent_view = buf.as_ptr();
            let buffer = Arc::new(Mutex::new(buf));

            let this = Self {
                done: done.clone(),
                buffer: buffer.clone(),
                concurrent_view,
            };

            (this, done, buffer)
        }

        pub fn lock_if_done(&self) -> Option<MutexGuard<'_, Box<[u32]>>> {
            let done = self.done.lock();
            if *done {
                let buf = self.buffer.lock();
                // SAFETY: if `done` is true, all the values in the buffer are valid u32 colors
                let buf = unsafe { mem::transmute(buf) };
                Some(buf)
            } else {
                None
            }
        }

        pub fn lock_when_done(&self) -> MutexGuard<'_, Box<[u32]>> {
            loop {
                let done = self.done.lock();
                if *done {
                    let buf = self.buffer.lock();
                    let buf = unsafe { mem::transmute(buf) };
                    return buf;
                }

                let _ = done;
                thread::yield_now();
                thread::sleep(Duration::from_millis(200));
            }
        }

        pub fn concurrent_view(&self) -> *const MaybePixel {
            self.concurrent_view
        }
    }
}

pub struct Handle {
    handle: thread::JoinHandle<()>,
    sender: wi::Sender<(CoordinatesBox, usize, Option<Point2D<usize, Pixels>>)>,
    buffer: RenderBuffer,
}

impl Handle {
    pub fn update(
        &self,
        coords: CoordinatesBox,
        iterations: usize,
        cursor_rel: Option<Point2D<usize, Pixels>>,
    ) {
        self.sender.send((coords, iterations, cursor_rel));
    }

    delegate! {
        to self.buffer {
            pub fn lock_if_done(&self) -> Option<MutexGuard<'_, Box<[u32]>>>;
            pub fn concurrent_view(&self) -> *const MaybePixel;
            pub fn lock_when_done(&self) -> MutexGuard<'_, Box<[u32]>>;
        }
    }

    pub fn terminate_and_join(self) -> thread::Result<()> {
        self.sender.terminate();
        self.handle.join()
    }
}

pub fn spawn(width: usize, height: usize) -> Handle {
    let (sender, receiver) = wi::channel();

    let (buffer, done_clone, buf_clone) = RenderBuffer::new(width, height);

    let tp = ThreadPoolBuilder::new().num_threads(12).build().unwrap();

    let handle = thread::spawn(move || {
        receiver.run_multithreaded(
            None,
            None,
            |(new_zoomed_coords, iterations, cursor_rel): (_, _, Option<_>), int| {
                *done_clone.lock() = false;

                let mut buf_lock = buf_clone.lock();

                for pixel in buf_lock.iter_mut() {
                    pixel.set_none()
                }

                let center = cursor_rel.unwrap_or(Point2D::<_, Pixels>::new(width / 2, height / 2));

                render(
                    &mut *buf_lock,
                    width,
                    height,
                    center,
                    new_zoomed_coords,
                    iterations,
                    int,
                    &tp,
                );
                *done_clone.lock() = true;
            },
        );
    });

    Handle {
        handle,
        sender,
        buffer,
    }
}
