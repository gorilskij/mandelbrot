use crate::drawing::maybe_pixel::MaybePixel;
use crate::rendering::{CoordinatesBox, Pixels, render};
use crate::support::Point;
use delegate::delegate;
use parking_lot::{Mutex, MutexGuard};
use rayon::ThreadPoolBuilder;
use render_buffer::RenderBuffer;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use std::{mem, thread};
use waker_interrupter as wi;

pub type BufGuard<'a, T> = MutexGuard<'a, Box<[T]>>;

pub mod render_buffer {
    use std::{cmp, slice};

    use crate::drawing::new_buf;

    use super::*;

    pub type Done = Arc<Mutex<bool>>;
    pub type Buffer = Arc<Mutex<Box<[MaybePixel]>>>;

    pub struct RenderBuffer {
        // when the `done` Mutex is held, buffer can be updated but not cleared
        done: Done,
        buffer: Buffer,
        len: usize,
        concurrent_view: *const MaybePixel,
        submitted_render_id: Arc<AtomicUsize>,
        running_render_id: Arc<AtomicUsize>,
    }

    impl RenderBuffer {
        pub fn new(
            width: usize,
            height: usize,
            submitted_render_id: Arc<AtomicUsize>,
            running_render_id: Arc<AtomicUsize>,
        ) -> (Self, Done, Buffer) {
            // TODO: make atomic
            let done = Arc::new(Mutex::new(false));

            let len = width * height;
            let buf = vec![MaybePixel::none(); len].into_boxed_slice();
            let concurrent_view = buf.as_ptr();
            let buffer = Arc::new(Mutex::new(buf));

            let this = Self {
                done: done.clone(),
                buffer: buffer.clone(),
                len,
                concurrent_view,
                submitted_render_id,
                running_render_id,
            };

            (this, done, buffer)
        }

        pub fn lock_if_done(&self) -> Option<MutexGuard<'_, Box<[u32]>>> {
            let done = self.done.lock();
            if *done {
                let buf = self.buffer.lock();
                // SAFETY: if `done` is true, all the values in the buffer are valid u32 colors
                let buf =
                    unsafe { mem::transmute::<BufGuard<'_, MaybePixel>, BufGuard<'_, u32>>(buf) };
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
                    let buf = unsafe {
                        mem::transmute::<BufGuard<'_, MaybePixel>, BufGuard<'_, u32>>(buf)
                    };
                    return buf;
                }

                drop(done);
                thread::yield_now();
                thread::sleep(Duration::from_millis(200));
            }
        }

        pub fn concurrent_view(&self) -> *const MaybePixel {
            self.concurrent_view
        }

        pub fn cloned_buffer(&self) -> Option<Box<[MaybePixel]>> {
            use std::cmp::Ordering::*;
            use std::sync::atomic::Ordering::*;

            match self
                .submitted_render_id
                .load(Acquire)
                .cmp(&self.running_render_id.load(Acquire))
            {
                Greater => None,
                Equal => {
                    // // SAFETY: self.concurrent_view just points to the beginning of a slice
                    Some(
                        unsafe { slice::from_raw_parts(self.concurrent_view, self.len) }
                            .to_vec() // clone
                            .into_boxed_slice(),
                    )
                }
                Less => unreachable!(),
            }

            // let mut buf = new_buf(self.len);
            // (0..self.len).for_each(|i| {
            //     buf[i] = unsafe { self.concurrent_view.offset(i as isize).read_volatile() }
            // });
            // buf
        }
    }
}

pub type RenderId = usize;
pub type MessageTuple = (
    RenderId,
    CoordinatesBox,
    usize,
    Option<Point<usize, Pixels>>,
);

pub struct Handle {
    handle: thread::JoinHandle<()>,
    sender: wi::Sender<MessageTuple>,
    buffer: RenderBuffer,
    submitted_render_id: Arc<AtomicUsize>,
}

impl Handle {
    pub fn update(
        &mut self,
        coords: &CoordinatesBox,
        iterations: usize,
        cursor_rel: Option<&Point<usize, Pixels>>,
    ) {
        let render_id = self.submitted_render_id.fetch_add(1, Ordering::AcqRel);
        self.sender
            .send((render_id, coords.clone(), iterations, cursor_rel.cloned()));
    }

    delegate! {
        to self.buffer {
            pub fn lock_if_done(&self) -> Option<BufGuard<'_, u32>>;
            pub fn concurrent_view(&self) -> *const MaybePixel;
            pub fn cloned_buffer(&self) -> Option<Box<[MaybePixel]>>;
            pub fn lock_when_done(&self) -> BufGuard<'_, u32>;
        }
    }

    pub fn terminate_and_join(self) -> thread::Result<()> {
        self.sender.terminate();
        self.handle.join()
    }
}

pub fn spawn(width: usize, height: usize) -> Handle {
    let (sender, receiver) = wi::channel();

    let submitted_render_id = Arc::new(AtomicUsize::new(0));
    let running_render_id = Arc::new(AtomicUsize::new(0));
    let (buffer, done_clone, buf_clone) = RenderBuffer::new(
        width,
        height,
        submitted_render_id.clone(),
        running_render_id.clone(),
    );

    let tp = ThreadPoolBuilder::new().num_threads(12).build().unwrap();

    let running_render_id_clone = running_render_id;
    let handle = thread::spawn(move || {
        receiver.run_multithreaded(
            None,
            None,
            |(render_id, new_zoomed_coords, iterations, cursor_rel): (_, _, _, Option<_>), int| {
                *done_clone.lock() = false;

                let center = cursor_rel.unwrap_or(Point::<_, Pixels>::new(width / 2, height / 2));

                let mut buf_lock = buf_clone.lock();
                println!("start clearing");
                buf_lock.iter_mut().for_each(|p| p.set_none());
                println!("== done clearing ==");

                running_render_id_clone.store(render_id, Ordering::Release);

                render(
                    &mut buf_lock,
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
        submitted_render_id,
    }
}
