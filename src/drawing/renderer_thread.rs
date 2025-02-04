use std::sync::Arc;
use std::{mem, thread};
use std::time::Duration;
use parking_lot::{Mutex, MutexGuard};
use crate::rendering::{render, CoordinatesBox};
use waker_interrupter::*;
use crate::drawing::maybe_pixel::MaybePixel;

pub struct Handle {
    handle: thread::JoinHandle<()>,
    sender: Sender<CoordinatesBox>,
    // when the `done` Mutex is held, buffer can be updated but not cleared
    done: Arc<Mutex<bool>>,
    buffer: Arc<Mutex<Box<[MaybePixel]>>>,
}

impl Handle {
    pub fn update(&self, coords: CoordinatesBox) {
        self.sender.send(coords)
    }

    pub fn get_partial_buffer(&self) -> MutexGuard<Box<[MaybePixel]>> {
        self.buffer.lock()
    }

    pub fn get_buffer_if_done(&self) -> Option<&[u32]> {
        let done = self.done.lock();
        if *done {
            let buf = self.buffer.lock();
            // SAFETY: if `done` is true, all the values in the buffer are valid u32 colors
            let buf: &[MaybePixel] = &buf;
            let buf: &[u32] = unsafe { mem::transmute(buf) };
            Some(buf)
        } else {
            None
        }
    }

    pub fn terminate_and_join(self) -> thread::Result<()> {
        self.sender.terminate();
        self.handle.join()
    }
}

pub fn spawn(width: usize, height: usize) -> Handle {
    let (sender, receiver) = channel();
    let done = Arc::new(Mutex::new(false));
    let buffer = Arc::new(Mutex::new(vec![MaybePixel::none(); width * height].into_boxed_slice()));

    let done_clone = done.clone();
    let buffer_clone = buffer.clone();
    let handle = thread::spawn(move || {
        receiver.run(Duration::from_millis(200), |new_zoomed_coords, int| {
            *done_clone.lock() = false;
            render(&mut *buffer_clone.lock(), width, height, new_zoomed_coords, int);
            *done_clone.lock() = true;
        });
    });

    Handle {
        handle,
        sender,
        done,
        buffer,
    }
}