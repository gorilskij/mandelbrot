mod buffer;

use crate::rendering::{Point, Units, View, render, sample_zoomed};
use buffer::Buffer;
use log::trace;
use parking_lot::{Condvar, Mutex};
use std::sync::Arc;
use std::time::Duration;
use std::{mem, thread};

#[derive(Debug)]
pub enum State {
    Run,
    Wait,
    Terminate,
}

pub struct Handle {
    handle: thread::JoinHandle<()>,
    pub buffer: Buffer,
    wake: Arc<(Mutex<State>, Condvar)>,
    missed_update: Arc<Mutex<bool>>,
}

impl Handle {
    pub fn join(self) -> thread::Result<()> {
        self.handle.join()
    }

    pub fn notify_run(&self) {
        if let Some(mut lock) = self.wake.0.try_lock() {
            *lock = State::Run;
            self.wake.1.notify_all();
        } else {
            *self.missed_update.lock() = true;
        }
    }

    pub fn notify_terminate(&self) {
        *self.wake.0.lock() = State::Terminate;
        self.wake.1.notify_all();
    }
}

pub fn spawn(width: usize, height: usize, origin: Point<Units>, view: View) -> Handle {
    let buffer = Buffer::new(width, height, origin, view);

    let wake = Arc::new((Mutex::new(State::Wait), Condvar::new()));
    let missed_update = Arc::new(Mutex::new(false));

    let handle = {
        let buffer = buffer.clone();
        let wake = wake.clone();
        let missed_update = missed_update.clone();

        thread::spawn(move || {
            let mut tmp_buffer = vec![0; width * height].into_boxed_slice();
            let (lock, cvar) = &*wake;

            let mut lock = lock.lock();
            loop {
                trace!("re: wait");

                cvar.wait_for(&mut lock, Duration::from_millis(200));

                trace!("re: woken {:?}", *lock);

                match mem::replace(&mut *lock, State::Wait) {
                    State::Run => {}
                    State::Wait => {
                        let missed_update = &mut *missed_update.lock();
                        if !mem::replace(missed_update, false) {
                            continue;
                        }
                    }
                    State::Terminate => break,
                };

                trace!("re: redrawing");

                let (origin, view) = {
                    let zoomed = buffer.zoomed.lock();
                    (zoomed.origin, zoomed.view)
                };

                trace!("re: rendering");
                render(&mut tmp_buffer, width, height, origin, view);
                trace!("re: done rendering");

                {
                    let base = &mut buffer.base.lock();
                    trace!("re: base acquired");

                    base.buffer.copy_from_slice(&tmp_buffer);
                    base.origin = origin;
                    base.view = view;

                    let zoomed = &mut buffer.zoomed.lock();

                    let zoomed_origin = zoomed.origin;
                    let zoomed_view = zoomed.view;

                    trace!("re: sample zooming");
                    sample_zoomed(
                        &base.buffer,
                        &mut zoomed.buffer,
                        //
                        width,
                        height,
                        //
                        base.origin,
                        base.view,
                        //
                        zoomed_origin,
                        zoomed_view,
                    );
                    trace!("re: done sample zooming");
                }
            }
        })
    };

    Handle {
        handle,
        buffer,
        wake,
        missed_update,
    }
}
