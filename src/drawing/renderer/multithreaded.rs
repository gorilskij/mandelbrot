use crate::rendering::{CoordinatesBox, Pixels};
use crate::support::Point;
use crate::tiles::perturb::Perturbator;
use crate::tiles::render::{GroupCache, run_generation};
use crate::tiles::store::TileStore;
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread;
use waker_interrupter as wi;

pub type Message = (
    CoordinatesBox,
    usize,
    Option<Point<usize, Pixels>>,
    usize,
    usize,
);

pub struct Handle {
    sender: wi::Sender<Message>,
}

impl Handle {
    /// Ask the render thread to (re)render the tiles visible in `coords` at
    /// the given display size. Interrupts any render in progress; tiles
    /// already completed stay in the store and are not redone.
    pub fn update(
        &self,
        coords: &CoordinatesBox,
        iterations: usize,
        cursor_rel: Option<&Point<usize, Pixels>>,
        width: usize,
        height: usize,
    ) {
        self.sender
            .send((coords.clone(), iterations, cursor_rel.cloned(), width, height));
    }

    /// Signal the render thread to stop, without waiting for the in-flight
    /// generation to finish. The `JoinHandle` is dropped (detached); the OS
    /// reclaims the thread at process exit. This keeps window close instant
    /// even mid-generation, where a join could block for a whole tile pass.
    pub fn terminate(self) {
        self.sender.terminate();
    }
}

pub fn spawn(
    store:   Arc<TileStore>,
    backend: Arc<dyn Perturbator + Send + Sync>,
) -> Handle {
    let (sender, receiver) = wi::channel();

    thread::spawn(move || {
        let group_cache = Mutex::new(GroupCache::new());

        receiver.run_multithreaded(
            None,
            None,
            |(coords, iterations, cursor, width, height): Message, int| {
                run_generation(
                    &store,
                    &group_cache,
                    width,
                    height,
                    &coords,
                    iterations,
                    cursor,
                    int,
                    backend.as_ref(),
                );
            },
        );
    });

    Handle { sender }
}
