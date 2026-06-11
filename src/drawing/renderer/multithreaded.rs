use crate::rendering::{CoordinatesBox, Pixels};
use crate::support::Point;
use crate::tiles::render::{GroupCache, run_generation};
use crate::tiles::store::TileStore;
use parking_lot::Mutex;
use rayon::ThreadPoolBuilder;
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
    handle: thread::JoinHandle<()>,
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

    pub fn terminate_and_join(self) -> thread::Result<()> {
        self.sender.terminate();
        self.handle.join()
    }
}

pub fn spawn(store: Arc<TileStore>) -> Handle {
    let (sender, receiver) = wi::channel();

    let tp = ThreadPoolBuilder::new().num_threads(12).build().unwrap();

    let handle = thread::spawn(move || {
        // memoized per-group shared reference lists; persists across
        // generations and is dropped only when the iteration count changes
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
                    &tp,
                );
            },
        );
    });

    Handle { handle, sender }
}
