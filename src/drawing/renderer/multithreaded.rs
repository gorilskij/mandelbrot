use crate::rendering::{CoordinatesBox, Pixels};
use crate::support::Point;
use crate::tiles::render::{OrbitPool, run_generation};
use crate::tiles::store::TileStore;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use std::thread;
use waker_interrupter as wi;

pub type Message = (CoordinatesBox, usize, Option<Point<usize, Pixels>>);

pub struct Handle {
    handle: thread::JoinHandle<()>,
    sender: wi::Sender<Message>,
}

impl Handle {
    /// Ask the render thread to (re)render the tiles visible in `coords`.
    /// Interrupts any render in progress; tiles already completed stay in
    /// the store and are not redone.
    pub fn update(
        &self,
        coords: &CoordinatesBox,
        iterations: usize,
        cursor_rel: Option<&Point<usize, Pixels>>,
    ) {
        self.sender
            .send((coords.clone(), iterations, cursor_rel.cloned()));
    }

    pub fn terminate_and_join(self) -> thread::Result<()> {
        self.sender.terminate();
        self.handle.join()
    }
}

pub fn spawn(width: usize, height: usize, store: Arc<TileStore>) -> Handle {
    let (sender, receiver) = wi::channel();

    let tp = ThreadPoolBuilder::new().num_threads(12).build().unwrap();

    let handle = thread::spawn(move || {
        // the orbit pool persists across render generations and is rebuilt
        // only when the viewport strays too far from its anchor (or the
        // iteration count changes)
        let mut pool: Option<OrbitPool> = None;

        receiver.run_multithreaded(None, None, |(coords, iterations, cursor): Message, int| {
            run_generation(
                &store, &mut pool, width, height, &coords, iterations, cursor, int, &tp,
            );
        });
    });

    Handle { handle, sender }
}
