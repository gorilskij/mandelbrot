use crate::rendering::{CoordinatesBox, Pixels};
use crate::support::Point;
use crate::tiles::perturb::{Toggle, gpu::{Gpu, GpuState}};
use crate::tiles::render::{GroupCache, run_generation};
use crate::tiles::store::TileStore;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
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

/// Start the compute thread (a Web Worker on the web; see
/// `platform::spawn_async`). It makes its own backends, since on the web
/// GPU objects stay on the thread that created them.
pub fn spawn(store: Arc<TileStore>, use_gpu: Arc<AtomicBool>) -> Handle {
    let (sender, receiver) = wi::channel();
    crate::platform::spawn_async("compute", move || compute_loop(store, receiver, use_gpu));
    Handle { sender }
}

/// Render each requested view until told to stop; a newer request
/// interrupts the current generation.
async fn compute_loop(store: Arc<TileStore>, receiver: wi::Receiver<Message>, use_gpu: Arc<AtomicBool>) {
    crate::platform::init_worker_threads();
    let backend = Toggle::new(Gpu(Arc::new(GpuState::new().await)), use_gpu);
    let group_cache = Mutex::new(GroupCache::new());
    while let Some(((coords, iterations, cursor, width, height), int)) = receiver.recv_multithreaded() {
        run_generation(&store, &group_cache, width, height, &coords, iterations, cursor, int, &backend).await;
    }
}
