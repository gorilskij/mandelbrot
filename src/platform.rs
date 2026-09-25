//! What differs between the native app and the web (wasm) build, behind one
//! interface, so the rest of the code is shared.

use parking_lot::Mutex;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

/// Wait `d` without blocking an event loop: natively the (compute) thread
/// sleeps; on the web the worker yields to its event loop meanwhile.
pub async fn sleep(d: Duration) {
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::sleep(d);
    #[cfg(target_arch = "wasm32")]
    gloo_timers::future::sleep(d).await;
}

/// A one-shot signal: `Signal::fire` from any thread (e.g. a GPU mapping
/// callback), `.await` the `Fired` side. Natively the GPU callbacks run
/// inside a blocking `device.poll`, so it is ready by the time it is awaited;
/// on the web the browser runs them from the event loop, which awaiting
/// yields to.
pub fn signal() -> (Signal, Fired) {
    let inner = Arc::new(Mutex::new((false, None::<Waker>)));
    (Signal(inner.clone()), Fired(inner))
}

pub struct Signal(Arc<Mutex<(bool, Option<Waker>)>>);

impl Signal {
    pub fn fire(self) {
        let waker = {
            let mut s = self.0.lock();
            s.0 = true;
            s.1.take()
        };
        if let Some(w) = waker { w.wake(); }
    }
}

pub struct Fired(Arc<Mutex<(bool, Option<Waker>)>>);

impl Future for Fired {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut s = self.0.lock();
        if s.0 {
            Poll::Ready(())
        } else {
            s.1 = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Run the future made by `f` on its own thread: natively a thread driving
/// it with a blocking executor; on the web a Web Worker running it on its
/// event loop (which GPU readbacks need).
pub fn spawn_async<F, Fut>(name: &str, f: F)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = ()> + 'static,
{
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::Builder::new()
        .name(name.into())
        .spawn(move || pollster::block_on(f()))
        .expect("spawn thread");
    #[cfg(target_arch = "wasm32")]
    web::spawn_async(name, f);
}

/// Set up what threads need before first use on this thread: natively
/// nothing (rayon starts its pool itself); on the web rayon's pool, whose
/// threads must be spawned as Web Workers.
pub fn init_worker_threads() {
    #[cfg(target_arch = "wasm32")]
    web::init_rayon();
}
