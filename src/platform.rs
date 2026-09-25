//! What differs between the native app and the web (wasm) build, behind one
//! interface, so the rest of the code is shared. On the web, threads are Web
//! Workers sharing the wasm memory (`wasm_thread`); the browser's main
//! thread must never block (a contended lock or a wait throws there).

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

/// Run `f` on a new thread (a Web Worker on the web), detached.
pub fn spawn(name: &str, f: impl FnOnce() + Send + 'static) {
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::Builder::new().name(name.into()).spawn(f).expect("spawn thread");
    #[cfg(target_arch = "wasm32")]
    wasm_thread::Builder::new()
        .name(name.into())
        .worker_script_url(web::worker_url(true))
        .spawn(f)
        .expect("spawn worker");
}

/// Set up what threads need before first use on this thread: natively
/// nothing (rayon starts its pool itself); on the web rayon's pool, whose
/// threads must be spawned as Web Workers.
pub fn init_worker_threads() {
    #[cfg(target_arch = "wasm32")]
    web::init_rayon();
}

/// `items.iter().map(f)` for work on the UI/render thread: in parallel
/// natively (rayon); sequential on the web, where that is the browser's
/// main thread, which must not block waiting for rayon's workers.
pub fn ui_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        items.par_iter().map(f).collect()
    }
    #[cfg(target_arch = "wasm32")]
    items.iter().map(f).collect()
}

/// Logging: natively to stderr and `gpu.log` in the working directory
/// (overwritten per launch; `RUST_LOG` overrides the `info` default); on the
/// web to the browser console, with panics there too.
pub fn init_logging() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut logger = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
        match std::fs::File::create("gpu.log") {
            Ok(file) => { logger.target(env_logger::Target::Pipe(Box::new(Tee(file)))); }
            Err(e)   => eprintln!("could not create gpu.log: {e}"),
        }
        logger.init();
        log::info!("logging to {}", std::env::current_dir().map(|d| d.join("gpu.log").display().to_string()).unwrap_or_default());
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).ok();
    }
}

/// DIAG: writes every log line to both stderr and `gpu.log` (flushed per
/// line, so the file is complete even if the app hangs and gets killed).
#[cfg(not(target_arch = "wasm32"))]
struct Tee(std::fs::File);

#[cfg(not(target_arch = "wasm32"))]
impl std::io::Write for Tee {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let _ = std::io::stderr().write_all(buf);
        self.0.write_all(buf)?;
        self.0.flush()?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { self.0.flush() }
}

/// Put `s` on the clipboard. On the web this is asynchronous (the browser
/// may ask for permission); a failure is only logged.
pub fn copy_text(s: String) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    return arboard::Clipboard::new().and_then(|mut cb| cb.set_text(s)).map_err(|e| e.to_string());
    #[cfg(target_arch = "wasm32")]
    {
        let clipboard = web_sys::window().ok_or("no window")?.navigator().clipboard();
        let promise = clipboard.write_text(&s);
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = wasm_bindgen_futures::JsFuture::from(promise).await {
                log::warn!("clipboard copy failed: {e:?}");
            }
        });
        Ok(())
    }
}

/// Read the clipboard and hand its text to `then`: natively right away; on
/// the web once the browser has it (asynchronously, maybe after asking for
/// permission). Empty on failure.
pub fn paste_text(then: impl FnOnce(String) + 'static) {
    #[cfg(not(target_arch = "wasm32"))]
    then(arboard::Clipboard::new().and_then(|mut cb| cb.get_text()).unwrap_or_default());
    #[cfg(target_arch = "wasm32")]
    {
        let Some(window) = web_sys::window() else { return then(String::new()) };
        let promise = window.navigator().clipboard().read_text();
        wasm_bindgen_futures::spawn_local(async move {
            let text = wasm_bindgen_futures::JsFuture::from(promise).await;
            then(text.ok().and_then(|t| t.as_string()).unwrap_or_default());
        });
    }
}

/// DIAG: show the debug line: natively the window title; on the web an
/// overlay on the page (a tab title is too short for it).
pub fn show_debug(window: &winit::window::Window, text: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    window.set_title(text);
    #[cfg(target_arch = "wasm32")]
    {
        let _ = window;
        if let Some(el) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id("debug")) {
            el.set_text_content(Some(text));
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use std::future::Future;

    /// Worker script for wasm_thread (its own uses wasm-bindgen's deprecated
    /// init call). With CLOSE the worker ends when the thread function
    /// returns, like a thread; without (`spawn_async`) it stays alive, since
    /// the function only starts an async task on the worker's event loop.
    const WORKER: &str = r#"
import init, {wasm_thread_entry_point} from "SHIM_URL";
self.onmessage = event => {
    let [module, memory, work] = event.data;
    init({ module_or_path: module, memory: memory }).then(() => {
        wasm_thread_entry_point(work);
        CLOSE
    }).catch(err => { setTimeout(() => { throw err; }); });
};
"#;

    /// A blob URL of `WORKER` (made once per variant).
    pub fn worker_url(close: bool) -> String {
        static URLS: std::sync::Mutex<[Option<String>; 2]> = std::sync::Mutex::new([None, None]);
        let slot = close as usize;
        // try_lock: a contended lock must not block (this may be the
        // browser's main thread); another caller is making the same URL
        loop {
            if let Ok(mut urls) = URLS.try_lock() {
                return urls[slot].get_or_insert_with(|| make_url(close)).clone();
            }
            std::hint::spin_loop();
        }
    }

    fn make_url(close: bool) -> String {
        let script = WORKER
            .replace("SHIM_URL", &wasm_thread::get_wasm_bindgen_shim_script_path())
            .replace("CLOSE", if close { "close();" } else { "" });
        let parts = js_sys::Array::of1(&wasm_bindgen::JsValue::from_str(&script));
        let options = web_sys::BlobPropertyBag::new();
        options.set_type("text/javascript");
        let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options).expect("blob");
        web_sys::Url::create_object_url_with_blob(&blob).expect("blob url")
    }

    pub fn spawn_async<F, Fut>(name: &str, f: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + 'static,
    {
        wasm_thread::Builder::new()
            .name(name.into())
            .worker_script_url(worker_url(false))
            .spawn(move || wasm_bindgen_futures::spawn_local(f()))
            .expect("spawn worker");
    }

    /// Rayon's global pool, its threads spawned as Web Workers (one per
    /// hardware thread, less the page's and the compute worker's).
    pub fn init_rayon() {
        let n = wasm_thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(2).max(2);
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .spawn_handler(|t| {
                wasm_thread::Builder::new()
                    .name(format!("rayon {}", t.index()))
                    .worker_script_url(worker_url(true))
                    .spawn(move || t.run())
                    .map(|_| ())
            })
            .build_global();
        match built {
            Ok(()) => log::info!("rayon: {n} worker threads"),
            Err(e) => log::warn!("rayon pool: {e}"),
        }
    }
}
