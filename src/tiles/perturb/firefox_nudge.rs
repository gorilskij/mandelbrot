//! FIREFOX WORKAROUND (web only): remove once Firefox fixes
//! <https://bugzilla.mozilla.org/show_bug.cgi?id=1870699>.
//!
//! Firefox's GPU process notices finished GPU work only on a 100 ms timer or
//! when the same device gets a `queue.submit`, so every readback resolves up
//! to 100 ms late (measured: 1024 px chunks of cheap pixels at ~100 ms each).
//! While a readback is late (past the chunk's estimated GPU time), submit an
//! empty command buffer every NUDGE_EVERY so Firefox checks. Browsers that
//! deliver on time get (almost) no nudges, so it needs no browser check.
//! See docs/reference/web-build.md, "Firefox reads back late".

use std::future::{poll_fn, Future};
use std::pin::{pin, Pin};
use std::task::Poll;
use std::time::Duration;

use crate::platform::Fired;

/// Interval between nudges once a readback is late. Browsers clamp nested
/// timers to ~4 ms anyway; the result arrives up to this much after the GPU
/// finished, which chunk sizing sees as cost.
const NUDGE_EVERY: Duration = Duration::from_millis(4);

/// Wait for `mapped`, nudging the device once `late_after` has passed (and
/// every NUDGE_EVERY after that). Returns the number of nudges.
pub async fn wait(device: &wgpu::Device, queue: &wgpu::Queue, mapped: &mut Fired, late_after: Duration) -> usize {
    let mut nudges = 0;
    let mut delay = late_after;
    while !fired_within(mapped, delay).await {
        queue.submit([device.create_command_encoder(&Default::default()).finish()]);
        nudges += 1;
        delay = NUDGE_EVERY;
    }
    nudges
}

/// Whether `mapped` fires within `d` (returning as soon as it does).
async fn fired_within(mapped: &mut Fired, d: Duration) -> bool {
    let mut timer = pin!(crate::platform::sleep(d));
    poll_fn(|cx| {
        if Pin::new(&mut *mapped).poll(cx).is_ready() { return Poll::Ready(true); }
        timer.as_mut().poll(cx).map(|()| false)
    }).await
}
