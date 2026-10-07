//! FIREFOX WORKAROUND (web only): remove once Firefox fixes
//! <https://bugzilla.mozilla.org/show_bug.cgi?id=1870699>.
//!
//! Firefox's GPU process notices finished GPU work only on a 100 ms timer or
//! when the same device gets a `queue.submit`, so every readback resolves up
//! to 100 ms late (measured: 1024 px chunks of cheap pixels at ~100 ms each).
//! From shortly before the chunk's estimated completion, submit an empty
//! command buffer every NUDGE_EVERY so Firefox checks. Browsers that deliver
//! on time get few nudges, so it needs no browser check.
//!
//! The nudges must start before the estimate: in Firefox a result arrives
//! only at a nudge, so one that starts at the estimate never sees a chunk
//! finish earlier, and an estimate made from readback times never falls
//! (measured: the GPU busy ~10 % of each chunk).
//! See docs/reference/web-build.md, "Firefox reads back late".

use std::future::{poll_fn, Future};
use std::pin::{pin, Pin};
use std::task::Poll;
use std::time::Duration;

use crate::platform::Fired;

/// Interval between nudges. Browsers clamp nested timers to ~4 ms anyway; the
/// result arrives up to this much after the GPU finished.
const NUDGE_EVERY: Duration = Duration::from_millis(4);

/// How much before an estimate from GPU timestamps the nudges start, to
/// catch a chunk that finishes early.
const START_EARLY: Duration = Duration::from_millis(4);

/// Wait for `mapped` (submitted `elapsed` ago, estimated to take `expected`),
/// nudging the device every NUDGE_EVERY from: START_EARLY before `expected`
/// when it comes from GPU timestamps (`exact`); else from half of it, since
/// an estimate from readback times includes Firefox's lateness and must be
/// able to fall. Returns the number of nudges.
pub async fn wait(
    device:   &wgpu::Device,
    queue:    &wgpu::Queue,
    mapped:   &mut Fired,
    expected: Duration,
    elapsed:  Duration,
    exact:    bool,
) -> usize {
    let start = if exact { expected.saturating_sub(START_EARLY) } else { expected / 2 };
    let mut nudges = 0;
    let mut delay = start.saturating_sub(elapsed);
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
