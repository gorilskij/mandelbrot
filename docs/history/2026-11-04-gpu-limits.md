# The high-iteration crash and the GPU's buffer limits (2026-11-04)

**The report.** At the owner's 2⁻⁴⁴⁴ view
(`-1.962806555174172021647685634946390363305838735256282905291651714974987176375827993719362540520237766236256620207342904928884631910114733336316139198285994494676342,-0.000000000000000000000000000000000000000000000000000000000000000042947767643305669232409999573568561541838549727767369923819385466754528090057229022273635574237949421053083159607872189092473578715032590666576834883379438641201333430314048627108223712715224994612480066102195044615076323678|2.346020205021077e-134/4096`),
raising the iterations froze rendering: the progress bar stuck, and lowering the iterations,
zooming or panning did not bring it back. Natively, "very consistent".

**Reproducing it.** The test harness runs each generation to completion, so it did not
(1M finished in 7.5 s); Chrome showed a different, slower problem (backlog: the centre's
bigfloat orbit is not interruptible, 17.5 s at 1M in wasm, and is discarded when interrupted).
Driving the real native app with System Events (paste the view, ↑ every 0.5 s), with the owner
taking it on to 8M and zooming, found it: the compute thread **panicked** on a fatal wgpu
validation error, `Device::create_buffer: Buffer size 536870880 is greater than the maximum
buffer size (268435456)`: the BLA table of an 8M-iteration orbit ((2²⁴ − 1) × 32 bytes). With
the compute thread dead nothing renders again.

**Why so low.** The device was created with wgpu's default limits: 256 MiB per buffer, 128 MiB
per storage binding. The M2 Pro's adapter allows 4 GiB for both. At the defaults 2M iterations
was the last count that fit (its table 134217696 bytes against a 134217728 binding).

**The plan** (agreed with the owner): drop the BLA table's unused level 0 (half its size);
request the adapter's limits; refuse ↑ when the next count would not fit (the owner: refuse,
rather than degrade the BLA table); make a crashed thread visible.

**Level 0 dropped.** The shaders search from level 1 (a single-step jump costs as much as a
plain iteration), so level 0 is only needed to build level 1; the upload now starts at level 1
(`BlaTable::gpu_index`, `bla_find`'s index − 2^L). Checked: the layout test compares every
uploaded entry with its block; all 16 GPU tests pass (BLA against plain iteration, the
pipeline against exact orbits).

**The adapter's limits, and ↑ refused past them.** The device now requests `adapter.limits()`;
`max_iterations` (unit-tested against wgpu's defaults, 2²², and 4 GiB, 2²⁷) bounds ↑ (refused,
with a title note and a warning, the owner's call over degrading the table) and pastes (capped).
Checked by driving the native app with System Events up to 8M at the owner's view: no panic,
all 7 passes (generation 12: pass 0 5.5 s after the nucleus search's 19.2 s, the whole 16.7 s);
the log reports the ceiling, 134217728. The refusal itself was not reached (2²⁷ would need
tens of GiB of RAM for the bigfloat orbit first).

