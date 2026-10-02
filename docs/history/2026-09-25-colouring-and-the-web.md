# 2026-09-25 – 26: palette scrolling, GPU colouring, the web build

## Palette scrolling

The palette became two phases (hue and lightness) shifted by scrolling with modifiers. The
modifier mapping moved three times the same day: Cmd-scroll for hue and Alt for lightness, then
Ctrl for lightness (slower hue steps), then Cmd+Ctrl for both, Shift inverting lightness, and
finally **Alt for hue** — Cmd's counterpart on Windows, the Windows key, opens the Start menu,
so Alt works everywhere; the owner unbound their own Alt binding for it.

## Colouring on the GPU

To make palette changes instant at high s, colouring moved to the compositor's GPU: tiles upload
their iteration counts, and a compute pass colours them. Per-tile dispatches took 365 ms for a
recolour at s = 4 (Metal serialises dispatches), so tiles were pooled in **2D array textures
per chunk**, coloured and drawn per chunk (25–30 ms at s = 4). A palette lookup table
(tried as an sRGB texture with hardware decode) was the pass's main cost on noisy tiles: the
palette is now computed in the shader. The iterations stay on the GPU (~1.5 GiB at s = 4),
accepted by the owner.

## The memory ceiling

A ceiling for s (the view's own tiles cannot be evicted) was added for all platforms, then
**limited to wasm** the same day: natively the owner wants "the full ungodly glory of s = 4".
When it bites it must be visible (owner): `s 4 (→3)` in the title.

## The web build

In steps: the compute path made **async** (backends return futures; on the web a worker cannot
block on a GPU readback), a shared compute loop that makes its own backend, then **wasm +
WebGPU with Web Worker threads** (`web/build.sh`, `web/serve.py`); a Display P3 canvas (sRGB
looked duller than native); views from the URL hash. `waker_interrupter` became a git
dependency and `Cargo.lock` was committed (the wasm-bindgen CLI must match), the nightly
pinned. Deployed as the Cloudflare Pages project `mandelbrot` (`mandelbrot.pages.dev` was
taken, so `mandelbrot-6pr.pages.dev`) behind the site's router at
`games.gorilskij.com/mandelbrot/`.

## Interior detection with a foreign nucleus

The owner saw black discs and misshapen minibrots on the web (`view_2026_09_25_black_disc`,
`diag_view_2026_09_25b_references`): with a cached nucleus of another period, interior
detection's windows were out of phase with the pixel's cycle. Fixed with the **return test**
(z must come back within 2⁻¹⁰ of where it was a window earlier). Space now also drops the GPU's
cached reference and search state, so it searches afresh.

On 2026-09-26 the notes were brought up to date (deploy, URL-hash views, the P3 canvas, the
2026-09-25 tests, the open threads).
