# Getting started

From a fresh clone to a deep zoom, natively and in the browser. A Mac with Apple silicon
(Metal) is what it is developed on.

## 1. Run it

```
git clone https://github.com/gorilskij/mandelbrot.git
cd mandelbrot
cargo run --release
```

The first build installs the pinned nightly (`rust-toolchain.toml`) and takes a while. A window
opens on the whole set, computed on the **GPU** (the title says `f32`).

## 2. Look around

- **Scroll** to zoom around the cursor; **drag** to pan. Tiles appear coarse first and refine
  (a white progress bar fills at the bottom while a view computes).
- **↑** a few times: more iterations; the boundary fills in without recomputing what was
  already right.
- **Alt-scroll** / **Ctrl-scroll**: shift the palette's hue / lightness — instant, nothing is
  recomputed.
- **→**: sampling ratio s = 1.5, 2, …: smoother edges, more work.
- **Escape**: switch to the CPU backend (title `cpu`) and back.

All controls: [controls](../reference/controls.md).

## 3. Go deep

Paste a deep view: copy this line, focus the window, press **Cmd-V**:

```
-1.9668397736674502471190915717933297855516229637138819293158370614586336,-0.00000000000000000000000000000000000000000000000000029220571175822584063984092359286899246883933607420262095710710521967394|1.0504810301241657e-54/32768
```

(units per pixel ~10⁻⁵⁴, 32768 iterations; from `bugs.txt`). Keep zooming: below ~2⁻¹⁰⁰ the
title switches to `fe` (the floatexp pipeline). Cmd-C copies wherever you are, so you can
come back.

## 4. In the browser

```
web/build.sh       # installs the matching wasm-bindgen CLI the first time
web/serve.py       # http://localhost:8000/mandelbrot/
```

Open it in Chrome (WebGPU). The debug line at the top replaces the window title. Append
`#<a Cmd-C string>` to the URL to open a view directly. The deployed version is at
[gorilskij.com/mandelbrot/](https://gorilskij.com/mandelbrot/).

## 5. Next

[Architecture](../explanation/architecture.md), [perturbation](../explanation/perturbation.md),
then [pitfalls](../reference/pitfalls.md) before changing anything.
