# Mandelbrot explorer — documentation

An interactive deep-zoom Mandelbrot viewer in Rust (winit + wgpu), native on macOS and on the
web (wasm + WebGPU) at [gorilskij.com/mandelbrot/](https://gorilskij.com/mandelbrot/). It
zooms far below `f64` (views around 2⁻³²⁰ are routine) with **perturbation theory**: one
high-precision reference orbit (`dashu` `FBig`), cheap low-precision delta iteration per
pixel, on the GPU (nucleus references, rebasing, BLA, interior detection) or on the CPU (an
older, simpler backend).

These docs are written so that someone — human or model — with **no prior knowledge** can
pick the project up without guessing. If you had to guess something, that is a gap in the
docs: fill it in the same change as whatever you were doing.

## How the docs are organised

The docs follow [Diátaxis](https://diataxis.fr/): four kinds of page, each answering a
different need. A fifth part, the history, records what was done and why.

| part | answers | start with |
|---|---|---|
| [Tutorials](tutorials/README.md) | "Teach me" — guided first steps | [Getting started](tutorials/getting-started.md) |
| [How-to guides](how-to/README.md) | "How do I …?" — recipes for real tasks | [Reproduce a visual bug](how-to/reproduce-a-visual-bug.md) |
| [Reference](reference/README.md) | "What exactly is …?" — precise facts, rules, layouts | [Repository layout](reference/repo-layout.md) |
| [Explanation](explanation/README.md) | "Why is it like this?" — design reasoning | [Architecture](explanation/architecture.md) |
| [History](history/README.md) | "What happened, when, and why?" — the work log | [The log](history/README.md) |

Two living pages sit beside them:

- [Status](status.md) — what works, what is deployed, branches, test baselines. **Read this
  first** when resuming work.
- [Backlog](backlog.md) — every open thread, TODO and open question.

And one page every newcomer should read before changing anything:

- [Pitfalls](reference/pitfalls.md) — the traps this project has already fallen into.

## Reading order for a newcomer

1. [Status](status.md) — where things stand.
2. [Architecture](explanation/architecture.md) — threads, tiles, backends, compositor.
3. [Perturbation](explanation/perturbation.md) — the math everything rests on.
4. [Getting started](tutorials/getting-started.md) — run it, zoom, break it.
5. [Pitfalls](reference/pitfalls.md) — what not to do.
6. Then whatever the task needs, from the reference.

## Keeping the docs true

- Every change that alters behaviour, a constant, a control, a command or a file updates the
  docs **in the same change**. Stale docs are worse than none: they are believed.
- Reference pages state the **current** truth only. How it came to be goes in the
  [history](history/README.md), with dates, measurements and reasons.
- `cargo test --test docs` fails on broken links between pages (and their `#anchors`), on
  pages no other page links to, and on a `CLAUDE.md` that stops pointing here.
- Hosting (Cloudflare Workers, the site's 404 page, Access) is shared with the owner's site
  and documented there, in the `gorilskij.com` repo's `docs/reference/hosting.md`; this repo
  documents its own side ([deploy](how-to/deploy.md)).
