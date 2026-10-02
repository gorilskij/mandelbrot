# Mandelbrot explorer

An interactive deep-zoom Mandelbrot viewer in Rust (winit + wgpu), native and on the web
(`gorilskij.com/mandelbrot/`), using perturbation theory on the GPU. **All documentation is in
[`docs/`](docs/README.md).**

## Start here

1. [`docs/status.md`](docs/status.md) — what works, what is deployed, branches, test baselines.
2. [`docs/backlog.md`](docs/backlog.md) — the open threads (talk them through with the owner
   before coding) and what was decided.
3. [`docs/reference/pitfalls.md`](docs/reference/pitfalls.md) — before touching the shaders,
   the GPU dispatch, precision or the web build.
4. Then the part of [`docs/`](docs/README.md) the task needs.

## Keep the docs true

- Every change updates `docs/` in the same change: the reference pages to the new truth, a
  dated [history](docs/history/README.md) entry with the reasons, measurements and rejected
  alternatives, and [status](docs/status.md) / [backlog](docs/backlog.md) when they change.
  Findings go into the docs the moment they surface, not only into the reply.
- Reference pages hold no history; the history holds no current rules.
- `cargo test --test docs` must pass (links, anchors, every page reachable).
- Keep this file short and pointing into `docs/`.

## How the owner likes to work

- **A visual bug with coordinates: reproduce it first** in the test harness, then A/B fixes
  there ([how](docs/how-to/reproduce-a-visual-bug.md)) — not theorising, not a CPU mirror of a
  shader.
- **Talk open threads through before coding them**; agreeing on *what* is not a go to start.
- **`Pf` is f32 on purpose**: ask before changing it.
- **Never deploy from `master`**: deploys go `master` → `test-website` → `pub-website`, each on
  a push ([deploy](docs/how-to/deploy.md)).
- **Commit each logical step separately**, with the measurements in the message.
- The owner types Dvorak: letter shortcuts match the character, not the key position.

## Most-used commands

```
cargo run --release                     # the app (starts on the GPU)
cargo test                              # regular tests, no GPU (baseline: docs/status.md)
cargo test --release -- --ignored       # GPU tests and diag_* measurements
web/build.sh && web/serve.py            # the web build at http://localhost:8000/mandelbrot/
```

Everything else: [`docs/reference/commands.md`](docs/reference/commands.md).
