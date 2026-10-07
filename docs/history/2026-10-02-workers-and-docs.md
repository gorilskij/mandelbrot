# 2026-10-02 – 03: hosting on Workers, and these docs

## Hosting on Workers (2026-10-02)

Together with the owner's site and hex snake, mandelbrot moved from Cloudflare Pages behind a
router to a Cloudflare Worker with static assets, and from `games.gorilskij.com/mandelbrot/` to
**`gorilskij.com/mandelbrot/`** (the owner's call: out of the games section). The full account,
with the reasons (Pages' public `*.pages.dev` addresses bypassed the test login) and how it was
checked, is in the `gorilskij.com` repo's history (`docs/history/2026-10-02-workers-migration.md`).

What changed here: `web/build.sh` stages under `web/dist/mandelbrot/` (the path it is served
at), `_headers` applies to `/mandelbrot/*`, `web/worker.js` passes misses to the site's 404 page,
`wrangler.toml` defines `mandelbrot` and `mandelbrot-test`, Workers Builds deploys on push to
`test-website` / `pub-website`. Checked in headless Chrome from the new path:
cross-origin isolated, renders.

The hosting commit was first made on `master` and the test Worker deployed from it; for
mandelbrot `master` differed only in docs, but it was rebased onto `test-website` like hex
snake's ([pitfalls](../reference/pitfalls.md#process)). On 2026-10-03 `test-website` was merged
back into `master`.

## The docs (2026-10-03)

The notes moved from `CLAUDE.md` into `docs/` (Diátaxis, plus history, status and backlog),
following the owner's other repos; the old `CLAUDE.md` is archived verbatim
([working notes](working-notes-2026-10-03.md)). Every constant, identifier and test named was
checked against the code. What the old notes had wrong or missing:

- the GPU dispatch chain (`dispatch_f32` / `dispatch_fe` do not exist; the main chunks go
  through `Deltas::pack` → `submit_deltas` → `finish`);
- why the web build has its own worker script (wasm-bindgen's deprecated init call in
  `wasm_thread`'s, and two variants: closing and staying alive);
- scrolling and dragging, the zoom-out limit, and what the title shows;
- the two benchmark files in `tests/`, and `bugs.txt`;
- a stale comment in `rust-toolchain.toml` (it said the Cloudflare **Pages** builder) — fixed.

`tests/docs.rs` now checks the docs' links and that every page is reachable.
