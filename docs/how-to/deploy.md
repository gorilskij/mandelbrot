# Deploy

The web build is served at `gorilskij.com/mandelbrot/` by the Cloudflare Worker `mandelbrot`,
and at `test.gorilskij.com/mandelbrot/` (behind Cloudflare Access) by `mandelbrot-test`. Both
deploy themselves when their branch is pushed (Workers Builds). The whole hosting picture,
shared with the owner's site and hex snake, is in the `gorilskij.com` repo
(`docs/reference/hosting.md`).

## The flow

```
master ──merge──▶ test-website ──merge──▶ pub-website
                   │                       │
                   ▼                       ▼
     test.gorilskij.com/mandelbrot/   gorilskij.com/mandelbrot/
```

1. Work on `master`. **`master` is never deployed directly** — it may carry work not meant to
   ship.
2. **Test:** merge `master` into `test-website`, push it. Workers Builds runs
   `bash web/cf-build.sh` then `npx wrangler deploy --env test` (~4.5 min: it installs Rust
   and rebuilds std).
3. Check it on `test.gorilskij.com/mandelbrot/` (the owner's login).
4. **Prod:** merge `test-website` into `pub-website` ("Merge branch 'test-website' into
   pub-website"), push it: `npx wrangler deploy`.

Builds of all the owner's projects queue one at a time on the account; see them in the
Cloudflare dashboard (Workers & Pages → `mandelbrot` → Deployments).

## What is deployed

- `wrangler.toml`: Worker `mandelbrot`, routes `gorilskij.com/mandelbrot` and
  `gorilskij.com/mandelbrot/*` (zone `gorilskij.com`); `[env.test]`: `mandelbrot-test` on the
  same paths of `test.gorilskij.com`. Both `workers_dev = false`, `preview_urls = false` (no
  address that bypasses the login). `name`, `routes`, `workers_dev`, `preview_urls` are
  restated under `[env.test]` because they do not inherit.
- Static assets: `web/dist` — the files under `mandelbrot/` (so a request path is a file), and
  `_headers` (COOP/COEP for `/mandelbrot/*`).
- `web/worker.js` runs only when no file matches, and passes the request on to the site's
  Worker, which answers with its 404 page. So `/mandelbrot/nope` gets the site's 404.
- The routes run in front of the site's Worker (which owns `gorilskij.com`).

## By hand

```
bash web/cf-build.sh && npx wrangler deploy --env test    # test
bash web/cf-build.sh && npx wrangler deploy               # prod
```

(`npx wrangler login` once.) A hand deploy is replaced by the next push to that branch. Check a
config change first with `npx wrangler deploy --dry-run` (and `--env test`).
