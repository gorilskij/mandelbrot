# Update the toolchain

The nightly is pinned in `rust-toolchain.toml` (`nightly-2026-09-25`, with `rust-src` and the
wasm target) because the web build needs `-Z build-std` and the nightly-only wasm threading
flags. It should move forward now and then ([backlog](../backlog.md)).

1. Change `channel` to a newer nightly.
2. `cargo test` and `cargo run --release` (native).
3. `web/build.sh`, then `web/serve.py`: the page must start and render, zoom and recolour.
   **The `+atomics` target feature is being phased out**: a newer nightly may need changes to
   `.cargo/config.toml` (already, recent nightlies stopped adding the shared-memory link args
   for `+atomics`, which the config now passes itself).
4. If `wasm-bindgen` is bumped in `Cargo.lock` too, `build.sh` reinstalls the matching CLI.
5. Deploy to test first ([deploy](deploy.md)): Cloudflare's builder starts from a clean
   toolchain and installs exactly what the file says.
