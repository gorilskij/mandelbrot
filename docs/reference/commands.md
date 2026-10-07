# Commands

| command | does |
|---|---|
| `cargo run --release` | the native app (macOS; Metal). Dev builds optimise dependencies too (`[profile.dev.package."*"] opt-level = 3`), so `cargo run` is usable; the crate itself stays at opt-level 0 |
| `RUST_LOG=debug cargo run --release` | more logging (default `info`; also teed to `gpu.log`) |
| `cargo test` | the regular tests, no GPU needed ([testing](testing.md)) |
| `cargo test --test docs` | the docs check |
| `cargo test --release -- --ignored` | the GPU tests and `diag_*` measurements (on this machine's GPU) |
| `cargo test --release -- --ignored diag_far_reference --nocapture` | one measurement, with its output |
| `DIAG_OUT=dir cargo test --release -- --ignored …` | cache exact values, dump renders as raw RGB |
| `web/build.sh` | the web build into `web/dist/mandelbrot/` ([web build](web-build.md)) |
| `web/serve.py [port]` | serve it at `http://localhost:8000/mandelbrot/` with the headers |
| `bash web/cf-build.sh` | what Cloudflare runs: install Rust if missing, then `build.sh` without debug info |
| `npx wrangler deploy --dry-run` / `--env test --dry-run` | check the Worker config |
| `npx wrangler deploy` / `--env test` | deploy by hand (normally a push does it: [deploy](../how-to/deploy.md)) |

The nightly toolchain is picked automatically from `rust-toolchain.toml` (rustup installs it,
`rust-src` and the wasm target on first use).
