#!/bin/sh
# Build the web version into web/dist/mandelbrot (served at /mandelbrot/: run
# web/serve.py, or deploy web/dist with `wrangler deploy`; _headers sets the
# headers it needs).
set -e
cd "$(dirname "$0")/.."

BINDGEN_VERSION=$(sed -n '/^name = "wasm-bindgen"$/{n;s/version = "\(.*\)"/\1/p;}' Cargo.lock)
BINDGEN=web/.tools/bin/wasm-bindgen
if [ ! -x "$BINDGEN" ] || [ "$("$BINDGEN" --version | cut -d' ' -f2)" != "$BINDGEN_VERSION" ]; then
    echo "installing wasm-bindgen-cli $BINDGEN_VERSION into web/.tools"
    cargo install wasm-bindgen-cli --version "$BINDGEN_VERSION" --locked --root web/.tools
fi

# std rebuilt with atomics (target flags in .cargo/config.toml)
cargo build --release --target wasm32-unknown-unknown -Z build-std=std,panic_abort

rm -rf web/dist && mkdir -p web/dist/mandelbrot
"$BINDGEN" target/wasm32-unknown-unknown/release/mandelbrot.wasm \
    --out-dir web/dist/mandelbrot --target web --no-typescript
cp web/index.html web/dist/mandelbrot/
cp web/_headers web/dist/
echo "built web/dist ($(du -sh web/dist | cut -f1))"
