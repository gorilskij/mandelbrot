#!/usr/bin/env bash
# Cloudflare Pages build script.
#
# Set in the Pages project settings:
#   Build command:           bash web/cf-build.sh
#   Build output directory:  web/dist
#   Production branch:       pub-website
#
# The Pages build image has no Rust toolchain, so install rustup;
# rust-toolchain.toml brings the pinned nightly, rust-src and the wasm target.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

# The release profile keeps debug info (for native profiling); Pages rejects
# files over 25 MiB, so drop it for the deployed build.
CARGO_PROFILE_RELEASE_DEBUG=false sh web/build.sh
