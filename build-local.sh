#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "ERROR: $*" >&2; exit 1; }

echo ""
echo "=== Voice Clone - Local macOS Build ==="
echo ""

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
  TARGET_TRIPLE="aarch64-apple-darwin"
else
  TARGET_TRIPLE="x86_64-apple-darwin"
fi

command -v node >/dev/null 2>&1 || err "node not found. Install via: brew install node"
command -v npm >/dev/null 2>&1 || err "npm not found."
command -v cargo >/dev/null 2>&1 || err "cargo not found. Install via: https://rustup.rs"

log "Architecture: $ARCH -> Tauri target: $TARGET_TRIPLE"
log "Installing frontend dependencies..."
npm install --silent

log "Building Tauri app without bundled Python sidecar..."
export APPLE_SIGNING_IDENTITY="${APPLE_SIGNING_IDENTITY:--}"
npm run tauri build -- --target "$TARGET_TRIPLE"

echo ""
echo "Build complete."
find src-tauri/target/*/release/bundle -name "*.dmg" -o -name "*.app" 2>/dev/null | head -5 || true
echo ""
echo "Runtime note: Python, FFmpeg, PyTorch, and models are installed automatically on first app launch into ~/.voiceclone."
