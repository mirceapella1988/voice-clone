#!/usr/bin/env bash
# =============================================================================
# build-local.sh — Local macOS build (no Developer ID required)
# Usage: bash build-local.sh
# =============================================================================
set -euo pipefail

RESET='\033[0m'; BOLD='\033[1m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
log()  { echo -e "${CYAN}${BOLD}[$(date '+%H:%M:%S')]${RESET} $*"; }
ok()   { echo -e "${GREEN}✅ $*${RESET}"; }
warn() { echo -e "${YELLOW}⚠️  $*${RESET}"; }
err()  { echo -e "${RED}❌ $*${RESET}"; exit 1; }

echo -e "\n${BOLD}=== Voice Clone — Local macOS Build ===${RESET}\n"

# ── Detect architecture ────────────────────────────────────────────────────
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    TARGET_TRIPLE="aarch64-apple-darwin"
else
    TARGET_TRIPLE="x86_64-apple-darwin"
fi
log "Architecture: $ARCH → Tauri target: $TARGET_TRIPLE"

# ── Check tools ────────────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || err "python3 not found. Install via: brew install python"
command -v node    >/dev/null 2>&1 || err "node not found. Install via: brew install node"
command -v cargo   >/dev/null 2>&1 || err "cargo not found. Install via: https://rustup.rs"
command -v npm     >/dev/null 2>&1 || err "npm not found."

# ── [1/4] Python dependencies ──────────────────────────────────────────────
log "[1/4] Installing Python dependencies..."
pip3 install -q -r src-python/requirements.txt
pip3 install -q pyinstaller
ok "Python dependencies installed"
PYINSTALLER="python3 -m PyInstaller"

# ── [2/4] Build Python sidecar ────────────────────────────────────────────
log "[2/4] Building Python sidecar with PyInstaller..."
mkdir -p src-tauri/binaries
rm -rf build
rm -f "sidecar-${TARGET_TRIPLE}.spec"

$PYINSTALLER \
    --clean -y \
    --onefile \
    --distpath src-tauri/binaries \
    --specpath build/spec \
    --collect-all omnivoice \
    --collect-all torch \
    --collect-all torchaudio \
    --collect-all transformers \
    --collect-all accelerate \
    --collect-all safetensors \
    --collect-all soundfile \
    --collect-all librosa \
    --collect-all pydub \
    --collect-data huggingface_hub \
    --collect-all hf_xet \
    --hidden-import transformers.models.qwen3 \
    --name "sidecar-${TARGET_TRIPLE}" \
    src-python/sidecar.py

ok "Sidecar built: src-tauri/binaries/sidecar-${TARGET_TRIPLE}"

# ── [3/4] Model bundle policy ──────────────────────────────────────────────
log "[3/4] OmniVoice model bundle policy..."
warn "OmniVoice model files are not bundled. Users download k2-fsa/OmniVoice from Hugging Face on first model load."
if [ -d "src-python/models" ]; then
    warn "Local src-python/models cache exists but is ignored and not configured as a Tauri resource."
fi

# ── [4/4] Build Tauri app ─────────────────────────────────────────────────
log "[4/4] Building Tauri app..."
npm install --silent

# Ad-hoc signing for local use — no Developer ID required
export APPLE_SIGNING_IDENTITY="-"

npm run tauri build -- --target "$TARGET_TRIPLE"

echo ""
ok "Build complete!"
echo ""
echo -e "${BOLD}Output:${RESET}"
find src-tauri/target/*/release/bundle -name "*.dmg" -o -name "*.app" 2>/dev/null | head -5 || true
echo ""
echo -e "${YELLOW}Lưu ý:${RESET} App build local sẽ chạy bình thường trên máy này."
echo -e "Để mở trực tiếp: open src-tauri/target/*/release/bundle/macos/*.app"
