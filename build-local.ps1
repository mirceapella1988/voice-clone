# =============================================================================
# build-local.ps1 — Local Windows build (no certificate required)
# Usage: Right-click → Run with PowerShell, or: powershell -ExecutionPolicy Bypass -File build-local.ps1
# =============================================================================
$ErrorActionPreference = "Stop"

function log  { param($msg) Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg" -ForegroundColor Cyan }
function ok   { param($msg) Write-Host "✅ $msg" -ForegroundColor Green }
function warn { param($msg) Write-Host "⚠️  $msg" -ForegroundColor Yellow }
function err  { param($msg) Write-Host "❌ $msg" -ForegroundColor Red; exit 1 }

Write-Host "`n=== Voice Clone — Local Windows Build ===`n" -ForegroundColor White

# ── Check tools ────────────────────────────────────────────────────────────
if (-not (Get-Command python -ErrorAction SilentlyContinue))  { err "python not found. Install from https://python.org" }
if (-not (Get-Command node   -ErrorAction SilentlyContinue))  { err "node not found. Install from https://nodejs.org" }
if (-not (Get-Command cargo  -ErrorAction SilentlyContinue))  { err "cargo not found. Install from https://rustup.rs" }

# ── [1/4] Python dependencies ──────────────────────────────────────────────
log "[1/4] Installing Python dependencies..."

# Detect CUDA availability. The base requirements install CPU/MPS wheels; NVIDIA
# Windows builds override PyTorch with CUDA wheels before packaging.
$hasCuda = $false
$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $cudaVer = & nvidia-smi 2>&1 | Select-String "CUDA Version"
    if ($cudaVer) {
        warn "NVIDIA GPU detected. PyTorch CUDA wheels will be installed."
        $hasCuda = $true
    }
}

pip install -q -r src-python/requirements.txt
if ($hasCuda) {
    pip install -q --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchaudio==2.8.0+cu128
}
pip install -q pyinstaller
ok "Python dependencies installed"

# ── [2/4] Build Python sidecar ────────────────────────────────────────────
log "[2/4] Building Python sidecar with PyInstaller..."
New-Item -ItemType Directory -Force -Path "src-tauri/binaries" | Out-Null
Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue
Remove-Item -Force "sidecar-x86_64-pc-windows-msvc.spec" -ErrorAction SilentlyContinue

$pyinstallerArgs = @(
    "--clean", "-y",
    "--onefile",
    "--distpath", "src-tauri/binaries",
    "--specpath", "build/spec",
    "--collect-all", "omnivoice",
    "--collect-all", "torch",
    "--collect-all", "torchaudio",
    "--collect-all", "transformers",
    "--collect-all", "accelerate",
    "--collect-all", "safetensors",
    "--collect-all", "soundfile",
    "--collect-all", "librosa",
    "--collect-all", "pydub",
    "--collect-data", "huggingface_hub",
    "--collect-all", "hf_xet",
    "--hidden-import", "transformers.models.qwen3",
    "--name", "sidecar-x86_64-pc-windows-msvc",
    "src-python/sidecar.py"
)

& pyinstaller @pyinstallerArgs
ok "Sidecar built: src-tauri/binaries/sidecar-x86_64-pc-windows-msvc.exe"

# ── [3/4] Optional model cache prewarm ─────────────────────────────────────
log "[3/4] OmniVoice model cache..."
if ($env:PREWARM_OMNIVOICE_MODELS -eq "1") {
    python src-python/download_models.py
    ok "Models cached"
} else {
    warn "Skipping model prewarm. Set PREWARM_OMNIVOICE_MODELS=1 to cache k2-fsa/OmniVoice during build."
}

# ── [4/4] Build Tauri app ─────────────────────────────────────────────────
log "[4/4] Building Tauri app (MSI + NSIS)..."
npm install --silent
npm run tauri build

Write-Host "`n" -NoNewline
ok "Build complete!"
Write-Host ""
Write-Host "Output:" -ForegroundColor White
Get-ChildItem -Recurse -Path "src-tauri/target/release/bundle" -Include "*.msi","*.exe" -ErrorAction SilentlyContinue | Select-Object FullName | Format-Table -HideTableHeaders
Write-Host ""
Write-Host "Lưu ý: Chạy file .msi hoặc .exe trong thư mục trên để cài đặt." -ForegroundColor Yellow
