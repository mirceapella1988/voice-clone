# =============================================================================
# build-local.ps1 — Local Windows build (no certificate required)
# Usage: Right-click → Run with PowerShell, or: powershell -ExecutionPolicy Bypass -File build-local.ps1
# =============================================================================
param(
    [switch]$ForceCuda
)

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

# Detect CUDA availability and version via nvidia-smi.
# Maps the installed driver CUDA version to the closest compatible PyTorch wheel.
# The base requirements install CPU/MPS wheels; NVIDIA Windows builds
# override PyTorch with CUDA wheels before packaging.
$hasCuda = $ForceCuda
$cudaWheel = $null
if (-not $hasCuda) {
    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        # Parse exact CUDA version reported by nvidia-smi
        $cudaVerMatch = & nvidia-smi 2>&1 | Select-String "CUDA Version\s*:\s*([0-9]+\.[0-9]+)"
        if ($cudaVerMatch) {
            $cudaVersion = $cudaVerMatch.Matches[0].Groups[1].Value
            log "NVIDIA GPU detected. CUDA Version: $cudaVersion"

            # Additional GPU info for diagnostics
            $gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
            if ($gpuName) { log "GPU: $($gpuName.Trim())" }
            $driverVer = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
            if ($driverVer) { log "Driver: $($driverVer.Trim())" }

            # Map CUDA driver version to PyTorch wheel index
            if ([version]$cudaVersion -ge [version]"12.4") {
                $cudaWheel = "cu128"
            } elseif ([version]$cudaVersion -ge [version]"12.0") {
                $cudaWheel = "cu124"
            } elseif ([version]$cudaVersion -ge [version]"11.8") {
                $cudaWheel = "cu118"
            } else {
                warn "CUDA version $cudaVersion is older than 11.8. PyTorch CUDA wheels may not be available; falling back to CPU build."
            }

            if ($cudaWheel) {
                $hasCuda = $true
                log "Selected PyTorch CUDA wheel: $cudaWheel"
            }
        } else {
            warn "nvidia-smi found but CUDA Version not detected in output. Skipping CUDA install."
        }
    } else {
        warn "nvidia-smi not found. Build will be CPU-only."
    }
}

pip install -q -r src-python/requirements.txt
if ($LASTEXITCODE -ne 0) { err "Failed to install Python requirements." }

if ($hasCuda -and $cudaWheel) {
    log "Installing PyTorch CUDA wheels ($cudaWheel)..."
    pip install -q --force-reinstall --index-url "https://download.pytorch.org/whl/$cudaWheel" torch==2.8.0+$cudaWheel torchaudio==2.8.0+$cudaWheel
    if ($LASTEXITCODE -ne 0) { err "Failed to install CUDA PyTorch packages." }
    ok "CUDA PyTorch wheels installed ($cudaWheel)"
} elseif ($ForceCuda) {
    # ForceCuda without a GPU present (e.g., CI build): default to cu128 as before
    log "ForceCuda enabled without detected GPU. Installing default CUDA wheels (cu128)..."
    pip install -q --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchaudio==2.8.0+cu128
    if ($LASTEXITCODE -ne 0) { err "Failed to install CUDA PyTorch packages." }
    ok "CUDA PyTorch wheels installed (cu128)"
}
pip install -q pyinstaller
if ($LASTEXITCODE -ne 0) { err "Failed to install PyInstaller." }
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
if ($LASTEXITCODE -ne 0) { err "PyInstaller build failed." }
ok "Sidecar built: src-tauri/binaries/sidecar-x86_64-pc-windows-msvc.exe"

# ── [3/4] Model bundle policy ──────────────────────────────────────────────
log "[3/4] OmniVoice model bundle policy..."
warn "OmniVoice model files are not bundled. Users download k2-fsa/OmniVoice from Hugging Face on first model load."
if (Test-Path "src-python/models") {
    warn "Local src-python/models cache exists but is ignored and not configured as a Tauri resource."
}

# ── [4/4] Build Tauri app ─────────────────────────────────────────────────
log "[4/4] Building Tauri app (MSI + NSIS)..."
npm install --silent
if ($LASTEXITCODE -ne 0) { err "npm install failed." }
npm run tauri build
if ($LASTEXITCODE -ne 0) { err "Tauri build failed." }

Write-Host "`n" -NoNewline
ok "Build complete!"
Write-Host ""
Write-Host "Output:" -ForegroundColor White
Get-ChildItem -Recurse -Path "src-tauri/target/release/bundle" -Include "*.msi","*.exe" -ErrorAction SilentlyContinue | Select-Object FullName | Format-Table -HideTableHeaders
Write-Host ""
Write-Host "Lưu ý: Chạy file .msi hoặc .exe trong thư mục trên để cài đặt." -ForegroundColor Yellow
