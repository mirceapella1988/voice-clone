# =============================================================================
# build-local.ps1 - Local Windows build (no certificate required)
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

Write-Host "`n=== Voice Clone - Local Windows Build ===`n" -ForegroundColor White

# ── Check tools ────────────────────────────────────────────────────────────
if (-not (Get-Command python -ErrorAction SilentlyContinue))  { err "python not found. Install from https://python.org" }
if (-not (Get-Command node   -ErrorAction SilentlyContinue))  { err "node not found. Install from https://nodejs.org" }
if (-not (Get-Command cargo  -ErrorAction SilentlyContinue))  { err "cargo not found. Install from https://rustup.rs" }

# ── [1/4] Python dependencies ──────────────────────────────────────────────
log "[1/4] Installing Python dependencies..."

# Detect CUDA availability and version via nvidia-smi.
$hasCuda = $ForceCuda
$cudaWheel = $null
if (-not $hasCuda) {
    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $cudaVerMatch = & nvidia-smi 2>&1 | Select-String "CUDA Version\s*:\s*([0-9]+\.[0-9]+)"
        if ($cudaVerMatch) {
            $cudaVersion = $cudaVerMatch.Matches[0].Groups[1].Value
            log "NVIDIA GPU detected. CUDA Version: $cudaVersion"

            $gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
            if ($gpuName) { log "GPU: $($gpuName.Trim())" }
            $driverVer = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
            if ($driverVer) { log "Driver: $($driverVer.Trim())" }

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
    $targetVersion = "2.8.0+$cudaWheel"
    $isInstalled = $false

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $checkTorch = python -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>$null
        if ($checkTorch -and $checkTorch.Contains($targetVersion) -and $checkTorch.Contains("True")) {
            $isInstalled = $true
        }
    }

    if ($isInstalled) {
        ok "PyTorch CUDA wheel ($targetVersion) is already installed and verified. Skipping installation."
    } else {
        log "Installing PyTorch CUDA wheels ($cudaWheel) with --no-cache-dir..."
        pip install -q --no-cache-dir --force-reinstall --index-url "https://download.pytorch.org/whl/$cudaWheel" torch==2.8.0+$cudaWheel torchaudio==2.8.0+$cudaWheel
        if ($LASTEXITCODE -ne 0) { err "Failed to install CUDA PyTorch packages." }
        ok "CUDA PyTorch wheels installed ($cudaWheel)"
    }
} elseif ($ForceCuda) {
    $targetVersion = "2.8.0+cu128"
    $isInstalled = $false
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $checkTorch = python -c 'import torch; print(torch.__version__)' 2>$null
        if ($checkTorch -and $checkTorch.Contains($targetVersion)) {
            $isInstalled = $true
        }
    }

    if ($isInstalled) {
        ok "PyTorch CUDA wheel ($targetVersion) is already installed. Skipping installation."
    } else {
        log "ForceCuda enabled without detected GPU. Installing default CUDA wheels (cu128) with --no-cache-dir..."
        pip install -q --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchaudio==2.8.0+cu128
        if ($LASTEXITCODE -ne 0) { err "Failed to install CUDA PyTorch packages." }
        ok "CUDA PyTorch wheels installed (cu128)"
    }
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
    "--onedir",
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
ok "Sidecar built: src-tauri/binaries/sidecar-x86_64-pc-windows-msvc/"

# Remove redundant static library files (.lib) and headers to save space
log "Removing redundant static libraries (.lib) and headers from sidecar package..."
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\*.lib"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\*.h"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\*.hpp"

# Remove redundant / unused CUDA and cuDNN DLLs to significantly reduce bundle size
log "Removing redundant/unused CUDA and cuDNN DLLs from sidecar package..."
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\_internal\torch\lib\cudnn_adv64_9.dll"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\_internal\torch\lib\cusolverMg64_11.dll"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\_internal\torch\lib\nvrtc64_120_0.alt.dll"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\_internal\torch\lib\curand64_10.dll"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\_internal\torch\lib\nvrtc-builtins64_128.dll"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\_internal\torch\lib\uv.dll"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\_internal\torch\lib\cufftw64_11.dll"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\_internal\torch\lib\zlibwapi.dll"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\_internal\torch\lib\libiompstubs5md.dll"
cmd.exe /c "del /s /q /f src-tauri\binaries\sidecar-x86_64-pc-windows-msvc\_internal\torch\lib\caffe2_nvrtc.dll"

# Create a small dummy file to satisfy Tauri's externalBin requirements on Windows.
log "Creating dummy sidecar file for Tauri build..."
New-Item -ItemType File -Force -Path "src-tauri/binaries/sidecar-x86_64-pc-windows-msvc.exe" -Value "" | Out-Null

# ── [3/4] Model bundle policy ──────────────────────────────────────────────
log "[3/4] OmniVoice model bundle policy..."
warn "OmniVoice model files are not bundled. Users download k2-fsa/OmniVoice from Hugging Face on first model load."
if (Test-Path "src-python/models") {
    warn "Local src-python/models cache exists but is ignored and not configured as a Tauri resource."
}

# ── [4/4] Build Tauri app ──────────────────────────────────────────────────
log "[4/4] Building Tauri app..."
npm install --silent
if ($LASTEXITCODE -ne 0) { err "npm install failed." }

# Phase 1: Compile Rust release binary
log "Compiling Rust release binary..."
npm run tauri build -- --no-bundle
if ($LASTEXITCODE -ne 0) { err "Tauri Rust compilation failed." }
ok "Rust binary compiled: src-tauri/target/release/tauri-app.exe"

# Phase 2: Create Portable distribution (Full - with embedded sidecar)
log "Creating portable distribution (Full)..."
$portableDir = "src-tauri/target/release/VoiceClone-Portable"
if (Test-Path $portableDir) { Remove-Item -Recurse -Force $portableDir }
New-Item -ItemType Directory -Force -Path $portableDir | Out-Null
Copy-Item "src-tauri/target/release/tauri-app.exe" "$portableDir/Voice Clone.exe"
New-Item -ItemType Directory -Force -Path "$portableDir/binaries" | Out-Null
Copy-Item -Recurse "src-tauri/binaries/sidecar-x86_64-pc-windows-msvc" "$portableDir/binaries/sidecar-x86_64-pc-windows-msvc"
$totalSize = [math]::Round((Get-ChildItem -Recurse $portableDir -File | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
ok "Portable package created: $portableDir ($totalSize GB)"

# Phase 3: Compress Portable distribution to ZIP
log "Zipping portable distribution..."
$zipPath = "src-tauri/target/release/VoiceClone-Portable.zip"
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path $portableDir -DestinationPath $zipPath -Force
ok "Portable ZIP archive created: $zipPath"

Write-Host "`n" -NoNewline
ok "Build complete!"
Write-Host ""
Write-Host "Output:" -ForegroundColor White
Write-Host "  Portable Full Directory: $(Resolve-Path $portableDir)" -ForegroundColor White
Write-Host "  Portable Full ZIP:       $(Resolve-Path $zipPath)" -ForegroundColor White
Write-Host ""
