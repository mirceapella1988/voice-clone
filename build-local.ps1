$ErrorActionPreference = "Stop"

function log { param($msg) Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg" -ForegroundColor Cyan }
function err { param($msg) Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "=== Voice Clone - Local Windows Build ==="
Write-Host ""

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { err "node not found. Install from https://nodejs.org" }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { err "npm not found." }
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { err "cargo not found. Install from https://rustup.rs" }

log "Installing frontend dependencies..."
npm install --silent
if ($LASTEXITCODE -ne 0) { err "npm install failed." }

log "Building Tauri app without bundled Python sidecar..."
npm run tauri build
if ($LASTEXITCODE -ne 0) { err "Tauri build failed." }

Write-Host ""
Write-Host "Build complete."
Write-Host "Runtime note: Python, FFmpeg, PyTorch, and models install into %LOCALAPPDATA%\Voice Clone."
