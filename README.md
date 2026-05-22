# Voice Clone

Tauri desktop app for zero-shot voice cloning with [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).

## Runtime

- Voice Clone supports Windows and macOS.
- Installers do not bundle a PyInstaller sidecar, Python runtime, PyTorch runtime, FFmpeg binary, or OmniVoice model weights.
- On first launch, the app installs runtime assets automatically into the app local data directory:

| Platform | Runtime root | Folder cache |
| --- | --- | --- |
| macOS | `~/Library/Application Support/Voice Clone/runtime` | `~/Library/Application Support/Voice Clone/models` |
| Windows | `%LOCALAPPDATA%\Voice Clone\runtime` | `%LOCALAPPDATA%\Voice Clone\models` |

Runtime layout:

```text
Voice Clone/
  runtime/
    python/
    ffmpeg/
  models/
```

The bundled app includes only the Tauri shell and Python source files required to run the local runtime. Model weights are downloaded from Hugging Face on first model load and cached in `models/`.

## Development

```bash
npm install
npm run build
```

Run the Tauri dev app:

```bash
npm run tauri dev
```

The dev app uses `src-python/sidecar.py` and the runtime installed in the app local data directory. If runtime packages are missing, the setup screen installs them before the main UI loads.

## Local Packaged Builds

Build on the same OS you want to package. macOS can build macOS apps; Windows can build Windows installers. Cross-building Windows installers on macOS is not supported by the current Tauri/NSIS setup.

### macOS

Requirements:

- macOS.
- Node.js/npm and Rust/Cargo.
- Xcode Command Line Tools.

```bash
bash build-local.sh
```

The script installs frontend dependencies and builds the Tauri app. Runtime dependencies install automatically on first launch.

### Windows

Requirements:

- Windows x64.
- Node.js/npm, Rust/Cargo, and NSIS for installer output.

```powershell
powershell -ExecutionPolicy Bypass -File build-local.ps1
```

The script installs frontend dependencies and builds the Tauri app. Runtime dependencies install automatically on first launch.

## GitHub Release Builds

GitHub Actions is the primary hosted release-build path for macOS and Windows installers.

The workflow is defined in `.github/workflows/build-release.yml` and runs on:

- `push` tags matching `v*`
- manual `workflow_dispatch`

Release jobs build the Tauri installers only. They assert that `externalBin`, PyInstaller sidecars, runtime binaries, and model directories are not bundled.

Trigger a GitHub release build:

```bash
git tag -a v0.4.9 -m "Release v0.4.9"
git push origin v0.4.9
```

The Tauri action creates a draft GitHub Release for the tag. Review the generated installers before publishing the release.

## Release Checklist

1. Bump version in:
   - `package.json`
   - `package-lock.json`
   - `src-tauri/Cargo.toml`
   - `src-tauri/Cargo.lock`
   - `src-tauri/tauri.conf.json`
2. Run verification:
   - `python3 -m unittest discover -s src-python/tests -q`
   - `npm run build -- --clearScreen false`
   - `cd src-tauri && cargo test`
3. Commit the version and source changes.
4. Create and push a tag like `v0.4.9`.
5. Let GitHub Actions build the installers.

## Troubleshooting

### First launch is slow

This is expected. The app installs Python, FFmpeg, PyTorch, and voice-cloning packages into the app local data runtime folder.

### First model load is slow

This is expected. The app downloads `k2-fsa/OmniVoice` to the app local data `models` folder the first time the user loads the model.

### macOS app is blocked by Gatekeeper

Local builds are ad-hoc signed for internal testing. Public distribution should use Apple Developer ID signing and notarization.

### Windows build cannot find NSIS

Install NSIS and ensure `makensis` is on `PATH`:

```powershell
choco install -y nsis
```
