# Voice Clone

Tauri desktop app for zero-shot voice cloning with [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).

## Runtime

- The Python sidecar loads `k2-fsa/OmniVoice` through the official `omnivoice` PyTorch API.
- Device selection prefers CUDA, then Apple Silicon MPS, then CPU.
- Installers do **not** bundle OmniVoice model files. Models are downloaded from https://huggingface.co/k2-fsa/OmniVoice on first model load and cached in `~/.voiceclone/models`.
- The bundled Python sidecar contains the runtime dependencies needed to load OmniVoice, but the model weights stay outside the installer.

## Model storage policy

Model files are intentionally excluded from Git, local installers, GitHub releases, and GitLab artifacts.

| Platform | Model cache path |
| --- | --- |
| macOS | `~/.voiceclone/models` |
| Windows | `%USERPROFILE%\.voiceclone\models` |
| Linux | `~/.voiceclone/models` |

If `src-python/models` exists locally, it is treated only as a developer cache and is not configured as a Tauri resource.

## Development

```bash
npm install
python3 -m pip install -r src-python/requirements.txt
npm run build
```

Run the Tauri dev app:

```bash
npm run tauri dev
```

In dev mode, the app prefers `src-python/sidecar.py` so Python-side changes are picked up without rebuilding the sidecar binary.

## Local packaged builds

Build on the same OS you want to package. macOS can build macOS apps; Windows can build Windows installers. Cross-building Windows installers on macOS is not supported by the current Tauri/NSIS setup.

### macOS build

Requirements:

- macOS, preferably Apple Silicon for the default `aarch64-apple-darwin` target.
- Python 3, Node.js/npm, Rust/Cargo.
- Xcode Command Line Tools.

Install common tools:

```bash
xcode-select --install
brew install node python
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Build:

```bash
bash build-local.sh
```

What the script does:

1. Detects `arm64` or `x86_64` and selects the matching Tauri target.
2. Installs Python dependencies from `src-python/requirements.txt`.
3. Builds the Python sidecar with PyInstaller into `src-tauri/binaries/`.
4. Verifies OmniVoice model files are not bundled.
5. Runs `npm run tauri build -- --target <target>`.

Outputs:

- `.app`: `src-tauri/target/*/release/bundle/macos/*.app`
- `.dmg`: `src-tauri/target/*/release/bundle/dmg/*.dmg`

Local macOS builds use ad-hoc signing (`APPLE_SIGNING_IDENTITY=-`). This is fine for internal testing, but public distribution should use Apple Developer ID signing and notarization.

### Windows build

Requirements:

- Windows x64.
- Python, Node.js/npm, Rust/Cargo.
- NSIS for `.exe` installer output.
- Optional: NVIDIA GPU/CUDA. The script installs CUDA PyTorch wheels when `nvidia-smi` is detected.

Install common tools with Chocolatey:

```powershell
choco install -y python nodejs-lts rustup.install nsis
```

Build:

```powershell
powershell -ExecutionPolicy Bypass -File build-local.ps1
```

To force packaging with PyTorch CUDA wheels (useful for CI/build servers without a local NVIDIA GPU):

```powershell
powershell -ExecutionPolicy Bypass -File build-local.ps1 -ForceCuda
```

What the script does:

1. Installs Python dependencies from `src-python/requirements.txt`.
2. If NVIDIA/CUDA is detected (or `-ForceCuda` is passed), replaces PyTorch with CUDA wheels.
3. Builds the Python sidecar with PyInstaller into `src-tauri/binaries/`.
4. Verifies OmniVoice model files are not bundled.
5. Runs `npm run tauri build`.

Outputs:

- NSIS `.exe`: `src-tauri/target/release/bundle/nsis/*.exe`
- MSI `.msi`: `src-tauri/target/release/bundle/msi/*.msi` if enabled by Tauri output.

The Windows installer includes runtime dependencies and the sidecar, but users still download `k2-fsa/OmniVoice` model weights on first model load.

## GitHub release builds

GitHub Actions is the primary hosted release-build path for macOS and Windows installers.

The workflow is defined in `.github/workflows/build-release.yml` and runs on:

- `push` tags matching `v*`
- manual `workflow_dispatch`

### GitHub jobs

| Runner | Target | Output |
| --- | --- | --- |
| `macos-latest` | `aarch64-apple-darwin` | macOS `.dmg` / `.app` |
| `windows-latest` | NSIS bundle | Windows `.exe` installer |

Both jobs build a PyInstaller sidecar first, assert that model files are not bundled, then run the Tauri release action.

### Trigger a GitHub release build

Push a version tag:

```bash
git tag -a v0.4.7 -m "Release v0.4.7"
git push origin v0.4.7
```

Or run **Build & Release** manually from the GitHub Actions UI.

The Tauri action creates a **draft GitHub Release** for the tag. Review the generated installers before publishing the release.

### macOS signing

Public macOS distribution should use Apple Developer ID signing and notarization. Configure all of these repository secrets together:

- `APPLE_CERTIFICATE`
- `APPLE_CERTIFICATE_PASSWORD`
- `APPLE_SIGNING_IDENTITY`
- `APPLE_ID`
- `APPLE_PASSWORD`
- `APPLE_TEAM_ID`

If no Apple secrets are configured, the workflow uses ad-hoc signing for internal testing. If only some Apple secrets are configured, the workflow fails early so it does not create a partially signed release.

### Windows CUDA support

The Windows release job installs PyTorch CUDA wheels (`cu128`) before building the sidecar. At runtime, the app uses CUDA when a compatible NVIDIA/CUDA environment is available and falls back to CPU otherwise.

### GitHub Actions billing

GitHub-hosted macOS and Windows runners may consume billable minutes depending on the account plan. Before pushing a release tag, confirm:

- GitHub Actions **Build & Release** workflow is enabled.
- Billing and spending limits allow hosted runners to start.
- You want the tag push to trigger hosted builds.

If GitHub Actions is disabled manually, pushing tags still works, but release builds will not start until the workflow is re-enabled.

## GitLab CI fallback

`.gitlab-ci.yml` is kept as an optional fallback for projects that want to build with GitLab runners instead of GitHub-hosted runners.

The GitLab pipeline is tag/manual only so normal pushes do not spend runner minutes.

- Default runner tags: `saas-macos-medium-m1`, `shared-windows`, `saas-linux-medium-amd64`.
- If your GitLab project shows different runner tags, override `MACOS_RUNNER_TAG`, `WINDOWS_RUNNER_TAG`, or `LINUX_RUNNER_TAG` in CI/CD variables.
- GitLab release artifacts are uploaded to the GitLab Generic Package Registry and linked from the GitLab Release.

## Release checklist

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
4. Create and push a tag like `v0.4.7`.
5. Let GitHub Actions build the installers, or use local scripts if GitHub-hosted runners are unavailable.

## Troubleshooting

### First model load is slow

This is expected. The app downloads `k2-fsa/OmniVoice` to `~/.voiceclone/models` the first time the user loads the model.

### macOS app is blocked by Gatekeeper

Local builds are ad-hoc signed for internal testing. For public distribution, sign and notarize with an Apple Developer ID.

### Windows build cannot find NSIS

Install NSIS and ensure `makensis` is on `PATH`:

```powershell
choco install -y nsis
```

### GitHub build does not start

Check whether the workflow was manually disabled or whether GitHub billing/spending limits prevent hosted runners from starting.
