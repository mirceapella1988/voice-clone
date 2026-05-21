# Voice Clone

Tauri desktop app for zero-shot voice cloning with [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).

## Runtime

- The Python sidecar loads `k2-fsa/OmniVoice` through the official `omnivoice` PyTorch API.
- Device selection prefers CUDA, then Apple Silicon MPS, then CPU.
- Installers do not bundle OmniVoice model files. Models are downloaded from https://huggingface.co/k2-fsa/OmniVoice on first model load and cached in `~/.voiceclone/models`.

## Development

```bash
npm install
python3 -m pip install -r src-python/requirements.txt
npm run build
```

For local packaged builds, use `build-local.sh` on macOS or `build-local.ps1` on Windows.

## GitLab CI release builds

The GitLab pipeline is tag/manual only so normal pushes do not spend runner minutes.

- Default runner tags: `saas-macos-medium-m1`, `shared-windows`, `saas-linux-medium-amd64`.
- If your GitLab project shows different runner tags, override `MACOS_RUNNER_TAG`, `WINDOWS_RUNNER_TAG`, or `LINUX_RUNNER_TAG` in CI/CD variables.
- Tag a version such as `v0.4.2` to build macOS + Windows artifacts and create a GitLab Release.
- The GitLab release job uploads installers to the Generic Package Registry and links them from the release.
