# Voice Clone

Tauri desktop app for zero-shot voice cloning with [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).

## Runtime

- The Python sidecar loads `k2-fsa/OmniVoice` through the official `omnivoice` PyTorch API.
- Device selection prefers CUDA, then Apple Silicon MPS, then CPU.
- Installers do not bundle OmniVoice model files. Models are downloaded from https://huggingface.co/k2-fsa/OmniVoice on first model load and cached in the user's app data directory.

## Development

```bash
npm install
python3 -m pip install -r src-python/requirements.txt
npm run build
```

For local packaged builds, use `build-local.sh` on macOS or `build-local.ps1` on Windows.
