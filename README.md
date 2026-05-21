# Voice Clone

Tauri desktop app for zero-shot voice cloning with [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).

## Runtime

- The Python sidecar loads `k2-fsa/OmniVoice` through the official `omnivoice` PyTorch API.
- Device selection prefers CUDA, then Apple Silicon MPS, then CPU.
- Model files are downloaded from Hugging Face on first load and cached locally instead of being bundled in the installer.

## Development

```bash
npm install
python3 -m pip install -r src-python/requirements.txt
npm run build
```

For local packaged builds, use `build-local.sh` on macOS or `build-local.ps1` on Windows.
