import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "k2-fsa/OmniVoice"


def default_model_cache_dir():
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return str(Path(root) / "Voice Clone" / "models")
    return str(Path.home() / "Library" / "Application Support" / "Voice Clone" / "models")


def main():
    cache_dir = (
        os.environ.get("VOICECLONE_MODELS")
        or os.environ.get("APP_MODEL_DIR")
        or default_model_cache_dir()
    )
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Prewarming OmniVoice folder cache: {cache_dir}")
    try:
        model_path = snapshot_download(
            repo_id=MODEL_REPO,
            cache_dir=cache_dir,
            token=False,
            local_files_only=False,
        )
    except Exception as exc:
        print(f"Error downloading {MODEL_REPO}: {exc}")
        sys.exit(1)

    print(f"OmniVoice cached at: {model_path}")


if __name__ == "__main__":
    main()
