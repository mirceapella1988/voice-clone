import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "k2-fsa/OmniVoice"


def main():
    cache_dir = os.environ.get(
        "APP_MODEL_DIR",
        str(Path.home() / ".voiceclone" / "models"),
    )
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Prewarming OmniVoice model cache: {cache_dir}")
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

    print(f"OmniVoice model cached at: {model_path}")


if __name__ == "__main__":
    main()
