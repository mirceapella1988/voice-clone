import os
import sys
from huggingface_hub import hf_hub_download

MODEL_REPO = "zxczx221312/OmniVoice-WebGPU-ONNX"
files = [
    "omnivoice-config.json",
    "tokenizer.json",
    "omnivoice-decoder-webgpu.onnx",
    "omnivoice-encoder-fixed.onnx",
    "omnivoice-main-kv-fp16-b1.onnx",
    "omnivoice-main-kv-fp16-b1.onnx_data"
]

def main():
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading models to: {local_dir}")
    for f in files:
        target_path = os.path.join(local_dir, f)
        if os.path.exists(target_path):
            print(f"File {f} already exists, skipping.")
            continue
            
        print(f"Downloading {f} from {MODEL_REPO}...")
        try:
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=f,
                local_dir=local_dir,
                local_files_only=False
            )
            print(f"Successfully downloaded {f}")
        except Exception as e:
            print(f"Error downloading {f}: {e}")
            sys.exit(1)
            
    print("All models downloaded successfully.")

if __name__ == "__main__":
    main()
