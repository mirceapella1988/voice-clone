import importlib
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from huggingface_hub import snapshot_download

MODEL_REPO = "k2-fsa/OmniVoice"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_FRAME_RATE = 75


class OmniVoiceDependencyError(RuntimeError):
    pass


class OmniVoiceValidationError(ValueError):
    pass


_ZH_RE = re.compile(r"[\u4e00-\u9fff]")


def _load_omnivoice_instruct_resolver():
    try:
        omnivoice_model_module = importlib.import_module("omnivoice.models.omnivoice")
    except ImportError:
        return None
    return getattr(omnivoice_model_module, "_resolve_instruct", None)


def normalize_instruct(instruct, target_text=None, resolver=None):
    raw = str(instruct or "").strip()
    if not raw:
        return None

    resolver = resolver or _load_omnivoice_instruct_resolver()
    if resolver is None:
        return raw

    try:
        return resolver(raw, use_zh=bool(target_text and _ZH_RE.search(target_text)))
    except ValueError as exc:
        raise OmniVoiceValidationError(str(exc)) from exc


def compute_rms(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def to_mono_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0 if audio.shape[0] <= audio.shape[-1] else 1)
    return audio.astype(np.float32, copy=False)


def _get_bool(params, key, default):
    value = params.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _get_float(params, key, default=None):
    value = params.get(key, default)
    if value is None or value == "":
        return default
    return float(value)


def _get_int(params, key, default):
    value = params.get(key, default)
    if value is None or value == "":
        return int(default)
    return int(value)


def default_model_cache_dir():
    return str(Path.home() / ".voiceclone" / "models")


def normalize_language_id(language_id):
    value = str(language_id or "").strip()
    if not value or value.lower() == "auto":
        return None

    if value.endswith(")") and "(" in value:
        value = value.rsplit("(", 1)[1].rstrip(")").strip()

    if len(value) <= 3 and value.replace("-", "").isalpha():
        return value.lower()

    return value


def import_torch():
    return importlib.import_module("torch")


def is_mps_available(torch_module):
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    return bool(mps and mps.is_available())


def _check_nvidia_smi():
    """Check if nvidia-smi is available and return basic GPU info."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            gpu_name = parts[0] if parts else "Unknown"
            driver_ver = parts[1] if len(parts) > 1 else "Unknown"
            return {"available": True, "gpu": gpu_name, "driver": driver_ver}
        return {"available": True, "gpu": "Unknown", "driver": "Unknown"}
    except Exception:
        return {"available": False}


def get_available_hardware_devices(torch_loader=import_torch):
    devices = [{"id": "cpu", "name": "CPU (Mac dinh)"}]
    auto_detect = "cpu"
    diagnostics = {"torch_available": False}

    try:
        torch_module = torch_loader()
    except Exception as e:
        import traceback
        import sys
        print(f"Error loading torch: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        diagnostics["error"] = str(e)
        return {
            "devices": devices,
            "auto_detect": auto_detect,
            "diagnostics": diagnostics,
        }

    diagnostics["torch_available"] = True

    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is not None and cuda_module.is_available():
        devices.append({"id": "cuda", "name": "NVIDIA GPU (CUDA)"})
        auto_detect = "cuda"
    elif cuda_module is not None:
        # CUDA module exists but is not available; gather extra diagnostics
        nsmi = _check_nvidia_smi()
        diagnostics["cuda_available_in_torch"] = False
        diagnostics["cuda_module_present"] = True
        diagnostics["nvidia_smi"] = nsmi
        if nsmi.get("available"):
            diagnostics["gpu_hint"] = (
                "nvidia-smi reports a GPU, but torch.cuda.is_available() is False. "
                "This usually means the PyTorch CUDA wheel does not match the CUDA driver version, "
                "or required CUDA runtime DLLs are missing."
            )
        else:
            diagnostics["gpu_hint"] = "nvidia-smi not found; no NVIDIA GPU detected or drivers not installed."
    else:
        diagnostics["cuda_module_present"] = False

    if is_mps_available(torch_module):
        devices.append({"id": "mps", "name": "Apple Silicon (MPS)"})
        if auto_detect == "cpu":
            auto_detect = "mps"

    return {
        "devices": devices,
        "auto_detect": auto_detect,
        "diagnostics": diagnostics,
    }


class OmniVoiceRuntime:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.model = None
        self.model_path = None
        self.device = "auto"
        self.actual_device = "cpu"
        self.sampling_rate = DEFAULT_SAMPLE_RATE
        self.last_diagnostics = {}
        self._torch = None
        self._generation_config_cls = None

    def emit(self, status, pct):
        if self.progress_callback:
            self.progress_callback(status, min(1.0, max(0.0, pct / 100.0)))

    def _load_modules(self):
        try:
            torch_module = import_torch()
            omnivoice_module = importlib.import_module("omnivoice")
        except ImportError as exc:
            import traceback
            import sys
            print(f"Error loading modules: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise OmniVoiceDependencyError(
                "Missing OmniVoice PyTorch dependencies. Run `pip install -r src-python/requirements.txt`."
            ) from exc

        return (
            torch_module,
            omnivoice_module.OmniVoice,
            omnivoice_module.OmniVoiceGenerationConfig,
        )

    def _resolve_device(self, requested_device, torch_module):
        requested = (requested_device or "auto").lower()

        cuda_available = bool(
            getattr(torch_module, "cuda", None) and torch_module.cuda.is_available()
        )
        mps_available = is_mps_available(torch_module)

        if requested == "cuda":
            if cuda_available:
                return "cuda"
            self.emit("CUDA unavailable, falling back to CPU", 20)
            return "cpu"

        if requested in ("mps", "coreml"):
            if mps_available:
                return "mps"
            self.emit("MPS unavailable, falling back to CPU", 20)
            return "cpu"

        if requested == "cpu":
            return "cpu"

        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    def _dtype_for_device(self, device, torch_module):
        return torch_module.float32 if device == "cpu" else torch_module.float16

    def _get_model_cache_dir(self):
        model_dir = os.environ.get("APP_MODEL_DIR")
        if model_dir:
            return model_dir
        return default_model_cache_dir()

    def _resolve_model_path(self):
        checkpoint = os.environ.get("OMNIVOICE_MODEL", MODEL_REPO)
        if os.path.isdir(checkpoint):
            return checkpoint

        cache_dir = self._get_model_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        self.emit("Loading OmniVoice model from Hugging Face...", 25)
        return snapshot_download(
            repo_id=checkpoint,
            cache_dir=cache_dir,
            token=False,
            local_files_only=False,
        )

    def load(self, device="auto"):
        self.device = device or "auto"
        self.emit("Loading OmniVoice PyTorch dependencies...", 5)

        torch_module, OmniVoice, OmniVoiceGenerationConfig = self._load_modules()
        self._torch = torch_module
        self._generation_config_cls = OmniVoiceGenerationConfig
        self.actual_device = self._resolve_device(self.device, torch_module)
        dtype = self._dtype_for_device(self.actual_device, torch_module)

        self.model_path = self._resolve_model_path()
        self.emit(f"Loading OmniVoice on {self.actual_device.upper()}...", 60)
        self.model = OmniVoice.from_pretrained(
            self.model_path,
            device_map=self.actual_device,
            dtype=dtype,
            load_asr=False,
        )
        self.sampling_rate = int(getattr(self.model, "sampling_rate", DEFAULT_SAMPLE_RATE))
        self.emit("OmniVoice ready", 100)
        return self

    def _ensure_loaded(self):
        if self.model is None:
            self.load(self.device)
        return self.model

    def _make_generation_config(self, params):
        config_cls = self._generation_config_cls
        if config_cls is None:
            config_cls = SimpleNamespace

        return config_cls(
            num_step=_get_int(params, "num_step", params.get("infer_steps", 32)),
            guidance_scale=_get_float(
                params,
                "guidance_scale",
                params.get("cfg_strength", 2.0),
            ),
            denoise=_get_bool(params, "denoise", True),
            preprocess_prompt=_get_bool(params, "preprocess_prompt", True),
            postprocess_output=_get_bool(params, "postprocess_output", True),
            audio_chunk_duration=_get_float(params, "audio_chunk_duration", 15.0),
            audio_chunk_threshold=_get_float(params, "audio_chunk_threshold", 30.0),
        )

    def _get_frame_rate(self):
        audio_tokenizer = getattr(self.model, "audio_tokenizer", None)
        config = getattr(audio_tokenizer, "config", None)
        return int(getattr(config, "frame_rate", DEFAULT_FRAME_RATE))

    def generate(self, text, ref_audio=None, ref_text=None, instruct=None, params=None, cancel_flag=None):
        params = params or {}
        model = self._ensure_loaded()
        torch_module = self._torch
        if torch_module is None:
            raise OmniVoiceDependencyError("Torch runtime is not initialized.")

        target_text = (text or "").strip()
        if not target_text:
            raise ValueError("Text to synthesize is required.")

        language_id = params.get("language_id")
        language = normalize_language_id(language_id)
        ref_sample_rate = int(params.get("ref_sample_rate") or self.sampling_rate)
        preprocess_prompt = _get_bool(params, "preprocess_prompt", True)
        generation_config = self._make_generation_config(params)
        speed = _get_float(params, "speed", None)
        duration = _get_float(params, "duration", None)

        self.last_diagnostics = {
            "model_repo": MODEL_REPO,
            "model_path": self.model_path,
            "requested_device": self.device,
            "actual_device": self.actual_device,
            "sample_rate": self.sampling_rate,
            "num_step": int(getattr(generation_config, "num_step", 32)),
            "guidance_scale": float(getattr(generation_config, "guidance_scale", 2.0)),
            "denoise": bool(getattr(generation_config, "denoise", True)),
            "preprocess_prompt": preprocess_prompt,
            "postprocess_output": bool(getattr(generation_config, "postprocess_output", True)),
            "raw_language_id": language_id,
            "language_id": language,
            "target_text_chars": len(target_text),
            "speed": speed,
            "duration_override_seconds": duration,
            "ref_text_chars": len((ref_text or "").strip()),
            "instruct_chars": len((instruct or "").strip()),
            "normalized_instruct": None,
        }
        normalized_instruct = normalize_instruct(instruct, target_text=target_text)
        self.last_diagnostics["normalized_instruct"] = normalized_instruct

        voice_clone_prompt = None
        if ref_audio is not None:
            if cancel_flag and cancel_flag():
                raise InterruptedError("Stopped")

            reference = to_mono_audio(ref_audio)
            self.last_diagnostics.update({
                "reference_raw_samples": int(len(reference)),
                "reference_raw_sample_rate": ref_sample_rate,
                "reference_raw_duration_seconds": len(reference) / ref_sample_rate if ref_sample_rate > 0 else 0,
                "reference_raw_rms": compute_rms(reference),
                "reference_resample_method": "omnivoice",
            })

            self.emit("Preparing voice clone prompt...", 10)
            waveform = torch_module.from_numpy(reference.astype(np.float32, copy=False))
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=(waveform, ref_sample_rate),
                ref_text=(ref_text or None),
                preprocess_prompt=preprocess_prompt,
            )

            token_count = None
            ref_tokens = getattr(voice_clone_prompt, "ref_audio_tokens", None)
            if ref_tokens is not None and hasattr(ref_tokens, "size"):
                token_count = int(ref_tokens.size(-1))
            frame_rate = self._get_frame_rate()
            self.last_diagnostics.update({
                "reference_token_count": token_count,
                "reference_processed_sample_rate": self.sampling_rate,
                "reference_processed_duration_seconds": (
                    token_count / frame_rate if token_count is not None and frame_rate > 0 else None
                ),
                "reference_processed_rms": getattr(voice_clone_prompt, "ref_rms", None),
            })

        if cancel_flag and cancel_flag():
            raise InterruptedError("Stopped")

        generate_kwargs = {
            "text": target_text,
            "language": language,
            "generation_config": generation_config,
        }
        if voice_clone_prompt is not None:
            generate_kwargs["voice_clone_prompt"] = voice_clone_prompt
        if normalized_instruct:
            generate_kwargs["instruct"] = normalized_instruct
        if speed is not None and speed != 1.0:
            generate_kwargs["speed"] = speed
        if duration is not None and duration > 0:
            generate_kwargs["duration"] = duration

        self.emit("Generating speech...", 20)
        audio_items = model.generate(**generate_kwargs)

        if cancel_flag and cancel_flag():
            raise InterruptedError("Stopped")

        if not audio_items:
            raise RuntimeError("OmniVoice returned no audio.")

        audio = to_mono_audio(audio_items[0])
        self.last_diagnostics.update({
            "output_samples": int(len(audio)),
            "output_duration_seconds": len(audio) / self.sampling_rate if self.sampling_rate > 0 else 0,
            "chunk_count": None,
        })
        self.emit("Done", 100)
        return audio.astype(np.float32, copy=False), self.sampling_rate
