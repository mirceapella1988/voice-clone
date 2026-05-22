import base64
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import omnivoice_runtime
from omnivoice_runtime import (
    MODEL_REPO,
    OmniVoiceRuntime,
    OmniVoiceValidationError,
    default_model_cache_dir,
    get_available_hardware_devices,
    normalize_instruct,
    normalize_language_id,
    to_mono_audio,
)
from sidecar import SidecarApp
from sidecar import read_next_command


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)


class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeMps:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeTorch:
    float16 = "float16"
    float32 = "float32"

    def __init__(self, cuda=False, mps=False):
        self.cuda = FakeCuda(cuda)
        self.backends = SimpleNamespace(mps=FakeMps(mps))
        self.from_numpy_calls = []

    def from_numpy(self, array):
        tensor = FakeTensor(array)
        self.from_numpy_calls.append(tensor)
        return tensor


class FakeGenerationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeRefTokens:
    def __init__(self, count):
        self.count = count

    def size(self, dim):
        self.last_dim = dim
        return self.count


class FakePrompt:
    def __init__(self):
        self.ref_audio_tokens = FakeRefTokens(25)
        self.ref_rms = 0.08
        self.ref_text = "reference text."


class FakeModel:
    def __init__(self):
        self.sampling_rate = 24000
        self.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(frame_rate=75))
        self.prompt = FakePrompt()
        self.prompt_kwargs = None
        self.generate_kwargs = None

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_kwargs = kwargs
        return self.prompt

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [np.array([0.0, 0.25, -0.25], dtype=np.float32)]


class FakeOmniVoice:
    load_kwargs = None
    model = None

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.load_kwargs = {"model_path": model_path, **kwargs}
        cls.model = FakeModel()
        return cls.model


class RuntimeLoadTests(unittest.TestCase):
    def setUp(self):
        self.original_voiceclone_models = os.environ.get("VOICECLONE_MODELS")
        self.original_app_model_dir = os.environ.get("APP_MODEL_DIR")
        self.original_omnivoice_model = os.environ.get("OMNIVOICE_MODEL")

    def tearDown(self):
        if self.original_voiceclone_models is None:
            os.environ.pop("VOICECLONE_MODELS", None)
        else:
            os.environ["VOICECLONE_MODELS"] = self.original_voiceclone_models

        if self.original_app_model_dir is None:
            os.environ.pop("APP_MODEL_DIR", None)
        else:
            os.environ["APP_MODEL_DIR"] = self.original_app_model_dir

        if self.original_omnivoice_model is None:
            os.environ.pop("OMNIVOICE_MODEL", None)
        else:
            os.environ["OMNIVOICE_MODEL"] = self.original_omnivoice_model

    def test_load_downloads_public_snapshot_without_token_and_uses_cuda_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_MODEL_DIR"] = tmp
            fake_torch = FakeTorch(cuda=True)
            calls = []

            def fake_snapshot_download(**kwargs):
                calls.append(kwargs)
                return str(Path(tmp) / "snapshot")

            runtime = OmniVoiceRuntime()
            runtime._load_modules = lambda: (fake_torch, FakeOmniVoice, FakeGenerationConfig)

            with patch.object(omnivoice_runtime, "snapshot_download", side_effect=fake_snapshot_download):
                runtime.load(device="auto")

            self.assertEqual(calls[0]["repo_id"], MODEL_REPO)
            self.assertEqual(calls[0]["cache_dir"], tmp)
            self.assertFalse(calls[0]["token"])
            self.assertEqual(runtime.actual_device, "cuda")
            self.assertEqual(FakeOmniVoice.load_kwargs["device_map"], "cuda")
            self.assertEqual(FakeOmniVoice.load_kwargs["dtype"], fake_torch.float16)
            self.assertFalse(FakeOmniVoice.load_kwargs["load_asr"])

    def test_load_uses_float32_for_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_MODEL_DIR"] = tmp
            fake_torch = FakeTorch(cuda=False, mps=False)
            runtime = OmniVoiceRuntime()
            runtime._load_modules = lambda: (fake_torch, FakeOmniVoice, FakeGenerationConfig)

            with patch.object(omnivoice_runtime, "snapshot_download", return_value=str(Path(tmp) / "snapshot")):
                runtime.load(device="cpu")

            self.assertEqual(runtime.actual_device, "cpu")
            self.assertEqual(FakeOmniVoice.load_kwargs["dtype"], fake_torch.float32)

    def test_default_model_cache_dir_uses_voiceclone_home_directory(self):
        with patch.object(omnivoice_runtime.Path, "home", return_value=Path("/Users/example")):
            self.assertEqual(
                default_model_cache_dir(),
                str(Path("/Users/example/.voiceclone/models")),
            )

    def test_default_model_cache_dir_uses_windows_app_data(self):
        with patch.object(omnivoice_runtime.os, "name", "nt"), patch.dict(
            os.environ,
            {"LOCALAPPDATA": r"C:\Users\example\AppData\Local"},
            clear=False,
        ):
            self.assertEqual(
                default_model_cache_dir(),
                r"C:\Users\example\AppData\Local\Voice Clone\models",
            )

    def test_voiceclone_models_env_takes_precedence_over_legacy_app_model_dir(self):
        os.environ["VOICECLONE_MODELS"] = "/tmp/voiceclone-models"
        os.environ["APP_MODEL_DIR"] = "/tmp/legacy-models"

        runtime = OmniVoiceRuntime()

        self.assertEqual(runtime._get_model_cache_dir(), "/tmp/voiceclone-models")


class RuntimeGenerateTests(unittest.TestCase):
    def make_loaded_runtime(self):
        runtime = OmniVoiceRuntime()
        runtime.model = FakeModel()
        runtime.model_path = "/tmp/omnivoice"
        runtime.device = "auto"
        runtime.actual_device = "cuda"
        runtime.sampling_rate = 24000
        runtime._torch = FakeTorch(cuda=True)
        runtime._generation_config_cls = FakeGenerationConfig
        return runtime

    def test_generate_matches_space_voice_clone_prompt_flow(self):
        runtime = self.make_loaded_runtime()

        with patch.object(
            omnivoice_runtime,
            "_load_omnivoice_instruct_resolver",
            return_value=lambda instruct, use_zh=False: "male, whisper",
        ):
            audio, sample_rate = runtime.generate(
                text="Xin chao",
                ref_audio=np.zeros(48000, dtype=np.float32),
                ref_text="Reference text",
                instruct="Male, whisper",
                params={
                    "ref_sample_rate": 48000,
                    "language_id": "vi",
                    "num_step": 24,
                    "guidance_scale": 1.5,
                    "speed": 1.2,
                    "duration": None,
                    "denoise": True,
                    "preprocess_prompt": True,
                    "postprocess_output": False,
                },
            )

        self.assertEqual(sample_rate, 24000)
        np.testing.assert_allclose(audio, np.array([0.0, 0.25, -0.25], dtype=np.float32))

        prompt_kwargs = runtime.model.prompt_kwargs
        self.assertEqual(prompt_kwargs["ref_text"], "Reference text")
        self.assertTrue(prompt_kwargs["preprocess_prompt"])
        waveform, prompt_sample_rate = prompt_kwargs["ref_audio"]
        self.assertIsInstance(waveform, FakeTensor)
        self.assertEqual(prompt_sample_rate, 48000)
        self.assertEqual(waveform.array.dtype, np.float32)

        generate_kwargs = runtime.model.generate_kwargs
        self.assertEqual(generate_kwargs["text"], "Xin chao")
        self.assertEqual(generate_kwargs["language"], "vi")
        self.assertEqual(generate_kwargs["voice_clone_prompt"], runtime.model.prompt)
        self.assertEqual(generate_kwargs["instruct"], "male, whisper")
        self.assertEqual(generate_kwargs["speed"], 1.2)
        self.assertNotIn("duration", generate_kwargs)

        gen_config = generate_kwargs["generation_config"]
        self.assertEqual(gen_config.num_step, 24)
        self.assertEqual(gen_config.guidance_scale, 1.5)
        self.assertTrue(gen_config.denoise)
        self.assertTrue(gen_config.preprocess_prompt)
        self.assertFalse(gen_config.postprocess_output)

        self.assertEqual(runtime.last_diagnostics["model_repo"], MODEL_REPO)
        self.assertEqual(runtime.last_diagnostics["reference_token_count"], 25)
        self.assertAlmostEqual(runtime.last_diagnostics["reference_processed_duration_seconds"], 25 / 75)

    def test_generate_sends_duration_when_positive(self):
        runtime = self.make_loaded_runtime()

        runtime.generate(
            text="hello",
            ref_audio=np.zeros(24000, dtype=np.float32),
            ref_text="hello",
            params={"duration": 2.5, "speed": 0.8},
        )

        self.assertEqual(runtime.model.generate_kwargs["duration"], 2.5)
        self.assertEqual(runtime.model.generate_kwargs["speed"], 0.8)

    def test_generate_normalizes_language_labels_to_omnivoice_id(self):
        runtime = self.make_loaded_runtime()

        runtime.generate(
            text="Xin chao",
            ref_audio=np.zeros(24000, dtype=np.float32),
            ref_text="hello",
            params={"language_id": "Vietnamese (vi)"},
        )

        self.assertEqual(runtime.model.generate_kwargs["language"], "vi")
        self.assertEqual(runtime.last_diagnostics["raw_language_id"], "Vietnamese (vi)")
        self.assertEqual(runtime.last_diagnostics["language_id"], "vi")

    def test_generate_rejects_upstream_instruct_validation_before_model_generate(self):
        runtime = self.make_loaded_runtime()

        def fake_resolver(instruct, use_zh=False):
            raise ValueError("Unsupported instruct items found in male, lightly")

        with patch.object(omnivoice_runtime, "_load_omnivoice_instruct_resolver", return_value=fake_resolver):
            with self.assertRaises(OmniVoiceValidationError) as context:
                runtime.generate(
                    text="Xin chao",
                    ref_audio=np.zeros(24000, dtype=np.float32),
                    ref_text="hello",
                    instruct="male, lightly",
                )

        self.assertIn("lightly", str(context.exception))
        self.assertIsNone(runtime.model.generate_kwargs)

    def test_generate_respects_cancel_before_prompt_creation(self):
        runtime = self.make_loaded_runtime()

        with self.assertRaises(InterruptedError):
            runtime.generate(
                text="hello",
                ref_audio=np.zeros(24000, dtype=np.float32),
                ref_text="hello",
                cancel_flag=lambda: True,
            )

        self.assertIsNone(runtime.model.prompt_kwargs)


class HardwareDetectionTests(unittest.TestCase):
    def test_detects_cuda_before_mps_for_auto(self):
        hardware = get_available_hardware_devices(lambda: FakeTorch(cuda=True, mps=True))

        self.assertEqual(hardware["auto_detect"], "cuda")
        self.assertEqual([device["id"] for device in hardware["devices"]], ["cpu", "cuda", "mps"])

    def test_missing_torch_reports_cpu_only(self):
        def missing_torch():
            raise ImportError("torch")

        hardware = get_available_hardware_devices(missing_torch)

        self.assertEqual(hardware["auto_detect"], "cpu")
        self.assertEqual([device["id"] for device in hardware["devices"]], ["cpu"])
        self.assertFalse(hardware["diagnostics"]["torch_available"])


class LanguageNormalizationTests(unittest.TestCase):
    def test_normalizes_auto_codes_and_display_labels(self):
        self.assertIsNone(normalize_language_id("Auto"))
        self.assertEqual(normalize_language_id("VI"), "vi")
        self.assertEqual(normalize_language_id("Vietnamese (vi)"), "vi")
        self.assertEqual(normalize_language_id("Vietnamese"), "Vietnamese")


class InstructNormalizationTests(unittest.TestCase):
    def test_normalizes_with_omnivoice_resolver(self):
        calls = []

        def fake_resolver(instruct, use_zh=False):
            calls.append((instruct, use_zh))
            return "male, low pitch"

        self.assertEqual(
            normalize_instruct("Male, low pitch", target_text="Xin chao", resolver=fake_resolver),
            "male, low pitch",
        )
        self.assertEqual(calls, [("Male, low pitch", False)])

    def test_uses_chinese_resolution_when_target_text_is_chinese(self):
        calls = []

        def fake_resolver(instruct, use_zh=False):
            calls.append((instruct, use_zh))
            return "男，低音调"

        self.assertEqual(
            normalize_instruct("male, low pitch", target_text="你好", resolver=fake_resolver),
            "男，低音调",
        )
        self.assertEqual(calls, [("male, low pitch", True)])

    def test_wraps_upstream_instruct_validation_errors(self):
        def fake_resolver(instruct, use_zh=False):
            raise ValueError("Unsupported instruct items found in male, lightly")

        with self.assertRaises(OmniVoiceValidationError) as context:
            normalize_instruct("male, lightly", resolver=fake_resolver)

        self.assertIn("lightly", str(context.exception))


class AudioHelperTests(unittest.TestCase):
    def test_to_mono_audio_averages_stereo_channels(self):
        audio = np.array(
            [
                [1.0, 0.5, -0.5, -1.0],
                [-1.0, -0.5, 0.5, 1.0],
            ],
            dtype=np.float32,
        )

        mono = to_mono_audio(audio)

        np.testing.assert_allclose(mono, np.zeros(4, dtype=np.float32))


class SidecarReferenceAudioTests(unittest.TestCase):
    def test_decode_reference_audio_reports_wav_metadata_and_mixes_channels(self):
        pcm = np.array([32767, -32767, 16384, 16384], dtype=np.int16)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48000)
            wav_file.writeframes(pcm.tobytes())

        encoded = base64.b64encode(wav_io.getvalue()).decode("ascii")
        app = SidecarApp()

        audio, sample_rate, metadata = app.decode_reference_audio(encoded, {})

        self.assertEqual(sample_rate, 48000)
        self.assertEqual(metadata["input_wav_channels"], 2)
        self.assertEqual(metadata["input_wav_sample_width"], 2)
        self.assertEqual(len(audio), 2)
        self.assertAlmostEqual(float(audio[0]), 0.0, places=5)
        self.assertAlmostEqual(float(audio[1]), 0.5, places=5)

    def test_handle_command_ignores_blank_stdin_lines(self):
        app = SidecarApp()
        output = io.StringIO()

        with patch("sys.stdout", output):
            app.handle_command("   \n")

        self.assertEqual(output.getvalue(), "")

    def test_handle_command_reports_invalid_json_cleanly(self):
        app = SidecarApp()
        output = io.StringIO()

        with patch("sys.stdout", output):
            app.handle_command("not-json\n")

        message = json.loads(output.getvalue())
        self.assertEqual(message["type"], "error")
        self.assertIn("Invalid JSON command:", message["message"])
        self.assertNotIn("command command", message["message"])

    def test_read_next_command_reads_content_length_frame(self):
        payload = json.dumps({
            "command": "generate",
            "text": "Xin chào",
            "ref_audio": "A" * 100_000,
        })
        frame = f"Content-Length: {len(payload.encode('utf-8'))}\n\n".encode("utf-8")
        framed_input = io.BytesIO(frame + payload.encode("utf-8") + b"\n")

        self.assertEqual(read_next_command(framed_input), payload)
        self.assertIsNone(read_next_command(framed_input))

    def test_read_next_command_keeps_legacy_line_json_fallback(self):
        payload = '{"command":"get_devices"}\n'

        self.assertEqual(read_next_command(io.BytesIO(payload.encode("utf-8"))), payload)

    def test_progress_heartbeat_emits_elapsed_generation_status(self):
        app = SidecarApp()
        events = []
        stop_event = threading.Event()
        app.on_progress = lambda status, progress: events.append((status, progress))

        thread = app.start_progress_heartbeat(
            stop_event,
            started_at=time.time() - 10,
            interval=0.01,
        )
        time.sleep(0.03)
        stop_event.set()
        thread.join(timeout=1.0)

        self.assertTrue(events)
        self.assertTrue(events[0][0].startswith("Generating speech..."))
        self.assertGreaterEqual(events[0][1], 0.25)
        self.assertLessEqual(events[0][1], 0.95)


if __name__ == "__main__":
    unittest.main()
