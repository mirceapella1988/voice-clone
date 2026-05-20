import sys
import unittest
import base64
import io
import os
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnivoice_runtime import OmniVoiceRuntime, compute_scores, resample_audio_mono
from sidecar import SidecarApp


class AudioResampleTests(unittest.TestCase):
    def test_resample_audio_mono_keeps_matching_sample_rate(self):
        audio = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)

        resampled = resample_audio_mono(audio, 24000, 24000)

        np.testing.assert_allclose(resampled, audio)
        self.assertEqual(resampled.dtype, np.float32)

    def test_resample_audio_mono_downsamples_to_target_length(self):
        audio = np.linspace(-1.0, 1.0, 48000, dtype=np.float32)

        resampled = resample_audio_mono(audio, 48000, 24000)

        self.assertEqual(len(resampled), 24000)
        self.assertEqual(resampled.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(resampled)))
        self.assertLessEqual(float(np.max(np.abs(resampled))), 1.1)

    def test_resample_audio_mono_averages_stereo_channels(self):
        audio = np.array(
            [
                [1.0, 0.5, -0.5, -1.0],
                [-1.0, -0.5, 0.5, 1.0],
            ],
            dtype=np.float32,
        )

        resampled = resample_audio_mono(audio, 24000, 24000)

        np.testing.assert_allclose(resampled, np.zeros(4, dtype=np.float32))

    def test_runtime_resamples_reference_audio_before_encoding(self):
        runtime = OmniVoiceRuntime()
        runtime.config = {
            "audio_mask_id": 1024,
            "audio_vocab_size": 1025,
            "sampling_rate": 24000,
            "frame_rate": 75,
        }

        captured = {}
        runtime.load_encoder = lambda: None

        def fake_encode(audio):
            captured["encoded_len"] = len(audio)
            return np.zeros((8, 25), dtype=np.int64)

        runtime._encode_audio = fake_encode
        runtime._generate_token_sequence = lambda *args, **kwargs: {
            "target_ids": np.zeros(8, dtype=np.int32),
            "T_target": 1,
        }
        runtime._decode_audio = lambda target_ids, target_len: np.zeros(2400, dtype=np.float32)

        runtime.generate(
            text="hello",
            ref_audio=np.zeros(48000, dtype=np.float32),
            ref_text="hello",
            params={
                "ref_sample_rate": 48000,
                "preprocess_prompt": False,
                "postprocess_output": False,
            },
        )

        self.assertEqual(captured["encoded_len"], 24000)
        self.assertEqual(runtime.last_diagnostics["reference_resample_method"], "scipy.signal.resample_poly")
        self.assertEqual(runtime.last_diagnostics["reference_token_count"], 25)
        self.assertGreater(runtime.last_diagnostics["target_token_count"], 0)
        self.assertEqual(runtime.last_diagnostics["chunk_count"], 1)

    def test_compute_scores_matches_log_softmax_guidance(self):
        cond_logits = np.array([[[2.0, 0.0, -1.0]]], dtype=np.float32)
        uncond_logits = np.array([[[0.0, 2.0, -1.0]]], dtype=np.float32)
        masked = np.array([1], dtype=np.uint8)

        original_uniform = np.random.uniform
        np.random.uniform = lambda *_args, **_kwargs: np.exp(-1.0)
        try:
            pred_tokens, scores = compute_scores(
                cond_logits,
                uncond_logits,
                masked,
                guidance_scale=2.0,
                num_cb=1,
                T=1,
                vocab_size=3,
                mask_id=2,
            )
        finally:
            np.random.uniform = original_uniform

        def log_softmax(values):
            stable = values - np.max(values)
            return stable - np.log(np.sum(np.exp(stable)))

        c_log = log_softmax(cond_logits[0, 0])
        u_log = log_softmax(uncond_logits[0, 0])
        expected_log_probs = log_softmax(c_log + 2.0 * (c_log - u_log))
        expected_log_probs[2] = -np.inf

        self.assertEqual(int(pred_tokens[0]), int(np.argmax(expected_log_probs)))
        self.assertAlmostEqual(float(scores[0]), float(np.max(expected_log_probs) / 5.0), places=6)


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


class ModelFileLookupTests(unittest.TestCase):
    def setUp(self):
        self.original_app_model_dir = os.environ.get("APP_MODEL_DIR")
        self.original_resources_dir = os.environ.get("APP_RESOURCES_DIR")

    def tearDown(self):
        if self.original_app_model_dir is None:
            os.environ.pop("APP_MODEL_DIR", None)
        else:
            os.environ["APP_MODEL_DIR"] = self.original_app_model_dir

        if self.original_resources_dir is None:
            os.environ.pop("APP_RESOURCES_DIR", None)
        else:
            os.environ["APP_RESOURCES_DIR"] = self.original_resources_dir

    def test_get_model_file_reads_bundled_resource_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "src-python" / "models"
            model_dir.mkdir(parents=True)
            expected = model_dir / "unit-test-model.bin"
            expected.write_bytes(b"ok")
            os.environ.pop("APP_MODEL_DIR", None)
            os.environ["APP_RESOURCES_DIR"] = tmp

            runtime = OmniVoiceRuntime()

            self.assertEqual(Path(runtime._get_model_file("unit-test-model.bin")), expected)

    def test_get_model_file_prefers_writable_app_model_cache(self):
        with tempfile.TemporaryDirectory() as app_tmp, tempfile.TemporaryDirectory() as resource_tmp:
            app_model_dir = Path(app_tmp) / "models"
            resource_model_dir = Path(resource_tmp) / "src-python" / "models"
            app_model_dir.mkdir(parents=True)
            resource_model_dir.mkdir(parents=True)
            expected = app_model_dir / "unit-test-model.bin"
            expected.write_bytes(b"cache")
            (resource_model_dir / "unit-test-model.bin").write_bytes(b"resource")
            os.environ["APP_MODEL_DIR"] = str(app_model_dir)
            os.environ["APP_RESOURCES_DIR"] = resource_tmp

            runtime = OmniVoiceRuntime()

            self.assertEqual(Path(runtime._get_model_file("unit-test-model.bin")), expected)


if __name__ == "__main__":
    unittest.main()
