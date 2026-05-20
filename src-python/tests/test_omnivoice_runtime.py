import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnivoice_runtime import OmniVoiceRuntime, resample_audio_mono


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
        self.assertAlmostEqual(float(resampled[0]), -1.0, places=5)
        self.assertAlmostEqual(float(resampled[-1]), 1.0, places=5)

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


if __name__ == "__main__":
    unittest.main()
