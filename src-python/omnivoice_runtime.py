import os
import re
import json
import time
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL_REPO = "zxczx221312/OmniVoice-WebGPU-ONNX"
NUM_LAYERS = 28
NUM_CB = 8
KV_HEADS = 8
KV_DIM = 128
ENCODER_HOP_LENGTH = 960
T_SHIFT = 0.1
LAYER_PEN = 5.0
POS_TEMP = 5.0

WEIGHTS = {
    "cjk": 3.0, "hangul": 2.5, "kana": 2.2, "ethiopic": 3.0, "yi": 3.0,
    "indic": 1.8, "thai_lao": 1.5, "khmer_myanmar": 1.8,
    "arabic": 1.5, "hebrew": 1.5,
    "latin": 1.0, "cyrillic": 1.0, "greek": 1.0, "armenian": 1.0, "georgian": 1.0,
    "punctuation": 0.5, "space": 0.2, "digit": 3.5, "mark": 0.0, "default": 1.0,
}

RANGES = [
    [0x02AF, "latin"], [0x03FF, "greek"], [0x052F, "cyrillic"],
    [0x058F, "armenian"], [0x05FF, "hebrew"], [0x077F, "arabic"],
    [0x089F, "arabic"], [0x08FF, "arabic"], [0x097F, "indic"],
    [0x09FF, "indic"], [0x0A7F, "indic"], [0x0AFF, "indic"],
    [0x0B7F, "indic"], [0x0BFF, "indic"], [0x0C7F, "indic"],
    [0x0CFF, "indic"], [0x0D7F, "indic"], [0x0DFF, "indic"],
    [0x0EFF, "thai_lao"], [0x0FFF, "indic"], [0x109F, "khmer_myanmar"],
    [0x10FF, "georgian"], [0x11FF, "hangul"], [0x137F, "ethiopic"],
    [0x139F, "ethiopic"], [0x13FF, "default"], [0x167F, "default"],
    [0x169F, "default"], [0x16FF, "default"], [0x171F, "default"],
    [0x173F, "default"], [0x175F, "default"], [0x177F, "default"],
    [0x17FF, "khmer_myanmar"], [0x18AF, "default"], [0x18FF, "default"],
    [0x194F, "indic"], [0x19DF, "indic"], [0x19FF, "khmer_myanmar"],
    [0x1A1F, "indic"], [0x1AAF, "indic"], [0x1B7F, "indic"],
    [0x1BBF, "indic"], [0x1BFF, "indic"], [0x1C4F, "indic"],
    [0x1C7F, "indic"], [0x1C8F, "cyrillic"], [0x1CBF, "georgian"],
    [0x1CCF, "indic"], [0x1CFF, "indic"], [0x1D7F, "latin"],
    [0x1DBF, "latin"], [0x1DFF, "default"], [0x1EFF, "latin"],
    [0x309F, "kana"], [0x30FF, "kana"], [0x312F, "cjk"],
    [0x318F, "hangul"], [0xa9f, "cjk"], [0x9FFF, "cjk"], [0xA4CF, "yi"],
    [0xA4FF, "default"], [0xA63F, "default"], [0xA69F, "cyrillic"],
    [0xA6FF, "default"], [0xA7FF, "latin"], [0xA82F, "indic"],
    [0xA87F, "default"], [0xA8DF, "indic"], [0xA8FF, "indic"],
    [0xA92F, "indic"], [0xA95F, "indic"], [0xA97F, "hangul"],
    [0xA9DF, "indic"], [0xA9FF, "khmer_myanmar"], [0xAA5F, "indic"],
    [0xAA7F, "khmer_myanmar"], [0xAADF, "indic"], [0xAAFF, "indic"],
    [0xAB2F, "ethiopic"], [0xAB6F, "latin"], [0xABBF, "default"],
    [0xABFF, "indic"], [0xD7AF, "hangul"], [0xFAFF, "cjk"],
    [0xFDFF, "arabic"], [0xFE6F, "default"], [0xFEFF, "arabic"],
    [0xFFEF, "latin"],
]

BREAKPOINTS = [range_item[0] for range_item in RANGES]
SPLIT_PUNCTUATION = set(".,;:!?。，；：！？")
CLOSING_MARKS = set(["\"", "'", "）", "]", "》", ">", "」", "】"])
END_PUNCTUATION = set([
    ";", ":", ",", ".", "!", "?", "…", ")", "]", "}", "\"", "'",
    "；", "：", "，", "。", "！", "？", "、", "……", "）", "】",
])
NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]"
)
ABBREVIATIONS = set([
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.",
    "Capt.", "Gen.", "Sen.", "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.", "Corp.",
    "Inc.", "Ltd.", "Est.", "Dept.", "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.", "Jan.",
    "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "i.e.", "e.g.", "vs.",
    "Vs.", "Etc.", "approx.", "fig.", "def.",
])


def get_time_steps(num_step, t_shift):
    ts = np.zeros(num_step + 1, dtype=np.float32)
    for i in range(num_step + 1):
        t = i / num_step
        ts[i] = t_shift * t / (1.0 + (t_shift - 1.0) * t)
    return ts


def build_mask_schedule(total_masked, num_step, timesteps):
    schedule = np.zeros(num_step, dtype=np.int32)
    remaining = total_masked
    for step in range(num_step):
        if step == num_step - 1:
            count = remaining
        else:
            count = min(int(np.ceil(total_masked * (timesteps[step + 1] - timesteps[step]))), remaining)
        schedule[step] = count
        remaining -= count
    return schedule


def make_zero_kv(num_layers, num_heads, seq_len, head_dim, batch_size=1):
    zeros = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float16)
    keys = [zeros.copy() for _ in range(num_layers)]
    values = [zeros.copy() for _ in range(num_layers)]
    return keys, values


def compute_scores(cond_logits, uncond_logits, masked, guidance_scale, num_cb, T, vocab_size, mask_id):
    total = num_cb * T
    pred_tokens = np.zeros(total, dtype=np.int32)
    scores = np.full(total, -np.inf, dtype=np.float32)
    use_cfg = uncond_logits is not None and guidance_scale > 0
    s_plus_1 = 1.0 + guidance_scale

    for flat in range(total):
        if not masked[flat]:
            continue
        cb = flat // T
        t = flat % T

        c_l = cond_logits[cb, t, :]
        if use_cfg:
            u_l = uncond_logits[cb, t, :]
            r = s_plus_1 * c_l - guidance_scale * u_l
        else:
            r = c_l

        r_ex = r.copy()
        r_ex[mask_id] = -np.inf
        best_tok = np.argmax(r_ex)

        stab = np.max(r)
        exp_r = np.exp(r - stab)
        sum_exp = np.sum(exp_r)
        log_p = r[best_tok] - stab - np.log(sum_exp)

        pred_tokens[flat] = best_tok
        scores[flat] = log_p - cb * LAYER_PEN

    inv_pos_temp = 1.0 / POS_TEMP
    for flat in range(total):
        if masked[flat]:
            u = np.random.uniform(0.0, 1.0)
            g = -np.log(-np.log(u + 1e-20) + 1e-20)
            scores[flat] = scores[flat] * inv_pos_temp + g

    return pred_tokens, scores


def bisect_left(arr, value):
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def get_char_category(code):
    if ((0x0300 <= code <= 0x036F) or (0x0483 <= code <= 0x0489) or (0x0591 <= code <= 0x05BD) or
            (0x064B <= code <= 0x065F) or (0x0900 <= code <= 0x0903) or (0x093A <= code <= 0x094F) or
            (0x0951 <= code <= 0x0957) or (0x0962 <= code <= 0x0963) or (0xFE20 <= code <= 0xFE2F)):
        return "M"
    if ((0x0021 <= code <= 0x002F) or (0x003A <= code <= 0x0040) or (0x005B <= code <= 0x0060) or
            (0x007B <= code <= 0x007E) or (0x2000 <= code <= 0x206F) or (0x3000 <= code <= 0x303F)):
        return "P"
    if 0x0030 <= code <= 0x0039:
        return "N"
    if (0x00A0 <= code <= 0x00BF) or (0x2100 <= code <= 0x27FF):
        return "S"
    if code in (0x00A0, 0x2000, 0x2001, 0x2002, 0x2003, 0x3000):
        return "Z"
    return "L"


def get_char_weight(char):
    code = ord(char)
    if (65 <= code <= 90) or (97 <= code <= 122):
        return WEIGHTS["latin"]
    if code == 32:
        return WEIGHTS["space"]
    if code == 0x0640:
        return WEIGHTS["mark"]

    category = get_char_category(code)
    if category == "M":
        return WEIGHTS["mark"]
    if category in ("P", "S"):
        return WEIGHTS["punctuation"]
    if category == "Z":
        return WEIGHTS["space"]
    if category == "N":
        return WEIGHTS["digit"]

    idx = bisect_left(BREAKPOINTS, code)
    if idx < len(RANGES):
        return WEIGHTS.get(RANGES[idx][1], WEIGHTS["default"])
    if code > 0x20000:
        return WEIGHTS["cjk"]
    return WEIGHTS["default"]


def calculate_total_weight(text):
    return sum(get_char_weight(char) for char in text)


def estimate_duration(target_text, ref_text, ref_duration, low_threshold=50, boost_strength=3):
    if ref_duration <= 0 or not ref_text:
        return 0
    ref_weight = calculate_total_weight(ref_text)
    if ref_weight == 0:
        return 0
    speed_factor = ref_weight / ref_duration
    target_weight = calculate_total_weight(target_text)
    estimated = target_weight / speed_factor

    if low_threshold is not None and estimated < low_threshold:
        alpha = 1.0 / boost_strength
        return low_threshold * ((estimated / low_threshold) ** alpha)
    return estimated


def estimate_target_tokens(text, ref_text=None, num_ref_audio_tokens=None, speed=1.0):
    effective_ref_text = ref_text
    effective_ref_tokens = num_ref_audio_tokens

    if effective_ref_tokens is None or effective_ref_text is None or len(effective_ref_text) == 0:
        effective_ref_text = "Nice to meet you."
        effective_ref_tokens = 25

    est = estimate_duration(text, effective_ref_text, effective_ref_tokens)
    if speed > 0 and speed != 1.0:
        est = est / speed
    return max(1, int(round(est)))


def compute_rms(audio):
    if len(audio) == 0:
        return 0
    return np.sqrt(np.mean(audio ** 2))


def db_to_amplitude(db):
    return 10 ** (db / 20.0)


def trim_long_audio_mono(audio, sample_rate, max_duration=15, min_duration=3, trim_threshold=20):
    if len(audio) == 0:
        return audio
    duration = len(audio) / sample_rate
    if duration <= trim_threshold:
        return audio

    frame_samples = max(1, int(sample_rate * 0.01))
    threshold = db_to_amplitude(-40)
    max_frames = int((max_duration * sample_rate) / frame_samples)
    min_frames = int((min_duration * sample_rate) / frame_samples)

    best_split = 0
    silent_start = -1
    num_frames = int(np.ceil(len(audio) / frame_samples))
    for frame in range(num_frames):
        start = frame * frame_samples
        end = min(len(audio), start + frame_samples)
        peak = np.max(np.abs(audio[start:end]))
        silent = peak < threshold

        if silent and silent_start < 0:
            silent_start = frame
        if not silent and silent_start >= 0:
            if silent_start <= max_frames:
                best_split = silent_start
            silent_start = -1
            if frame > max_frames:
                break

    split_frame = best_split
    if split_frame < min_frames:
        split_frame = min(max_frames, int(np.ceil(len(audio) / frame_samples)))
    clip_samples = min(len(audio), split_frame * frame_samples)
    return audio[:clip_samples].copy()


def remove_silence_mono(audio, sample_rate, mid_sil_ms=300, lead_sil_ms=100, trail_sil_ms=300):
    if len(audio) == 0:
        return audio

    frame_samples = max(1, int(sample_rate * 0.01))
    silence_threshold = db_to_amplitude(-50)
    frames = []
    for start in range(0, len(audio), frame_samples):
        end = min(len(audio), start + frame_samples)
        peak = np.max(np.abs(audio[start:end]))
        frames.append({"start": start, "end": end, "silent": peak < silence_threshold})

    first_active = next((i for i, f in enumerate(frames) if not f["silent"]), -1)
    if first_active < 0:
        return np.zeros(0, dtype=np.float32)
    last_active = len(frames) - 1
    while last_active >= 0 and frames[last_active]["silent"]:
        last_active -= 1

    keep_lead_frames = int(lead_sil_ms / 10)
    keep_trail_frames = int(trail_sil_ms / 10)
    keep_mid_frames = int(mid_sil_ms / 10)

    trimmed_start_frame = max(0, first_active - keep_lead_frames)
    trimmed_end_frame = min(len(frames) - 1, last_active + keep_trail_frames)

    out = []
    index = trimmed_start_frame
    while index <= trimmed_end_frame:
        if not frames[index]["silent"]:
            out.append(audio[frames[index]["start"]:frames[index]["end"]])
            index += 1
            continue

        run_end = index
        while run_end <= trimmed_end_frame and frames[run_end]["silent"]:
            run_end += 1
        run_length = run_end - index
        keep_frames = min(run_length, keep_mid_frames)
        for i in range(keep_frames):
            out.append(audio[frames[index + i]["start"]:frames[index + i]["end"]])
        index = run_end

    if len(out) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(out)


def chunk_text_punctuation(text, chunk_len, min_chunk_len=3):
    sentences = []
    current_sentence = []
    tokens = list(text)

    for token in tokens:
        if len(current_sentence) == 0 and len(sentences) != 0 and (token in SPLIT_PUNCTUATION or token in CLOSING_MARKS):
            sentences[-1].append(token)
            continue

        current_sentence.append(token)
        if token not in SPLIT_PUNCTUATION:
            continue

        is_abbreviation = False
        if token == ".":
            temp = "".join(current_sentence).strip()
            if temp:
                words = temp.split()
                last_word = words[-1]
                if last_word in ABBREVIATIONS:
                    is_abbreviation = True

        if not is_abbreviation:
            sentences.append(current_sentence)
            current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    merged_chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_len:
            current_chunk.extend(sentence)
        else:
            if len(current_chunk) > 0:
                merged_chunks.append(current_chunk)
            current_chunk = sentence.copy()
    if current_chunk:
        merged_chunks.append(current_chunk)

    final_chunks = merged_chunks
    if min_chunk_len is not None:
        first_chunk_short = len(merged_chunks) > 0 and len(merged_chunks[0]) < min_chunk_len
        final_chunks = []
        for i, chunk in enumerate(merged_chunks):
            if i == 1 and first_chunk_short:
                final_chunks[-1].extend(chunk)
            elif len(chunk) >= min_chunk_len:
                final_chunks.append(chunk.copy())
            elif len(final_chunks) == 0:
                final_chunks.append(chunk.copy())
            else:
                final_chunks[-1].extend(chunk)

    return ["".join(chunk).strip() for chunk in final_chunks if "".join(chunk).strip()]


def cross_fade_chunks_mono(chunks, sample_rate, silence_duration=0.3):
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].copy()

    output = chunks[0].copy()
    total_samples = max(1, int(silence_duration * sample_rate))
    fade_samples = total_samples // 3
    silence_samples = fade_samples

    for i in range(1, len(chunks)):
        next_chunk = chunks[i].copy()
        fout = min(fade_samples, len(output))
        if fout > 0:
            fade_out = 1.0 - np.linspace(0.0, 1.0, fout)
            output[-fout:] *= fade_out

        fin = min(fade_samples, len(next_chunk))
        if fin > 0:
            fade_in = np.linspace(0.0, 1.0, fin)
            next_chunk[:fin] *= fade_in

        combined = np.zeros(len(output) + silence_samples + len(next_chunk), dtype=np.float32)
        combined[:len(output)] = output
        combined[len(output) + silence_samples:] = next_chunk
        output = combined

    return output


def add_punctuation(text):
    trimmed = text.strip()
    if not trimmed:
        return trimmed
    if trimmed[-1] in END_PUNCTUATION:
        return trimmed

    is_chinese = any(0x4e00 <= ord(char) <= 0x9fff for char in trimmed)
    return f"{trimmed}。" if is_chinese else f"{trimmed}."


def fade_and_pad_mono(audio, sample_rate, pad_duration=0.1, fade_duration=0.1):
    if len(audio) == 0:
        return audio

    processed = audio.copy()
    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)
    k = min(fade_samples, len(processed) // 2)

    if k > 0:
        fade_curve = np.linspace(0.0, 1.0, k)
        processed[:k] *= fade_curve
        processed[-k:] *= fade_curve[::-1]

    if pad_samples <= 0:
        return processed

    out = np.zeros(len(processed) + pad_samples * 2, dtype=np.float32)
    out[pad_samples:pad_samples + len(processed)] = processed
    return out


def post_process_audio(audio, sample_rate, ref_rms, postprocess_output):
    processed = audio.copy()
    if postprocess_output:
        processed = remove_silence_mono(processed, sample_rate, 500, 100, 100)

    if ref_rms is not None and 0 < ref_rms < 0.1:
        scale = ref_rms / 0.1
        processed *= scale
    elif ref_rms is None:
        peak = np.max(np.abs(processed))
        if peak > 1e-6:
            scale = 0.5 / peak
            processed *= scale

    return fade_and_pad_mono(processed, sample_rate)


def combine_text(text, ref_text=None):
    full_text = f"{ref_text.strip()} {text.strip()}" if ref_text else text.strip()
    full_text = full_text.replace("\r", "").replace("\n", "")
    full_text = full_text.replace("\uFF08", "(").replace("\uFF09", ")")
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"(?<=[\u4e00-\u9fff])\s+|\s+(?=[\u4e00-\u9fff])", "", full_text)
    return full_text


class OmniVoiceRuntime:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.config = None
        self.tokenizer = None
        self.sessions = {}
        self.static_tensor_cache = {}
        self.device = "auto"

    def emit(self, status, pct):
        if self.progress_callback:
            self.progress_callback(status, min(1.0, max(0.0, pct / 100.0)))

    def _get_ort_providers(self, device):
        available = ort.get_available_providers()
        providers = []
        if device == "cuda":
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
        elif device == "mps" or device == "coreml":
            if "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
            providers.append("CPUExecutionProvider")
        elif device == "cpu":
            providers.append("CPUExecutionProvider")
        else: # "auto"
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            # Tránh tự động dùng CoreML trên Mac trong chế độ "auto" vì graph bị chia cắt 113 phần làm chậm hiệu năng.
            # Người dùng vẫn có thể chọn thủ công trong dropdown nếu muốn thử nghiệm.
            providers.append("CPUExecutionProvider")
        return providers

    def _get_model_file(self, filename):
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        # Thử tải từ local cache trước (Offline mode)
        try:
            return hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                local_dir=local_dir,
                local_files_only=True
            )
        except Exception:
            # Nếu chưa có file cục bộ, kết nối internet tải về thư mục local_dir
            self.emit(f"Downloading {filename} (Online)...", 25)
            return hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                local_dir=local_dir,
                local_files_only=False
            )

    def load(self, device="auto"):
        self.device = device
        self.emit("Checking model files...", 5)
        
        # Tải/Đọc local config
        config_path = self._get_model_file("omnivoice-config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        tokenizer_path = self._get_model_file("tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.emit("Checking main model graph...", 20)
        main_model_path = self._get_model_file("omnivoice-main-kv-fp16-b1.onnx")
        # Đảm bảo tải cả file data đi kèm
        self._get_model_file("omnivoice-main-kv-fp16-b1.onnx_data")

        # Khởi tạo Inference Session với sess_options để ẩn cảnh báo spam
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # Chỉ hiện Error, ẩn các cảnh báo Warning/Info
        
        providers = self._get_ort_providers(device)
        self.sessions["main"] = ort.InferenceSession(main_model_path, sess_options=sess_opts, providers=providers)

        self.emit("Checking decoder model...", 70)
        decoder_path = self._get_model_file("omnivoice-decoder-webgpu.onnx")
        self.sessions["decoder"] = ort.InferenceSession(decoder_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

        self.emit("OmniVoice ready", 100)

    def load_encoder(self):
        if "encoder" in self.sessions:
            return
        self.emit("Checking encoder model...", 10)
        encoder_path = self._get_model_file("omnivoice-encoder-fixed.onnx")
        
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        
        self.sessions["encoder"] = ort.InferenceSession(encoder_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    def generate(self, text, ref_audio=None, ref_text=None, instruct=None, params=None, cancel_flag=None):
        if params is None:
            params = {}

        num_step = params.get("num_step", 32)
        guidance_scale = params.get("guidance_scale", 2.0)
        denoise = params.get("denoise", True)
        preprocess_prompt = params.get("preprocess_prompt", True)
        postprocess_output = params.get("postprocess_output", True)
        audio_chunk_duration = params.get("audio_chunk_duration", 15.0)
        audio_chunk_threshold = params.get("audio_chunk_threshold", 30.0)
        language_id = params.get("language_id", None)
        speed = params.get("speed", 1.0)
        duration = params.get("duration", None)

        lang = None if not language_id or language_id == "Auto" else language_id

        # 1. Preprocess prompt
        processed_ref_audio = np.array(ref_audio, dtype=np.float32) if ref_audio is not None else None
        processed_ref_text = (ref_text or "").strip()
        ref_rms = None

        if processed_ref_audio is not None:
            ref_rms = compute_rms(processed_ref_audio)
            if 0 < ref_rms < 0.1:
                scale = 0.1 / ref_rms
                processed_ref_audio *= scale

            if preprocess_prompt:
                if not processed_ref_text:
                    processed_ref_audio = trim_long_audio_mono(processed_ref_audio, self.config["sampling_rate"])
                processed_ref_audio = remove_silence_mono(processed_ref_audio, self.config["sampling_rate"], 200, 100, 200)

                if len(processed_ref_audio) == 0:
                    raise ValueError("Reference audio is empty after silence removal.")

            if preprocess_prompt and processed_ref_text:
                processed_ref_text = add_punctuation(processed_ref_text)

        # 2. Encode reference audio
        ref_codes = None
        if processed_ref_audio is not None:
            if cancel_flag and cancel_flag():
                raise InterruptedError("Stopped")
            self.emit("Encoding reference audio...", 5)
            self.load_encoder()
            ref_codes = self._encode_audio(processed_ref_audio)

        # 4. Generate tokens
        total_target = self._estimate_target_length(text, processed_ref_text, ref_codes, speed, duration)
        should_chunk = audio_chunk_duration > 0 and total_target > int(round(audio_chunk_threshold * self.config["frame_rate"]))

        if should_chunk:
            token_chunks = self._generate_chunked(
                text, processed_ref_text, ref_codes, instruct, lang,
                num_step, guidance_scale, denoise, speed, duration,
                total_target, audio_chunk_duration, cancel_flag
            )
        else:
            token_chunks = [self._generate_token_sequence(
                text, processed_ref_text, ref_codes, instruct, lang,
                num_step, guidance_scale, denoise, speed, duration,
                cancel_flag=cancel_flag
            )]

        # 5. Decode audio
        if cancel_flag and cancel_flag():
            raise InterruptedError("Stopped")

        if len(token_chunks) == 1:
            self.emit("Decoding audio...", 90)
            audio = self._decode_audio(token_chunks[0]["target_ids"], token_chunks[0]["T_target"])
        else:
            self.emit(f"Decoding {len(token_chunks)} chunks...", 88)
            chunk_audios = []
            for i, chunk in enumerate(token_chunks):
                if cancel_flag and cancel_flag():
                    raise InterruptedError("Stopped")
                self.emit(f"Decoding chunk {i+1}/{len(token_chunks)}...", 88 + int((i / len(token_chunks)) * 8))
                chunk_audios.append(self._decode_audio(chunk["target_ids"], chunk["T_target"]))
            audio = cross_fade_chunks_mono(chunk_audios, self.config["sampling_rate"])

        audio = post_process_audio(audio, self.config["sampling_rate"], ref_rms, postprocess_output)
        self.emit("Done", 100)
        return audio, self.config["sampling_rate"]

    def _estimate_target_length(self, text, ref_text, ref_codes, speed=1.0, duration=None):
        if duration is not None:
            return max(1, int(round(duration * self.config["frame_rate"])))
        has_ref = ref_codes is not None and ref_text and len(ref_text) > 0
        est_ref_text = ref_text if has_ref else "Nice to meet you."
        est_ref_tokens = ref_codes.shape[1] if has_ref else 25
        return estimate_target_tokens(text, est_ref_text, est_ref_tokens, speed)

    def _generate_chunked(self, text, ref_text, ref_codes, instruct, language_id,
                          num_step, guidance_scale, denoise, speed, duration,
                          total_target, chunk_duration, cancel_flag=None):
        avg_tokens_per_char = total_target / max(1, len(text))
        chunk_len = max(3, int(np.floor((chunk_duration * self.config["frame_rate"]) / max(avg_tokens_per_char, 1e-3))))
        chunks = chunk_text_punctuation(text, chunk_len, 3)

        if len(chunks) <= 1:
            return [self._generate_token_sequence(
                text, ref_text, ref_codes, instruct, language_id,
                num_step, guidance_scale, denoise, speed, duration,
                cancel_flag=cancel_flag
            )]

        self.emit(f"Chunking long synthesis into {len(chunks)} parts", 8)
        raw_estimate = self._estimate_target_length(text, ref_text, ref_codes, speed=1.0, duration=None)
        chunk_speed = raw_estimate / total_target if duration is not None and total_target > 0 else speed

        outputs = []
        if ref_codes is not None:
            for i, chunk in enumerate(chunks):
                if cancel_flag and cancel_flag():
                    raise InterruptedError("Stopped")
                outputs.append(self._generate_token_sequence(
                    chunk, ref_text, ref_codes, instruct, language_id,
                    num_step, guidance_scale, denoise, chunk_speed, None,
                    log_prefix=f"[chunk {i+1}/{len(chunks)}] ", cancel_flag=cancel_flag
                ))
            return outputs

        first_chunk = self._generate_token_sequence(
            chunks[0], None, None, instruct, language_id,
            num_step, guidance_scale, denoise, chunk_speed, None,
            log_prefix=f"[chunk 1/{len(chunks)}] ", cancel_flag=cancel_flag
        )
        outputs.append(first_chunk)

        # Chuyển tokens của chunk 1 làm reference codes cho chunk sau
        first_ref_codes = first_chunk["target_ids"].reshape(NUM_CB, first_chunk["T_target"])
        first_ref_text = chunks[0]

        for i in range(1, len(chunks)):
            if cancel_flag and cancel_flag():
                raise InterruptedError("Stopped")
            outputs.append(self._generate_token_sequence(
                chunks[i], first_ref_text, first_ref_codes, instruct, language_id,
                num_step, guidance_scale, denoise, chunk_speed, None,
                log_prefix=f"[chunk {i+1}/{len(chunks)}] ", cancel_flag=cancel_flag
            ))

        return outputs

    def _generate_token_sequence(self, text, ref_text, ref_codes, instruct, language_id,
                                 num_step, guidance_scale, denoise, speed, duration,
                                 log_prefix="", cancel_flag=None):
        mask_id = self.config["audio_mask_id"]
        vocab_size = self.config["audio_vocab_size"]

        style_str = ""
        if denoise and ref_codes is not None:
            style_str += "<|denoise|>"
        style_str += f"<|lang_start|>{language_id or 'None'}<|lang_end|>"
        style_str += f"<|instruct_start|>{(instruct or 'None').strip() or 'None'}<|instruct_end|>"

        full_text = combine_text(text, ref_text)
        text_str = f"<|text_start|>{full_text}<|text_end|>"

        style_ids = self.tokenizer.encode(style_str, add_special_tokens=False).ids
        # Tokenize text_str handles nonverbal patterns
        text_ids = self._tokenize_text_nonverbal(text_str)

        has_ref = ref_codes is not None and ref_text and len(ref_text) > 0
        est_ref_text = ref_text if has_ref else "Nice to meet you."
        est_ref_tokens = ref_codes.shape[1] if has_ref else 25

        T_target = max(1, int(round(duration * self.config["frame_rate"]))) if duration is not None else estimate_target_tokens(text, est_ref_text, est_ref_tokens, speed)

        self.emit(f"{log_prefix}Target estimate: {T_target} tokens (ref={est_ref_tokens})", 8)

        target_ids = np.full(NUM_CB * T_target, mask_id, dtype=np.int32)
        masked = np.ones(NUM_CB * T_target, dtype=np.uint8)
        remaining = NUM_CB * T_target
        schedule = build_mask_schedule(remaining, num_step, get_time_steps(num_step, T_SHIFT))

        # Build conditional inputs
        style_len = len(style_ids)
        text_len = len(text_ids)
        ref_len = ref_codes.shape[1] if ref_codes is not None else 0
        target_offset = style_len + text_len + ref_len
        total_len = target_offset + T_target

        cond_ids_buf = np.zeros((NUM_CB, total_len), dtype=np.int64)
        for c in range(NUM_CB):
            cond_ids_buf[c, :style_len] = style_ids
            cond_ids_buf[c, style_len:style_len+text_len] = text_ids
            if ref_codes is not None:
                cond_ids_buf[c, style_len+text_len:target_offset] = ref_codes[c, :]
            cond_ids_buf[c, target_offset:] = mask_id

        cond_audio_mask = np.zeros(total_len, dtype=bool)
        audio_start = style_len + text_len if ref_codes is not None else target_offset
        cond_audio_mask[audio_start:] = True

        uncond_ids_buf = np.zeros((NUM_CB, T_target), dtype=np.int64) if guidance_scale > 0 else None
        uncond_audio_mask = np.ones(T_target, dtype=bool) if guidance_scale > 0 else None

        cond_zero_kv = make_zero_kv(NUM_LAYERS, KV_HEADS, total_len, KV_DIM, 1)
        uncond_zero_kv = make_zero_kv(NUM_LAYERS, KV_HEADS, T_target, KV_DIM, 1) if guidance_scale > 0 else None

        for step in range(num_step):
            if cancel_flag and cancel_flag():
                raise InterruptedError("Stopped")
            if remaining <= 0:
                break

            k = schedule[step]
            if k <= 0:
                continue

            self.emit(f"{log_prefix}Diffusion step {step + 1}/{num_step}", int(15 + (step / num_step) * 72))

            # Cập nhật target region trong cond
            for c in range(NUM_CB):
                cond_ids_buf[c, target_offset:] = target_ids[c * T_target : (c+1) * T_target]

            # COND forward
            cond_out = self._run_main(cond_ids_buf, cond_audio_mask, total_len, total_len, 0, cond_zero_kv)
            # audio_logits shape: (1, num_cb, seq_len, vocab_size)
            # Lấy slice của target region và đổi layout
            cond_logits = cond_out[0, :, target_offset:, :] # (num_cb, T_target, vocab_size)

            uncond_logits = None
            if guidance_scale > 0:
                for c in range(NUM_CB):
                    uncond_ids_buf[c, :] = target_ids[c * T_target : (c+1) * T_target]

                uncond_out = self._run_main(uncond_ids_buf, uncond_audio_mask, T_target, T_target, 0, uncond_zero_kv)
                uncond_logits = uncond_out[0, :, :, :] # (num_cb, T_target, vocab_size)

            pred_tokens, scores = compute_scores(
                cond_logits, uncond_logits, masked, guidance_scale, NUM_CB, T_target, vocab_size, mask_id
            )

            # Chọn k phần tử có điểm cao nhất để unmask
            candidates = [i for i, m in enumerate(masked) if m]
            candidates.sort(key=lambda idx: scores[idx], reverse=True)
            chosen = candidates[:k]

            for flat in chosen:
                target_ids[flat] = pred_tokens[flat]
                masked[flat] = 0
            remaining -= len(chosen)

        return {"target_ids": target_ids, "T_target": T_target}

    def _run_main(self, input_ids_data, audio_mask_data, seq_new, seq_full, position_start, past_kv):
        # input_ids_data shape: (NUM_CB, seq_new) -> model nhận (1, NUM_CB, seq_new)
        # audio_mask_data shape: (seq_new) -> model nhận (1, seq_new)
        input_ids = np.expand_dims(input_ids_data, axis=0)
        audio_mask = np.expand_dims(audio_mask_data, axis=0)

        cache_key = f"{seq_new}|{seq_full}|{position_start}"
        static_tensors = self.static_tensor_cache.get(cache_key)
        if not static_tensors:
            att_data = np.ones((1, 1, seq_new, seq_full), dtype=bool)
            single = np.arange(position_start, position_start + seq_new, dtype=np.int64)
            pos_ids = np.expand_dims(single, axis=0)
            tgt_pos = pos_ids.copy()
            static_tensors = {
                "att_mask": att_data,
                "pos_ids": pos_ids,
                "tgt_pos": tgt_pos,
            }
            self.static_tensor_cache[cache_key] = static_tensors

        feeds = {
            "input_ids": input_ids,
            "audio_mask": audio_mask,
            "attention_mask": static_tensors["att_mask"],
            "position_ids": static_tensors["pos_ids"],
            "target_positions": static_tensors["tgt_pos"],
        }
        for i in range(NUM_LAYERS):
            feeds[f"past_key_{i}"] = past_kv[0][i]
            feeds[f"past_value_{i}"] = past_kv[1][i]

        out = self.sessions["main"].run(None, feeds)
        return out[0] # output audio_logits

    def _encode_audio(self, audio_data):
        clip_len = len(audio_data) - (len(audio_data) % ENCODER_HOP_LENGTH)
        if clip_len < ENCODER_HOP_LENGTH:
            raise ValueError("Reference audio is too short.")

        aligned = audio_data[:clip_len].reshape(1, 1, clip_len)
        out = self.sessions["encoder"].run(None, {"input_values": aligned})
        codes = out[0] # [1, NUM_CB, T] int64
        return codes[0] # -> [NUM_CB, T]

    def _decode_audio(self, target_ids, T_target):
        codes_data = target_ids.reshape(1, NUM_CB, T_target).astype(np.int64)
        out = self.sessions["decoder"].run(None, {"audio_codes": codes_data})
        # values shape (1, 1, samples) hoặc (samples,)
        return out[0].flatten().astype(np.float32)

    def _tokenize_text_nonverbal(self, text):
        parts = []
        last_end = 0
        for match in NONVERBAL_PATTERN.finditer(text):
            start, end = match.span()
            if start > last_end:
                sub_ids = self.tokenizer.encode(text[last_end:start], add_special_tokens=False).ids
                parts.extend(sub_ids)
            tag_ids = self.tokenizer.encode(match.group(0), add_special_tokens=False).ids
            parts.extend(tag_ids)
            last_end = end

        if last_end < len(text):
            sub_ids = self.tokenizer.encode(text[last_end:], add_special_tokens=False).ids
            parts.extend(sub_ids)

        return parts
