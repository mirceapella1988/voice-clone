import sys
import json
import base64
import time
import io
import wave
import threading
import traceback
import numpy as np

from omnivoice_runtime import OmniVoiceRuntime

# Lock để đồng bộ hóa việc ghi ra stdout
stdout_lock = threading.Lock()

def send_json(data):
    with stdout_lock:
        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()

class SidecarApp:
    def __init__(self):
        self.runtime = OmniVoiceRuntime(self.on_progress)
        self.cancel_event = threading.Event()
        self.inference_thread = None
        self.is_generating = False

    def on_progress(self, status, progress):
        send_json({
            "type": "progress",
            "status": status,
            "progress": progress
        })

    def cancel_flag(self):
        return self.cancel_event.is_set()

    def run_inference(self, text, ref_audio, ref_text, instruct, params):
        self.is_generating = True
        self.cancel_event.clear()
        
        start_time = time.time()
        try:
            # Giải mã ref_audio nếu có
            ref_audio_np = None
            if ref_audio is not None:
                if isinstance(ref_audio, str):
                    # Giả định ref_audio là base64 WAV
                    audio_bytes = base64.b64decode(ref_audio)
                    with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
                        params_wav = wav.getparams()
                        frames = wav.readframes(params_wav.nframes)
                        # Chuyển đổi sang float32 mono
                        if params_wav.sampwidth == 2:
                            ref_audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        elif params_wav.sampwidth == 1:
                            ref_audio_np = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                        else:
                            raise ValueError("Unsupported wav sample width")
                        
                        if params_wav.nchannels > 1:
                            # Trộn sang mono
                            ref_audio_np = ref_audio_np.reshape(-1, params_wav.nchannels).mean(axis=1)
                elif isinstance(ref_audio, list):
                    ref_audio_np = np.array(ref_audio, dtype=np.float32)

            audio, sample_rate = self.runtime.generate(
                text=text,
                ref_audio=ref_audio_np,
                ref_text=ref_text,
                instruct=instruct,
                params=params,
                cancel_flag=self.cancel_flag
            )

            # Chuyển đổi output audio sang base64 WAV
            int_audio = np.clip(audio, -1.0, 1.0)
            int_audio = (int_audio * 32767.0).astype(np.int16)
            
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(int_audio.tobytes())
            
            wav_bytes = wav_io.getvalue()
            audio_base64 = base64.b64encode(wav_bytes).decode('ascii')
            
            elapsed = time.time() - start_time
            duration_sec = len(audio) / sample_rate
            rtf = elapsed / duration_sec if duration_sec > 0 else 0
            
            # Tính toán tokens/s ước lượng
            tokens_generated = len(audio) / (sample_rate / 25.0) # Frame rate = 25
            tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

            send_json({
                "type": "complete",
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "metrics": {
                    "elapsedSeconds": elapsed,
                    "durationSeconds": duration_sec,
                    "rtf": rtf,
                    "tokensPerSec": tokens_per_sec
                }
            })

        except InterruptedError:
            send_json({
                "type": "stopped",
                "message": "Generation stopped by user request."
            })
        except Exception as e:
            traceback.print_exc()
            send_json({
                "type": "error",
                "message": str(e)
            })
        finally:
            self.is_generating = False

    def handle_command(self, line):
        try:
            data = json.loads(line.strip())
            cmd = data.get("command")

            if cmd == "load":
                device = data.get("device", "auto")
                # Chạy load trong background hoặc synchronous
                def load_task(dev):
                    try:
                        self.runtime.load(device=dev)
                        actual_providers = self.runtime.sessions["main"].get_providers()
                        actual_device = "cpu"
                        if "CUDAExecutionProvider" in actual_providers:
                            actual_device = "cuda"
                        elif "CoreMLExecutionProvider" in actual_providers:
                            actual_device = "mps"
                        send_json({
                            "type": "status",
                            "status": "ready",
                            "success": True,
                            "actual_device": actual_device,
                            "providers": actual_providers
                        })
                    except Exception as e:
                        send_json({"type": "error", "message": f"Failed to load model: {str(e)}"})
                threading.Thread(target=load_task, args=(device,), daemon=True).start()

            elif cmd == "get_devices":
                import onnxruntime as ort
                available = ort.get_available_providers()
                devices = [{"id": "cpu", "name": "CPU (Mặc định)"}]
                auto_detect = "cpu"
                if "CUDAExecutionProvider" in available:
                    devices.append({"id": "cuda", "name": "NVIDIA GPU (CUDA)"})
                    auto_detect = "cuda"
                if "CoreMLExecutionProvider" in available:
                    devices.append({"id": "mps", "name": "Apple Silicon (CoreML)"})
                    auto_detect = "mps"
                send_json({
                    "type": "devices",
                    "devices": devices,
                    "auto_detect": auto_detect
                })

            elif cmd == "generate":
                if self.is_generating:
                    send_json({"type": "error", "message": "Already generating speech. Send 'stop' first."})
                    return

                text = data.get("text", "")
                ref_audio = data.get("ref_audio")
                ref_text = data.get("ref_text")
                instruct = data.get("instruct")
                params = data.get("params", {})

                self.inference_thread = threading.Thread(
                    target=self.run_inference,
                    args=(text, ref_audio, ref_text, instruct, params),
                    daemon=True
                )
                self.inference_thread.start()

            elif cmd == "stop":
                self.cancel_event.set()
                send_json({"type": "status", "status": "stopping"})

            else:
                send_json({"type": "error", "message": f"Unknown command: {cmd}"})

        except Exception as e:
            send_json({"type": "error", "message": f"Invalid JSON command command: {str(e)}"})

    def main_loop(self):
        send_json({"type": "status", "status": "started"})
        for line in sys.stdin:
            if not line:
                break
            self.handle_command(line)

if __name__ == "__main__":
    app = SidecarApp()
    app.main_loop()
