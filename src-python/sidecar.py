import sys
import json
import base64
import time
import io
import wave
import threading
import traceback
import numpy as np

from omnivoice_runtime import OmniVoiceRuntime, OmniVoiceValidationError, get_available_hardware_devices

# Pre-import heavy packages in the main thread to prevent deadlocks (Loader Lock)
# when importing them dynamically inside a background thread on Windows.
try:
    import torch
    import omnivoice
except ImportError:
    pass



# Lock để đồng bộ hóa việc ghi ra stdout
stdout_lock = threading.Lock()

def send_json(data):
    with stdout_lock:
        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()


def read_next_command(input_buffer):
    while True:
        line = input_buffer.readline()
        if not line:
            return None

        stripped = line.strip()
        if not stripped:
            continue

        if stripped.lower().startswith(b"content-length:"):
            try:
                length = int(stripped.split(b":", 1)[1].strip())
            except ValueError as exc:
                raise ValueError("Invalid command frame content length") from exc

            separator = input_buffer.readline()
            if separator not in (b"\n", b"\r\n", b""):
                raise ValueError("Invalid command frame separator")

            payload = input_buffer.read(length)
            if len(payload) != length:
                raise EOFError("Incomplete command frame payload")
            return payload.decode("utf-8")

        return line.decode("utf-8")


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

    def start_progress_heartbeat(self, stop_event, started_at, interval=2.0):
        def heartbeat():
            while not stop_event.wait(interval):
                elapsed = max(0.0, time.time() - started_at)
                progress = min(0.95, 0.25 + min(elapsed, 180.0) / 180.0 * 0.70)
                self.on_progress(f"Generating speech... {elapsed:.0f}s elapsed", progress)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        return thread

    def cancel_flag(self):
        return self.cancel_event.is_set()

    def decode_reference_audio(self, ref_audio, params):
        ref_audio_np = None
        ref_sample_rate = params.get("ref_sample_rate")
        metadata = {}

        if ref_audio is None:
            return ref_audio_np, ref_sample_rate, metadata

        if isinstance(ref_audio, str):
            audio_bytes = base64.b64decode(ref_audio)
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
                params_wav = wav.getparams()
                ref_sample_rate = params_wav.framerate
                frames = wav.readframes(params_wav.nframes)
                metadata = {
                    "input_source": "base64_wav",
                    "input_wav_sample_rate": params_wav.framerate,
                    "input_wav_channels": params_wav.nchannels,
                    "input_wav_sample_width": params_wav.sampwidth,
                    "input_wav_frames": params_wav.nframes,
                }
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
            metadata = {
                "input_source": "float_list",
                "input_samples": len(ref_audio_np),
            }
        else:
            raise ValueError("Unsupported reference audio payload type")

        return ref_audio_np, ref_sample_rate, metadata

    def run_inference(self, text, ref_audio, ref_text, instruct, params):
        self.is_generating = True
        self.cancel_event.clear()
        params = params or {}
        diagnostics = {}
        
        start_time = time.time()
        heartbeat_stop = threading.Event()
        heartbeat_thread = None
        try:
            self.on_progress("Decoding reference audio...", 0.05)
            ref_audio_np, ref_sample_rate, diagnostics = self.decode_reference_audio(ref_audio, params)

            heartbeat_thread = self.start_progress_heartbeat(heartbeat_stop, start_time)
            audio, sample_rate = self.runtime.generate(
                text=text,
                ref_audio=ref_audio_np,
                ref_text=ref_text,
                instruct=instruct,
                params={**params, "ref_sample_rate": ref_sample_rate},
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
            
            # Tính toán tokens/s ước lượng theo frame rate của audio tokenizer.
            audio_tokenizer = getattr(getattr(self.runtime, "model", None), "audio_tokenizer", None)
            tokenizer_config = getattr(audio_tokenizer, "config", None)
            frame_rate = float(getattr(tokenizer_config, "frame_rate", 75.0))
            tokens_generated = len(audio) / (sample_rate / frame_rate)
            tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
            diagnostics = {**diagnostics, **getattr(self.runtime, "last_diagnostics", {})}

            send_json({
                "type": "complete",
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "metrics": {
                    "elapsedSeconds": elapsed,
                    "durationSeconds": duration_sec,
                    "rtf": rtf,
                    "tokensPerSec": tokens_per_sec
                },
                "diagnostics": diagnostics
            })

        except InterruptedError:
            send_json({
                "type": "stopped",
                "message": "Generation stopped by user request."
            })
        except OmniVoiceValidationError as e:
            diagnostics = {**diagnostics, **getattr(self.runtime, "last_diagnostics", {})}
            send_json({
                "type": "error",
                "message": str(e),
                "diagnostics": diagnostics
            })
        except Exception as e:
            traceback.print_exc()
            diagnostics = {**diagnostics, **getattr(self.runtime, "last_diagnostics", {})}
            send_json({
                "type": "error",
                "message": str(e),
                "diagnostics": diagnostics
            })
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
            self.is_generating = False

    def handle_command(self, line):
        line = line.strip()
        if not line:
            return

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            send_json({"type": "error", "message": f"Invalid JSON command: {str(e)}"})
            return

        try:
            cmd = data.get("command")

            if cmd == "load":
                device = data.get("device", "auto")
                # Chạy load trong background hoặc synchronous
                def load_task(dev):
                    try:
                        self.runtime.load(device=dev)
                        send_json({
                            "type": "status",
                            "status": "ready",
                            "success": True,
                            "actual_device": self.runtime.actual_device,
                            "providers": [self.runtime.actual_device]
                        })
                    except Exception as e:
                        send_json({"type": "error", "message": f"Failed to load model: {str(e)}"})
                threading.Thread(target=load_task, args=(device,), daemon=True).start()

            elif cmd == "get_devices":
                hardware = get_available_hardware_devices()
                devices = hardware["devices"]
                for device_info in devices:
                    if device_info["id"] == "cpu":
                        device_info["name"] = "CPU (Mặc định)"
                    elif device_info["id"] == "mps":
                        device_info["name"] = "Apple Silicon (MPS)"
                send_json({
                    "type": "devices",
                    "devices": devices,
                    "auto_detect": hardware["auto_detect"],
                    "diagnostics": hardware.get("diagnostics", {})
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

                send_json({
                    "type": "status",
                    "status": "generate_received",
                })
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
            send_json({"type": "error", "message": f"Failed to handle sidecar command: {str(e)}"})

    def main_loop(self):
        send_json({"type": "status", "status": "started"})
        input_buffer = sys.stdin.buffer
        while True:
            try:
                command = read_next_command(input_buffer)
            except Exception as e:
                send_json({"type": "error", "message": f"Invalid command frame: {str(e)}"})
                continue

            if command is None:
                break
            self.handle_command(command)

if __name__ == "__main__":
    app = SidecarApp()
    app.main_loop()
