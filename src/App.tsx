import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { AudioPlayer } from "./components/AudioPlayer";
import voiceCloneIcon from "./assets/voice-clone-icon.png";
import "./index.css";

interface PresetVoice {
  id: string;
  name: string;
  gender: string;
  region: string;
  audio: string;
  text: string;
}

interface Metrics {
  elapsedSeconds: number;
  durationSeconds: number;
  rtf: number;
  tokensPerSec: number;
}

interface ReferenceDiagnostics {
  input_source?: string;
  input_wav_sample_rate?: number;
  input_wav_channels?: number;
  input_wav_sample_width?: number;
  input_wav_frames?: number;
  reference_raw_sample_rate?: number;
  reference_raw_duration_seconds?: number;
  reference_raw_rms?: number;
  reference_resample_method?: string;
  reference_processed_sample_rate?: number;
  reference_processed_duration_seconds?: number;
  reference_processed_rms?: number;
  reference_token_count?: number;
  target_token_count?: number;
  chunk_count?: number;
  guidance_scale?: number;
  denoise?: boolean;
  preprocess_prompt?: boolean;
  postprocess_output?: boolean;
  language_id?: string | null;
}

interface LanguageOption {
  value: string;
  label: string;
}

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { value: "Auto", label: "Auto" },
  { value: "vi", label: "Vietnamese (vi)" },
  { value: "en", label: "English (en)" },
  { value: "zh", label: "Chinese (zh)" },
  { value: "ja", label: "Japanese (ja)" },
  { value: "ko", label: "Korean (ko)" },
  { value: "fr", label: "French (fr)" },
  { value: "de", label: "German (de)" },
  { value: "es", label: "Spanish (es)" },
  { value: "th", label: "Thai (th)" },
  { value: "id", label: "Indonesian (id)" },
];

export default function App() {
  // Model state
  const [modelStatus, setModelStatus] = useState<"unloaded" | "loading" | "ready">("unloaded");
  const [modelProgress, setModelProgress] = useState(0);
  const [modelProgressStatus, setModelProgressStatus] = useState("");

  // Presets list
  const [presets, setPresets] = useState<PresetVoice[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>("");

  // Target speech input
  const [targetText, setTargetText] = useState("Chào bạn! Đây là bản thử nghiệm tính năng nhân bản giọng nói tiếng Việt chạy local trên máy tính.");
  const [instructText, setInstructText] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("vi");

  // Tab state: "preset" | "custom"
  const [activeTab, setActiveTab] = useState<"preset" | "custom">("preset");

  // Reference voice state (Preset / Custom)
  const [refAudioData, setRefAudioData] = useState<Float32Array | null>(null);
  const [croppedRefAudioData, setCroppedRefAudioData] = useState<Float32Array | null>(null);
  const [refText, setRefText] = useState("");

  // Inference advanced parameters
  const [cfgStrength, setCfgStrength] = useState(2.0);
  const [inferSteps, setInferSteps] = useState(32);
  const [speed, setSpeed] = useState(1.0);
  const [duration, setDuration] = useState<number | "">("");

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateProgress, setGenerateProgress] = useState(0);
  const [generateStatus, setGenerateStatus] = useState("");

  // Output voice
  const [outputAudioData, setOutputAudioData] = useState<Float32Array | null>(null);
  const [refSampleRate, setRefSampleRate] = useState<number>(24000);
  const [outputSampleRate, setOutputSampleRate] = useState<number>(24000);

  // Device detection state
  const [devices, setDevices] = useState<{ id: string; name: string }[]>([
    { id: "cpu", name: "CPU (Mặc định)" }
  ]);
  const [selectedDevice, setSelectedDevice] = useState<string>("auto");
  const [actualDevice, setActualDevice] = useState<string>("");

  // Logs & Metrics
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [referenceDiagnostics, setReferenceDiagnostics] = useState<ReferenceDiagnostics | null>(null);

  // Offline Audio Context to decode reference wav files
  const audioContextRef = useRef<AudioContext | null>(null);

  const appendLog = (msg: string) => {
    const cleanMessage = msg.replace(/\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g, "").trim();
    setLogs((prev) => [...prev.slice(-100), `[${new Date().toLocaleTimeString()}] ${cleanMessage}`]);
  };

  // Convert Base64 to Float32Array (For result audio)
  const base64ToFloat32Array = (base64: string): Promise<{ audio: Float32Array; sampleRate: number }> => {
    return new Promise((resolve, reject) => {
      try {
        const binaryString = atob(base64);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        const arrayBuffer = bytes.buffer;
        // Decode WAV header to get PCM raw float data
        const dataView = new DataView(arrayBuffer);
        const sampleRate = dataView.getUint32(24, true);
        const numChannels = dataView.getUint16(22, true);
        const blockAlign = dataView.getUint16(32, true);
        const bitsPerSample = dataView.getUint16(34, true);

        // WAV data offset usually starts at index 44
        const dataOffset = 44;
        const dataBytes = arrayBuffer.byteLength - dataOffset;
        const samplesCount = dataBytes / (bitsPerSample / 8);

        const floatData = new Float32Array(samplesCount / numChannels);

        if (bitsPerSample === 16) {
          let floatIdx = 0;
          for (let i = dataOffset; i < arrayBuffer.byteLength; i += blockAlign) {
            // Lấy channel 0 (mono)
            const val = dataView.getInt16(i, true);
            floatData[floatIdx++] = val / 32768.0;
          }
          resolve({ audio: floatData, sampleRate });
        } else {
          reject(new Error("Unsupported WAV bits per sample"));
        }
      } catch (err) {
        reject(err);
      }
    });
  };

  // WAV header generator helper for Float32Array (to send base64 to sidecar)
  const convertToWavBlob = (buffer: Float32Array, sRate: number): Blob => {
    const length = buffer.length * 2 + 44;
    const bufferArr = new ArrayBuffer(length);
    const view = new DataView(bufferArr);

    const writeString = (v: DataView, offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        v.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + buffer.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // Mono channel
    view.setUint32(24, sRate, true);
    view.setUint32(28, sRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); // 16-bit
    writeString(view, 36, "data");
    view.setUint32(40, buffer.length * 2, true);

    let offset = 44;
    for (let i = 0; i < buffer.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, buffer[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return new Blob([view], { type: "audio/wav" });
  };

  const getBase64Wav = (buffer: Float32Array, sRate: number): Promise<string> => {
    const blob = convertToWavBlob(buffer, sRate);
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64data = (reader.result as string).split(",")[1];
        resolve(base64data);
      };
      reader.readAsDataURL(blob);
    });
  };

  const mixAudioBufferToMono = (audioBuffer: AudioBuffer): Float32Array => {
    if (audioBuffer.numberOfChannels === 1) {
      return new Float32Array(audioBuffer.getChannelData(0));
    }

    const mono = new Float32Array(audioBuffer.length);
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      const channelData = audioBuffer.getChannelData(channel);
      for (let i = 0; i < mono.length; i++) {
        mono[i] += channelData[i] / audioBuffer.numberOfChannels;
      }
    }
    return mono;
  };

  // Manual WAV parser fallback (khi AudioContext không hoạt động)
  const parseWavManual = (buffer: ArrayBuffer): { data: Float32Array; sampleRate: number } => {
    const view = new DataView(buffer);
    // Validate RIFF header
    const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (riff !== "RIFF") throw new Error("Not a valid WAV file");

    const sr = view.getUint32(24, true);
    const numChannels = view.getUint16(22, true);
    const bitsPerSample = view.getUint16(34, true);
    const blockAlign = view.getUint16(32, true);
    const bytesPerSample = bitsPerSample / 8;

    // Find data chunk
    let dataOffset = 44;
    let dataSize = buffer.byteLength - 44;
    // Try to find actual 'data' subchunk
    let offset = 12;
    while (offset + 8 < buffer.byteLength) {
      const chunkId = String.fromCharCode(
        view.getUint8(offset), view.getUint8(offset + 1),
        view.getUint8(offset + 2), view.getUint8(offset + 3)
      );
      const chunkSize = view.getUint32(offset + 4, true);
      if (chunkId === "data") {
        dataOffset = offset + 8;
        dataSize = chunkSize;
        break;
      }
      offset += 8 + chunkSize;
    }

    const samplesPerChannel = Math.floor(dataSize / blockAlign);
    const floatData = new Float32Array(samplesPerChannel);

    if (bitsPerSample === 16) {
      for (let i = 0; i < samplesPerChannel; i++) {
        let sum = 0;
        for (let channel = 0; channel < numChannels; channel++) {
          const byteIdx = dataOffset + i * blockAlign + channel * bytesPerSample;
          if (byteIdx + 1 < buffer.byteLength) {
            sum += view.getInt16(byteIdx, true) / 32768.0;
          }
        }
        floatData[i] = sum / numChannels;
      }
    } else if (bitsPerSample === 8) {
      for (let i = 0; i < samplesPerChannel; i++) {
        let sum = 0;
        for (let channel = 0; channel < numChannels; channel++) {
          const byteIdx = dataOffset + i * blockAlign + channel * bytesPerSample;
          if (byteIdx < buffer.byteLength) {
            sum += (view.getUint8(byteIdx) - 128) / 128.0;
          }
        }
        floatData[i] = sum / numChannels;
      }
    } else if (bitsPerSample === 32) {
      for (let i = 0; i < samplesPerChannel; i++) {
        let sum = 0;
        for (let channel = 0; channel < numChannels; channel++) {
          const byteIdx = dataOffset + i * blockAlign + channel * bytesPerSample;
          if (byteIdx + 3 < buffer.byteLength) {
            sum += view.getFloat32(byteIdx, true);
          }
        }
        floatData[i] = sum / numChannels;
      }
    } else {
      throw new Error(`Unsupported bits per sample: ${bitsPerSample}`);
    }

    return { data: floatData, sampleRate: sr };
  };

  // Decode audio file to Float32Array with fallback
  const decodeWavFile = async (urlOrFile: string | File): Promise<{ data: Float32Array; sampleRate: number }> => {
    let arrayBuffer: ArrayBuffer;
    if (typeof urlOrFile === "string") {
      const response = await fetch(urlOrFile);
      arrayBuffer = await response.arrayBuffer();
    } else {
      arrayBuffer = await urlOrFile.arrayBuffer();
    }

    // Try AudioContext first (supports many formats)
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      const audioCtx = audioContextRef.current;
      if (audioCtx) {
        // Clone buffer vì decodeAudioData detach gốc
        const bufferCopy = arrayBuffer.slice(0);
        const audioBuffer = await Promise.race([
          audioCtx.decodeAudioData(bufferCopy),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error("AudioContext decode timeout")), 10000)
          ),
        ]);
        return {
          data: mixAudioBufferToMono(audioBuffer),
          sampleRate: audioBuffer.sampleRate,
        };
      }
    } catch (err) {
      console.warn("AudioContext decode failed, trying manual WAV parser:", err);
    }

    // Fallback: manual WAV parsing
    return parseWavManual(arrayBuffer);
  };

  // Listen to Tauri events from sidecar
  useEffect(() => {
    let unlisten: (() => void) | null = null;

    const setupListener = async () => {
      unlisten = await listen("sidecar-event", (event) => {
        try {
          const data = JSON.parse(event.payload as string);
          
          if (data.type === "status") {
            appendLog(`Sidecar Status: ${data.status}`);
            if (data.status === "ready") {
              setModelStatus("ready");
              setModelProgress(100);
              setModelProgressStatus("Sẵn sàng");
              if (data.actual_device) {
                setActualDevice(data.actual_device);
                appendLog(`Model loaded on actual device: ${data.actual_device.toUpperCase()}`);
              }
            }
          } else if (data.type === "devices") {
            setDevices(data.devices);
            setSelectedDevice(data.auto_detect);
            appendLog(`Available hardware devices: ${data.devices.map((d: any) => d.name).join(", ")}. Auto-detect selected: ${data.auto_detect.toUpperCase()}`);
          } else if (data.type === "progress") {
            const p = Math.round(data.progress * 100);
            if (data.status.includes("Tải mô hình") || data.status.includes("Loading")) {
              setModelStatus("loading");
              setModelProgress(p);
              setModelProgressStatus(data.status);
            } else {
              setGenerateProgress(p);
              setGenerateStatus(data.status);
            }
            appendLog(`${data.status}: ${p}%`);
          } else if (data.type === "complete") {
            appendLog("Inference completed successfully!");
            base64ToFloat32Array(data.audio_base64).then(({ audio, sampleRate }) => {
              setOutputAudioData(audio);
              setOutputSampleRate(sampleRate);
            });
            setMetrics(data.metrics);
            if (data.diagnostics) {
              const processedDuration = data.diagnostics.reference_processed_duration_seconds;
              setReferenceDiagnostics(data.diagnostics);
              appendLog(
                `Reference diagnostics: ${typeof processedDuration === "number" ? processedDuration.toFixed(2) : "?"}s, ` +
                `${data.diagnostics.reference_token_count ?? "?"} ref tokens, resample=${data.diagnostics.reference_resample_method ?? "unknown"}`
              );
            }
            setIsGenerating(false);
            setGenerateProgress(100);
            setGenerateStatus("Hoàn thành");
          } else if (data.type === "stopped") {
            appendLog("Generation stopped.");
            setIsGenerating(false);
            setGenerateStatus("Đã ngắt");
          } else if (data.type === "error") {
            appendLog(`Sidecar Error: ${data.message}`);
            if (data.diagnostics) {
              setReferenceDiagnostics(data.diagnostics);
            }
            setIsGenerating(false);
            setGenerateStatus(`Lỗi: ${data.message}`);
          } else if (data.type === "stderr") {
            appendLog(`Sidecar stderr: ${data.message}`);
          }
        } catch (err) {
          appendLog(`Failed to parse sidecar message: ${event.payload}`);
        }
      });

      // Query devices after 500ms
      setTimeout(async () => {
        try {
          await invoke("send_to_sidecar", {
            msg: JSON.stringify({ command: "get_devices" })
          });
        } catch (err: any) {
          appendLog(`Failed to query devices: ${err.message || err}`);
        }
      }, 500);
    };

    setupListener();

    // Fetch presets manifest
    fetch("samples/manifest.json")
      .then((res) => res.json())
      .then((data) => {
        setPresets(data);
        if (data.length > 0) {
          handlePresetSelect(data[0]);
        }
      })
      .catch((err) => appendLog(`Failed to load presets: ${err.message}`));

    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  const handlePresetSelect = async (preset: PresetVoice) => {
    setSelectedPresetId(preset.id);
    setRefAudioData(null);
    setCroppedRefAudioData(null);
    setRefText("");

    try {
      appendLog(`Loading preset ${preset.name}...`);
      // Fetch text
      const txtRes = await fetch(preset.text);
      const textVal = await txtRes.text();
      setRefText(textVal);

      // Decode audio
      const audioRes = await decodeWavFile(preset.audio);
      setRefAudioData(audioRes.data);
      setRefSampleRate(audioRes.sampleRate);
      appendLog(`Loaded audio preset ${preset.name} (${audioRes.data.length} samples, ${audioRes.sampleRate} Hz)`);
    } catch (err: any) {
      appendLog(`Failed to load preset voice details: ${err.message}`);
    }
  };

  const handleCustomAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setRefAudioData(null);
    setCroppedRefAudioData(null);
    setRefText("");

    try {
      appendLog(`Decoding uploaded file: ${file.name}...`);
      const audioRes = await decodeWavFile(file);
      setRefAudioData(audioRes.data);
      setRefSampleRate(audioRes.sampleRate);
      appendLog(`Loaded custom audio file (${audioRes.data.length} samples, ${audioRes.sampleRate} Hz)`);
    } catch (err: any) {
      appendLog(`Failed to decode custom audio file: ${err.message}`);
      alert("Định dạng file không được hỗ trợ hoặc bị lỗi. Vui lòng tải file .wav.");
    }
  };

  const handleLoadModel = async () => {
    setModelStatus("loading");
    setModelProgress(10);
    setModelProgressStatus("Đang khởi tạo sidecar...");
    appendLog(`Sending load command to Python sidecar with device: ${selectedDevice.toUpperCase()}...`);
    try {
      await invoke("send_to_sidecar", {
        msg: JSON.stringify({ command: "load", device: selectedDevice })
      });
    } catch (err: any) {
      appendLog(`Failed to communicate with sidecar: ${err.message || err}`);
      setModelStatus("unloaded");
    }
  };

  const handleGenerate = async () => {
    if (modelStatus !== "ready") {
      alert("Vui lòng tải mô hình trước khi thực hiện.");
      return;
    }

    const audioToProcess = croppedRefAudioData || refAudioData;
    if (!audioToProcess) {
      alert("Vui lòng chọn Giọng hệ thống hoặc tải lên Giọng của bạn làm âm thanh mẫu (Reference Audio)!");
      return;
    }

    if (!targetText.trim()) {
      alert("Vui lòng nhập văn bản cần sinh giọng (Target Text)!");
      return;
    }

    if (activeTab === "custom" && !refText.trim()) {
      alert("Vui lòng nhập chính xác nội dung đang nói trong file mẫu (Reference Text) để mô hình có thể bắt chước chính xác nhất!");
      return;
    }

    setIsGenerating(true);
    setGenerateProgress(0);
    setGenerateStatus("Đang xử lý âm thanh mẫu...");
    setOutputAudioData(null);
    setMetrics(null);
    setReferenceDiagnostics(null);

    try {
      let refAudioB64 = null;
      if (audioToProcess) {
        appendLog("Encoding reference audio to Base64 WAV...");
        refAudioB64 = await getBase64Wav(audioToProcess, refSampleRate);
      }

      const payload = {
        command: "generate",
        text: targetText,
        ref_audio: refAudioB64,
        ref_text: refText,
        instruct: instructText,
        params: {
          cfg_strength: cfgStrength,
          guidance_scale: cfgStrength,
          infer_steps: inferSteps,
          num_step: inferSteps,
          language_id: selectedLanguage,
          ref_sample_rate: refSampleRate,
          speed,
          duration: typeof duration === "number" && duration > 0 ? duration : null,
          denoise: true,
          preprocess_prompt: true,
          postprocess_output: true,
        },
      };

      const languageLabel = LANGUAGE_OPTIONS.find((option) => option.value === selectedLanguage)?.label || selectedLanguage;
      appendLog(`Sending generate command to Python sidecar (language: ${languageLabel})...`);
      await invoke("send_to_sidecar", {
        msg: JSON.stringify(payload),
      });
    } catch (err: any) {
      appendLog(`Generation request failed: ${err.message || err}`);
      setIsGenerating(false);
    }
  };

  const handleStop = async () => {
    appendLog("Sending stop command to Python sidecar...");
    try {
      await invoke("send_to_sidecar", {
        msg: JSON.stringify({ command: "stop" }),
      });
    } catch (err: any) {
      appendLog(`Failed to stop: ${err.message || err}`);
    }
  };

  const consoleEndRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const formatDiagnosticSeconds = (value?: number) =>
    typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)}s` : "—";

  const formatDiagnosticNumber = (value?: number, digits = 0) =>
    typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";

  return (
    <main className="app-shell">
      {/* ─── Header ─── */}
      <header className="app-header">
        <div className="app-header-left">
          <img className="app-logo" src={voiceCloneIcon} alt="" aria-hidden="true" />
          <div className="app-header-text">
            <h1>Voice Clone</h1>
            <div className="subtitle">Vietnamese Voice Cloning · Local-first · React + Tauri + Python</div>
          </div>
        </div>

        <div className="app-header-right">
          {modelStatus === "ready" && (
            <div className="status-badge ready">
              <span className="status-dot"></span>
              Model sẵn sàng {actualDevice && `· ${actualDevice.toUpperCase()}`}
            </div>
          )}
        </div>
      </header>

      {/* ─── Model Control ─── */}
      {modelStatus !== "ready" && (
        <section className="model-control-card glass-panel">
          <div className="model-control-inner">
            <div className="model-control-info">
              <div className="model-control-icon">
                {modelStatus === "loading" ? (
                  <svg className="spinner-icon" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" strokeLinecap="round"/>
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" strokeLinejoin="round" strokeLinecap="round"/>
                    <path d="M3.27 6.96L12 12.01l8.73-5.05M12 22.08V12" strokeLinejoin="round" strokeLinecap="round"/>
                  </svg>
                )}
              </div>
              <div>
                <div className="model-control-title">
                  {modelStatus === "loading" ? "Đang tải mô hình..." : "Mô hình chưa được tải"}
                </div>
                <div className="model-control-desc">
                  {modelStatus === "loading"
                    ? modelProgressStatus
                    : "Chọn thiết bị phần cứng và tải model AI để bắt đầu sinh giọng nói"
                  }
                </div>
              </div>
            </div>

            {modelStatus === "unloaded" && (
              <div className="model-control-actions">
                <div className="device-select-group">
                  <label className="device-select-label">Thiết bị</label>
                  <select
                    className="device-select"
                    value={selectedDevice}
                    onChange={(e) => setSelectedDevice(e.target.value)}
                  >
                    <option value="auto">⚡ Tự động phát hiện</option>
                    {devices.map((dev) => (
                      <option key={dev.id} value={dev.id}>{dev.name}</option>
                    ))}
                  </select>
                </div>
                <button className="btn-primary" onClick={handleLoadModel}>
                  Tải Model
                </button>
              </div>
            )}
          </div>

          {modelStatus === "loading" && (
            <div className="model-control-progress">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${modelProgress}%` }} />
              </div>
              <div className="model-control-progress-label">{modelProgress}%</div>
            </div>
          )}
        </section>
      )}

      {/* ─── Workspace ─── */}
      <div className="workspace-container">
        {/* ── Main Column ── */}
        <section className="main-column">
          {/* Reference Audio Section */}
          <div className="glass-panel panel-flex gap-md">
            <div className="tabs-header">
              <button
                className={`tab-btn ${activeTab === "preset" ? "active" : ""}`}
                onClick={() => {
                  setActiveTab("preset");
                  if (presets.length > 0) {
                    const found = presets.find((p) => p.id === selectedPresetId) || presets[0];
                    handlePresetSelect(found);
                  }
                }}
              >
                Giọng Hệ Thống
              </button>
              <button
                className={`tab-btn ${activeTab === "custom" ? "active" : ""}`}
                onClick={() => {
                  setActiveTab("custom");
                  setRefAudioData(null);
                  setCroppedRefAudioData(null);
                  setRefText("");
                }}
              >
                Giọng Của Bạn
              </button>
            </div>

            {activeTab === "custom" && (
              <div className="form-group">
                <label>Tải file âm thanh mẫu (WAV, khuyến nghị 5-10s)</label>
                <input type="file" accept="audio/wav,audio/*" onChange={handleCustomAudioUpload} />
              </div>
            )}

            <div className="form-group">
              <label>Tham chiếu giọng (Reference Audio)</label>
              <AudioPlayer
                key={activeTab + (selectedPresetId || "custom")}
                audioData={refAudioData}
                sampleRate={refSampleRate}
                onCrop={(cropped) => {
                  setCroppedRefAudioData(cropped);
                  appendLog(cropped ? "Đã cập nhật vùng crop." : "Đã hủy vùng crop.");
                }}
                label={activeTab === "preset" ? "Giọng Preset" : "Giọng Tải Lên"}
                idPrefix="ref-player"
              />
            </div>

            <div className="form-group">
              <label>Nội dung giọng mẫu (Reference Text)</label>
              <textarea
                className="textarea-sm"
                value={refText}
                onChange={(e) => setRefText(e.target.value)}
                placeholder="Nhập chính xác nội dung đang nói trong file mẫu..."
                readOnly={activeTab === "preset"}
              />
            </div>
          </div>

          {/* Target Text & Generation */}
          <div className="glass-panel panel-flex gap-md">
            <div className="form-group">
              <label>Language</label>
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
              >
                {LANGUAGE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Văn bản cần sinh giọng (Target Text)</label>
              <textarea
                value={targetText}
                onChange={(e) => setTargetText(e.target.value)}
                placeholder="Nhập nội dung văn bản muốn sinh giọng nói..."
              />
            </div>

             <div className="form-group">
              <label>Chỉ dẫn giọng điệu / Thiết kế giọng (Instruct)</label>
              <input
                type="text"
                value={instructText}
                onChange={(e) => setInstructText(e.target.value)}
                placeholder="Để trống để giữ nguyên giọng mẫu. Ví dụ: male, whisper, low pitch..."
              />
              <span className="input-helper">
                Để trống để clone thuần theo giọng mẫu. Instruct có thể thay đổi giới tính/cao độ/phong cách và làm giọng lệch khỏi reference audio.
              </span>
            </div>

            <div className="params-grid">
              <div className="form-group">
                <label>CFG Strength: {cfgStrength.toFixed(2)}</label>
                <input
                  type="range" min="0.5" max="5.0" step="0.1"
                  value={cfgStrength}
                  onChange={(e) => setCfgStrength(parseFloat(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label>Bước Unmasking: {inferSteps}</label>
                <input
                  type="number" min="5" max="100"
                  value={inferSteps}
                  onChange={(e) => setInferSteps(parseInt(e.target.value) || 33)}
                />
              </div>
              <div className="form-group">
                <label>Tốc độ (Speed): {speed.toFixed(2)}x</label>
                <input
                  type="range" min="0.5" max="2.0" step="0.05"
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label>Thời lượng (Duration): {duration === "" ? "Tự động" : `${duration} giây`}</label>
                <input
                  type="number" min="0" max="60" step="0.1"
                  value={duration}
                  onChange={(e) => {
                    const v = e.target.value;
                    setDuration(v === "" ? "" : parseFloat(v));
                  }}
                  placeholder="Bỏ trống = Tự động"
                />
              </div>
            </div>

            {isGenerating ? (
              <div className="generate-progress">
                <div className="generate-progress-info">
                  <span>{generateStatus}</span>
                  <span>{generateProgress}%</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${generateProgress}%` }} />
                </div>
                <button className="btn-primary btn-danger" onClick={handleStop}>
                  Ngắt Tiến Trình
                </button>
              </div>
            ) : (
              <button
                className="btn-primary"
                onClick={handleGenerate}
                disabled={modelStatus !== "ready"}
              >
                Sinh Giọng Nói
              </button>
            )}
          </div>

          {/* Output Audio */}
          {outputAudioData && (
            <div className="glass-panel">
              <AudioPlayer
                audioData={outputAudioData}
                sampleRate={outputSampleRate}
                label="Giọng Nói Sinh Ra"
                idPrefix="output-player"
                downloadFileName="voice-clone-output.wav"
              />
            </div>
          )}
        </section>

        {/* ── Sidebar ── */}
        <aside className="sidebar-panel">
          {activeTab === "preset" && (
            <div className="glass-panel panel-flex gap-sm">
              <div className="section-title">Chọn Mẫu Giọng</div>
              <div className="presets-container">
                {presets.map((preset) => (
                  <div
                    key={preset.id}
                    className={`preset-card ${selectedPresetId === preset.id ? "active" : ""}`}
                    onClick={() => handlePresetSelect(preset)}
                  >
                    <div className="preset-name">{preset.name}</div>
                    <div className="preset-desc">
                      {preset.gender} · {preset.region}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {metrics && (
            <div className="glass-panel panel-flex gap-sm">
              <div className="section-title">Hiệu năng</div>
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Thời gian</div>
                  <div className="metric-value">{metrics.elapsedSeconds.toFixed(1)}s</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Thời lượng</div>
                  <div className="metric-value">{metrics.durationSeconds.toFixed(1)}s</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Tốc độ</div>
                  <div className="metric-value">{metrics.tokensPerSec.toFixed(1)} t/s</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">RTF</div>
                  <div className="metric-value">{metrics.rtf.toFixed(2)}</div>
                </div>
              </div>
            </div>
          )}

          {referenceDiagnostics && (
            <div className="glass-panel panel-flex gap-sm">
              <div className="section-title">Reference diagnostics</div>
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Ref raw</div>
                  <div className="metric-value">{formatDiagnosticSeconds(referenceDiagnostics.reference_raw_duration_seconds)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Ref processed</div>
                  <div className="metric-value">{formatDiagnosticSeconds(referenceDiagnostics.reference_processed_duration_seconds)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Ref tokens</div>
                  <div className="metric-value">{referenceDiagnostics.reference_token_count ?? "—"}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Target tokens</div>
                  <div className="metric-value">{referenceDiagnostics.target_token_count ?? "—"}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Ref RMS</div>
                  <div className="metric-value">{formatDiagnosticNumber(referenceDiagnostics.reference_processed_rms, 3)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Resample</div>
                  <div className="metric-value">{referenceDiagnostics.reference_resample_method ?? "—"}</div>
                </div>
              </div>
            </div>
          )}

          <div className="glass-panel console-panel">
            <div className="section-title">Nhật ký (Logs)</div>
            <div className="console-output">
              {logs.map((log, index) => (
                <div key={index}>{log}</div>
              ))}
              <div ref={consoleEndRef} />
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}
