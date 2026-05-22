import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent, ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  Activity,
  AudioLines,
  BadgeCheck,
  Bot,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleStop,
  Cpu,
  Gauge,
  Headphones,
  Languages,
  Loader2,
  Mic2,
  PanelBottomOpen,
  Play,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Terminal,
  UploadCloud,
  WandSparkles,
  X,
  Zap,
} from "lucide-react";
import { AudioPlayer } from "./components/AudioPlayer";
import { SetupScreen } from "./components/SetupScreen";
import { useSetup } from "./hooks/useSetup";
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

interface InstructOption {
  value: string;
  label: string;
  group: string;
  category: "gender" | "age" | "pitch" | "style" | "region";
  aliases: string[];
}

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { value: "Auto", label: "Tự động" },
  { value: "vi", label: "Tiếng Việt" },
  { value: "en", label: "Tiếng Anh" },
  { value: "zh", label: "Tiếng Trung" },
  { value: "ja", label: "Tiếng Nhật" },
  { value: "ko", label: "Tiếng Hàn" },
  { value: "fr", label: "Tiếng Pháp" },
  { value: "de", label: "Tiếng Đức" },
  { value: "es", label: "Tiếng Tây Ban Nha" },
  { value: "th", label: "Tiếng Thái" },
  { value: "id", label: "Tiếng Indonesia" },
];

const INSTRUCT_OPTIONS: InstructOption[] = [
  { value: "male", label: "Nam", group: "Giới tính", category: "gender", aliases: ["male", "nam", "giong nam"] },
  { value: "female", label: "Nữ", group: "Giới tính", category: "gender", aliases: ["female", "nu", "giong nu"] },
  { value: "child", label: "Trẻ em", group: "Độ tuổi", category: "age", aliases: ["child", "tre em", "nhi dong"] },
  { value: "teenager", label: "Thiếu niên", group: "Độ tuổi", category: "age", aliases: ["teenager", "thieu nien", "tuoi teen"] },
  { value: "young adult", label: "Thanh niên", group: "Độ tuổi", category: "age", aliases: ["young adult", "thanh nien", "tre"] },
  { value: "middle-aged", label: "Trung niên", group: "Độ tuổi", category: "age", aliases: ["middle-aged", "trung nien"] },
  { value: "elderly", label: "Người lớn tuổi", group: "Độ tuổi", category: "age", aliases: ["elderly", "nguoi lon tuoi", "cao tuoi", "gia"] },
  { value: "very low pitch", label: "Rất trầm", group: "Cao độ", category: "pitch", aliases: ["very low pitch", "rat tram", "cuc tram"] },
  { value: "low pitch", label: "Trầm", group: "Cao độ", category: "pitch", aliases: ["low pitch", "tram", "giong tram"] },
  { value: "moderate pitch", label: "Cao độ vừa", group: "Cao độ", category: "pitch", aliases: ["moderate pitch", "vua", "trung binh"] },
  { value: "high pitch", label: "Cao", group: "Cao độ", category: "pitch", aliases: ["high pitch", "cao", "giong cao"] },
  { value: "very high pitch", label: "Rất cao", group: "Cao độ", category: "pitch", aliases: ["very high pitch", "rat cao", "cuc cao"] },
  { value: "whisper", label: "Thì thầm", group: "Phong cách", category: "style", aliases: ["whisper", "thi tham", "noi nho"] },
  { value: "american accent", label: "Giọng Mỹ", group: "Vùng/giọng", category: "region", aliases: ["american accent", "my", "hoa ky"] },
  { value: "australian accent", label: "Giọng Úc", group: "Vùng/giọng", category: "region", aliases: ["australian accent", "uc"] },
  { value: "british accent", label: "Giọng Anh", group: "Vùng/giọng", category: "region", aliases: ["british accent", "anh", "anh quoc"] },
  { value: "canadian accent", label: "Giọng Canada", group: "Vùng/giọng", category: "region", aliases: ["canadian accent", "canada"] },
  { value: "chinese accent", label: "Giọng Trung khi nói tiếng Anh", group: "Vùng/giọng", category: "region", aliases: ["chinese accent", "trung quoc"] },
  { value: "indian accent", label: "Giọng Ấn Độ", group: "Vùng/giọng", category: "region", aliases: ["indian accent", "an do"] },
  { value: "japanese accent", label: "Giọng Nhật", group: "Vùng/giọng", category: "region", aliases: ["japanese accent", "nhat"] },
  { value: "korean accent", label: "Giọng Hàn", group: "Vùng/giọng", category: "region", aliases: ["korean accent", "han quoc"] },
  { value: "portuguese accent", label: "Giọng Bồ Đào Nha", group: "Vùng/giọng", category: "region", aliases: ["portuguese accent", "bo dao nha"] },
  { value: "russian accent", label: "Giọng Nga", group: "Vùng/giọng", category: "region", aliases: ["russian accent", "nga"] },
  { value: "东北话", label: "Tiếng Đông Bắc", group: "Vùng/giọng Trung", category: "region", aliases: ["dong bac", "dongbei"] },
  { value: "云南话", label: "Tiếng Vân Nam", group: "Vùng/giọng Trung", category: "region", aliases: ["van nam", "yunnan"] },
  { value: "四川话", label: "Tiếng Tứ Xuyên", group: "Vùng/giọng Trung", category: "region", aliases: ["tu xuyen", "sichuan"] },
  { value: "宁夏话", label: "Tiếng Ninh Hạ", group: "Vùng/giọng Trung", category: "region", aliases: ["ninh ha", "ningxia"] },
  { value: "桂林话", label: "Tiếng Quế Lâm", group: "Vùng/giọng Trung", category: "region", aliases: ["que lam", "guilin"] },
  { value: "河南话", label: "Tiếng Hà Nam", group: "Vùng/giọng Trung", category: "region", aliases: ["ha nam", "henan"] },
  { value: "济南话", label: "Tiếng Tế Nam", group: "Vùng/giọng Trung", category: "region", aliases: ["te nam", "jinan"] },
  { value: "甘肃话", label: "Tiếng Cam Túc", group: "Vùng/giọng Trung", category: "region", aliases: ["cam tuc", "gansu"] },
  { value: "石家庄话", label: "Tiếng Thạch Gia Trang", group: "Vùng/giọng Trung", category: "region", aliases: ["thach gia trang", "shijiazhuang"] },
  { value: "贵州话", label: "Tiếng Quý Châu", group: "Vùng/giọng Trung", category: "region", aliases: ["quy chau", "guizhou"] },
  { value: "陕西话", label: "Tiếng Thiểm Tây", group: "Vùng/giọng Trung", category: "region", aliases: ["thiem tay", "shaanxi"] },
  { value: "青岛话", label: "Tiếng Thanh Đảo", group: "Vùng/giọng Trung", category: "region", aliases: ["thanh dao", "qingdao"] },
];

const INSTRUCT_OPTION_BY_VALUE = new Map(INSTRUCT_OPTIONS.map((option) => [option.value, option]));

const normalizeSearchText = (value: string) =>
  value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

const cx = (...classes: Array<string | false | null | undefined>) => classes.filter(Boolean).join(" ");

const formatDiagnosticSeconds = (value?: number) =>
  typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)}s` : "—";

const formatDiagnosticNumber = (value?: number, digits = 0) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";

const formatVoiceBadge = (value: string) => {
  const normalized = normalizeSearchText(value);
  if (normalized.includes("nam")) return "Nam";
  if (normalized.includes("nu")) return "Nữ";
  return value;
};

const getEfficiencyLabel = (rtf?: number) => {
  if (typeof rtf !== "number" || !Number.isFinite(rtf)) return "Đang chờ";
  if (rtf <= 0.35) return "Nhanh";
  if (rtf <= 0.8) return "Ổn định";
  return "Chậm";
};

function MetricTile({ label, value, accent }: { label: string; value: ReactNode; accent?: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.04] p-3">
      <div className="text-[11px] font-semibold uppercase text-zinc-500">{label}</div>
      <div className={cx("mt-1 font-mono text-lg font-bold text-zinc-100", accent)}>{value}</div>
    </div>
  );
}

function StepHeader({ number, title, subtitle, icon }: { number: string; title: string; subtitle: string; icon: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold uppercase text-emerald-300">
          <span className="flex h-6 w-6 items-center justify-center rounded-full border border-emerald-400/30 bg-emerald-400/10 font-mono">
            {number}
          </span>
          Bước {number}
        </div>
        <h2 className="mt-3 text-xl font-bold text-zinc-50">{title}</h2>
        <p className="mt-1 text-sm text-zinc-400">{subtitle}</p>
      </div>
      <div className="rounded-xl border border-white/10 bg-white/[0.06] p-3 text-indigo-200 shadow-lg shadow-indigo-950/20">
        {icon}
      </div>
    </div>
  );
}

export default function App() {
  const [modelStatus, setModelStatus] = useState<"unloaded" | "loading" | "ready">("unloaded");
  const [modelProgress, setModelProgress] = useState(0);
  const [modelProgressStatus, setModelProgressStatus] = useState("");

  const [presets, setPresets] = useState<PresetVoice[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>("");

  const [targetText, setTargetText] = useState("Chào bạn! Đây là bản thử nghiệm tính năng nhân bản giọng nói tiếng Việt chạy local trên máy tính.");
  const [selectedInstructValues, setSelectedInstructValues] = useState<string[]>([]);
  const [instructQuery, setInstructQuery] = useState("");
  const [isInstructMenuOpen, setIsInstructMenuOpen] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("vi");

  const [activeTab, setActiveTab] = useState<"preset" | "custom">("preset");
  const [refAudioData, setRefAudioData] = useState<Float32Array | null>(null);
  const [croppedRefAudioData, setCroppedRefAudioData] = useState<Float32Array | null>(null);
  const [refText, setRefText] = useState("");
  const [uploadStatus, setUploadStatus] = useState("Chưa có tệp WAV nào được tải lên.");
  const [isDragActive, setIsDragActive] = useState(false);

  const [cfgStrength, setCfgStrength] = useState(2.0);
  const [inferSteps, setInferSteps] = useState(32);
  const [speed, setSpeed] = useState(1.0);
  const [duration, setDuration] = useState<number | "">("");

  const [isGenerating, setIsGenerating] = useState(false);
  const [generateProgress, setGenerateProgress] = useState(0);
  const [generateStatus, setGenerateStatus] = useState("");

  const [outputAudioData, setOutputAudioData] = useState<Float32Array | null>(null);
  const [refSampleRate, setRefSampleRate] = useState<number>(24000);
  const [outputSampleRate, setOutputSampleRate] = useState<number>(24000);

  const [devices, setDevices] = useState<{ id: string; name: string }[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>("auto");
  const [devicesStatus, setDevicesStatus] = useState<"loading" | "ready" | "error">("loading");
  const [actualDevice, setActualDevice] = useState<string>("");

  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [referenceDiagnostics, setReferenceDiagnostics] = useState<ReferenceDiagnostics | null>(null);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [isLogsOpen, setIsLogsOpen] = useState(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const generationInFlightRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const consoleEndRef = useRef<HTMLDivElement | null>(null);

  const appendLog = (msg: string) => {
    const cleanMessage = msg.replace(/\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g, "").trim();
    setLogs((prev) => [...prev.slice(-100), `[${new Date().toLocaleTimeString()}] ${cleanMessage}`]);
  };

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
        const dataView = new DataView(arrayBuffer);
        const sampleRate = dataView.getUint32(24, true);
        const numChannels = dataView.getUint16(22, true);
        const blockAlign = dataView.getUint16(32, true);
        const bitsPerSample = dataView.getUint16(34, true);

        const dataOffset = 44;
        const dataBytes = arrayBuffer.byteLength - dataOffset;
        const samplesCount = dataBytes / (bitsPerSample / 8);
        const floatData = new Float32Array(samplesCount / numChannels);

        if (bitsPerSample === 16) {
          let floatIdx = 0;
          for (let i = dataOffset; i < arrayBuffer.byteLength; i += blockAlign) {
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
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sRate, true);
    view.setUint32(28, sRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
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

  const parseWavManual = (buffer: ArrayBuffer): { data: Float32Array; sampleRate: number } => {
    const view = new DataView(buffer);
    const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (riff !== "RIFF") throw new Error("Not a valid WAV file");

    const sr = view.getUint32(24, true);
    const numChannels = view.getUint16(22, true);
    const bitsPerSample = view.getUint16(34, true);
    const blockAlign = view.getUint16(32, true);
    const bytesPerSample = bitsPerSample / 8;

    let dataOffset = 44;
    let dataSize = buffer.byteLength - 44;
    let offset = 12;
    while (offset + 8 < buffer.byteLength) {
      const chunkId = String.fromCharCode(
        view.getUint8(offset),
        view.getUint8(offset + 1),
        view.getUint8(offset + 2),
        view.getUint8(offset + 3),
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
          if (byteIdx + 1 < buffer.byteLength) sum += view.getInt16(byteIdx, true) / 32768.0;
        }
        floatData[i] = sum / numChannels;
      }
    } else if (bitsPerSample === 8) {
      for (let i = 0; i < samplesPerChannel; i++) {
        let sum = 0;
        for (let channel = 0; channel < numChannels; channel++) {
          const byteIdx = dataOffset + i * blockAlign + channel * bytesPerSample;
          if (byteIdx < buffer.byteLength) sum += (view.getUint8(byteIdx) - 128) / 128.0;
        }
        floatData[i] = sum / numChannels;
      }
    } else if (bitsPerSample === 32) {
      for (let i = 0; i < samplesPerChannel; i++) {
        let sum = 0;
        for (let channel = 0; channel < numChannels; channel++) {
          const byteIdx = dataOffset + i * blockAlign + channel * bytesPerSample;
          if (byteIdx + 3 < buffer.byteLength) sum += view.getFloat32(byteIdx, true);
        }
        floatData[i] = sum / numChannels;
      }
    } else {
      throw new Error(`Unsupported bits per sample: ${bitsPerSample}`);
    }

    return { data: floatData, sampleRate: sr };
  };

  const decodeWavFile = async (urlOrFile: string | File): Promise<{ data: Float32Array; sampleRate: number }> => {
    let arrayBuffer: ArrayBuffer;
    if (typeof urlOrFile === "string") {
      const response = await fetch(urlOrFile);
      arrayBuffer = await response.arrayBuffer();
    } else {
      arrayBuffer = await urlOrFile.arrayBuffer();
    }

    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      const audioCtx = audioContextRef.current;
      if (audioCtx) {
        const bufferCopy = arrayBuffer.slice(0);
        const audioBuffer = await Promise.race([
          audioCtx.decodeAudioData(bufferCopy),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error("AudioContext decode timeout")), 10000),
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

    return parseWavManual(arrayBuffer);
  };

  const setup = useSetup(appendLog);

  useEffect(() => {
    if (!setup.isReady) return;

    let disposed = false;
    let unlisten: (() => void) | null = null;
    let deviceTimer: number | null = null;
    const presetsAbortController = new AbortController();

    const setupListener = async () => {
      const cleanup = await listen("sidecar-event", (event) => {
        if (disposed) return;

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
            setDevicesStatus("ready");
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
            generationInFlightRef.current = false;
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
                `${data.diagnostics.reference_token_count ?? "?"} ref tokens, resample=${data.diagnostics.reference_resample_method ?? "unknown"}`,
              );
            }
            setIsGenerating(false);
            setGenerateProgress(100);
            setGenerateStatus("Hoàn thành");
          } else if (data.type === "stopped") {
            generationInFlightRef.current = false;
            appendLog("Generation stopped.");
            setIsGenerating(false);
            setGenerateStatus("Đã ngắt");
          } else if (data.type === "error") {
            generationInFlightRef.current = false;
            appendLog(`Sidecar Error: ${data.message}`);
            if (data.diagnostics) setReferenceDiagnostics(data.diagnostics);
            setIsGenerating(false);
            setGenerateStatus(`Lỗi: ${data.message}`);
          } else if (data.type === "stderr") {
            appendLog(`Sidecar stderr: ${data.message}`);
          }
        } catch {
          appendLog(`Failed to parse sidecar message: ${event.payload}`);
        }
      });

      if (disposed) {
        cleanup();
        return;
      }

      unlisten = cleanup;

      deviceTimer = window.setTimeout(async () => {
        if (disposed) return;

        try {
          await invoke("send_to_sidecar", {
            msg: JSON.stringify({ command: "get_devices" }),
          });
        } catch (err: any) {
          if (disposed) return;

          appendLog(`Failed to query devices: ${err.message || err}`);
          setDevices([{ id: "cpu", name: "CPU (Mặc định)" }]);
          setSelectedDevice("cpu");
          setDevicesStatus("error");
        }
      }, 500);
    };

    setupListener();

    fetch("samples/manifest.json", { signal: presetsAbortController.signal })
      .then((res) => res.json())
      .then((data) => {
        if (disposed) return;

        setPresets(data);
        if (data.length > 0) {
          handlePresetSelect(data[0]);
        }
      })
      .catch((err) => {
        if (disposed || err.name === "AbortError") return;

        appendLog(`Failed to load presets: ${err.message}`);
      });

    return () => {
      disposed = true;
      presetsAbortController.abort();
      if (deviceTimer !== null) window.clearTimeout(deviceTimer);
      if (unlisten) unlisten();
    };
  }, [setup.isReady]);

  const handlePresetSelect = async (preset: PresetVoice) => {
    setSelectedPresetId(preset.id);
    setRefAudioData(null);
    setCroppedRefAudioData(null);
    setRefText("");

    try {
      appendLog(`Loading preset ${preset.name}...`);
      const txtRes = await fetch(preset.text);
      const textVal = await txtRes.text();
      setRefText(textVal);

      const audioRes = await decodeWavFile(preset.audio);
      setRefAudioData(audioRes.data);
      setRefSampleRate(audioRes.sampleRate);
      appendLog(`Loaded audio preset ${preset.name} (${audioRes.data.length} samples, ${audioRes.sampleRate} Hz)`);
    } catch (err: any) {
      appendLog(`Failed to load preset voice details: ${err.message}`);
    }
  };

  const loadCustomAudioFile = async (file: File) => {
    setActiveTab("custom");
    setRefAudioData(null);
    setCroppedRefAudioData(null);
    setRefText("");

    try {
      setUploadStatus(`Đang giải mã ${file.name}...`);
      appendLog(`Decoding uploaded file: ${file.name}...`);
      const audioRes = await decodeWavFile(file);
      setRefAudioData(audioRes.data);
      setRefSampleRate(audioRes.sampleRate);
      setUploadStatus(`${file.name} · ${(audioRes.data.length / audioRes.sampleRate).toFixed(1)}s · ${audioRes.sampleRate} Hz`);
      appendLog(`Loaded custom audio file (${audioRes.data.length} samples, ${audioRes.sampleRate} Hz)`);
    } catch (err: any) {
      setUploadStatus("Không thể đọc tệp. Vui lòng chọn WAV hợp lệ.");
      appendLog(`Failed to decode custom audio file: ${err.message}`);
      alert("Định dạng file không được hỗ trợ hoặc bị lỗi. Vui lòng tải file .wav.");
    }
  };

  const handleCustomAudioUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await loadCustomAudioFile(file);
  };

  const handleDrop = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (!file) return;
    await loadCustomAudioFile(file);
  };

  const handleLoadModel = async () => {
    if (devicesStatus === "loading") {
      appendLog("Hardware devices are still loading. Please wait.");
      return;
    }

    setModelStatus("loading");
    setModelProgress(10);
    setModelProgressStatus("Đang khởi tạo sidecar...");
    appendLog(`Sending load command to Python sidecar with device: ${selectedDevice.toUpperCase()}...`);
    try {
      await invoke("send_to_sidecar", {
        msg: JSON.stringify({ command: "load", device: selectedDevice }),
      });
    } catch (err: any) {
      appendLog(`Failed to communicate with sidecar: ${err.message || err}`);
      setModelStatus("unloaded");
    }
  };

  const selectedInstructOptions = selectedInstructValues
    .map((value) => INSTRUCT_OPTION_BY_VALUE.get(value))
    .filter((option): option is InstructOption => Boolean(option));
  const selectedInstructSet = new Set(selectedInstructValues);
  const selectedInstructCategories = new Set(selectedInstructOptions.map((option) => option.category));
  const instructText = selectedInstructOptions.map((option) => option.value).join(", ");
  const normalizedInstructQuery = normalizeSearchText(instructQuery.trim());
  const filteredInstructOptions = INSTRUCT_OPTIONS.filter((option) => {
    if (selectedInstructSet.has(option.value)) return false;
    if (selectedInstructCategories.has(option.category)) return false;
    if (!normalizedInstructQuery) return true;

    const haystack = normalizeSearchText([
      option.label,
      option.value,
      option.group,
      ...option.aliases,
    ].join(" "));
    return haystack.includes(normalizedInstructQuery);
  }).slice(0, 16);

  const groupedInstructOptions = useMemo(() => {
    const groups = new Map<string, InstructOption[]>();
    filteredInstructOptions.forEach((option) => {
      groups.set(option.group, [...(groups.get(option.group) || []), option]);
    });
    return Array.from(groups.entries());
  }, [filteredInstructOptions]);

  const addInstructOption = (option: InstructOption) => {
    setSelectedInstructValues((current) => [
      ...current.filter((value) => INSTRUCT_OPTION_BY_VALUE.get(value)?.category !== option.category),
      option.value,
    ]);
    setInstructQuery("");
    setIsInstructMenuOpen(true);
  };

  const removeInstructOption = (value: string) => {
    setSelectedInstructValues((current) => current.filter((item) => item !== value));
  };

  const handleGenerate = async () => {
    if (generationInFlightRef.current || isGenerating) {
      appendLog("Generation request ignored because another request is already running.");
      return;
    }

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

    generationInFlightRef.current = true;
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
      generationInFlightRef.current = false;
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

  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const isDeviceSelectionDisabled = devicesStatus === "loading" || modelStatus !== "unloaded";
  const isLoadModelDisabled = devicesStatus === "loading";
  const selectedPreset = presets.find((preset) => preset.id === selectedPresetId);
  const currentDeviceName =
    actualDevice ||
    devices.find((device) => device.id === selectedDevice)?.name ||
    (devicesStatus === "loading" ? "Đang phát hiện" : selectedDevice.toUpperCase());
  const referenceDuration = refAudioData ? refAudioData.length / refSampleRate : 0;
  const modelBadgeText =
    modelStatus === "ready"
      ? `Model Loaded on ${actualDevice ? actualDevice.toUpperCase() : currentDeviceName}`
      : modelStatus === "loading"
        ? `Loading Model... ${modelProgress}%`
        : "No Model Loaded";

  if (!setup.isReady) {
    return <SetupScreen setup={setup} />;
  }

  return (
    <main className="relative flex h-screen min-h-0 flex-col overflow-hidden bg-[radial-gradient(circle_at_top_left,rgba(16,185,129,0.16),transparent_28%),radial-gradient(circle_at_top_right,rgba(99,102,241,0.16),transparent_30%),hsl(222_47%_5%)] text-zinc-100">
      <header className="z-20 border-b border-white/10 bg-zinc-950/55 px-5 py-4 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1800px] items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-emerald-400 to-indigo-500 blur-md opacity-50" />
              <img className="relative h-11 w-11 rounded-2xl border border-white/15 bg-zinc-900 object-contain p-1.5" src={voiceCloneIcon} alt="Voice Clone Studio" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-xl font-extrabold text-white">Voice Clone Studio</h1>
              <p className="truncate text-xs text-zinc-400">Local desktop voice cloning · Tauri · OmniVoice runtime</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className={cx(
              "hidden items-center gap-2 rounded-full border px-3 py-2 text-xs font-semibold sm:flex",
              modelStatus === "ready" && "border-emerald-400/25 bg-emerald-400/10 text-emerald-200",
              modelStatus === "loading" && "border-indigo-400/25 bg-indigo-400/10 text-indigo-200",
              modelStatus === "unloaded" && "border-zinc-600/50 bg-white/[0.04] text-zinc-300",
            )}>
              {modelStatus === "loading" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <span className={cx("h-2 w-2 rounded-full", modelStatus === "ready" ? "bg-emerald-400" : "bg-zinc-500")} />}
              {modelBadgeText}
            </div>
            {modelStatus === "unloaded" && (
              <button
                className="rounded-full bg-indigo-500 px-4 py-2 text-sm font-bold text-white shadow-lg shadow-indigo-500/25 transition hover:scale-[1.02] hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
                onClick={handleLoadModel}
                disabled={isLoadModelDisabled}
              >
                Tải model
              </button>
            )}
            <button
              className="rounded-full border border-white/10 bg-white/[0.06] p-2.5 text-zinc-200 transition hover:scale-105 hover:border-indigo-300/40 hover:text-white"
              onClick={() => setIsAdvancedOpen(true)}
              aria-label="Mở tham số nâng cao"
              title="Tham số nâng cao"
            >
              <Settings className="h-5 w-5" />
            </button>
          </div>
        </div>
      </header>

      {modelStatus === "loading" && (
        <div className="z-10 border-b border-indigo-400/20 bg-indigo-500/10 px-5 py-3">
          <div className="mx-auto flex max-w-[1800px] items-center gap-3">
            <BrainCircuit className="h-5 w-5 text-indigo-200" />
            <div className="min-w-0 flex-1">
              <div className="flex justify-between gap-3 text-xs font-semibold text-indigo-100">
                <span className="truncate">{modelProgressStatus || "Đang tải mô hình..."}</span>
                <span>{modelProgress}%</span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
                <div className="h-full rounded-full bg-gradient-to-r from-indigo-400 via-teal-300 to-emerald-300 transition-all duration-500" style={{ width: `${modelProgress}%` }} />
              </div>
            </div>
          </div>
        </div>
      )}

      <section className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
        <div className="mx-auto grid max-w-[1800px] grid-cols-1 gap-4 xl:grid-cols-[1.05fr_1fr_1fr]">
          <article className="glass-panel flex min-h-[720px] flex-col gap-5 p-5">
            <StepHeader number="1" title="Chọn giọng tham chiếu" subtitle="Chọn preset có sẵn hoặc tải WAV của bạn để làm reference voice." icon={<Mic2 className="h-6 w-6" />} />

            <div className="grid grid-cols-2 gap-2 rounded-2xl border border-white/10 bg-zinc-950/50 p-1">
              <button
                className={cx("rounded-xl px-3 py-2 text-sm font-bold transition", activeTab === "preset" ? "bg-gradient-to-r from-emerald-400 to-teal-400 text-zinc-950 shadow-lg shadow-emerald-500/20" : "text-zinc-400 hover:bg-white/[0.06] hover:text-white")}
                onClick={() => {
                  setActiveTab("preset");
                  if (presets.length > 0) handlePresetSelect(presets.find((p) => p.id === selectedPresetId) || presets[0]);
                }}
              >
                Giọng hệ thống
              </button>
              <button
                className={cx("rounded-xl px-3 py-2 text-sm font-bold transition", activeTab === "custom" ? "bg-gradient-to-r from-emerald-400 to-teal-400 text-zinc-950 shadow-lg shadow-emerald-500/20" : "text-zinc-400 hover:bg-white/[0.06] hover:text-white")}
                onClick={() => {
                  setActiveTab("custom");
                  setRefAudioData(null);
                  setCroppedRefAudioData(null);
                  setRefText("");
                }}
              >
                Giọng của tôi
              </button>
            </div>

            {activeTab === "preset" ? (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {presets.map((preset) => {
                  const isActive = selectedPresetId === preset.id;
                  return (
                    <button
                      key={preset.id}
                      className={cx(
                        "group rounded-2xl border p-4 text-left transition hover:scale-[1.015]",
                        isActive ? "border-emerald-300/50 bg-emerald-400/10 ring-2 ring-emerald-300/20" : "border-white/10 bg-white/[0.04] hover:border-indigo-300/40 hover:bg-white/[0.07]",
                      )}
                      onClick={() => handlePresetSelect(preset)}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-base font-bold text-white">{preset.name}</span>
                            {isActive && <Check className="h-4 w-4 text-emerald-300" />}
                          </div>
                          <div className="mt-2 flex flex-wrap gap-2">
                            <span className="rounded-full border border-indigo-300/20 bg-indigo-400/10 px-2 py-1 text-[11px] font-semibold text-indigo-100">{formatVoiceBadge(preset.gender)}</span>
                            <span className="rounded-full border border-teal-300/20 bg-teal-400/10 px-2 py-1 text-[11px] font-semibold text-teal-100">{preset.region}</span>
                          </div>
                        </div>
                        <span className="rounded-full bg-white/10 p-2 text-zinc-300 transition group-hover:bg-indigo-400 group-hover:text-white">
                          <Play className="h-4 w-4 fill-current" />
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div
                className={cx(
                  "rounded-2xl border border-dashed p-5 text-center transition",
                  isDragActive ? "border-emerald-300 bg-emerald-400/10 ring-4 ring-emerald-400/10" : "border-white/15 bg-white/[0.04] hover:border-indigo-300/40",
                )}
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragActive(true);
                }}
                onDragLeave={() => setIsDragActive(false)}
                onDrop={handleDrop}
              >
                <input ref={fileInputRef} className="hidden" type="file" accept="audio/wav,audio/*" onChange={handleCustomAudioUpload} />
                <UploadCloud className="mx-auto h-10 w-10 text-emerald-300" />
                <div className="mt-3 text-base font-bold text-white">Kéo thả WAV hoặc chọn tệp</div>
                <p className="mt-1 text-sm text-zinc-400">Khuyến nghị 5-10 giây, ít nhiễu, đúng nội dung transcript.</p>
                <button
                  className="mt-4 rounded-full border border-emerald-300/30 bg-emerald-400/10 px-4 py-2 text-sm font-bold text-emerald-100 transition hover:scale-[1.02] hover:bg-emerald-400 hover:text-zinc-950"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Chọn WAV
                </button>
                <div className="mt-3 text-xs text-zinc-500">{uploadStatus}</div>
              </div>
            )}

            <div className="rounded-2xl border border-white/10 bg-zinc-950/35 p-3">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div className="text-sm font-bold text-white">Waveform tham chiếu</div>
                <div className="rounded-full bg-white/[0.06] px-2.5 py-1 font-mono text-xs text-zinc-400">
                  {referenceDuration ? `${referenceDuration.toFixed(1)}s · ${refSampleRate} Hz` : "Đang chờ audio"}
                </div>
              </div>
              <AudioPlayer
                key={activeTab + (selectedPresetId || "custom")}
                audioData={refAudioData}
                sampleRate={refSampleRate}
                onCrop={(cropped) => {
                  setCroppedRefAudioData(cropped);
                  appendLog(cropped ? "Đã cập nhật vùng crop." : "Đã hủy vùng crop.");
                }}
                label={activeTab === "preset" ? selectedPreset?.name || "Giọng preset" : "Giọng tải lên"}
                idPrefix="ref-player"
                variant="reference"
              />
            </div>

            <label className="flex flex-1 flex-col gap-2">
              <span className="text-xs font-bold uppercase text-zinc-500">Reference Text</span>
              <textarea
                className="min-h-[118px] flex-1 resize-none rounded-2xl border border-white/10 bg-zinc-950/55 p-4 text-sm leading-6 text-zinc-100 outline-none transition placeholder:text-zinc-600 focus:border-emerald-300/50 focus:ring-4 focus:ring-emerald-400/10"
                value={refText}
                onChange={(e) => setRefText(e.target.value)}
                placeholder="Nhập chính xác nội dung đang nói trong file mẫu..."
                readOnly={activeTab === "preset"}
              />
              <span className="text-xs text-amber-200/80">Transcription alignment rất quan trọng: Reference Text phải khớp nội dung WAV để clone ổn định.</span>
            </label>
          </article>

          <article className="glass-panel flex min-h-[720px] flex-col gap-5 p-5">
            <StepHeader number="2" title="Cấu hình tổng hợp" subtitle="Nhập nội dung mục tiêu, chọn ngôn ngữ và style giọng." icon={<SlidersHorizontal className="h-6 w-6" />} />

            <label className="flex flex-1 flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase text-zinc-500">Target Text</span>
                <span className="font-mono text-xs text-zinc-500">{targetText.length} ký tự</span>
              </div>
              <textarea
                className="min-h-[220px] flex-1 resize-none rounded-3xl border border-white/10 bg-zinc-950/55 p-5 text-base leading-7 text-zinc-100 outline-none transition placeholder:text-zinc-600 focus:border-indigo-300/50 focus:ring-4 focus:ring-indigo-400/10"
                value={targetText}
                onChange={(e) => setTargetText(e.target.value)}
                placeholder="Nhập nội dung bạn muốn clone thành giọng nói..."
              />
            </label>

            <label className="flex flex-col gap-2">
              <span className="text-xs font-bold uppercase text-zinc-500">Ngôn ngữ</span>
              <div className="relative">
                <Languages className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-indigo-200" />
                <select
                  className="w-full appearance-none rounded-2xl border border-white/10 bg-zinc-950/55 py-3 pl-11 pr-10 text-sm font-semibold text-zinc-100 outline-none transition focus:border-indigo-300/50 focus:ring-4 focus:ring-indigo-400/10"
                  value={selectedLanguage}
                  onChange={(e) => setSelectedLanguage(e.target.value)}
                >
                  {LANGUAGE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute right-4 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
              </div>
            </label>

            <div className="relative" onBlur={() => window.setTimeout(() => setIsInstructMenuOpen(false), 120)}>
              <span className="mb-2 block text-xs font-bold uppercase text-zinc-500">Instruct / Voice Styling</span>
              <div className="min-h-[60px] rounded-2xl border border-white/10 bg-zinc-950/55 p-2 transition focus-within:border-emerald-300/50 focus-within:ring-4 focus-within:ring-emerald-400/10" onClick={() => setIsInstructMenuOpen(true)}>
                <div className="flex flex-wrap items-center gap-2">
                  {selectedInstructOptions.map((option) => (
                    <span className="inline-flex max-w-full items-center gap-2 rounded-full border border-emerald-300/25 bg-emerald-400/10 px-3 py-1.5 text-xs font-bold text-emerald-50" key={option.value}>
                      <span>{option.label}</span>
                      <small className="font-mono text-emerald-200/70">{option.value}</small>
                      <button
                        type="button"
                        aria-label={`Bỏ ${option.label}`}
                        className="rounded-full p-0.5 text-emerald-100/70 hover:bg-red-400/20 hover:text-red-100"
                        onClick={(event) => {
                          event.stopPropagation();
                          removeInstructOption(option.value);
                        }}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </span>
                  ))}
                  <input
                    type="text"
                    value={instructQuery}
                    className="min-w-[180px] flex-1 border-none bg-transparent px-2 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
                    onFocus={() => setIsInstructMenuOpen(true)}
                    onChange={(e) => {
                      setInstructQuery(e.target.value);
                      setIsInstructMenuOpen(true);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && filteredInstructOptions.length > 0) {
                        event.preventDefault();
                        addInstructOption(filteredInstructOptions[0]);
                      } else if (event.key === "Backspace" && !instructQuery && selectedInstructValues.length > 0) {
                        removeInstructOption(selectedInstructValues[selectedInstructValues.length - 1]);
                      } else if (event.key === "Escape") {
                        setIsInstructMenuOpen(false);
                      }
                    }}
                    placeholder={selectedInstructOptions.length === 0 ? "Tìm Male, Whisper, High pitch, American accent..." : "Thêm tag..."}
                  />
                </div>
              </div>
              {isInstructMenuOpen && (
                <div className="absolute z-30 mt-2 max-h-80 w-full overflow-y-auto rounded-2xl border border-indigo-300/30 bg-zinc-950/95 p-2 shadow-2xl shadow-black/60 backdrop-blur-md" onMouseDown={(event) => event.preventDefault()}>
                  {groupedInstructOptions.length > 0 ? (
                    groupedInstructOptions.map(([group, options]) => (
                      <div key={group} className="mb-2 last:mb-0">
                        <div className="px-3 py-2 text-[11px] font-bold uppercase text-zinc-500">{group}</div>
                        <div className="grid gap-1">
                          {options.map((option) => (
                            <button
                              type="button"
                              className="flex items-center justify-between gap-3 rounded-xl px-3 py-2 text-left text-sm text-zinc-300 transition hover:bg-indigo-400/15 hover:text-white"
                              key={option.value}
                              onClick={() => addInstructOption(option)}
                            >
                              <span className="font-semibold">{option.label}</span>
                              <code className="text-xs text-indigo-200">{option.value}</code>
                            </button>
                          ))}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="p-4 text-sm text-zinc-500">Không có tag OmniVoice hợp lệ phù hợp.</div>
                  )}
                </div>
              )}
              <p className="mt-2 text-xs text-zinc-500">App hiển thị tiếng Việt nhưng gửi sang OmniVoice: {instructText || "không có instruct"}.</p>
            </div>

            {isGenerating ? (
              <div className="rounded-3xl border border-indigo-300/20 bg-indigo-400/10 p-4">
                <div className="mb-3 flex items-center justify-between gap-3 text-sm font-bold text-indigo-100">
                  <span className="truncate">{generateStatus}</span>
                  <span>{generateProgress}%</span>
                </div>
                <div className="h-3 overflow-hidden rounded-full bg-white/10">
                  <div className="h-full rounded-full bg-gradient-to-r from-indigo-400 via-teal-300 to-emerald-300 transition-all duration-500" style={{ width: `${generateProgress}%` }} />
                </div>
                <button className="mt-4 flex w-full items-center justify-center gap-2 rounded-2xl bg-red-500 px-5 py-3 text-sm font-extrabold text-white shadow-lg shadow-red-500/20 transition hover:scale-[1.01] hover:bg-red-400" onClick={handleStop}>
                  <CircleStop className="h-5 w-5" />
                  Dừng / Cancel
                </button>
              </div>
            ) : (
              <button
                className="group mt-auto flex w-full items-center justify-center gap-3 rounded-3xl bg-gradient-to-r from-indigo-500 via-teal-500 to-emerald-400 px-6 py-5 text-base font-extrabold text-white shadow-2xl shadow-indigo-500/25 transition hover:scale-[1.012] hover:shadow-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-45 disabled:shadow-none"
                onClick={handleGenerate}
                disabled={modelStatus !== "ready"}
              >
                <WandSparkles className="h-5 w-5 transition group-hover:rotate-6" />
                Clone Voice
              </button>
            )}
          </article>

          <article className="glass-panel flex min-h-[720px] flex-col gap-5 p-5">
            <StepHeader number="3" title="Kết quả & xuất file" subtitle="Nghe lại, tải WAV và theo dõi hiệu năng khi audio sẵn sàng." icon={<Headphones className="h-6 w-6" />} />

            {outputAudioData ? (
              <>
                <div className="rounded-3xl border border-emerald-300/20 bg-emerald-400/10 p-4">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-bold text-white">Output Player</div>
                      <div className="text-xs text-emerald-100/70">Bản clone đã render</div>
                    </div>
                    <BadgeCheck className="h-5 w-5 text-emerald-300" />
                  </div>
                  <AudioPlayer
                    audioData={outputAudioData}
                    sampleRate={outputSampleRate}
                    label="Giọng nói sinh ra"
                    idPrefix="output-player"
                    downloadFileName="voice-clone-output.wav"
                    variant="output"
                    showShareButton
                  />
                </div>

                <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-4">
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <div className="text-sm font-bold text-white">Performance Metrics</div>
                      <div className="text-xs text-zinc-500">Runtime feedback từ sidecar</div>
                    </div>
                    <span className="rounded-full border border-emerald-300/25 bg-emerald-400/10 px-3 py-1 text-xs font-bold text-emerald-200">
                      RTF: {metrics ? metrics.rtf.toFixed(2) : "—"} · {getEfficiencyLabel(metrics?.rtf)}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricTile label="Generate" value={metrics ? `${metrics.elapsedSeconds.toFixed(1)}s` : "—"} />
                    <MetricTile label="Duration" value={metrics ? `${metrics.durationSeconds.toFixed(1)}s` : "—"} />
                    <MetricTile label="Token/s" value={metrics ? metrics.tokensPerSec.toFixed(1) : "—"} />
                    <MetricTile label="Sample rate" value={`${outputSampleRate} Hz`} />
                  </div>
                </div>
              </>
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center rounded-3xl border border-dashed border-white/15 bg-white/[0.03] p-8 text-center">
                <div className="rounded-full border border-white/10 bg-white/[0.06] p-5 text-zinc-400">
                  <AudioLines className="h-10 w-10" />
                </div>
                <h3 className="mt-5 text-lg font-bold text-white">Chưa có audio đầu ra</h3>
                <p className="mt-2 max-w-sm text-sm text-zinc-500">Sau khi model hoàn tất, waveform player và metrics sẽ xuất hiện tại đây.</p>
                <div className="mt-5 grid w-full max-w-sm grid-cols-2 gap-3">
                  <MetricTile label="RTF" value="—" />
                  <MetricTile label="Elapsed" value="—" />
                </div>
              </div>
            )}
          </article>
        </div>
      </section>

      <footer className="z-20 border-t border-white/10 bg-zinc-950/70 px-5 py-2 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1800px] items-center justify-between gap-3 text-xs text-zinc-400">
          <div className="flex min-w-0 items-center gap-4">
            <span className="flex items-center gap-2"><Cpu className="h-4 w-4 text-emerald-300" /> {currentDeviceName}</span>
            <span className="hidden items-center gap-2 sm:flex"><Activity className="h-4 w-4 text-indigo-300" /> Runs: {metrics ? `${metrics.elapsedSeconds.toFixed(1)}s` : "idle"}</span>
            <span className="hidden items-center gap-2 md:flex"><Gauge className="h-4 w-4 text-teal-300" /> {metrics ? `RTF ${metrics.rtf.toFixed(2)}` : "RTF —"}</span>
          </div>
          <button className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.05] px-3 py-1.5 font-bold text-zinc-200 transition hover:border-emerald-300/40 hover:text-white" onClick={() => setIsLogsOpen(true)}>
            <PanelBottomOpen className="h-4 w-4" />
            Show Terminal Logs
          </button>
        </div>
      </footer>

      {isAdvancedOpen && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/55 backdrop-blur-sm" onClick={() => setIsAdvancedOpen(false)}>
          <aside className="h-full w-full max-w-md border-l border-white/10 bg-zinc-950/95 p-5 shadow-2xl shadow-black/60" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm font-bold text-emerald-200"><Settings className="h-4 w-4" /> Advanced Parameters</div>
                <h2 className="mt-2 text-2xl font-extrabold text-white">Bộ tinh chỉnh kỹ thuật</h2>
              </div>
              <button className="rounded-full border border-white/10 bg-white/[0.06] p-2 text-zinc-300 hover:text-white" onClick={() => setIsAdvancedOpen(false)} aria-label="Đóng tham số nâng cao">
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="mt-6 space-y-5">
              <label className="block">
                <span className="mb-2 flex items-center justify-between text-xs font-bold uppercase text-zinc-500">
                  Hardware Device
                  <span className="rounded-full border border-emerald-300/25 bg-emerald-400/10 px-2 py-1 text-[10px] text-emerald-200">{currentDeviceName}</span>
                </span>
                <select
                  className="w-full rounded-2xl border border-white/10 bg-zinc-900/90 p-3 text-sm font-semibold text-white outline-none focus:border-emerald-300/50 focus:ring-4 focus:ring-emerald-400/10"
                  value={selectedDevice}
                  onChange={(e) => setSelectedDevice(e.target.value)}
                  disabled={isDeviceSelectionDisabled}
                >
                  <option value="auto">{devicesStatus === "loading" ? "Đang phát hiện thiết bị..." : "Auto"}</option>
                  {devices.map((dev) => (
                    <option key={dev.id} value={dev.id}>{dev.name}</option>
                  ))}
                </select>
              </label>

              <label className="block">
                <span className="mb-2 flex justify-between text-xs font-bold uppercase text-zinc-500"><span>CFG Strength</span><span>{cfgStrength.toFixed(2)}</span></span>
                <input type="range" min="0.5" max="5.0" step="0.1" value={cfgStrength} onChange={(e) => setCfgStrength(parseFloat(e.target.value))} />
              </label>

              <label className="block">
                <span className="mb-2 block text-xs font-bold uppercase text-zinc-500">Unmasking Steps</span>
                <input className="w-full rounded-2xl border border-white/10 bg-zinc-900/90 p-3 text-sm text-white outline-none focus:border-indigo-300/50 focus:ring-4 focus:ring-indigo-400/10" type="number" min="5" max="100" value={inferSteps} onChange={(e) => setInferSteps(parseInt(e.target.value) || 33)} />
              </label>

              <label className="block">
                <span className="mb-2 flex justify-between text-xs font-bold uppercase text-zinc-500"><span>Pitch / Speed</span><span>{speed.toFixed(2)}x</span></span>
                <input type="range" min="0.5" max="2.0" step="0.05" value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))} />
              </label>

              <label className="block">
                <span className="mb-2 block text-xs font-bold uppercase text-zinc-500">Duration Constraint</span>
                <input
                  className="w-full rounded-2xl border border-white/10 bg-zinc-900/90 p-3 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-indigo-300/50 focus:ring-4 focus:ring-indigo-400/10"
                  type="number"
                  min="0"
                  max="60"
                  step="0.1"
                  value={duration}
                  onChange={(e) => {
                    const v = e.target.value;
                    setDuration(v === "" ? "" : parseFloat(v));
                  }}
                  placeholder="Auto"
                />
              </label>
            </div>
          </aside>
        </div>
      )}

      {isLogsOpen && (
        <div className="fixed inset-x-0 bottom-0 z-50 border-t border-white/10 bg-zinc-950/95 p-4 shadow-2xl shadow-black/60 backdrop-blur-md">
          <div className="mx-auto max-w-[1800px]">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-bold text-white"><Terminal className="h-4 w-4 text-emerald-300" /> System Logs & Diagnostics</div>
              <button className="rounded-full border border-white/10 bg-white/[0.06] p-2 text-zinc-300 hover:text-white" onClick={() => setIsLogsOpen(false)} aria-label="Đóng logs">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="grid max-h-[44vh] grid-cols-1 gap-4 md:grid-cols-[1.4fr_1fr]">
              <div className="min-h-0 overflow-y-auto rounded-2xl border border-white/10 bg-black p-4 font-mono text-xs leading-6 text-emerald-300">
                {logs.length > 0 ? logs.map((log, index) => <div key={`${log}-${index}`}>{log}</div>) : <div className="text-zinc-600">Đang chờ log từ runtime...</div>}
                <div ref={consoleEndRef} />
              </div>
              <div className="grid grid-cols-2 gap-3 overflow-y-auto">
                <MetricTile label="Ref duration" value={formatDiagnosticSeconds(referenceDiagnostics?.reference_raw_duration_seconds)} />
                <MetricTile label="Processed" value={formatDiagnosticSeconds(referenceDiagnostics?.reference_processed_duration_seconds)} />
                <MetricTile label="RMS" value={formatDiagnosticNumber(referenceDiagnostics?.reference_processed_rms, 3)} />
                <MetricTile label="Ref tokens" value={referenceDiagnostics?.reference_token_count ?? "—"} />
                <MetricTile label="Target tokens" value={referenceDiagnostics?.target_token_count ?? "—"} />
                <MetricTile label="Sample rate" value={referenceDiagnostics?.reference_processed_sample_rate ? `${referenceDiagnostics.reference_processed_sample_rate} Hz` : "—"} />
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="pointer-events-none fixed bottom-16 right-8 hidden rounded-full border border-emerald-300/20 bg-emerald-400/10 p-3 text-emerald-200 shadow-2xl shadow-emerald-950/40 lg:block">
        <Sparkles className="h-5 w-5" />
      </div>
      <div className="pointer-events-none fixed left-8 top-24 hidden rounded-full border border-indigo-300/20 bg-indigo-400/10 p-3 text-indigo-200 shadow-2xl shadow-indigo-950/40 lg:block">
        <Bot className="h-5 w-5" />
      </div>
      <div className="pointer-events-none fixed right-24 top-24 hidden rounded-full border border-teal-300/20 bg-teal-400/10 p-3 text-teal-200 shadow-2xl shadow-teal-950/40 lg:block">
        <Zap className="h-5 w-5" />
      </div>
    </main>
  );
}
