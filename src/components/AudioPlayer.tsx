import { useCallback, useEffect, useRef, useState } from "react";
import type { FC, PointerEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import {
  Download,
  Loader2,
  Pause,
  Play,
  RefreshCcw,
  RotateCcw,
  RotateCw,
  Scissors,
  Volume2,
  VolumeX,
} from "lucide-react";

interface AudioPlayerProps {
  audioData: Float32Array | null;
  sampleRate?: number;
  onCrop?: (croppedData: Float32Array | null) => void;
  label?: string;
  idPrefix?: string;
  downloadFileName?: string;
  variant?: "reference" | "output";
}

const PLAYBACK_RATES = [1.0, 1.2, 1.5, 2.0, 0.75, 0.85];

const formatTime = (secs: number) => {
  if (!Number.isFinite(secs) || secs <= 0) return "0:00";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
};

export const AudioPlayer: FC<AudioPlayerProps> = ({
  audioData,
  sampleRate = 24000,
  onCrop,
  label = "Audio Player",
  idPrefix = "audio-player",
  downloadFileName,
  variant = "reference",
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(0.9);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const [cropStart, setCropStart] = useState<number | null>(null);
  const [cropEnd, setCropEnd] = useState<number | null>(null);
  const [selectingCrop, setSelectingCrop] = useState(false);
  const [cropMoved, setCropMoved] = useState(false);
  const [pointerStartRatio, setPointerStartRatio] = useState<number | null>(null);

  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [currentAudioData, setCurrentAudioData] = useState<Float32Array | null>(null);
  const [originalAudioData, setOriginalAudioData] = useState<Float32Array | null>(null);
  const [waveformPeaks, setWaveformPeaks] = useState<Float32Array | null>(null);

  const canCrop = Boolean(onCrop);

  const convertToWavBlob = useCallback((buffer: Float32Array, sRate: number): Blob => {
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
  }, []);

  const computeWaveformPeaks = useCallback((samples: Float32Array, numBars: number): Float32Array => {
    const peaks = new Float32Array(numBars);
    const step = Math.max(1, Math.floor(samples.length / numBars));
    for (let i = 0; i < numBars; i++) {
      let peak = 0;
      const start = i * step;
      const end = Math.min(samples.length, start + step);
      for (let j = start; j < end; j++) {
        const abs = Math.abs(samples[j]);
        if (abs > peak) peak = abs;
      }
      peaks[i] = peak;
    }
    return peaks;
  }, []);

  useEffect(() => {
    if (audioData) {
      setOriginalAudioData(audioData);
      setCurrentAudioData(audioData);
      setCurrentTime(0);
      setDuration(0);
      setCropStart(null);
      setCropEnd(null);
    } else {
      setOriginalAudioData(null);
      setCurrentAudioData(null);
      setAudioUrl(null);
      setWaveformPeaks(null);
      setCurrentTime(0);
      setDuration(0);
    }
  }, [audioData]);

  useEffect(() => {
    if (!currentAudioData) {
      setAudioUrl(null);
      setWaveformPeaks(null);
      return;
    }

    setIsProcessing(true);

    const timeoutId = setTimeout(() => {
      try {
        const peaks = computeWaveformPeaks(currentAudioData, 360);
        setWaveformPeaks(peaks);

        const blob = convertToWavBlob(currentAudioData, sampleRate);
        const url = URL.createObjectURL(blob);

        setAudioUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
      } catch (err) {
        console.error("Failed to process audio data:", err);
      } finally {
        setIsProcessing(false);
      }
    }, 30);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [currentAudioData, sampleRate, convertToWavBlob, computeWaveformPeaks]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume;
      audioRef.current.muted = isMuted;
    }
  }, [volume, isMuted]);

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.parentElement?.clientWidth || 720;
    const height = 86;

    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);

    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const background = ctx.createLinearGradient(0, 0, width, height);
    background.addColorStop(0, variant === "output" ? "rgba(6, 78, 59, 0.42)" : "rgba(24, 24, 27, 0.92)");
    background.addColorStop(1, "rgba(17, 24, 39, 0.92)");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);

    const centerY = height / 2;

    ctx.save();
    ctx.setLineDash([2, 6]);
    ctx.strokeStyle = "rgba(226, 232, 240, 0.16)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();
    ctx.restore();

    const progress = duration ? currentTime / duration : 0;
    const progressX = width * Math.max(0, Math.min(1, progress));

    const barGap = 3;
    const barWidth = 2.5;
    const stride = barWidth + barGap;
    const barCount = Math.max(1, Math.floor(width / stride));
    const activeGradient = ctx.createLinearGradient(0, 0, width, 0);
    activeGradient.addColorStop(0, "#818cf8");
    activeGradient.addColorStop(0.55, "#2dd4bf");
    activeGradient.addColorStop(1, "#34d399");

    ctx.lineCap = "round";
    ctx.lineWidth = barWidth;

    if (waveformPeaks) {
      const peakStep = waveformPeaks.length / barCount;
      for (let i = 0; i < barCount; i++) {
        const x = i * stride + barWidth;
        const peakIdx = Math.min(waveformPeaks.length - 1, Math.floor(i * peakStep));
        const peak = waveformPeaks[peakIdx];
        const amp = Math.max(3, peak * height * 0.42);

        ctx.strokeStyle = x <= progressX ? activeGradient : "rgba(161, 161, 170, 0.45)";
        ctx.globalAlpha = x <= progressX ? 1 : 0.72;
        ctx.beginPath();
        ctx.moveTo(x, centerY - amp);
        ctx.lineTo(x, centerY + amp);
        ctx.stroke();
      }
    } else {
      for (let i = 0; i < barCount; i++) {
        const x = i * stride + barWidth;
        ctx.strokeStyle = "rgba(161, 161, 170, 0.2)";
        ctx.globalAlpha = 0.4;
        ctx.beginPath();
        ctx.moveTo(x, centerY - 4);
        ctx.lineTo(x, centerY + 4);
        ctx.stroke();
      }
    }

    ctx.globalAlpha = 1.0;

    if (cropStart !== null && cropEnd !== null) {
      const startX = Math.min(cropStart, cropEnd) * width;
      const endX = Math.max(cropStart, cropEnd) * width;

      ctx.fillStyle = "rgba(52, 211, 153, 0.14)";
      ctx.fillRect(startX, 4, endX - startX, height - 8);

      ctx.strokeStyle = "rgba(110, 231, 183, 0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(startX, 5);
      ctx.lineTo(startX, height - 5);
      ctx.moveTo(endX, 5);
      ctx.lineTo(endX, height - 5);
      ctx.stroke();
    }

    if (duration && progressX > 0) {
      ctx.strokeStyle = "#c4b5fd";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(progressX, 5);
      ctx.lineTo(progressX, height - 5);
      ctx.stroke();
    }
  }, [waveformPeaks, currentTime, duration, cropStart, cropEnd, variant]);

  useEffect(() => {
    drawWaveform();
  }, [drawWaveform]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;

    const observer = new ResizeObserver(() => {
      requestAnimationFrame(drawWaveform);
    });
    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [drawWaveform]);

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const handlePlayPause = async () => {
    const audio = audioRef.current;
    if (!audio || !audioUrl) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      try {
        await audio.play();
        setIsPlaying(true);
      } catch (err) {
        console.error("Playback error:", err);
      }
    }
  };

  const handleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.muted = !audio.muted;
    setIsMuted(audio.muted);
  };

  const handleVolumeChange = (nextVolume: number) => {
    setVolume(nextVolume);
    if (nextVolume > 0 && isMuted) {
      setIsMuted(false);
    }
  };

  const handleSpeedChange = () => {
    const audio = audioRef.current;
    if (!audio) return;
    const currentIndex = PLAYBACK_RATES.indexOf(playbackRate);
    const nextRate = PLAYBACK_RATES[(currentIndex + 1) % PLAYBACK_RATES.length];
    audio.playbackRate = nextRate;
    setPlaybackRate(nextRate);
  };

  const handleSeek = (seconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, Math.min(duration, audio.currentTime + seconds));
  };

  const readCanvasRatio = (e: PointerEvent<HTMLCanvasElement>): number => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return 0;
    return Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  };

  const handlePointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!currentAudioData || !canCrop) return;
    const ratio = readCanvasRatio(e);
    setSelectingCrop(true);
    setCropMoved(false);
    setPointerStartRatio(ratio);
    setCropStart(ratio);
    setCropEnd(ratio);
    canvasRef.current?.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!selectingCrop || pointerStartRatio === null || !canCrop) return;
    const ratio = readCanvasRatio(e);
    setCropEnd(ratio);
    setCropMoved(Math.abs(ratio - pointerStartRatio) > 0.01);
  };

  const handlePointerUp = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!selectingCrop || !canCrop) return;
    setSelectingCrop(false);
    canvasRef.current?.releasePointerCapture(e.pointerId);

    if (!cropMoved && pointerStartRatio !== null) {
      if (audioRef.current && duration) {
        audioRef.current.currentTime = pointerStartRatio * duration;
      }
      setCropStart(null);
      setCropEnd(null);
      return;
    }

    if (cropStart !== null && cropEnd !== null) {
      const start = Math.min(cropStart, cropEnd);
      const end = Math.max(cropStart, cropEnd);
      setCropStart(start);
      setCropEnd(end);
    }
  };

  const applyCrop = () => {
    if (!currentAudioData || cropStart === null || cropEnd === null) return;

    const startIdx = Math.floor(Math.min(cropStart, cropEnd) * currentAudioData.length);
    const endIdx = Math.floor(Math.max(cropStart, cropEnd) * currentAudioData.length);

    if (endIdx - startIdx < 2400) {
      alert("Vùng chọn quá ngắn (yêu cầu tối thiểu 0.1 giây).");
      return;
    }

    const cropped = currentAudioData.slice(startIdx, endIdx);
    setCurrentAudioData(cropped);
    setCropStart(null);
    setCropEnd(null);

    if (onCrop) {
      onCrop(cropped);
    }
  };

  const resetCrop = () => {
    if (originalAudioData) {
      setCurrentAudioData(originalAudioData);
      setCropStart(null);
      setCropEnd(null);
      if (onCrop) {
        onCrop(null);
      }
    }
  };

  const handleDownload = async () => {
    if (!currentAudioData || isProcessing || !downloadFileName) return;

    try {
      setIsSaving(true);
      const selectedPath = await save({
        title: "Lưu file WAV",
        defaultPath: downloadFileName,
        filters: [{ name: "WAV audio", extensions: ["wav"] }],
      });
      if (!selectedPath) return;

      const wavBytes = new Uint8Array(await convertToWavBlob(currentAudioData, sampleRate).arrayBuffer());
      await invoke("save_audio_file", {
        path: selectedPath,
        bytes: Array.from(wavBytes),
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      alert(`Không thể lưu file WAV: ${message}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="audio-player-container" ref={containerRef}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-sm font-bold text-white">{label}</div>
          <div className="mt-1 font-mono text-xs text-zinc-500">{isProcessing ? "Đang xử lý waveform..." : `${formatTime(currentTime)} / ${formatTime(duration)}`}</div>
        </div>
        <div className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 font-mono text-xs text-zinc-400">
          {sampleRate} Hz
        </div>
      </div>

      <div className="audio-player-waveform-wrapper">
        {isProcessing && (
          <div className="absolute inset-0 z-10 flex items-center justify-center rounded-2xl bg-zinc-950/80 text-xs font-semibold text-zinc-400 backdrop-blur-sm">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Đang tải âm thanh...
          </div>
        )}
        {!audioData && !isProcessing && (
          <div className="absolute inset-0 z-10 flex items-center justify-center rounded-2xl text-xs font-semibold text-zinc-600">
            Chưa có waveform
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="audio-player-canvas"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          aria-label="Audio waveform"
          style={{ cursor: currentAudioData && canCrop ? "crosshair" : "pointer" }}
        />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          id={`${idPrefix}-play`}
          className="flex h-11 w-11 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-emerald-400 text-white shadow-lg shadow-indigo-500/25 transition hover:scale-105 disabled:cursor-not-allowed disabled:opacity-40"
          onClick={handlePlayPause}
          disabled={!audioUrl || isProcessing}
          title={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? <Pause className="h-5 w-5 fill-current" /> : <Play className="h-5 w-5 fill-current" />}
        </button>

        <button className="player-icon-button" onClick={() => handleSeek(-5)} disabled={!audioUrl || isProcessing} title="Lùi 5s">
          <RotateCcw className="h-4 w-4" />
        </button>
        <button className="player-icon-button" onClick={() => handleSeek(5)} disabled={!audioUrl || isProcessing} title="Tới 5s">
          <RotateCw className="h-4 w-4" />
        </button>
        <button className="h-9 rounded-full border border-white/10 bg-white/[0.05] px-3 font-mono text-xs font-bold text-zinc-300 transition hover:scale-105 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40" onClick={handleSpeedChange} disabled={!audioUrl || isProcessing} title="Tốc độ">
          {playbackRate}x
        </button>
        <button className="player-icon-button" onClick={handleMute} disabled={!audioUrl || isProcessing} title="Mute">
          {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
        </button>
        <input
          className="w-24 accent-emerald-400"
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={volume}
          onChange={(event) => handleVolumeChange(parseFloat(event.target.value))}
          disabled={!audioUrl || isProcessing}
          aria-label="Âm lượng"
        />

        <div className="ml-auto flex items-center gap-2">
          {downloadFileName && (
            <button
              type="button"
              className="player-icon-button"
              disabled={!currentAudioData || isProcessing || isSaving}
              onClick={handleDownload}
              aria-label="Tải WAV"
              title="Chọn nơi lưu WAV"
            >
              {isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            </button>
          )}
        </div>

        {cropStart !== null && cropEnd !== null && (
          <button className="rounded-full border border-emerald-300/30 bg-emerald-400/10 px-3 py-2 text-xs font-bold text-emerald-100 transition hover:bg-emerald-400 hover:text-zinc-950" onClick={applyCrop}>
            <Scissors className="mr-1 inline h-3.5 w-3.5" />
            Cắt
          </button>
        )}

        {currentAudioData !== originalAudioData && (
          <button className="rounded-full border border-red-300/25 bg-red-400/10 px-3 py-2 text-xs font-bold text-red-100 transition hover:bg-red-500 hover:text-white" onClick={resetCrop}>
            <RefreshCcw className="mr-1 inline h-3.5 w-3.5" />
            Reset
          </button>
        )}
      </div>

      <audio
        ref={audioRef}
        src={audioUrl || undefined}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
        style={{ display: "none" }}
      />
    </div>
  );
};
