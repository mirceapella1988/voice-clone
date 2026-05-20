import React, { useRef, useEffect, useState, useCallback } from "react";

interface AudioPlayerProps {
  audioData: Float32Array | null;
  sampleRate?: number;
  onCrop?: (croppedData: Float32Array | null) => void;
  label?: string;
  idPrefix?: string;
  downloadFileName?: string;
}

const PLAYBACK_RATES = [1.0, 1.2, 1.5, 2.0, 0.75, 0.85];

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  audioData,
  sampleRate = 24000,
  onCrop,
  label = "Audio Player",
  idPrefix = "audio-player",
  downloadFileName,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isMuted, setIsMuted] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const [cropStart, setCropStart] = useState<number | null>(null);
  const [cropEnd, setCropEnd] = useState<number | null>(null);
  const [selectingCrop, setSelectingCrop] = useState(false);
  const [cropMoved, setCropMoved] = useState(false);
  const [pointerStartRatio, setPointerStartRatio] = useState<number | null>(null);

  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [currentAudioData, setCurrentAudioData] = useState<Float32Array | null>(null);
  const [originalAudioData, setOriginalAudioData] = useState<Float32Array | null>(null);

  // Precomputed waveform peaks for fast drawing
  const [waveformPeaks, setWaveformPeaks] = useState<Float32Array | null>(null);

  // Ghi file WAV từ Float32Array — chunked để không block UI
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

  // Precompute waveform peaks (downsampled) for fast rendering
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

  // Đồng bộ hóa với prop audioData đầu vào
  useEffect(() => {
    if (audioData) {
      setOriginalAudioData(audioData);
      setCurrentAudioData(audioData);
    } else {
      setOriginalAudioData(null);
      setCurrentAudioData(null);
      setAudioUrl(null);
      setWaveformPeaks(null);
    }
  }, [audioData]);

  // Tạo URL + Waveform khi currentAudioData thay đổi (deferred)
  useEffect(() => {
    if (!currentAudioData) {
      setAudioUrl(null);
      setWaveformPeaks(null);
      return;
    }

    setIsProcessing(true);

    // Defer heavy work to next frame so React can render loading state
    const timeoutId = setTimeout(() => {
      try {
        // Compute waveform peaks first (lighter)
        const peaks = computeWaveformPeaks(currentAudioData, 300);
        setWaveformPeaks(peaks);

        // Then create WAV blob (heavier)
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

  // Cleanup URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, []);

  // Vẽ Waveform
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.parentElement?.clientWidth || 720;
    const height = 64;

    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);

    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = "#131825";
    ctx.fillRect(0, 0, width, height);

    const centerY = height / 2;

    // Center line
    ctx.save();
    ctx.setLineDash([1, 4]);
    ctx.strokeStyle = "rgba(226, 232, 240, 0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();
    ctx.restore();

    const progress = duration ? currentTime / duration : 0;
    const progressX = width * Math.max(0, Math.min(1, progress));

    const barGap = 3;
    const barWidth = 2;
    const stride = barWidth + barGap;
    const barCount = Math.max(1, Math.floor(width / stride));

    ctx.lineCap = "round";
    ctx.lineWidth = barWidth;

    if (waveformPeaks) {
      // Use precomputed peaks
      const peakStep = waveformPeaks.length / barCount;
      for (let i = 0; i < barCount; i++) {
        const x = i * stride + barWidth;
        const peakIdx = Math.min(waveformPeaks.length - 1, Math.floor(i * peakStep));
        const peak = waveformPeaks[peakIdx];
        const amp = Math.max(2, peak * height * 0.43);

        ctx.strokeStyle = x <= progressX ? "#9d8ff8" : "#555d74";
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.moveTo(x, centerY - amp);
        ctx.lineTo(x, centerY + amp);
        ctx.stroke();
      }
    } else {
      // Placeholder bars
      for (let i = 0; i < barCount; i++) {
        const x = i * stride + barWidth;
        ctx.strokeStyle = "#555d74";
        ctx.globalAlpha = 0.25;
        ctx.beginPath();
        ctx.moveTo(x, centerY - 3);
        ctx.lineTo(x, centerY + 3);
        ctx.stroke();
      }
    }

    ctx.globalAlpha = 1.0;

    // Crop region
    if (cropStart !== null && cropEnd !== null) {
      const startX = Math.min(cropStart, cropEnd) * width;
      const endX = Math.max(cropStart, cropEnd) * width;

      ctx.fillStyle = "rgba(124, 109, 240, 0.15)";
      ctx.fillRect(startX, 2, endX - startX, height - 4);

      ctx.strokeStyle = "rgba(157, 143, 248, 0.8)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(startX, 2);
      ctx.lineTo(startX, height - 2);
      ctx.moveTo(endX, 2);
      ctx.lineTo(endX, height - 2);
      ctx.stroke();
    }

    // Playback cursor
    if (duration && progressX > 0) {
      ctx.strokeStyle = "#7c6df0";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(progressX, 4);
      ctx.lineTo(progressX, height - 4);
      ctx.stroke();
    }
  }, [waveformPeaks, currentTime, duration, cropStart, cropEnd]);

  // Re-draw waveform khi thay đổi
  useEffect(() => {
    drawWaveform();
  }, [drawWaveform]);

  // ResizeObserver vẽ lại khi đổi kích thước màn hình
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

  // Audio event handlers
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

  // Crop pointer handlers
  const readCanvasRatio = (e: React.PointerEvent<HTMLCanvasElement>): number => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return 0;
    return Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!currentAudioData) return;
    const ratio = readCanvasRatio(e);
    setSelectingCrop(true);
    setCropMoved(false);
    setPointerStartRatio(ratio);
    setCropStart(ratio);
    setCropEnd(ratio);
    canvasRef.current?.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!selectingCrop || pointerStartRatio === null) return;
    const ratio = readCanvasRatio(e);
    setCropEnd(ratio);
    setCropMoved(Math.abs(ratio - pointerStartRatio) > 0.01);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!selectingCrop) return;
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

  const formatTime = (secs: number) => {
    if (!Number.isFinite(secs) || secs <= 0) return "0:00";
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  };

  return (
    <div className="audio-player-container" ref={containerRef}>
      <div className="audio-player-header">
        <span className="audio-player-label">{label}</span>
        <span className="audio-player-time">
          {isProcessing ? "Đang xử lý..." : `${formatTime(currentTime)} / ${formatTime(duration)}`}
        </span>
      </div>

      <div className="audio-player-waveform-wrapper">
        {isProcessing && (
          <div style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(19, 24, 37, 0.9)",
            zIndex: 2,
            borderRadius: "6px",
            fontSize: "0.72rem",
            color: "#7a839a"
          }}>
            Đang tải âm thanh...
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="audio-player-canvas"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          aria-label="Audio waveform"
          style={{ cursor: currentAudioData ? "crosshair" : "default" }}
        />
      </div>

      <div className="audio-player-controls">
        <button
          id={`${idPrefix}-play`}
          className="btn-control btn-play"
          onClick={handlePlayPause}
          disabled={!audioUrl || isProcessing}
          title={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
              <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        <button
          className="btn-control"
          onClick={() => handleSeek(-5)}
          disabled={!audioUrl || isProcessing}
          title="Back 5s"
        >
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
            <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z" />
          </svg>
        </button>

        <button
          className="btn-control"
          onClick={() => handleSeek(5)}
          disabled={!audioUrl || isProcessing}
          title="Forward 5s"
        >
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
            <path d="M18 13c0 3.31-2.69 6-6 6s-6-2.69-6-6 2.69-6 6-6v4l5-5-5-5v4c-4.42 0-8 3.58-8 8s3.58 8 8 8 8-3.58 8-8h-2z" />
          </svg>
        </button>

        <button
          className="btn-control btn-speed"
          onClick={handleSpeedChange}
          disabled={!audioUrl || isProcessing}
          title="Tốc độ"
        >
          {playbackRate}x
        </button>

        <button
          className="btn-control"
          onClick={handleMute}
          disabled={!audioUrl || isProcessing}
          title="Mute"
        >
          {isMuted ? (
            <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
              <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.21.05-.42.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73 4.27 3zM12 4L9.91 6.09 12 8.18V4z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
              <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
            </svg>
          )}
        </button>

        {downloadFileName && (
          <a
            className="btn-control btn-download"
            href={audioUrl || undefined}
            download={downloadFileName}
            aria-disabled={!audioUrl || isProcessing}
            onClick={(e) => {
              if (!audioUrl || isProcessing) e.preventDefault();
            }}
            aria-label="Tải WAV"
            title="Tải WAV"
          >
            <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
              <path d="M5 20h14v-2H5v2zM19 9h-4V3H9v6H5l7 7 7-7z" />
            </svg>
          </a>
        )}

        {cropStart !== null && cropEnd !== null && (
          <button className="btn-action btn-crop-apply" onClick={applyCrop}>
            Cắt (Crop)
          </button>
        )}

        {currentAudioData !== originalAudioData && (
          <button className="btn-action btn-crop-reset" onClick={resetCrop}>
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
