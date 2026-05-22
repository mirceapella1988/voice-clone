import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface RuntimeStatus {
  has_python: boolean;
  has_ffmpeg: boolean;
  has_torch: boolean;
  models_path: string;
}

export interface SetupProgress {
  stage: string;
  message: string;
  percent: number;
}

export type SetupStatus = "checking" | "installing" | "starting" | "ready" | "error";

export interface SetupState {
  status: SetupStatus;
  progress: SetupProgress;
  runtime: RuntimeStatus | null;
  gpu: string;
  error: string;
  isReady: boolean;
  retry: () => void;
}

const initialProgress: SetupProgress = {
  stage: "checking",
  message: "Đang kiểm tra cấu hình môi trường...",
  percent: 0,
};

const parseProgressPayload = (payload: unknown): SetupProgress | null => {
  const data = typeof payload === "string" ? JSON.parse(payload) : payload;
  if (!data || typeof data !== "object") return null;
  const value = data as Partial<SetupProgress>;

  return {
    stage: String(value.stage || "installing"),
    message: String(value.message || "Đang chuẩn bị môi trường chạy lần đầu..."),
    percent: Math.max(0, Math.min(100, Number(value.percent || 0))),
  };
};

export function useSetup(appendLog?: (message: string) => void): SetupState {
  const [status, setStatus] = useState<SetupStatus>("checking");
  const [progress, setProgress] = useState<SetupProgress>(initialProgress);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [gpu, setGpu] = useState("");
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const activeRunRef = useRef(0);
  const appendLogRef = useRef(appendLog);

  useEffect(() => {
    appendLogRef.current = appendLog;
  }, [appendLog]);

  const retry = useCallback(() => {
    setAttempt((value) => value + 1);
  }, []);

  useEffect(() => {
    let disposed = false;
    const runId = activeRunRef.current + 1;
    activeRunRef.current = runId;

    const runSetup = async () => {
      setStatus("checking");
      setError("");
      setProgress(initialProgress);

      let unlistenProgress: (() => void) | null = null;
      let unlistenComplete: (() => void) | null = null;

      try {
        unlistenProgress = await listen("install-progress", (event) => {
          if (disposed || activeRunRef.current !== runId) return;
          try {
            const parsed = parseProgressPayload(event.payload);
            if (parsed) {
              setStatus("installing");
              setProgress(parsed);
              appendLogRef.current?.(`${parsed.message} ${parsed.percent}%`);
            }
          } catch (err) {
            appendLogRef.current?.(`Failed to parse install progress: ${String(err)}`);
          }
        });

        unlistenComplete = await listen("install-complete", (event) => {
          if (disposed || activeRunRef.current !== runId) return;
          try {
            const data = typeof event.payload === "string" ? JSON.parse(event.payload) : event.payload;
            if (data?.models_path) {
              appendLogRef.current?.(`Model cache ready at ${data.models_path}`);
            }
          } catch {
            appendLogRef.current?.("Runtime installation completed.");
          }
        });

        const firstStatus = await invoke<RuntimeStatus>("check_runtime");
        if (disposed || activeRunRef.current !== runId) return;
        setRuntime(firstStatus);

        if (!firstStatus.has_python || !firstStatus.has_ffmpeg || !firstStatus.has_torch) {
          setStatus("installing");
          setProgress({
            stage: "detecting",
            message: "Đang phát hiện GPU và chuẩn bị cài đặt runtime...",
            percent: 1,
          });
          const detectedGpu = await invoke<string>("get_gpu_type");
          if (disposed || activeRunRef.current !== runId) return;
          setGpu(detectedGpu);
          appendLogRef.current?.(`Selected runtime GPU profile: ${detectedGpu.toUpperCase()}`);

          await invoke("install_runtime", { gpu: detectedGpu });
          if (disposed || activeRunRef.current !== runId) return;
        }

        const readyStatus = await invoke<RuntimeStatus>("check_runtime");
        if (disposed || activeRunRef.current !== runId) return;
        setRuntime(readyStatus);

        if (!readyStatus.has_python || !readyStatus.has_ffmpeg || !readyStatus.has_torch) {
          throw new Error("Runtime installation finished but required components are still missing.");
        }

        setStatus("starting");
        setProgress({
          stage: "starting",
          message: "Đang khởi động Python runtime...",
          percent: 100,
        });
        await invoke("start_runtime_sidecar");
        if (disposed || activeRunRef.current !== runId) return;

        appendLogRef.current?.("Python runtime sidecar is ready.");
        setStatus("ready");
      } catch (err) {
        if (disposed || activeRunRef.current !== runId) return;
        const message = err instanceof Error ? err.message : String(err);
        setStatus("error");
        setError(message);
        appendLogRef.current?.(`Setup failed: ${message}`);
      } finally {
        unlistenProgress?.();
        unlistenComplete?.();
      }
    };

    runSetup();

    return () => {
      disposed = true;
    };
  }, [attempt]);

  return {
    status,
    progress,
    runtime,
    gpu,
    error,
    isReady: status === "ready",
    retry,
  };
}
