import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface RuntimeStatus {
  has_python: boolean;
  has_ffmpeg: boolean;
  has_torch: boolean;
  has_runtime_packages?: boolean;
  has_source_files?: boolean;
  is_ready?: boolean;
  models_path: string;
  missing_components?: string[];
  details?: string[];
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
  message: "Checking runtime environment...",
  percent: 0,
};

const parseProgressPayload = (payload: unknown): SetupProgress | null => {
  const data = typeof payload === "string" ? JSON.parse(payload) : payload;
  if (!data || typeof data !== "object") return null;
  const value = data as Partial<SetupProgress>;

  return {
    stage: String(value.stage || "installing"),
    message: String(value.message || "Preparing the runtime for first launch..."),
    percent: Math.max(0, Math.min(100, Number(value.percent || 0))),
  };
};

const runtimeIsReady = (runtime: RuntimeStatus) =>
  runtime.is_ready ?? (
    runtime.has_python &&
    runtime.has_ffmpeg &&
    runtime.has_torch &&
    runtime.has_runtime_packages !== false &&
    runtime.has_source_files !== false
  );

const formatMissingRuntime = (runtime: RuntimeStatus) => {
  const missing = runtime.missing_components?.filter(Boolean) || [];
  const detail = runtime.details?.filter(Boolean) || [];
  const missingText = missing.length > 0 ? missing.join(", ") : "unknown runtime components";
  const detailText = detail.length > 0 ? `\n\nDetails:\n${detail.join("\n\n")}` : "";
  return `${missingText}${detailText}`;
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
              appendLogRef.current?.(`Folder cache ready at ${data.models_path}`);
            }
          } catch {
            appendLogRef.current?.("Runtime installation completed.");
          }
        });

        const firstStatus = await invoke<RuntimeStatus>("check_runtime");
        if (disposed || activeRunRef.current !== runId) return;
        setRuntime(firstStatus);

        if (firstStatus.has_source_files === false) {
          throw new Error(`Application runtime files are missing: ${formatMissingRuntime(firstStatus)}`);
        }

        if (!runtimeIsReady(firstStatus)) {
          setStatus("installing");
          setProgress({
            stage: "detecting",
            message: `Repairing missing runtime components: ${firstStatus.missing_components?.join(", ") || "runtime packages"}...`,
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

        if (!runtimeIsReady(readyStatus)) {
          throw new Error(`Runtime repair finished but required components are still missing: ${formatMissingRuntime(readyStatus)}`);
        }

        setStatus("starting");
        setProgress({
          stage: "starting",
          message: "Starting Python runtime...",
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
