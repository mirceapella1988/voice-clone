import voiceCloneIcon from "../assets/voice-clone-icon.png";
import type { SetupState } from "../hooks/useSetup";

interface SetupScreenProps {
  setup: SetupState;
}

const formatRuntimePath = (modelsPath?: string) => {
  if (modelsPath) return modelsPath;
  if (navigator.platform.toLowerCase().includes("win")) {
    return "%USERPROFILE%\\.voiceclone\\models";
  }
  return "~/.voiceclone/models";
};

export function SetupScreen({ setup }: SetupScreenProps) {
  const isError = setup.status === "error";
  const isChecking = setup.status === "checking";
  const message = isError
    ? setup.error
    : setup.progress.message || "Đang chuẩn bị môi trường chạy lần đầu...";
  const percent = isChecking ? 0 : setup.progress.percent;

  return (
    <div className="setup-fullscreen-bg">
      <section className={`setup-glass-card setup-runtime-card${isError ? " error-card" : ""}`}>
        <img className={`setup-logo${isError ? "" : " pulse"}`} src={voiceCloneIcon} alt="Voice Clone" />
        <h2>{isError ? "Không thể chuẩn bị Runtime" : "Đang chuẩn bị môi trường chạy lần đầu..."}</h2>
        <p className="setup-runtime-message">{message}</p>

        {!isError && (
          <div className="setup-progress-container">
            <div className="setup-progress-track" aria-label="Runtime install progress">
              <div className="setup-progress-fill" style={{ width: `${percent}%` }} />
            </div>
            <div className="setup-progress-info">
              <span>{setup.progress.stage}</span>
              <span>{percent}%</span>
            </div>
          </div>
        )}

        <div className="setup-runtime-path">
          <span>Model cache</span>
          <code>{formatRuntimePath(setup.runtime?.models_path)}</code>
        </div>

        {setup.gpu && !isError && (
          <div className="setup-runtime-chip">Runtime profile: {setup.gpu.toUpperCase()}</div>
        )}

        {isError && (
          <button className="btn-primary" onClick={setup.retry}>
            Thử lại
          </button>
        )}
      </section>
    </div>
  );
}
