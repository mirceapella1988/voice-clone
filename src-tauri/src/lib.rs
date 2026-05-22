// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod setup;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use serde::Serialize;
use sysinfo::System;
use tauri::{Emitter, Manager, State};

struct SidecarState {
    child: Mutex<Option<std::process::Child>>,
    stdin: Mutex<Option<std::process::ChildStdin>>,
    startup_error: Mutex<Option<String>>,
}

struct SystemMetricsState {
    system: Mutex<System>,
}

#[derive(Serialize)]
struct SystemUsage {
    cpu_percent: f32,
    ram_percent: f32,
    gpu_percent: Option<f32>,
}

fn resolve_sidecar_script(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();

    if cfg!(debug_assertions) {
        if let Ok(current_dir) = std::env::current_dir() {
            candidates.push(current_dir.join("src-python").join("sidecar.py"));
            if current_dir
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case("src-tauri"))
            {
                if let Some(parent) = current_dir.parent() {
                    candidates.push(parent.join("src-python").join("sidecar.py"));
                }
            }
        }

        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let manifest_dir = PathBuf::from(manifest_dir);
            if let Some(parent) = manifest_dir.parent() {
                candidates.push(parent.join("src-python").join("sidecar.py"));
            }
        }
    }

    if let Ok(resource_dir) = app_handle.path().resource_dir() {
        candidates.push(resource_dir.join("src-python").join("sidecar.py"));
    }

    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| "Cannot find src-python/sidecar.py in app resources".to_string())
}

fn sidecar_is_running(state: &SidecarState) -> Result<bool, String> {
    let mut child_guard = state.child.lock().map_err(|e| e.to_string())?;
    if let Some(child) = child_guard.as_mut() {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            *child_guard = None;
            *state.stdin.lock().map_err(|e| e.to_string())? = None;
            *state.startup_error.lock().map_err(|e| e.to_string())? =
                Some(format!("Python sidecar exited: {status}"));
            return Ok(false);
        }
        return Ok(true);
    }
    Ok(false)
}

fn prepend_path_dir(command: &mut Command, dir: &Path) -> Result<(), String> {
    let existing = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![dir.to_path_buf()];
    paths.extend(std::env::split_paths(&existing));
    let joined = std::env::join_paths(paths).map_err(|e| format!("Failed to build PATH: {e}"))?;
    command.env("PATH", joined);
    Ok(())
}

fn spawn_runtime_sidecar(
    app_handle: &tauri::AppHandle,
    state: &SidecarState,
) -> Result<(), String> {
    if sidecar_is_running(state)? {
        return Ok(());
    }

    let base = setup::get_base_dir(app_handle)?;
    let python = setup::python_path(&base);
    let ffmpeg_dir = setup::ffmpeg_dir(&base);
    let models_dir = setup::models_dir(app_handle)?;
    let sidecar_script = resolve_sidecar_script(app_handle)?;

    if !python.is_file() {
        return Err(format!(
            "Python runtime is not installed at {}",
            python.display()
        ));
    }
    if !setup::ffmpeg_path(&base).is_file() {
        return Err(format!(
            "FFmpeg runtime is not installed at {}",
            setup::ffmpeg_path(&base).display()
        ));
    }

    std::fs::create_dir_all(&models_dir).map_err(|e| {
        format!(
            "Failed to create folder cache directory {}: {e}",
            models_dir.display()
        )
    })?;

    let mut command = Command::new(&python);
    command
        .arg(&sidecar_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("VOICECLONE_MODELS", &models_dir)
        .env("APP_MODEL_DIR", &models_dir);

    prepend_path_dir(&mut command, &ffmpeg_dir)?;

    if let Some(script_dir) = sidecar_script.parent() {
        command.current_dir(script_dir);
        command.env("PYTHONPATH", script_dir);
    }
    if let Ok(resource_dir) = app_handle.path().resource_dir() {
        command.env("APP_RESOURCES_DIR", resource_dir);
    }

    hide_subprocess_window(&mut command);

    match command.spawn() {
        Ok(mut child) => {
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| "Failed to open Python sidecar stdin".to_string())?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| "Failed to open Python sidecar stdout".to_string())?;
            let stderr = child.stderr.take();

            *state.child.lock().map_err(|e| e.to_string())? = Some(child);
            *state.stdin.lock().map_err(|e| e.to_string())? = Some(stdin);
            *state.startup_error.lock().map_err(|e| e.to_string())? = None;

            let app_handle_clone = app_handle.clone();
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                let reader = BufReader::new(stdout);
                for line in reader.lines().map_while(Result::ok) {
                    let _ = app_handle_clone.emit("sidecar-event", line);
                }
            });

            if let Some(stderr) = stderr {
                let app_handle_clone = app_handle.clone();
                std::thread::spawn(move || {
                    use std::io::{BufRead, BufReader};
                    let reader = BufReader::new(stderr);
                    for line in reader.lines().map_while(Result::ok) {
                        emit_sidecar_message(&app_handle_clone, "stderr", &line);
                    }
                });
            }

            Ok(())
        }
        Err(e) => {
            let msg = format!("Failed to start Python sidecar: {e}");
            *state.startup_error.lock().map_err(|e| e.to_string())? = Some(msg.clone());
            Err(msg)
        }
    }
}

#[tauri::command]
fn start_runtime_sidecar(
    app_handle: tauri::AppHandle,
    state: State<'_, SidecarState>,
) -> Result<(), String> {
    spawn_runtime_sidecar(&app_handle, &state)
}

#[tauri::command]
fn send_to_sidecar(state: State<'_, SidecarState>, msg: String) -> Result<(), String> {
    let msg = msg.trim();
    if msg.is_empty() {
        return Err("Refusing to send an empty command to Python sidecar".to_string());
    }
    match serde_json::from_str::<serde_json::Value>(msg)
        .map_err(|e| format!("Refusing to send invalid JSON to Python sidecar: {e}"))?
    {
        serde_json::Value::Object(_) => {}
        _ => return Err("Refusing to send non-object JSON to Python sidecar".to_string()),
    }

    {
        let mut child_guard = state.child.lock().map_err(|e| e.to_string())?;
        if let Some(child) = child_guard.as_mut() {
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                *child_guard = None;
                *state.stdin.lock().map_err(|e| e.to_string())? = None;
                *state.startup_error.lock().map_err(|e| e.to_string())? =
                    Some(format!("Sidecar exited before handling command: {status}"));
                return Err(format!("Sidecar exited before handling command: {status}"));
            }
        } else {
            if let Some(error) = state
                .startup_error
                .lock()
                .map_err(|e| e.to_string())?
                .as_ref()
            {
                return Err(format!("Python sidecar is not running: {error}"));
            }
            return Err("Python sidecar is not running".to_string());
        }
    }

    let mut stdin_guard = state.stdin.lock().map_err(|e| e.to_string())?;
    if let Some(ref mut stdin) = *stdin_guard {
        let msg_bytes = msg.as_bytes();
        if let Err(e) = write!(stdin, "Content-Length: {}\n\n", msg_bytes.len())
            .and_then(|_| stdin.write_all(msg_bytes))
            .and_then(|_| stdin.write_all(b"\n"))
            .and_then(|_| stdin.flush())
        {
            *stdin_guard = None;
            return Err(format!("Failed to write to Python sidecar: {e}"));
        }
        Ok(())
    } else {
        Err("Python sidecar is not running".to_string())
    }
}

#[tauri::command]
fn save_audio_file(path: String, bytes: Vec<u8>) -> Result<(), String> {
    if bytes.is_empty() {
        return Err("Refusing to save an empty audio file".to_string());
    }

    let mut path = PathBuf::from(path);
    if path.extension().and_then(|extension| extension.to_str()) != Some("wav") {
        path.set_extension("wav");
    }

    std::fs::write(&path, bytes)
        .map_err(|e| format!("Failed to save WAV file {}: {e}", path.display()))
}

fn read_nvidia_gpu_usage() -> Option<f32> {
    let mut command = Command::new("nvidia-smi");
    command.args([
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]);
    hide_subprocess_window(&mut command);

    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.trim().parse::<f32>().ok())
}

fn parse_percent_metric(output: &str, key: &str) -> Option<f32> {
    let needle = format!("\"{key}\"=");
    output.match_indices(&needle).filter_map(|(index, _)| {
        let value_start = index + needle.len();
        let value = output[value_start..]
            .chars()
            .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
            .collect::<String>();
        value.parse::<f32>().ok()
    }).max_by(|left, right| left.total_cmp(right))
}

#[cfg(target_os = "macos")]
fn read_apple_gpu_usage() -> Option<f32> {
    let output = Command::new("ioreg")
        .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_percent_metric(&stdout, "Device Utilization %")
        .or_else(|| parse_percent_metric(&stdout, "Renderer Utilization %"))
        .or_else(|| parse_percent_metric(&stdout, "Tiler Utilization %"))
}

#[cfg(not(target_os = "macos"))]
fn read_apple_gpu_usage() -> Option<f32> {
    None
}

fn read_gpu_usage() -> Option<f32> {
    read_apple_gpu_usage().or_else(read_nvidia_gpu_usage)
}

#[tauri::command]
fn get_system_usage(state: State<'_, SystemMetricsState>) -> Result<SystemUsage, String> {
    let mut system = state.system.lock().map_err(|e| e.to_string())?;
    system.refresh_memory();
    system.refresh_cpu_usage();

    let total_memory = system.total_memory();
    let ram_percent = if total_memory > 0 {
        (system.used_memory() as f32 / total_memory as f32) * 100.0
    } else {
        0.0
    };

    Ok(SystemUsage {
        cpu_percent: system.global_cpu_usage(),
        ram_percent,
        gpu_percent: read_gpu_usage(),
    })
}

fn emit_sidecar_message(app_handle: &tauri::AppHandle, event_type: &str, message: &str) {
    let payload = serde_json::json!({
        "type": event_type,
        "message": message,
    })
    .to_string();
    let _ = app_handle.emit("sidecar-event", payload);
}

#[cfg(target_os = "windows")]
fn hide_subprocess_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;

    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(target_os = "windows"))]
fn hide_subprocess_window(_command: &mut Command) {}

#[cfg(test)]
mod tests {
    use super::{parse_percent_metric, prepend_path_dir};
    use std::path::Path;
    use std::process::Command;

    #[test]
    fn prepend_path_dir_keeps_existing_path_entries() {
        let mut command = Command::new("python");
        prepend_path_dir(&mut command, Path::new("/tmp/voiceclone-ffmpeg")).unwrap();

        let path = command
            .get_envs()
            .find(|(key, _)| *key == "PATH")
            .and_then(|(_, value)| value)
            .unwrap()
            .to_string_lossy()
            .to_string();

        assert!(path.starts_with("/tmp/voiceclone-ffmpeg"));
    }

    #[test]
    fn parse_percent_metric_reads_macos_ioreg_gpu_utilization() {
        let output = r#""PerformanceStatistics" = {"Tiler Utilization %"=20,"Renderer Utilization %"=29,"Device Utilization %"=33}"#;

        assert_eq!(parse_percent_metric(output, "Device Utilization %"), Some(33.0));
        assert_eq!(parse_percent_metric(output, "Renderer Utilization %"), Some(29.0));
    }

    #[test]
    fn parse_percent_metric_uses_highest_value_when_multiple_gpus_exist() {
        let output = r#""Device Utilization %"=12 "Device Utilization %"=47"#;

        assert_eq!(parse_percent_metric(output, "Device Utilization %"), Some(47.0));
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(SidecarState {
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            startup_error: Mutex::new(None),
        })
        .manage(SystemMetricsState {
            system: Mutex::new(System::new_all()),
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.state::<SidecarState>();
                let mut child_guard = state.child.lock().unwrap();
                if let Some(mut child) = child_guard.take() {
                    let _ = child.kill();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_system_usage,
            save_audio_file,
            send_to_sidecar,
            start_runtime_sidecar,
            setup::check_runtime,
            setup::get_gpu_type,
            setup::install_runtime
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
