// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::{Emitter, Manager, State};

struct SidecarState {
    child: Mutex<Option<std::process::Child>>,
    stdin: Mutex<Option<std::process::ChildStdin>>,
}

fn push_sidecar_candidate(candidates: &mut Vec<PathBuf>, path: PathBuf) {
    if !candidates.iter().any(|candidate| candidate == &path) {
        candidates.push(path);
    }
}

fn is_usable_sidecar_candidate(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }

    if path.extension().is_some_and(|ext| ext == "py") {
        return cfg!(debug_assertions);
    }

    path.metadata()
        .map(|metadata| metadata.len() > 1024)
        .unwrap_or(false)
}

fn voiceclone_model_dir_from_home(home_dir: PathBuf) -> PathBuf {
    home_dir.join(".voiceclone").join("models")
}

fn resolve_voiceclone_model_dir() -> Result<PathBuf, String> {
    let home_dir = if cfg!(target_os = "windows") {
        std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
    } else {
        std::env::var_os("HOME")
    }
    .map(PathBuf::from)
    .filter(|path| !path.as_os_str().is_empty())
    .ok_or_else(|| "Failed to resolve user home directory for .voiceclone model cache".to_string())?;

    Ok(voiceclone_model_dir_from_home(home_dir))
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
    use super::{is_usable_sidecar_candidate, push_sidecar_candidate, voiceclone_model_dir_from_home};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        std::env::temp_dir().join(format!("voice-clone-{name}-{millis}"))
    }

    #[test]
    fn sidecar_candidate_filter_skips_dummy_binary() {
        let dir = temp_dir("dummy-sidecar");
        fs::create_dir_all(&dir).unwrap();
        let dummy = dir.join("sidecar.exe");
        fs::write(&dummy, b"# dummy\n").unwrap();

        assert!(!is_usable_sidecar_candidate(&dummy));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sidecar_candidate_filter_accepts_dev_script_in_debug() {
        let dir = temp_dir("script-sidecar");
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("sidecar.py");
        fs::write(&script, b"print('ok')\n").unwrap();

        assert!(is_usable_sidecar_candidate(&script));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn push_sidecar_candidate_keeps_order_without_duplicates() {
        let mut candidates = Vec::new();
        let first = std::path::PathBuf::from("sidecar.exe");
        let second = std::path::PathBuf::from("sidecar-x86_64-pc-windows-msvc.exe");

        push_sidecar_candidate(&mut candidates, first.clone());
        push_sidecar_candidate(&mut candidates, second.clone());
        push_sidecar_candidate(&mut candidates, first.clone());

        assert_eq!(candidates, vec![first, second]);
    }

    #[test]
    fn voiceclone_model_dir_uses_hidden_home_directory() {
        let home = std::path::PathBuf::from("/Users/example");

        assert_eq!(
            voiceclone_model_dir_from_home(home),
            std::path::PathBuf::from("/Users/example/.voiceclone/models")
        );
    }
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
                return Err(format!("Sidecar exited before handling command: {status}"));
            }
        } else {
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(SidecarState {
            child: Mutex::new(None),
            stdin: Mutex::new(None),
        })
        .setup(|app| {
            let target_triple = if cfg!(target_os = "windows") {
                "x86_64-pc-windows-msvc"
            } else if cfg!(target_os = "macos") {
                if cfg!(target_arch = "aarch64") {
                    "aarch64-apple-darwin"
                } else {
                    "x86_64-apple-darwin"
                }
            } else {
                "x86_64-unknown-linux-gnu"
            };

            #[cfg(target_os = "windows")]
            let ext = ".exe";
            #[cfg(not(target_os = "windows"))]
            let ext = "";

            let binary_name = format!("sidecar-{}{}", target_triple, ext);
            let runtime_binary_name = format!("sidecar{}", ext);

            let mut sidecar_candidates = Vec::new();
            if cfg!(debug_assertions) {
                if let Ok(current_dir) = std::env::current_dir() {
                    if current_dir.ends_with("src-tauri") {
                        if let Some(parent) = current_dir.parent() {
                            push_sidecar_candidate(&mut sidecar_candidates, parent.join("src-python").join("sidecar.py"));
                        }
                    }
                    push_sidecar_candidate(&mut sidecar_candidates, current_dir.join("src-python").join("sidecar.py"));
                }
            }
            if let Ok(current_exe) = std::env::current_exe() {
                if let Some(exe_dir) = current_exe.parent() {
                    push_sidecar_candidate(&mut sidecar_candidates, exe_dir.join(&runtime_binary_name));
                    push_sidecar_candidate(&mut sidecar_candidates, exe_dir.join(&binary_name));
                }
            }
            if let Ok(resource_dir) = app.path().resource_dir() {
                push_sidecar_candidate(&mut sidecar_candidates, resource_dir.join(&runtime_binary_name));
                push_sidecar_candidate(&mut sidecar_candidates, resource_dir.join(&binary_name));
                push_sidecar_candidate(&mut sidecar_candidates, resource_dir.join("binaries").join(&binary_name));
                if cfg!(debug_assertions) {
                    push_sidecar_candidate(&mut sidecar_candidates, resource_dir.join("src-python").join("sidecar.py"));
                }
            }

            let Some(sidecar_path) = sidecar_candidates
                .into_iter()
                .find(|path| is_usable_sidecar_candidate(path))
            else {
                let message = if cfg!(debug_assertions) {
                    "Failed to find a usable Python sidecar binary or dev script"
                } else {
                    "Failed to find bundled Python sidecar binary. Please reinstall the app or rebuild the Windows release."
                };
                eprintln!("{message}");
                let _ = app.emit(
                    "sidecar-event",
                    format!(r#"{{"type":"error","message":"{message}"}}"#),
                );
                return Ok(());
            };
            let sidecar_dir = sidecar_path.parent().map(|path| path.to_path_buf());
            let is_script = sidecar_path.extension().map_or(false, |ext| ext == "py");
            let resource_dir = app.path().resource_dir().ok();
            let app_model_dir = match resolve_voiceclone_model_dir() {
                Ok(dir) => {
                    if let Err(e) = std::fs::create_dir_all(&dir) {
                        let message = format!("Failed to create .voiceclone model cache directory: {e}");
                        eprintln!("{message}");
                        let _ = app.emit(
                            "sidecar-event",
                            serde_json::json!({"type":"error","message":message}).to_string(),
                        );
                        return Ok(());
                    }
                    Some(dir)
                }
                Err(message) => {
                    eprintln!("{message}");
                    let _ = app.emit(
                        "sidecar-event",
                        serde_json::json!({"type":"error","message":message}).to_string(),
                    );
                    return Ok(());
                }
            };

            let child_process = if !is_script {
                // Chạy trực tiếp binary sidecar đóng gói
                let mut cmd = Command::new(&sidecar_path);
                cmd.stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .env("PYTHONUNBUFFERED", "1")
                    .env("PYTHONIOENCODING", "utf-8");
                if let Some(resource_dir) = &resource_dir {
                    cmd.env("APP_RESOURCES_DIR", resource_dir);
                }
                if let Some(model_dir) = &app_model_dir {
                    cmd.env("APP_MODEL_DIR", model_dir);
                }
                if let Some(dir) = &sidecar_dir {
                    cmd.current_dir(dir);
                }
                hide_subprocess_window(&mut cmd);
                cmd.spawn()
            } else {
                // Chạy script python (chế độ dev)
                let mut python3 = Command::new("python3");
                python3
                    .arg(&sidecar_path)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .env("PYTHONUNBUFFERED", "1")
                    .env("PYTHONIOENCODING", "utf-8");
                if let Some(resource_dir) = &resource_dir {
                    python3.env("APP_RESOURCES_DIR", resource_dir);
                }
                if let Some(model_dir) = &app_model_dir {
                    python3.env("APP_MODEL_DIR", model_dir);
                }
                if let Some(dir) = &sidecar_dir {
                    python3.current_dir(dir);
                }
                hide_subprocess_window(&mut python3);
                let spawn_res = python3.spawn();

                if spawn_res.is_err() {
                    let mut python = Command::new("python");
                    python
                        .arg(&sidecar_path)
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .env("PYTHONUNBUFFERED", "1")
                        .env("PYTHONIOENCODING", "utf-8");
                    if let Some(resource_dir) = &resource_dir {
                        python.env("APP_RESOURCES_DIR", resource_dir);
                    }
                    if let Some(model_dir) = &app_model_dir {
                        python.env("APP_MODEL_DIR", model_dir);
                    }
                    if let Some(dir) = &sidecar_dir {
                        python.current_dir(dir);
                    }
                    hide_subprocess_window(&mut python);
                    python.spawn()
                } else {
                    spawn_res
                }
            };

            match child_process {
                Ok(mut child) => {
                    let stdin = child.stdin.take().unwrap();
                    let stdout = child.stdout.take().unwrap();
                    let stderr = child.stderr.take();

                    // Lưu child và stdin vào state
                    let state = app.state::<SidecarState>();
                    *state.child.lock().unwrap() = Some(child);
                    *state.stdin.lock().unwrap() = Some(stdin);

                    // Spawn thread đọc stdout bất đồng bộ
                    let app_handle = app.handle().clone();
                    std::thread::spawn(move || {
                        use std::io::{BufRead, BufReader};
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            if let Ok(l) = line {
                                // Gửi event sidecar-event tới frontend
                                let _ = app_handle.emit("sidecar-event", l);
                            }
                        }
                    });

                    if let Some(stderr) = stderr {
                        let app_handle = app.handle().clone();
                        std::thread::spawn(move || {
                            use std::io::{BufRead, BufReader};
                            let reader = BufReader::new(stderr);
                            for line in reader.lines() {
                                if let Ok(l) = line {
                                    emit_sidecar_message(&app_handle, "stderr", &l);
                                }
                            }
                        });
                    }
                }
                Err(e) => {
                    let message = format!("Failed to start Python sidecar: {e}");
                    eprintln!("{message}");
                    let _ = app.emit(
                        "sidecar-event",
                        serde_json::json!({"type":"error","message":message}).to_string(),
                    );
                }
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // Kill child process khi cửa sổ bị đóng
                let state = window.state::<SidecarState>();
                let mut child_guard = state.child.lock().unwrap();
                if let Some(mut child) = child_guard.take() {
                    let _ = child.kill();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![send_to_sidecar])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
