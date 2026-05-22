// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{Emitter, Manager, State};

struct SidecarState {
    child: Mutex<Option<std::process::Child>>,
    stdin: Mutex<Option<std::process::ChildStdin>>,
    startup_error: Mutex<Option<String>>,
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

fn append_release_sidecar_candidates(
    candidates: &mut Vec<PathBuf>,
    base_dir: &Path,
    runtime_binary_name: &str,
    binary_name: &str,
) {
    push_sidecar_candidate(candidates, base_dir.join(runtime_binary_name));
    push_sidecar_candidate(candidates, base_dir.join(binary_name));
    push_sidecar_candidate(candidates, base_dir.join("binaries").join(runtime_binary_name));
    push_sidecar_candidate(candidates, base_dir.join("binaries").join(binary_name));

    // Support PyInstaller --onedir directory structure
    let binary_name_no_ext = Path::new(binary_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(binary_name);
    push_sidecar_candidate(candidates, base_dir.join(binary_name_no_ext).join(binary_name));
    push_sidecar_candidate(candidates, base_dir.join("binaries").join(binary_name_no_ext).join(binary_name));

    let runtime_binary_name_no_ext = Path::new(runtime_binary_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(runtime_binary_name);
    push_sidecar_candidate(candidates, base_dir.join(runtime_binary_name_no_ext).join(runtime_binary_name));
    push_sidecar_candidate(candidates, base_dir.join("binaries").join(runtime_binary_name_no_ext).join(runtime_binary_name));
}

fn find_sidecar_path(app_handle: &tauri::AppHandle) -> Option<PathBuf> {
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
            let in_src_tauri = current_dir
                .components()
                .last()
                .and_then(|c| c.as_os_str().to_str())
                .map_or(false, |name| name.eq_ignore_ascii_case("src-tauri"));
            if in_src_tauri {
                if let Some(parent) = current_dir.parent() {
                    push_sidecar_candidate(&mut sidecar_candidates, parent.join("src-python").join("sidecar.py"));
                }
            }
            push_sidecar_candidate(&mut sidecar_candidates, current_dir.join("src-python").join("sidecar.py"));
            if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
                push_sidecar_candidate(&mut sidecar_candidates, PathBuf::from(&manifest_dir).parent().unwrap_or(Path::new("")).join("src-python").join("sidecar.py"));
            }
        }
    }
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(exe_dir) = current_exe.parent() {
            append_release_sidecar_candidates(
                &mut sidecar_candidates,
                exe_dir,
                &runtime_binary_name,
                &binary_name,
            );
        }
    }
    if let Ok(resource_dir) = app_handle.path().resource_dir() {
        append_release_sidecar_candidates(
            &mut sidecar_candidates,
            &resource_dir,
            &runtime_binary_name,
            &binary_name,
        );
        if cfg!(debug_assertions) {
            push_sidecar_candidate(&mut sidecar_candidates, resource_dir.join("src-python").join("sidecar.py"));
        }
    }
    if let Ok(app_data_dir) = app_handle.path().app_data_dir() {
        let dynamic_bin_dir = app_data_dir.join("binaries");
        append_release_sidecar_candidates(
            &mut sidecar_candidates,
            &dynamic_bin_dir,
            &runtime_binary_name,
            &binary_name,
        );
    }

    sidecar_candidates
        .into_iter()
        .find(|path| is_usable_sidecar_candidate(path))
}

fn spawn_sidecar(
    app_handle: &tauri::AppHandle,
    state: &SidecarState,
    sidecar_path: &Path,
) -> Result<(), String> {
    let sidecar_dir = sidecar_path.parent().map(|path| path.to_path_buf());
    let is_script = sidecar_path.extension().map_or(false, |ext| ext == "py");
    let resource_dir = app_handle.path().resource_dir().ok();

    let app_model_dir = match resolve_voiceclone_model_dir() {
        Ok(dir) => {
            if let Err(e) = std::fs::create_dir_all(&dir) {
                return Err(format!("Failed to create .voiceclone model cache directory: {e}"));
            }
            Some(dir)
        }
        Err(e) => return Err(e),
    };

    let child_process = if !is_script {
        let mut cmd = Command::new(sidecar_path);
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
        fn is_python_available(cmd: &str) -> bool {
            Command::new(cmd).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok()
        }

        let python_cmd = if cfg!(target_os = "windows") {
            if is_python_available("py") {
                "py"
            } else if is_python_available("python") {
                "python"
            } else {
                "python3"
            }
        } else {
            if is_python_available("python3") {
                "python3"
            } else {
                "python"
            }
        };

        let mut python = Command::new(python_cmd);
        python
            .arg(sidecar_path)
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
    };

    match child_process {
        Ok(mut child) => {
            let stdin = child.stdin.take().unwrap();
            let stdout = child.stdout.take().unwrap();
            let stderr = child.stderr.take();

            *state.child.lock().unwrap() = Some(child);
            *state.stdin.lock().unwrap() = Some(stdin);
            *state.startup_error.lock().unwrap() = None;

            let app_handle_clone = app_handle.clone();
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    if let Ok(l) = line {
                        let _ = app_handle_clone.emit("sidecar-event", l);
                    }
                }
            });

            if let Some(stderr) = stderr {
                let app_handle_clone = app_handle.clone();
                std::thread::spawn(move || {
                    use std::io::{BufRead, BufReader};
                    let reader = BufReader::new(stderr);
                    for line in reader.lines() {
                        if let Ok(l) = line {
                            emit_sidecar_message(&app_handle_clone, "stderr", &l);
                        }
                    }
                });
            }
            Ok(())
        }
        Err(e) => {
            let msg = format!("Failed to start Python sidecar: {e}");
            *state.startup_error.lock().unwrap() = Some(msg.clone());
            Err(msg)
        }
    }
}

fn extract_zip(zip_path: &Path, extract_to: &Path) -> Result<(), String> {
    let file = File::open(zip_path).map_err(|e| format!("Failed to open zip file: {e}"))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| format!("Failed to read zip archive: {e}"))?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| format!("Zip entry read error: {e}"))?;
        let outpath = match file.enclosed_name() {
            Some(path) => extract_to.join(path),
            None => continue,
        };

        if file.name().ends_with('/') {
            std::fs::create_dir_all(&outpath)
                .map_err(|e| format!("Failed to create directory in zip: {e}"))?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    std::fs::create_dir_all(p)
                        .map_err(|e| format!("Failed to create parent dir: {e}"))?;
                }
            }
            let mut outfile = File::create(&outpath)
                .map_err(|e| format!("Failed to create file: {e}"))?;
            std::io::copy(&mut file, &mut outfile)
                .map_err(|e| format!("Failed to write file from zip: {e}"))?;
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Some(mode) = file.unix_mode() {
                let _ = std::fs::set_permissions(&outpath, std::fs::Permissions::from_mode(mode));
            }
        }
    }
    Ok(())
}

fn perform_download_and_install(app_handle: &tauri::AppHandle) -> Result<(), String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data dir: {e}"))?;
    let binaries_dir = app_data_dir.join("binaries");
    std::fs::create_dir_all(&binaries_dir)
        .map_err(|e| format!("Failed to create binaries directory: {e}"))?;

    let url = if cfg!(target_os = "windows") {
        "https://github.com/mirceapella1988/voice-clone/releases/download/v0.4.7/VoiceClone-Sidecar-Windows.zip"
    } else if cfg!(target_os = "macos") {
        "https://github.com/mirceapella1988/voice-clone/releases/download/v0.4.7/VoiceClone-Sidecar-MacOS.zip"
    } else {
        return Err("Unsupported OS for dynamic sidecar download".to_string());
    };

    let temp_file_path = binaries_dir.join("download.tmp");

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(1800))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

    let _ = app_handle.emit(
        "download-progress",
        serde_json::json!({
            "type": "connecting",
            "message": "Đang kết nối tới máy chủ..."
        })
        .to_string(),
    );

    let mut response = client
        .get(url)
        .send()
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Server returned error status: {}", response.status()));
    }

    let total_bytes = response
        .content_length()
        .ok_or_else(|| "Failed to get Content-Length from response".to_string())?;

    let mut file = File::create(&temp_file_path)
        .map_err(|e| format!("Failed to create temporary file: {e}"))?;

    let mut buffer = vec![0; 65536];
    let mut downloaded_bytes = 0;
    let start_time = Instant::now();
    let mut last_emit_time = Instant::now();

    loop {
        let bytes_read = response
            .read(&mut buffer)
            .map_err(|e| format!("Error downloading: {e}"))?;

        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read])
            .map_err(|e| format!("Failed to write to temporary file: {e}"))?;

        downloaded_bytes += bytes_read as u64;

        let now = Instant::now();
        if now.duration_since(last_emit_time) >= Duration::from_millis(200) || downloaded_bytes == total_bytes {
            let percent = (downloaded_bytes as f64 / total_bytes as f64 * 100.0) as u32;
            let elapsed = now.duration_since(start_time).as_secs_f64();
            let speed = if elapsed > 0.0 {
                (downloaded_bytes as f64 / 1024.0 / 1024.0) / elapsed
            } else {
                0.0
            };

            let _ = app_handle.emit(
                "download-progress",
                serde_json::json!({
                    "type": "downloading",
                    "downloaded": downloaded_bytes,
                    "total": total_bytes,
                    "percent": percent,
                    "speed": speed
                })
                .to_string(),
            );
            last_emit_time = now;
        }
    }

    drop(file);

    let _ = app_handle.emit(
        "download-progress",
        serde_json::json!({
            "type": "extracting",
            "message": "Đang giải nén bộ thư viện (quá trình này có thể mất 1-2 phút)..."
        })
        .to_string(),
    );

    extract_zip(&temp_file_path, &binaries_dir)?;

    let _ = std::fs::remove_file(&temp_file_path);

    let _ = app_handle.emit(
        "download-progress",
        serde_json::json!({
            "type": "completed",
            "message": "Cài đặt thành công!"
        })
        .to_string(),
    );

    Ok(())
}

#[tauri::command]
fn check_sidecar_installed(app_handle: tauri::AppHandle) -> bool {
    find_sidecar_path(&app_handle).is_some()
}

#[tauri::command]
fn download_and_install_sidecar(app_handle: tauri::AppHandle) -> Result<(), String> {
    std::thread::spawn(move || {
        if let Err(e) = perform_download_and_install(&app_handle) {
            let _ = app_handle.emit(
                "download-progress",
                serde_json::json!({
                    "type": "error",
                    "message": e
                })
                .to_string(),
            );
        }
    });
    Ok(())
}

#[tauri::command]
fn start_sidecar_dynamically(
    app_handle: tauri::AppHandle,
    state: State<'_, SidecarState>,
) -> Result<(), String> {
    if let Some(sidecar_path) = find_sidecar_path(&app_handle) {
        spawn_sidecar(&app_handle, &state, &sidecar_path)
    } else {
        Err("Cannot start sidecar: sidecar executable not found".to_string())
    }
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
    use super::{
        append_release_sidecar_candidates, is_usable_sidecar_candidate, push_sidecar_candidate,
        voiceclone_model_dir_from_home,
    };
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
    fn release_sidecar_candidates_include_tauri_resource_binary_folder() {
        let mut candidates = Vec::new();
        let base = std::path::Path::new("resources");

        append_release_sidecar_candidates(
            &mut candidates,
            base,
            "sidecar.exe",
            "sidecar-x86_64-pc-windows-msvc.exe",
        );

        assert_eq!(
            candidates,
            vec![
                std::path::PathBuf::from("resources/sidecar.exe"),
                std::path::PathBuf::from("resources/sidecar-x86_64-pc-windows-msvc.exe"),
                std::path::PathBuf::from("resources/binaries/sidecar.exe"),
                std::path::PathBuf::from("resources/binaries/sidecar-x86_64-pc-windows-msvc.exe"),
                std::path::PathBuf::from("resources/sidecar-x86_64-pc-windows-msvc/sidecar-x86_64-pc-windows-msvc.exe"),
                std::path::PathBuf::from("resources/binaries/sidecar-x86_64-pc-windows-msvc/sidecar-x86_64-pc-windows-msvc.exe"),
                std::path::PathBuf::from("resources/sidecar/sidecar.exe"),
                std::path::PathBuf::from("resources/binaries/sidecar/sidecar.exe"),
            ]
        );
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
                *state.startup_error.lock().map_err(|e| e.to_string())? =
                    Some(format!("Sidecar exited before handling command: {status}"));
                return Err(format!("Sidecar exited before handling command: {status}"));
            }
        } else {
            if let Some(error) = state.startup_error.lock().map_err(|e| e.to_string())?.as_ref() {
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(SidecarState {
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            startup_error: Mutex::new(None),
        })
        .setup(|app| {
            if let Some(sidecar_path) = find_sidecar_path(app.handle()) {
                let state = app.state::<SidecarState>();
                if let Err(e) = spawn_sidecar(app.handle(), &state, &sidecar_path) {
                    eprintln!("Failed to spawn sidecar: {e}");
                }
            } else {
                println!("Sidecar binary not found. Waiting for user setup.");
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
        .invoke_handler(tauri::generate_handler![
            send_to_sidecar,
            check_sidecar_installed,
            download_and_install_sidecar,
            start_sidecar_dynamically
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
