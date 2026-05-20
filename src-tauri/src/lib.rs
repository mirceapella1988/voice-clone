// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::{Emitter, Manager, State};

struct SidecarState {
    child: Mutex<Option<std::process::Child>>,
    stdin: Mutex<Option<std::process::ChildStdin>>,
}

#[tauri::command]
fn send_to_sidecar(state: State<'_, SidecarState>, msg: String) -> Result<(), String> {
    {
        let mut child_guard = state.child.lock().map_err(|e| e.to_string())?;
        if let Some(child) = child_guard.as_mut() {
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                *child_guard = None;
                *state.stdin.lock().map_err(|e| e.to_string())? = None;
                return Err(format!(
                    "Python sidecar exited before handling command: {status}"
                ));
            }
        } else {
            return Err("Python sidecar is not running".to_string());
        }
    }

    let mut stdin_guard = state.stdin.lock().map_err(|e| e.to_string())?;
    if let Some(ref mut stdin) = *stdin_guard {
        if let Err(e) = writeln!(stdin, "{}", msg).and_then(|_| stdin.flush()) {
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
            let mut sidecar_candidates = Vec::new();
            if let Ok(resource_dir) = app.path().resource_dir() {
                sidecar_candidates.push(resource_dir.join("src-python").join("sidecar.py"));
            }
            if let Ok(current_dir) = std::env::current_dir() {
                if current_dir.ends_with("src-tauri") {
                    if let Some(parent) = current_dir.parent() {
                        sidecar_candidates.push(parent.join("src-python").join("sidecar.py"));
                    }
                }
                sidecar_candidates.push(current_dir.join("src-python").join("sidecar.py"));
            }

            let Some(sidecar_path) = sidecar_candidates.into_iter().find(|path| path.exists())
            else {
                let message = "Failed to find Python sidecar in app resources or project directory";
                eprintln!("{message}");
                let _ = app.emit(
                    "sidecar-event",
                    format!(r#"{{"type":"error","message":"{message}"}}"#),
                );
                return Ok(());
            };
            let sidecar_dir = sidecar_path.parent().map(|path| path.to_path_buf());

            // Thử chạy python3, fallback sang python nếu lỗi
            let mut python3 = Command::new("python3");
            python3
                .arg(&sidecar_path)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit());
            if let Some(dir) = &sidecar_dir {
                python3.current_dir(dir);
            }
            let mut child_process = python3.spawn();

            if child_process.is_err() {
                let mut python = Command::new("python");
                python
                    .arg(&sidecar_path)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit());
                if let Some(dir) = &sidecar_dir {
                    python.current_dir(dir);
                }
                child_process = python.spawn();
            }

            match child_process {
                Ok(mut child) => {
                    let stdin = child.stdin.take().unwrap();
                    let stdout = child.stdout.take().unwrap();

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
                }
                Err(e) => {
                    eprintln!("Failed to start Python sidecar: {}", e);
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
