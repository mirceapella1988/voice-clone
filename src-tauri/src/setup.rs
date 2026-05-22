use flate2::read::GzDecoder;
use serde::Serialize;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tar::Archive;
use tauri::{Emitter, Manager};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use zip::ZipArchive;

const PYTHON_WINDOWS_URL: &str =
    "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip";
const GET_PIP_URL: &str = "https://bootstrap.pypa.io/get-pip.py";
const FFMPEG_WINDOWS_URL: &str = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip";
const PYTHON_MACOS_AARCH64_URL: &str = "https://github.com/astral-sh/python-build-standalone/releases/download/20240713/cpython-3.11.9%2B20240713-aarch64-apple-darwin-install_only.tar.gz";
const PYTHON_MACOS_X86_64_URL: &str = "https://github.com/astral-sh/python-build-standalone/releases/download/20240713/cpython-3.11.9%2B20240713-x86_64-apple-darwin-install_only.tar.gz";
const FFMPEG_MACOS_URL: &str = "https://evermeet.cx/ffmpeg/ffmpeg-6.1.zip";
const DOWNLOAD_RETRIES: usize = 3;

#[derive(Debug, Serialize)]
pub struct RuntimeStatus {
    pub has_python: bool,
    pub has_ffmpeg: bool,
    pub has_torch: bool,
    pub models_path: String,
}

#[derive(Debug, Serialize)]
struct InstallProgress {
    stage: &'static str,
    message: String,
    percent: u8,
}

pub fn get_base_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    let local_data = app_handle
        .path()
        .app_local_data_dir()
        .or_else(|_| app_handle.path().app_data_dir())
        .map_err(|e| format!("Failed to resolve app local data directory: {e}"))?;

    if cfg!(target_os = "windows") {
        if let Some(parent) = local_data.parent() {
            Ok(parent.join("Voice Clone"))
        } else {
            Ok(local_data)
        }
    } else {
        Ok(local_data)
    }
}

pub fn models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(get_base_dir(app_handle)?.join("models"))
}

pub fn python_path(base: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        base.join("runtime").join("python").join("python.exe")
    } else {
        base.join("runtime")
            .join("python")
            .join("bin")
            .join("python3")
    }
}

pub fn ffmpeg_dir(base: &Path) -> PathBuf {
    base.join("runtime").join("ffmpeg")
}

pub fn ffmpeg_path(base: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        ffmpeg_dir(base).join("ffmpeg.exe")
    } else {
        ffmpeg_dir(base).join("ffmpeg")
    }
}

fn python_site_packages_dir(base: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        base.join("runtime")
            .join("python")
            .join("Lib")
            .join("site-packages")
    } else {
        base.join("runtime")
            .join("python")
            .join("lib")
            .join("python3.11")
            .join("site-packages")
    }
}

pub fn torch_dir(base: &Path) -> PathBuf {
    python_site_packages_dir(base).join("torch")
}

#[tauri::command]
pub async fn check_runtime(app_handle: tauri::AppHandle) -> Result<RuntimeStatus, String> {
    let base = get_base_dir(&app_handle)?;
    let models = models_dir(&app_handle)?;

    Ok(RuntimeStatus {
        has_python: python_path(&base).is_file(),
        has_ffmpeg: ffmpeg_path(&base).is_file(),
        has_torch: torch_dir(&base).is_dir(),
        models_path: models.to_string_lossy().to_string(),
    })
}

#[tauri::command]
pub async fn get_gpu_type() -> Result<String, String> {
    if cfg!(target_os = "macos") {
        return Ok("mps".to_string());
    }
    if !cfg!(target_os = "windows") {
        return Ok("cpu".to_string());
    }

    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
        ])
        .output()
        .await;

    let Ok(output) = output else {
        return Ok("cpu".to_string());
    };
    if !output.status.success() {
        return Ok("cpu".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(gpu_type_from_driver_version(&stdout))
}

#[tauri::command]
pub async fn install_runtime(app_handle: tauri::AppHandle, gpu: String) -> Result<(), String> {
    if !(cfg!(target_os = "windows") || cfg!(target_os = "macos")) {
        return Err("Voice Clone runtime installation supports Windows and macOS only".to_string());
    }

    let base = get_base_dir(&app_handle)?;
    let runtime = base.join("runtime");
    let python_dir = runtime.join("python");
    let ffmpeg = ffmpeg_dir(&base);
    let models = models_dir(&app_handle)?;
    let downloads = runtime.join("downloads");

    tokio::fs::create_dir_all(&python_dir)
        .await
        .map_err(|e| format!("Failed to create Python runtime directory: {e}"))?;
    tokio::fs::create_dir_all(&ffmpeg)
        .await
        .map_err(|e| format!("Failed to create FFmpeg runtime directory: {e}"))?;
    tokio::fs::create_dir_all(&models)
        .await
        .map_err(|e| format!("Failed to create folder cache directory: {e}"))?;
    tokio::fs::create_dir_all(&downloads)
        .await
        .map_err(|e| format!("Failed to create downloads directory: {e}"))?;

    emit_progress(
        &app_handle,
        "prepare",
        "Preparing runtime directories...",
        2,
    );

    if cfg!(target_os = "windows") {
        install_windows_runtime(&app_handle, &downloads, &python_dir, &ffmpeg, &gpu).await?;
    } else {
        install_macos_runtime(&app_handle, &downloads, &python_dir, &ffmpeg).await?;
    }

    emit_progress(
        &app_handle,
        "complete",
        "Runtime installation complete.",
        100,
    );
    let _ = app_handle.emit(
        "install-complete",
        serde_json::json!({
            "models_path": models.to_string_lossy().to_string()
        })
        .to_string(),
    );

    Ok(())
}

async fn install_windows_runtime(
    app_handle: &tauri::AppHandle,
    downloads: &Path,
    python_dir: &Path,
    ffmpeg_dir: &Path,
    gpu: &str,
) -> Result<(), String> {
    let python_zip = downloads.join("python-embed.zip");
    download_with_retry(
        app_handle,
        PYTHON_WINDOWS_URL,
        &python_zip,
        "Downloading Python runtime...",
        5,
        20,
    )
    .await?;
    extract_zip_all(&python_zip, python_dir).await?;
    fix_windows_python_pth(python_dir).await?;

    let get_pip = python_dir.join("get-pip.py");
    download_with_retry(
        app_handle,
        GET_PIP_URL,
        &get_pip,
        "Downloading pip bootstrap...",
        20,
        25,
    )
    .await?;
    emit_progress(app_handle, "python", "Installing pip...", 26);
    run_command(
        app_handle,
        &python_dir.join("python.exe"),
        &[get_pip.to_string_lossy().as_ref()],
        None,
    )
    .await?;

    let ffmpeg_zip = downloads.join("ffmpeg-windows.zip");
    download_with_retry(
        app_handle,
        FFMPEG_WINDOWS_URL,
        &ffmpeg_zip,
        "Downloading FFmpeg...",
        30,
        42,
    )
    .await?;
    extract_ffmpeg_from_zip(&ffmpeg_zip, ffmpeg_dir, true).await?;

    install_python_packages(app_handle, &python_dir.join("python.exe"), gpu).await
}

async fn install_macos_runtime(
    app_handle: &tauri::AppHandle,
    downloads: &Path,
    python_dir: &Path,
    ffmpeg_dir: &Path,
) -> Result<(), String> {
    let python_url = if cfg!(target_arch = "aarch64") {
        PYTHON_MACOS_AARCH64_URL
    } else {
        PYTHON_MACOS_X86_64_URL
    };
    let python_tar = downloads.join("python-standalone.tar.gz");
    download_with_retry(
        app_handle,
        python_url,
        &python_tar,
        "Downloading Python runtime...",
        5,
        24,
    )
    .await?;
    extract_python_standalone_tar(&python_tar, python_dir).await?;
    chmod_executable(&python_dir.join("bin").join("python3")).await?;

    let python = python_dir.join("bin").join("python3");
    emit_progress(app_handle, "python", "Bootstrapping pip...", 28);
    run_command(app_handle, &python, &["-m", "ensurepip"], None).await?;

    let ffmpeg_zip = downloads.join("ffmpeg-macos.zip");
    download_with_retry(
        app_handle,
        FFMPEG_MACOS_URL,
        &ffmpeg_zip,
        "Downloading FFmpeg...",
        30,
        42,
    )
    .await?;
    extract_ffmpeg_from_zip(&ffmpeg_zip, ffmpeg_dir, false).await?;
    chmod_executable(&ffmpeg_dir.join("ffmpeg")).await?;

    install_python_packages(app_handle, &python, "mps").await
}

async fn install_python_packages(
    app_handle: &tauri::AppHandle,
    python: &Path,
    gpu: &str,
) -> Result<(), String> {
    emit_progress(app_handle, "packages", "Upgrading pip...", 45);
    run_command(
        app_handle,
        python,
        &["-m", "pip", "install", "--upgrade", "pip"],
        None,
    )
    .await?;

    let torch_index = match gpu {
        "cu128" => Some("https://download.pytorch.org/whl/cu128"),
        "cu124" => Some("https://download.pytorch.org/whl/cu124"),
        "cpu" if cfg!(target_os = "windows") => Some("https://download.pytorch.org/whl/cpu"),
        _ => None,
    };

    emit_progress(
        app_handle,
        "packages",
        "Installing PyTorch runtime packages...",
        50,
    );
    let mut torch_args = vec!["-m", "pip", "install", "torch", "torchvision", "torchaudio"];
    if let Some(index) = torch_index {
        torch_args.push("--index-url");
        torch_args.push(index);
    }
    run_command(app_handle, python, &torch_args, None).await?;

    emit_progress(
        app_handle,
        "packages",
        "Installing Voice Clone Python packages...",
        78,
    );
    run_command(
        app_handle,
        python,
        &[
            "-m",
            "pip",
            "install",
            "omnivoice==0.1.5",
            "transformers==5.3.0",
            "accelerate",
            "pydub",
            "soundfile",
            "librosa",
            "huggingface_hub",
            "hf_xet",
            "hf_transfer",
            "faster-whisper",
            "numpy",
        ],
        None,
    )
    .await
}

async fn download_with_retry(
    app_handle: &tauri::AppHandle,
    url: &str,
    dest: &Path,
    message: &str,
    start_percent: u8,
    end_percent: u8,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1800))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_RETRIES {
        match download_once(
            &client,
            app_handle,
            url,
            dest,
            message,
            start_percent,
            end_percent,
            attempt,
        )
        .await
        {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_error = Some(e);
                let _ = tokio::fs::remove_file(dest).await;
                tokio::time::sleep(Duration::from_secs(attempt as u64)).await;
            }
        }
    }

    Err(last_error.unwrap_or_else(|| format!("Failed to download {url}")))
}

async fn download_once(
    client: &reqwest::Client,
    app_handle: &tauri::AppHandle,
    url: &str,
    dest: &Path,
    message: &str,
    start_percent: u8,
    end_percent: u8,
    attempt: usize,
) -> Result<(), String> {
    emit_progress(
        app_handle,
        "download",
        &format!("{message} (attempt {attempt}/{DOWNLOAD_RETRIES})"),
        start_percent,
    );

    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Download request failed for {url}: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Download failed for {url}: HTTP {}",
            response.status()
        ));
    }

    let total = response.content_length().unwrap_or(0);
    let mut downloaded = 0u64;
    let mut last_percent = start_percent;
    let mut file = tokio::fs::File::create(dest)
        .await
        .map_err(|e| format!("Failed to create download file {}: {e}", dest.display()))?;

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| format!("Failed reading download stream for {url}: {e}"))?
    {
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Failed writing download file {}: {e}", dest.display()))?;
        downloaded += chunk.len() as u64;

        if total > 0 {
            let range = end_percent.saturating_sub(start_percent) as u64;
            let pct = start_percent as u64 + (downloaded.saturating_mul(range) / total);
            let pct = pct.min(end_percent as u64) as u8;
            if pct != last_percent || downloaded == total {
                emit_progress(app_handle, "download", message, pct);
                last_percent = pct;
            }
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("Failed to flush download file {}: {e}", dest.display()))?;
    emit_progress(app_handle, "download", message, end_percent);
    Ok(())
}

async fn extract_zip_all(zip_path: &Path, extract_to: &Path) -> Result<(), String> {
    let zip = zip_path.to_path_buf();
    let dest = extract_to.to_path_buf();
    tokio::task::spawn_blocking(move || extract_zip_all_sync(&zip, &dest))
        .await
        .map_err(|e| format!("Zip extraction task failed: {e}"))?
}

fn extract_zip_all_sync(zip_path: &Path, extract_to: &Path) -> Result<(), String> {
    let file = File::open(zip_path)
        .map_err(|e| format!("Failed to open zip {}: {e}", zip_path.display()))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|e| format!("Failed to read zip {}: {e}", zip_path.display()))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("Failed to read zip entry: {e}"))?;
        let Some(path) = safe_zip_entry_path(entry.name()) else {
            continue;
        };
        let outpath = extract_to.join(path);
        if entry.is_dir() {
            std::fs::create_dir_all(&outpath)
                .map_err(|e| format!("Failed to create directory {}: {e}", outpath.display()))?;
            continue;
        }
        if let Some(parent) = outpath.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {}: {e}", parent.display()))?;
        }
        let mut outfile = File::create(&outpath)
            .map_err(|e| format!("Failed to create file {}: {e}", outpath.display()))?;
        std::io::copy(&mut entry, &mut outfile)
            .map_err(|e| format!("Failed to extract file {}: {e}", outpath.display()))?;
    }

    Ok(())
}

async fn extract_ffmpeg_from_zip(
    zip_path: &Path,
    ffmpeg_dir: &Path,
    windows: bool,
) -> Result<(), String> {
    let zip = zip_path.to_path_buf();
    let dest = ffmpeg_dir.to_path_buf();
    tokio::task::spawn_blocking(move || extract_ffmpeg_from_zip_sync(&zip, &dest, windows))
        .await
        .map_err(|e| format!("FFmpeg extraction task failed: {e}"))?
}

fn extract_ffmpeg_from_zip_sync(
    zip_path: &Path,
    ffmpeg_dir: &Path,
    windows: bool,
) -> Result<(), String> {
    let file = File::open(zip_path)
        .map_err(|e| format!("Failed to open FFmpeg zip {}: {e}", zip_path.display()))?;
    let mut archive =
        ZipArchive::new(file).map_err(|e| format!("Failed to read FFmpeg zip: {e}"))?;
    let wanted = if windows { "ffmpeg.exe" } else { "ffmpeg" };

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("Failed to read FFmpeg zip entry: {e}"))?;
        let name = entry.name().replace('\\', "/");
        if !name.ends_with(wanted) || (windows && !name.contains("/bin/")) {
            continue;
        }

        std::fs::create_dir_all(ffmpeg_dir).map_err(|e| {
            format!(
                "Failed to create FFmpeg directory {}: {e}",
                ffmpeg_dir.display()
            )
        })?;
        let target = ffmpeg_dir.join(wanted);
        let mut outfile = File::create(&target)
            .map_err(|e| format!("Failed to create FFmpeg binary {}: {e}", target.display()))?;
        std::io::copy(&mut entry, &mut outfile)
            .map_err(|e| format!("Failed to extract FFmpeg binary {}: {e}", target.display()))?;
        return Ok(());
    }

    Err(format!("Could not find {wanted} in FFmpeg archive"))
}

async fn extract_python_standalone_tar(tar_path: &Path, python_dir: &Path) -> Result<(), String> {
    let tar = tar_path.to_path_buf();
    let dest = python_dir.to_path_buf();
    tokio::task::spawn_blocking(move || extract_python_standalone_tar_sync(&tar, &dest))
        .await
        .map_err(|e| format!("Python tar extraction task failed: {e}"))?
}

fn extract_python_standalone_tar_sync(tar_path: &Path, python_dir: &Path) -> Result<(), String> {
    let file = File::open(tar_path)
        .map_err(|e| format!("Failed to open Python tar {}: {e}", tar_path.display()))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    for entry in archive
        .entries()
        .map_err(|e| format!("Failed to read Python tar: {e}"))?
    {
        let mut entry = entry.map_err(|e| format!("Failed to read Python tar entry: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("Failed to resolve Python tar entry path: {e}"))?;
        let stripped = strip_python_archive_prefix(&path);
        if stripped.as_os_str().is_empty() {
            continue;
        }
        let target = python_dir.join(stripped);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {}: {e}", parent.display()))?;
        }
        entry
            .unpack(&target)
            .map_err(|e| format!("Failed to extract Python file {}: {e}", target.display()))?;
    }

    Ok(())
}

fn strip_python_archive_prefix(path: &Path) -> PathBuf {
    let mut components = path.components();
    if components
        .next()
        .and_then(|component| component.as_os_str().to_str())
        .is_some_and(|name| name == "python")
    {
        components.as_path().to_path_buf()
    } else {
        path.to_path_buf()
    }
}

fn safe_zip_entry_path(name: &str) -> Option<PathBuf> {
    let path = Path::new(name);
    if path.is_absolute() {
        return None;
    }
    if path.components().any(|component| {
        matches!(
            component,
            std::path::Component::ParentDir | std::path::Component::Prefix(_)
        )
    }) {
        return None;
    }
    Some(path.to_path_buf())
}

async fn fix_windows_python_pth(python_dir: &Path) -> Result<(), String> {
    let pth_path = python_dir.join("python311._pth");
    let mut contents = tokio::fs::read_to_string(&pth_path)
        .await
        .map_err(|e| format!("Failed to read {}: {e}", pth_path.display()))?;
    contents = contents.replace("#import site", "import site");
    tokio::fs::write(&pth_path, contents)
        .await
        .map_err(|e| format!("Failed to update {}: {e}", pth_path.display()))
}

async fn chmod_executable(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = tokio::fs::metadata(path)
            .await
            .map_err(|e| format!("Failed to read permissions for {}: {e}", path.display()))?;
        let mut permissions = metadata.permissions();
        permissions.set_mode(0o755);
        tokio::fs::set_permissions(path, permissions)
            .await
            .map_err(|e| format!("Failed to chmod {}: {e}", path.display()))?;
    }
    Ok(())
}

async fn run_command(
    app_handle: &tauri::AppHandle,
    program: &Path,
    args: &[&str],
    current_dir: Option<&Path>,
) -> Result<(), String> {
    let label = format!("Running {} {}", program.display(), args.join(" "));
    let _ = app_handle;

    let mut command = Command::new(program);
    command.args(args);
    if let Some(current_dir) = current_dir {
        command.current_dir(current_dir);
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    let output = command
        .output()
        .await
        .map_err(|e| format!("Failed to run command {label}: {e}"))?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    Err(format!(
        "Command failed: {label}\nstdout: {stdout}\nstderr: {stderr}"
    ))
}

fn emit_progress(app_handle: &tauri::AppHandle, stage: &'static str, message: &str, percent: u8) {
    let _ = app_handle.emit(
        "install-progress",
        serde_json::to_string(&InstallProgress {
            stage,
            message: message.to_string(),
            percent: percent.min(100),
        })
        .unwrap_or_else(|_| "{}".to_string()),
    );
}

pub fn gpu_type_from_driver_version(version_output: &str) -> String {
    let first = version_output
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    let major = first
        .split(|c: char| !c.is_ascii_digit())
        .find(|part| !part.is_empty())
        .and_then(|part| part.parse::<u32>().ok())
        .unwrap_or(0);

    if major == 0 {
        "cpu".to_string()
    } else if major >= 560 {
        "cu128".to_string()
    } else {
        "cu124".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ffmpeg_path, gpu_type_from_driver_version, python_path, safe_zip_entry_path, torch_dir,
    };
    use std::path::PathBuf;

    #[test]
    fn runtime_paths_use_app_local_data_root() {
        let base = PathBuf::from("/Users/example/Library/Application Support/com.kiyooo.clone");

        if cfg!(target_os = "windows") {
            assert_eq!(python_path(&base), base.join("runtime/python/python.exe"));
            assert_eq!(ffmpeg_path(&base), base.join("runtime/ffmpeg/ffmpeg.exe"));
        } else {
            assert_eq!(python_path(&base), base.join("runtime/python/bin/python3"));
            assert_eq!(ffmpeg_path(&base), base.join("runtime/ffmpeg/ffmpeg"));
        }
    }

    #[test]
    fn torch_detection_path_targets_site_packages() {
        let base = PathBuf::from("/Users/example/Library/Application Support/com.kiyooo.clone");
        let torch = torch_dir(&base).to_string_lossy().replace('\\', "/");

        assert!(torch.ends_with("site-packages/torch"));
    }

    #[test]
    fn gpu_driver_version_selects_cuda_wheel() {
        assert_eq!(gpu_type_from_driver_version("561.09"), "cu128");
        assert_eq!(gpu_type_from_driver_version("552.44"), "cu124");
        assert_eq!(gpu_type_from_driver_version(""), "cpu");
    }

    #[test]
    fn zip_entry_path_rejects_traversal() {
        assert!(safe_zip_entry_path("python.exe").is_some());
        assert!(safe_zip_entry_path("dir/bin/ffmpeg").is_some());
        assert!(safe_zip_entry_path("../escape").is_none());
        assert!(safe_zip_entry_path("/absolute").is_none());
    }
}
