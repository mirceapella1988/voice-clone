import fs from 'fs';
import path from 'path';
import https from 'https';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rootDir = path.resolve(__dirname, '..');
const binariesDir = path.join(rootDir, 'src-tauri', 'binaries');

const targets = {
  'win32-x64': {
    triple: 'x86_64-pc-windows-msvc',
    url: 'https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-win32-x64',
    filename: 'ffmpeg.exe'
  },
  'darwin-arm64': {
    triple: 'aarch64-apple-darwin',
    url: 'https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-darwin-arm64',
    filename: 'ffmpeg'
  },
  'darwin-x64': {
    triple: 'x86_64-apple-darwin',
    url: 'https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-darwin-x64',
    filename: 'ffmpeg'
  }
};

function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);
    console.log(`Downloading ffmpeg from ${url}...`);

    const request = (targetUrl) => {
      https.get(targetUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      }, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          request(response.headers.location);
          return;
        }

        if (response.statusCode !== 200) {
          reject(new Error(`Failed to download ffmpeg, status code: ${response.statusCode}`));
          return;
        }

        response.pipe(file);

        file.on('finish', () => {
          file.close();
          console.log(`Successfully saved ffmpeg to ${destPath}`);
          resolve();
        });
      }).on('error', (err) => {
        fs.unlink(destPath, () => {});
        reject(err);
      });
    };

    request(url);
  });
}

async function main() {
  const platform = process.platform;
  const arch = process.arch;
  const key = `${platform}-${arch}`;

  const config = targets[key];
  if (!config) {
    console.log(`Unsupported platform/architecture for ffmpeg embedding: ${key}. Skipping.`);
    return;
  }

  const targetSubdir = path.join(binariesDir, `sidecar-${config.triple}`);

  if (fs.existsSync(targetSubdir) && fs.statSync(targetSubdir).isFile()) {
    console.log(`Converting single-file sidecar to directory structure for packaging: ${targetSubdir}`);
    const tempFile = `${targetSubdir}-temp`;
    fs.renameSync(targetSubdir, tempFile);
    fs.mkdirSync(targetSubdir, { recursive: true });
    fs.renameSync(tempFile, path.join(targetSubdir, `sidecar-${config.triple}`));
  }

  if (!fs.existsSync(targetSubdir)) {
    fs.mkdirSync(targetSubdir, { recursive: true });
  }

  const destPath = path.join(targetSubdir, config.filename);

  if (fs.existsSync(destPath) && fs.statSync(destPath).size > 1000000) {
    console.log(`ffmpeg static binary already exists at ${destPath}. Skipping download.`);
  } else {
    try {
      await downloadFile(config.url, destPath);
    } catch (error) {
      console.error(`Error downloading ffmpeg: ${error.message}`);
      process.exit(1);
    }
  }

  if (platform !== 'win32') {
    fs.chmodSync(destPath, '755');
    console.log(`Set executable permissions (755) for ${destPath}`);

    const sidecarPath = path.join(targetSubdir, `sidecar-${config.triple}`);
    if (fs.existsSync(sidecarPath)) {
      fs.chmodSync(sidecarPath, '755');
      console.log(`Set executable permissions (755) for ${sidecarPath}`);
    }
  }
}

main();
