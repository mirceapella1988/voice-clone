import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rootDir = path.resolve(__dirname, '..');
const binariesDir = path.join(rootDir, 'src-tauri', 'binaries');
const onedirDir = path.join(binariesDir, 'sidecar-x86_64-pc-windows-msvc');
const dummyExe = path.join(binariesDir, 'sidecar-x86_64-pc-windows-msvc.exe');

// Ensure directory src-tauri/binaries/sidecar-x86_64-pc-windows-msvc exists
if (!fs.existsSync(onedirDir)) {
  fs.mkdirSync(onedirDir, { recursive: true });
  console.log(`Created directory: ${onedirDir}`);
}

// Create placeholder file to satisfy Tauri's glob resource check
const placeholderFile = path.join(onedirDir, 'placeholder.txt');
if (!fs.existsSync(placeholderFile)) {
  fs.writeFileSync(placeholderFile, 'placeholder');
  console.log(`Created placeholder: ${placeholderFile}`);
}

// Create dummy exe if it doesn't exist to satisfy Tauri's externalBin check
if (!fs.existsSync(dummyExe)) {
  fs.writeFileSync(dummyExe, '');
  console.log(`Created dummy exe: ${dummyExe}`);
}

console.log('Tauri binaries pre-requisites prepared successfully.');
