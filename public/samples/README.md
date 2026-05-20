# VieNeu-TTS Preset Reference Voices

These samples were downloaded from the Hugging Face Space
`pnnbao-ump/VieNeu-TTS`, folder `sample/`.

The static UI loads these presets through `samples/manifest.json`. When a user
selects a voice, `app.js` fetches the matching `.txt` file and reads audio from
`samples/audio-data.json` first. The JSON audio fallback avoids custom-domain
browser fetches being redirected to Hugging Face/Xet binary storage for `.wav`
files.

- `binh_nam_bac.wav`
- `dung_nu_nam.wav`
- `huong_nu_bac.wav`
- `ly_nu_bac.wav`
- `nguyen_nam_nam.wav`
- `ngoc_nu_bac.wav`
- `son_nam_nam.wav`
- `tuyen_nam_bac.wav`
- `vinh_nam_nam.wav`
- `doan_nu_nam.wav`

Regenerate `manifest.json` from the repository root after adding or renaming
samples:

```bash
python3 tools/generate_samples_manifest.py
```
