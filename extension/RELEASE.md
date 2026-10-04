# Releasing GenClass 0.1.0 (for the maintainer; nothing here is automated)

`node scripts/build.mjs --release` produces:

```
dist/
  genclass/                       unpacked extension (Load unpacked / what the zip contains)
  genclass-0.1.0.zip              Web Store upload + GitHub release asset
  release/                        everything to attach to the GitHub release v0.1.0
    genclass-q8.onnx              56,931,453 B  sha256 346b3d3f…da3c   (WASM; default on CPUs)
    genclass-fp16.onnx            67,154,907 B  sha256 eabc2096…9ab    (WebGPU with shader-f16)
    tokenizer.json, calibration.json, meta.json, model.json
    genclass-0.1.0.zip
    ASSETS.json                   names, sizes, sha256 of every asset + the speech models fetched at runtime
  store/
    listing.md  permissions.md  privacy.md  privacy.html  icon-128.png
    screenshots/01…09-*.png       1280×800
```

## 1. GitHub release (repo MeharPro/GenClass)
1. Push the contents of `extension/genclass/` as the repo root (the `.gitignore` keeps `node_modules/`, `dist/` and the `.onnx` files out).
2. Create release **tag `v0.1.0`** and attach every file in `dist/release/`. The extension downloads from
   `https://github.com/MeharPro/GenClass/releases/download/v0.1.0/genclass-q8.onnx` (and `-fp16.onnx`), and checks the sha256 recorded in `src/model/model.json`. Asset names must match exactly.
3. Check: `curl -sIL https://github.com/MeharPro/GenClass/releases/download/v0.1.0/genclass-q8.onnx | grep -i content-length` should print 56931453.

## 2. Speech models (no action needed)
Moonshine and Whisper are fetched by Transformers.js from Hugging Face (`onnx-community/moonshine-base-ONNX`, `onnx-community/whisper-base.en`, `onnx-community/whisper-large-v3-turbo`) and are not redistributed. ASSETS.json lists the exact files per device. To mirror them later, set `env.remoteHost` in `src/offscreen/offscreen.js`.

## 3. Chrome Web Store
1. Upload `dist/genclass-0.1.0.zip`.
2. Paste the listing from `store/listing.md`, the icon `store/icon-128.png`, and up to 5 screenshots from `store/screenshots/`.
3. Privacy tab: single purpose and permission justifications from `store/permissions.md`; data-use answers from `store/listing.md`; privacy policy URL `https://github.com/MeharPro/GenClass/blob/main/extension/store/privacy.md`.
4. Declare **no remote code** (model weights are data loaded by the bundled ONNX Runtime).
