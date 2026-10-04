# GenClass

**Voice control for your browser that acts mid-sentence.** Say “click the headphones link and then scroll down a little” and the click happens while you're still saying “and then scroll down”. A small typed-decision model runs inside your browser (WebGPU, or WASM on the CPU), and speech recognition runs on your device by default.

> GenClass is an independent open-source project. It is not affiliated with, endorsed by, or sponsored by TypeSafe AI or its Jev product.

## What it does

| Feature | Default | How it works |
|---|---|---|
| **Voice browser control** | on | Click, type into fields, web search, open URLs, scroll, back/forward, new/close/switch tab, press keys, all acting on partial speech. |
| **Content filter** | off | Hides ad and sponsored blocks found by their markers (ad-network frames, “Sponsored”/“Promoted” labels, ad slots). The toolbar badge counts hidden blocks, there is a per-site toggle, and every hidden block has a “show” link. Model judgement of unmarked blocks (clickbait, off-topic) is experimental and off. |
| **Focus mode** | off | You type what you're working on. Every open or newly opened tab is scored for relevance (title, URL, meta description). Off-task tabs are parked (discarded) or closed after a grace period, with an undo list. |
| **RAM manager** | off | Reads `chrome.system.memory` every minute. When memory is low it parks idle tabs, lowest relevance (when focus mode has a task) and least recently used first, and offers one-click restore. |
| Video processing | — | Planned, not built. |

**Never touched:** pinned tabs, tabs playing audio, the active tab, and tabs with unsaved form input (focus mode and RAM manager). GenClass never types into password or card fields, and never acts on `chrome://` pages, the Web Store, banking and payment sites, or password managers.

## Install

### From the release zip (“Load unpacked”)
1. Download `genclass-0.1.0.zip` from the [GitHub release](https://github.com/MeharPro/GenClass/releases/tag/v0.1.0) and unzip it.
2. Open `chrome://extensions`, turn on **Developer mode** (top right), click **Load unpacked**, and pick the unzipped folder (the one containing `manifest.json`).
3. Pin GenClass from the puzzle-piece menu and click its icon to open the side panel.
4. A welcome tab asks for the microphone once. The first run downloads the decision model (57 MB on WASM, 67 MB on WebGPU) from the GitHub release, plus the Moonshine speech model (63 MB on WASM, 154 MB on WebGPU) from Hugging Face. Both are cached.

### Chrome Web Store
Coming soon. The same package is being submitted.

### From source
```bash
npm ci
npm run build              # -> dist/genclass/ (Load unpacked)
npm run package            # -> dist/genclass-0.1.0.zip
node scripts/build.mjs --bundle-model   # include the model in the extension (offline; needs release-assets/)
```
Requires Chrome 116 or newer (side panel, offscreen documents). WebGPU (Chrome 113+, most desktops) makes the model about 10x faster than WASM.

## Using it

Open the side panel and press **Start listening** (or `Alt+Shift+G`). Say things like:

- “search for weather in Toronto” · “go to wikipedia dot org”
- “click sign in” · “click the images link”
- “type hello world in the search box and press enter”
- “scroll down a little” · “go back” · “go forward” · “open a new tab” · “switch to YouTube” (an open tab) · “reload the page” · “copy that” · “undo that”
- “close this tab” (asks you to say “confirm”)

On a fixed test page, 16 of 18 everyday commands resolve to the right action (`test/eval/commands_eval.mjs`). The two misses: “open GitHub” when no GitHub tab is open (say “go to github dot com”), and ordinals over different links (“click the first result”).

When GenClass is unsure which element you mean, it puts numbered badges on the top candidates; say “number two”. **Stop everything:** press `Esc` on the page, click **Stop all**, or use `Alt+Shift+K`. You can also type a command into the panel; it is streamed word by word as if spoken.

**Dry run** is on until you switch on live mode in Settings. In dry run, GenClass highlights what it would do and does nothing.

### Speech engines (Settings → Speech engine)

| Engine | Where audio goes | Download | First partial* | Real-time factor* |
|---|---|---|---|---|
| **Moonshine base** (default) | stays on device | 63 MB (WASM) / 154 MB (WebGPU) | 149 ms | 0.08 |
| Chrome built-in | **sent to Google** by Chrome | none | depends on network | — |
| Whisper base.en | stays on device | 77 MB / 206 MB | 1238 ms | 0.48 |
| Whisper large-v3 Turbo | stays on device | 564–759 MB, **WebGPU only** | not measured (no GPU in CI) | — |

\*Measured in Chromium on a CPU-only Linux VM (WASM, 4 threads) with a 3.1 s clip played in real time through the same voice-activity detection and streaming pipeline the microphone uses. Local engines stream by re-decoding the open utterance every 250 ms (Moonshine), 500 ms (Whisper base) or 900 ms (Turbo).

## Safety model
- **Select, don't generate.** Typed and searched text is always an exact span of what you said. The model picks the span; it never writes text.
- **Risky actions need a spoken “confirm”.** This covers send, submit, buy, delete, close, sign out and similar words in the target, the payload or what you said; submit buttons and Enter in non-search forms; and anything the model's `destructive` question flags. The “confirm” must come from a later utterance, after the prompt, with an explicit yes-word. “No” or “cancel” drops it, and it expires after 8 s.
- **Never twice.** Words that triggered an action are consumed. If the recognizer later revises them, nothing re-runs.
- **Limits:** at most 3 actions per second. A kill switch stops everything, including words already heard.

## How it works

```
mic ──► speech (Moonshine/Whisper via transformers.js, or Web Speech) ──► partial transcripts
                                     offscreen document ("brain")
  Stream (consumed-prefix cursor) ─► questions + state ─► GenClass model (onnxruntime-web, WebGPU|WASM)
         ─► policy (gates) ─► safety ─► service worker ─► content script (observe page / click / type / scroll)
side panel = view + settings          service worker = tabs, navigation, features, RAM alarm
```

- **The model** is a 32M-parameter encoder (Ettin/ModernBERT) with a block-masked “one question per branch” layout and typed heads: choice (softmax over options), noul (an independent yes/no probability) and score. Each request asks about 10 questions at once (intent, complete, is_command, destructive, target element, app/tab, key, text span, URL span, scroll amount). Every on-screen element is an option of the `target` question. Questions are isolated by the attention mask, so GenClass asks in two exact stages: first intent/complete/is_command/destructive (about 40% of the tokens), then only the argument question the chosen intent needs. The answers are identical to asking everything at once.
- **Mid-sentence policy:** it re-decides on every partial. A closed-set command such as click or scroll acts as soon as the model says the command is complete and two partials agree. Payloads (typed or searched text) wait for the end of speech or 600 ms of silence, so “type hello wor…” is never typed.
- **Page observation:** the visible actionable elements (links, buttons, inputs, selects; accessible names, at most 60), numbered `e01…` in screen order. This is the same format the model was trained on.

### Model files and parity
The model was trained on synthetic computer-use data only. It was exported from PyTorch to ONNX with the block mask built inside the graph (`tools/genclass_export.py`) and checked on 50 real harness requests (503 answers):

| Variant | Size | Used for | Max \|Δ logit\| vs PyTorch | Same decision |
|---|---|---|---|---|
| fp32 (reference, not shipped) | 134 MB | — | 2e-5 | 503/503 |
| fp16 | 67 MB | WebGPU with shader-f16 | 0.018 | 503/503 |
| q8 (8-bit weight-only MatMul, fp16 embeddings) | 57 MB | WASM, and WebGPU without f16 | 0.37 | 502/503 |

The JavaScript tokenizer (byte-level BPE) and packer produce token ids, positions and attention groups identical to Python on all 50 requests. The full JS pipeline (packer → onnxruntime-web → calibration) reproduces the Python ONNX results exactly.

**Latency** (one decision, in Chromium): staged pass 1 takes 212 ms p50 and a full fan-out 713 ms on WASM ×4 (CPU-only VM); 251 ms p50 per decision in the end-to-end tests. WebGPU numbers are not measured yet: the CI machines have no GPU.

### Honest limits
- The model is small and was trained on synthetic data, so it sometimes misses a target. GenClass then asks (“which element?”) or, when one on-screen label appears word for word in what you said, uses that element (shown as “rescue” in the log).
- Content classification beyond ad markers is weak: the model was not trained for it. That part stays experimental and off by default.
- Focus-mode relevance is zero-shot. On a held-out set of 36 tabs across 3 tasks it flagged 18/19 off-task tabs but would also have parked 6/17 on-task ones. That is why parking is reversible, waits a grace period, and skips any tab you use during the session or open from an on-task tab.

## Development
```bash
npm test                     # parity tests vs the Python harness + tokenizer/packer + ONNX (needs release-assets/)
npm run test:e2e             # Playwright: unpacked extension on a local test site, dry-run and live
node test/e2e/server.mjs     # serve the test site and model assets on :8737
```
`release-assets/` holds the model files. They are published as GitHub release assets, not committed.

## Repository layout
```
manifest + pages   static/ (manifest.json, panel.html/css, offscreen.html, welcome.html, audio-worklet.js, icons/)
src/core/          pure JS: tokenizer, packer, engine, questions, state, spans, policy, safety, stream, controller, features
src/offscreen/     brain: model loading, speech engines, decision loop, feature runtime
src/background/    service worker: tabs, navigation, content filter rules, focus/RAM plumbing
src/content/       page observation, actions, overlays, ad-block hiding
src/sidepanel/     panel + welcome page
src/model/         model.json (URLs, sha256), tokenizer.json, calibration.json, meta.json
test/              unit (node:test) + e2e (Playwright) + fixtures from the Python harness
tools/             ONNX export script (needs the training code; kept for reproducibility)
store/             Web Store listing, privacy policy, permissions, screenshots
```

## License
Apache-2.0 for code and model weights. See `LICENSE` and `NOTICE` (third-party components: ONNX Runtime Web, Transformers.js, Ettin base model, Moonshine, Whisper).
