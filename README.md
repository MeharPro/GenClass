# GenClass

**The version of Jev that runs in your browser. Fully open source (Apache-2.0).**

![license](https://img.shields.io/badge/license-Apache--2.0-blue) ![runs](https://img.shields.io/badge/runs-100%25%20local-brightgreen) ![model](https://img.shields.io/badge/model-32M%20params-orange) ![api](https://img.shields.io/badge/API-Jev%20wire--compatible-purple)

GenClass is a small, fast typed-decision model and a Chrome extension built on it. You speak, and your browser acts, often before you finish the sentence. Everything runs locally: the model executes in the browser with WebGPU/WASM, and speech recognition uses Moonshine on-device by default.

> GenClass is an independent open-source project. It is not affiliated with or endorsed by TypeSafe AI. "Jev" is TypeSafe's product name, used here only to describe what GenClass is comparable to.

## What it does
- **Insanely fast browser use.** "Open GitHub and search for whisper": the open fires mid-sentence and the typing follows as soon as you finish the phrase.
- **Ads and content filtering.** Classify page blocks as ad, sponsored, clickbait or off-topic, then hide them.
- **Focus modes.** Tell it what you're working on, and it closes or parks tabs that don't fit.
- **RAM management.** Chrome eating all your memory? GenClass discards idle and irrelevant tabs before your machine starts swapping.
- **Video and more** (planned). Any decision that is a choice, yes/no or score over text runs through the same model.

## GenClass vs Jev (direct, same inputs)

| Decision | Jev | GenClass |
|---|---|---|
| Recognise the command mid-sentence | 66.4% | **90.4%** |
| Pick the exact words to type | 75.5% | **94.9%** |
| Full-command action | 91.4% | 92.0% |
| Pick the on-screen element | **86.6%** | 82.4% |
| General held-out questions | **94.7%** | 80.5% |
| Option-order flips | 10–13% (third-party) | **0** |
| Cost / privacy | paid cloud API | free, on-device |

Methodology, per-question tables and caveats are in [BENCHMARKS.md](BENCHMARKS.md). In short, GenClass wins at computer control and Jev wins at general questions.

## How it works
GenClass answers typed questions about a state in one forward pass:
- `choice` over options;
- `noul` for yes/no;
- `score` over ordered levels.

The questions use the same request format as Jev's System One API. The browser layer turns each partial transcript plus the visible page elements into one request. It acts on closed-set commands as soon as they are complete, and it asks for confirmation before anything risky.

## Open source: everything is here
- **The model.** A ModernBERT/Ettin encoder with typed decision heads, and per-option attention isolation so the order of the options can't change the answer (`jev_local/engine/encoder/`).
- **A Jev-compatible API server.** `POST /v1/systemone`, the same request and response shapes, so existing Jev clients work by changing the base URL (`jev_local/server/`).
- **The voice computer-use harness.** It re-decides on every partial transcript and acts mid-sentence (`jev_local/harness/`, `docs/DEMO.md`).
- **Training and data pipelines.** Synthetic data, multi-node CPU training and decontamination (`jev_local/train/`, `jev_local/data/`).
- **The benchmark harness.** 549 published Jev numbers, adapters for each publisher's own test code, and a pre-registration file (`jev_local/bench/`, `bench/`).
- **Design docs and results.** `docs/`, `results/`.

```bash
git clone https://github.com/MeharPro/GenClass && cd GenClass
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m jev_local.server.app --ckpt <weights dir>   # Jev-compatible API on 127.0.0.1:8765
```
The model weights (Apache-2.0) and the Chrome extension ship as release assets in v0.1.0.

## Status
Under active development. The extension, model weights (Apache-2.0) and install instructions are coming in the first release.

## License
Apache-2.0
