# GenClass

**The version of Jev that runs in your browser.**

GenClass is a small, fast typed-decision model and a Chrome extension built on it. You speak, and your browser acts, often before you finish the sentence. Everything runs locally: the model executes in the browser with WebGPU/WASM, and speech recognition uses Moonshine on-device by default.

> GenClass is an independent open-source project. It is not affiliated with or endorsed by TypeSafe AI. "Jev" is TypeSafe's product name, used here only to describe what GenClass is comparable to.

## What it does
- **Insanely fast browser use.** "Open GitHub and search for whisper": the open fires mid-sentence and the typing follows as soon as you finish the phrase.
- **Ads and content filtering.** Classify page blocks as ad, sponsored, clickbait or off-topic, then hide them.
- **Focus modes.** Tell it what you're working on, and it closes or parks tabs that don't fit.
- **RAM management.** Chrome eating all your memory? GenClass discards idle and irrelevant tabs before your machine starts swapping.
- **Video and more** (planned). Any decision that is a choice, yes/no or score over text runs through the same model.

## How it works
GenClass answers typed questions about a state in one forward pass:
- `choice` over options;
- `noul` for yes/no;
- `score` over ordered levels.

The questions use the same request format as Jev's System One API. The browser layer turns each partial transcript plus the visible page elements into one request. It acts on closed-set commands as soon as they are complete, and it asks for confirmation before anything risky.

## Status
Under active development. The extension, model weights (Apache-2.0) and install instructions are coming in the first release.

## License
Apache-2.0
