# GenClass privacy policy

_Last updated: 2026-10-04_

GenClass is a browser extension that lets you control Chrome by voice. It is an independent open-source project (https://github.com/MeharPro/GenClass) and is not affiliated with TypeSafe or its Jev product.

## The short version
- **Your audio stays on your device by default.** The default speech engine, Moonshine, and the optional Whisper engines run inside your browser.
- **Chrome built-in speech is opt-in and goes to Google.** If you choose “Chrome built-in” in Settings, Chrome sends your microphone audio to Google's speech service while GenClass is listening. Google's privacy policy applies to that audio.
- **The decision model runs on your device.** Page contents, the elements GenClass reads, your transcripts and its decisions never leave your computer.
- **We collect nothing.** No analytics, no accounts, no servers run by us.

## What GenClass reads, and why
| Data | Why | Where it goes |
|---|---|---|
| Microphone audio (only while listening) | to turn speech into text | your device (Moonshine/Whisper), or Google if you chose Chrome built-in speech |
| The visible links, buttons and fields of the current page | to find what you are referring to | your device only |
| Tab titles, URLs, page descriptions | switching tabs; focus mode and RAM manager (when you turn them on) | your device only |
| Page blocks' text and markers | content filter (when you turn it on) | your device only |
| Free/total system memory | RAM manager (when you turn it on) | your device only |

## Downloads
On first use GenClass downloads model weights. These are data files, not code:
- the GenClass decision model from GitHub (github.com/MeharPro/GenClass releases), and
- speech models from Hugging Face (huggingface.co, onnx-community) for the engine you choose.

These requests are ordinary file downloads. They carry no information about you beyond what any download does (your IP address, seen by GitHub or Hugging Face).

## Storage
Settings, the undo lists (parked and freed tabs) and the downloaded models are stored locally in your browser (extension storage and Cache Storage). Remove them with “Clear downloaded models” in Settings or by uninstalling the extension.

## Children
GenClass is not directed at children under 13 and collects no personal information.

## Changes and contact
Changes to this policy are published in the repository. Questions: open an issue at https://github.com/MeharPro/GenClass/issues.
