# Permission rationale (for the Chrome Web Store review)

| Permission | Why GenClass needs it |
|---|---|
| `sidePanel` | The GenClass interface (mic button, live transcript, decisions, confirmations, settings) lives in the side panel. |
| `offscreen` | Microphone capture, speech recognition and the on-device model run in an offscreen document so they keep working while you use the page. Reasons declared: `USER_MEDIA`, `WORKERS`. |
| `storage` | Saves your settings and the undo lists for tabs parked by focus mode and the RAM manager. |
| `scripting` | Injects the GenClass content script into tabs that were already open when the extension was installed, so voice commands work there without a reload. |
| `alarms` | Runs the RAM manager's once-a-minute memory check (only when you turn the RAM manager on). |
| `system.memory` | Reads free/total memory for the RAM manager (only when on). |
| Host access `<all_urls>` | Voice commands must work on whatever site you are on: GenClass reads that page's visible buttons, links and fields to find what you referred to, then clicks, types or scrolls there. The same access lets it download model files (GitHub, Hugging Face) and hide ad blocks when the content filter is on. Nothing read from pages leaves your device. |
| Commands (`Alt+Shift+G`, `Alt+Shift+K`) | Start/stop listening and the kill switch. |

**Not requested:** `tabs` (titles and URLs come from host access), `debugger`, `history`, `cookies`, `webRequest`, `clipboardRead`. **Remote code:** none. Model weights are data files run by the bundled ONNX Runtime.
