# Chrome Web Store listing: GenClass

**Name:** GenClass: voice control for your browser
**Short description (≤132 chars):** Talk to your browser and it acts mid-sentence. Private on-device AI: clicks, types, searches, scrolls and manages tabs.
**Category:** Accessibility (secondary: Productivity)
**Language:** English

## Description

GenClass lets you control Chrome with your voice, and it doesn't wait for you to finish. Say “click the headphones link and then scroll down a little” and the click happens while you're still talking.

**Private by default.** Speech recognition (Moonshine) and the decision model both run on your device, on your GPU via WebGPU or on your CPU. Your audio and the pages you visit are not sent anywhere. You can opt into Chrome's built-in speech recognition, which sends audio to Google.

**What you can say**
• “search for weather in Toronto” · “go to wikipedia dot org”
• “click sign in” · “type hello world in the search box and press enter”
• “scroll down a little” · “go back” · “open a new tab” · “switch to YouTube” · “reload the page”
• Unsure which button you meant? GenClass numbers the candidates; say “number two”.

**Safe by design**
• Starts in dry-run mode: it shows what it would do before you let it act.
• Anything that sends, submits, buys, deletes or closes waits for you to say “confirm”.
• Never types into password or card fields. Never acts on browser pages, banking sites or password managers.
• Kill switch: Esc, the Stop button, or Alt+Shift+K.

**Optional features (all off until you turn them on)**
• Content filter: hides sponsored and ad blocks on pages, with a count on the toolbar badge and a per-site switch.
• Focus mode: tell it what you're working on; tabs unrelated to that task are parked after a grace period. One-click undo. Pinned tabs, tabs playing audio and tabs with unsaved input are never touched.
• RAM manager: when your computer runs low on memory, it parks idle tabs first and lets you restore them with one click.

**Speech engines:** Moonshine (default; local, fastest), Chrome built-in (cloud), Whisper base.en (local), Whisper large-v3 Turbo (local, most accurate; needs WebGPU).

**First run** downloads the GenClass model (57–67 MB) and the Moonshine speech model (63–154 MB) once. Both are cached.

GenClass is free and open source (Apache-2.0): https://github.com/MeharPro/GenClass
GenClass is an independent project. It is not affiliated with TypeSafe or its Jev product.

## Single purpose (for the review form)
Control the browser by voice, using an on-device AI model that decides which click, typing, navigation or tab action the user asked for. The optional content filter, focus mode and RAM manager use the same on-device model and tab access to manage the pages and tabs the user is controlling.

## Screenshots (1280×800, in screenshots/)
01-dry-run, 02-live-type, 03-mid-sentence, 04-confirm, 05-content-filter, 06-focus-mode, 07-focus-parked, 08-ram-manager, 09-moonshine-voice.
The store allows up to 5: use 03-mid-sentence, 04-confirm, 05-content-filter, 06-focus-mode, 08-ram-manager.

## Data usage disclosures (Privacy practices tab)
- Collects: **nothing** is sent to the developer. No analytics.
- Audio: processed on device by default. If the user selects Chrome built-in speech, audio is sent to Google by Chrome's speech service (Google's privacy policy applies).
- Website content: read on device only, to find the buttons and links you refer to and to classify page blocks. Never transmitted.
- Certify: not sold, not used for unrelated purposes, not used for creditworthiness.
- Remote code: none. All code ships in the package. Model weights (data, not code) are downloaded from the GitHub release and Hugging Face.

Privacy policy URL: https://github.com/MeharPro/GenClass/blob/main/store/privacy.md
