// Compose Web Store screenshots (1280x800): the page (860 px) next to the GenClass side panel (420 px), with a
// caption strip. Input: dist/store/screenshots/raw/<name>-{page,panel}.png from the e2e runs.
import { chromium } from "@playwright/test";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(import.meta.dirname, "..");
const RAW = join(ROOT, "dist/store/screenshots/raw");
const OUT = join(ROOT, "dist/store/screenshots");
const CAPTIONS = {
  "01-dry-run": "Dry run first: GenClass shows what it would click before you let it act",
  "02-live-type": "“type wireless headphones in the search box”: the exact words, typed",
  "03-mid-sentence": "Acts mid-sentence: the click fires at word 4 while you keep talking",
  "04-confirm": "Buying, sending or deleting always waits for a spoken “confirm”",
  "05-content-filter": "Content filter: sponsored and ad blocks hidden, counted on the badge",
  "06-focus-mode": "Focus mode: tabs scored against what you are working on",
  "07-focus-parked": "Off-task tabs are parked after a grace period, with one-click restore",
  "08-ram-manager": "RAM manager: parks idle tabs when memory runs low, never pinned or audible ones",
  "09-moonshine-voice": "On-device speech (Moonshine): audio never leaves your computer",
};
const b64 = (p) => `data:image/png;base64,${readFileSync(p).toString("base64")}`;
const names = [...new Set(readdirSync(RAW).filter((f) => f.endsWith("-page.png")).map((f) => f.replace("-page.png", "")))].sort();
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
for (const n of names) {
  const cap = CAPTIONS[n] || n;
  await page.setContent(`<!doctype html><html><body style="margin:0;width:1280px;height:800px;overflow:hidden;font-family:system-ui,sans-serif;background:#0f172a">
    <div style="display:flex;height:744px">
      <img src="${b64(join(RAW, `${n}-page.png`))}" style="width:860px;height:744px;object-fit:cover;object-position:top left">
      <img src="${b64(join(RAW, `${n}-panel.png`))}" style="width:420px;height:744px;object-fit:cover;object-position:top left;border-left:1px solid #334155">
    </div>
    <div style="height:56px;display:flex;align-items:center;gap:12px;padding:0 24px;color:#f8fafc;font-size:21px;font-weight:600">
      <span style="background:#2563eb;border-radius:6px;padding:2px 10px;font-size:15px">GenClass</span>${cap}</div></body></html>`);
  await page.screenshot({ path: join(OUT, `${n}.png`) });
  console.log("wrote", n);
}
await browser.close();
