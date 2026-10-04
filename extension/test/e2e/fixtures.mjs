// Shared Playwright setup: launch Chromium with the unpacked extension, a local server, the panel as a page.
import { chromium } from "@playwright/test";
import { mkdtempSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startServer } from "./server.mjs";

export const EXT = process.env.GENCLASS_EXT || join(import.meta.dirname, "..", "..", "dist", "genclass");
export const SHOTS = join(import.meta.dirname, "..", "..", "dist", "store", "screenshots", "raw");
mkdirSync(SHOTS, { recursive: true });

export async function launch({ settings = {}, headless = process.env.HEADED ? false : true } = {}) {
  const { server, port } = await startServer();
  const base = `http://127.0.0.1:${port}`;
  const ctx = await chromium.launchPersistentContext(mkdtempSync(join(tmpdir(), "genclass-")), {
    channel: "chromium", headless, viewport: { width: 860, height: 800 },
    args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`, "--enable-unsafe-webgpu", "--autoplay-policy=no-user-gesture-required",
      ...(process.env.SWIFTSHADER ? ["--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader", "--enable-features=Vulkan,WebGPUExperimentalFeatures", "--use-angle=swiftshader"] : [])],
  });
  let sw = ctx.serviceWorkers()[0];
  if (!sw) sw = await ctx.waitForEvent("serviceworker");
  const extId = new URL(sw.url()).host;
  for (const p of ctx.pages()) if (p.url().includes("welcome.html")) await p.close().catch(() => {});

  const shop = await ctx.newPage();
  await shop.goto(`${base}/site/shop.html`);
  const tabId = await sw.evaluate(async (u) => (await chrome.tabs.query({})).find((t) => t.url && t.url.startsWith(u)).id, `${base}/site/shop.html`);
  const all = {
    speechEngine: "moonshine", dryRun: true, compute: process.env.GENCLASS_COMPUTE || "auto", searchUrl: `${base}/site/shop.html?q={q}`,
    maxElements: 60, staged: true, modelBaseUrl: `${base}/assets/`, targetTabId: tabId, preloadSpeech: false, ...settings,
  };
  await sw.evaluate((s) => chrome.storage.local.set({ settings: s }), all);
  const panel = await ctx.newPage();
  await panel.setViewportSize({ width: 420, height: 800 });
  await panel.goto(`chrome-extension://${extId}/panel.html`);
  await panel.waitForFunction(() => /Model:/.test(document.getElementById("engineLine").textContent), null, { timeout: 300000 });
  await shop.bringToFront();
  const brain = (msg) => panel.evaluate((m) => chrome.runtime.sendMessage({ to: "brain", ...m }), msg);
  // A pause before each command: back-to-back identical commands within 350 ms of a final are treated as a
  // recognizer duplicate by the Stream (uid echo rule), which no human speaker could produce.
  const say = async (text, options = { stream: true }) => { await new Promise((r) => setTimeout(r, 700)); return brain({ type: "text", text, options }); };
  const close = async () => { await ctx.close(); server.close(); };
  return { ctx, sw, extId, shop, panel, tabId, base, brain, say, close, settings: all };
}

export async function setSettings(t, patch) {
  t.settings = { ...t.settings, ...patch };
  await t.panel.evaluate((s) => chrome.storage.local.set({ settings: s }), t.settings);
  await t.brain({ type: "settings", settings: t.settings });
}

export async function shot(t, name) {
  await t.shop.screenshot({ path: join(SHOTS, `${name}-page.png`) });
  await t.panel.screenshot({ path: join(SHOTS, `${name}-panel.png`) });
}
