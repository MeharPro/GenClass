// The decision loop with the real model (q8, wasm, Node) on streamed text: acts mid-sentence, never twice,
// asks before risky actions. Fake page + executor (records actions). Skipped when the model is not built.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { join } from "node:path";
import * as ort from "onnxruntime-web";
import { Tokenizer } from "../../src/core/tokenizer.js";
import { Engine } from "../../src/core/engine.js";
import { Controller } from "../../src/core/controller.js";
import { appCatalog } from "../../src/core/sites.js";
import { defaultConfig } from "../../src/core/types.js";
import { ASSETS, readJson } from "./helpers.mjs";

const MODEL = join(ASSETS, "genclass-q8.onnx");
ort.env.wasm.numThreads = 4;

const PAGE = {
  id: 1, appName: "Google Chrome", windowTitle: "GenClass test shop", url: "http://localhost/shop", browser: true, tabId: 7,
  elements: [
    { eid: "e01", role: "search field", label: "Search products", value: null, context: "header", focused: false, enabled: true },
    { eid: "e02", role: "button", label: "Search", context: "header", enabled: true, inForm: true, searchForm: true, submits: true },
    { eid: "e03", role: "link", label: "Home", context: "navigation", enabled: true },
    { eid: "e04", role: "link", label: "Laptops", context: "navigation", enabled: true },
    { eid: "e05", role: "link", label: "Headphones", context: "navigation", enabled: true },
    { eid: "e06", role: "button", label: "Add to cart", context: "post", enabled: true },
    { eid: "e07", role: "button", label: "Buy now", context: "post", enabled: true, inForm: true, submits: true },
    { eid: "e08", role: "button", label: "Delete account", context: "footer", enabled: true },
    { eid: "e09", role: "text field", label: "Email", context: "form", enabled: true, inForm: true },
    { eid: "e10", role: "link", label: "Contact us", context: "footer", enabled: true },
  ],
  focusedEid: null,
};

async function harness() {
  const session = await ort.InferenceSession.create(readFileSync(MODEL));
  const engine = new Engine({ ort, session, tokenizer: new Tokenizer(readJson(join(ASSETS, "tokenizer.json"))), calibration: readJson(join(ASSETS, "calibration.json")), meta: readJson(join(ASSETS, "meta.json")) });
  const acts = [];
  const events = [];
  const cfg = defaultConfig();
  let catalog = null;
  const c = new Controller({
    engine, cfg, browser: true, staged: true,
    observer: { snapshot: async () => PAGE },
    executor: { run: async (a) => { acts.push({ ...a, at: Date.now() }); return { ok: true, changed: true, detail: a.kind }; } },
    apps: (t) => { catalog = appCatalog([{ id: 1, title: "GenClass test shop", url: "http://localhost/shop" }], t); return catalog.names; },
    ui: (ev, p) => events.push([ev, p]),
  });
  return { c, acts, events };
}

async function speak(c, uid, text, wordMs = 230, final = true) {
  const words = text.split(" ");
  const heardAt = [];
  c.onEvent({ kind: "speech_start", uid, text: "" });
  for (let i = 1; i <= words.length; i++) {
    c.onEvent({ kind: "partial", uid, text: words.slice(0, i).join(" ") });
    heardAt.push(Date.now());
    await new Promise((r) => setTimeout(r, wordMs));
  }
  await new Promise((r) => setTimeout(r, 300));
  if (final) c.onEvent({ kind: "final", uid, text });
  await new Promise((r) => setTimeout(r, 1200));
  await c.idle();
  return heardAt;
}

const skip = !existsSync(MODEL) && "model not built";

test("acts mid-sentence on a closed-set command, then runs the chained payload once", { skip, timeout: 120000 }, async () => {
  const { c, acts, events } = await harness();
  const heard = await speak(c, "u1", "scroll down a little and then search for noise cancelling headphones");
  const kinds = acts.map((a) => a.kind);
  console.log("[mid-sentence] actions:", acts.map((a) => `${a.kind}(${a.text || (a.amount ?? "")})`).join(", "));
  const ex = events.filter(([e]) => e === "executed").map(([, p]) => `fired at word ${p.firedAtWord} of ${p.wordsHeard} heard`);
  console.log("[mid-sentence]", ex.join("; "));
  const lat = events.filter(([e]) => e === "decision").map(([, p]) => p.timings.model);
  console.log("[mid-sentence] model ms per decision p50:", Math.round(lat.sort((a, b) => a - b)[lat.length >> 1]), "n", lat.length);
  assert.equal(kinds.filter((k) => k === "scroll_down").length, 1, "scroll exactly once");
  assert.ok(kinds.includes("search_web"), "search ran");
  assert.equal(kinds.filter((k) => k === "search_web").length, 1, "search exactly once");
  const sw = acts.find((a) => a.kind === "search_web");
  assert.ok(/noise cancelling headphones/.test(sw.text), `verbatim span: ${sw.text}`);
  const scroll = acts.find((a) => a.kind === "scroll_down");
  assert.ok(scroll.at < heard[heard.length - 1], "scroll fired before the last word was heard");
});

test("risky click waits for a spoken confirm from a later utterance", { skip, timeout: 120000 }, async () => {
  const { c, acts, events } = await harness();
  await speak(c, "u1", "click buy now");
  assert.equal(acts.length, 0, "nothing ran before confirm");
  assert.ok(c.pending && c.pending.kind === "click", `pending: ${JSON.stringify(c.pending)}`);
  await new Promise((r) => setTimeout(r, 900)); // reaction time after the prompt
  await speak(c, "u2", "confirm");
  console.log("[confirm] actions:", acts.map((a) => `${a.kind}(${a.targetLabel})`).join(", "), "| decisions:", events.filter(([e]) => e === "decision").map(([, p]) => p.verdict).join(","));
  assert.equal(acts.length, 1);
  assert.equal(acts[0].targetEid, "e07");
});

test("side talk is ignored, cancel drops a pending action", { skip, timeout: 120000 }, async () => {
  const { c, acts } = await harness();
  await speak(c, "u1", "I think we should get lunch soon");
  assert.equal(acts.length, 0);
  await speak(c, "u2", "click delete account");
  assert.ok(c.pending, "delete waits for confirmation");
  await new Promise((r) => setTimeout(r, 900));
  await speak(c, "u3", "no cancel that");
  assert.equal(c.pending, null);
  assert.equal(acts.length, 0);
});
