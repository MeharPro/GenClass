// GenClass "brain" (offscreen document): speech -> decision loop -> actions. The side panel is only a view.

import * as ort from "onnxruntime-web/webgpu";
import * as transformers from "@huggingface/transformers";
import { Controller } from "../core/controller.js";
import { Engine } from "../core/engine.js";
import { Tokenizer } from "../core/tokenizer.js";
import { appCatalog } from "../core/sites.js";
import { buildQuestions } from "../core/questions.js";
import { defaultConfig } from "../core/types.js";
import { Features } from "./features_rt.js";
import { LocalAsr, LocalSpeechSource, SPEECH_ENGINES, WebSpeechSource, gpuInfo, speechPlan, summarizeStats } from "./speech.js";

const URL_OF = (p) => chrome.runtime.getURL(p);
ort.env.wasm.wasmPaths = URL_OF("ort/");
ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
ort.env.wasm.proxy = false;
transformers.env.allowLocalModels = false;
transformers.env.useBrowserCache = true;
if (transformers.env.backends && transformers.env.backends.onnx && transformers.env.backends.onnx.wasm) {
  transformers.env.backends.onnx.wasm.wasmPaths = URL_OF("ort/");
}

const CACHE = "genclass-models-v1";

const B = {
  settings: null,
  engine: null,
  engineInfo: null,
  controller: null,
  source: null,
  asr: null,
  asrKey: null,
  listening: false,
  tabs: [],
  catalog: null,
  snapCache: null,
  undo: [],
  gpu: null,
  progress: {},
  textSeq: 0,
  log: [],
  logSeq: 0,
};

const send = (type, payload) => chrome.runtime.sendMessage({ to: "panel", type, payload }).catch(() => {});
const sw = (msg) => chrome.runtime.sendMessage({ to: "sw", ...msg });
const ui = (event, payload) => send("ui", { event, payload: jsonable(payload) });

function jsonable(x) {
  return JSON.parse(JSON.stringify(x, (_k, v) => (v instanceof Map ? Object.fromEntries(v) : v)));
}

function status(extra = {}) {
  send("status", {
    listening: B.listening,
    halted: !!(B.controller && B.controller.halted),
    engine: B.engineInfo,
    speech: B.settings && B.settings.speechEngine,
    speechReady: !!(B.asr || (B.settings && B.settings.speechEngine === "chrome")),
    gpu: B.gpu,
    isolated: globalThis.crossOriginIsolated,
    progress: B.progress,
    pending: B.controller && B.controller.pending,
    dryRun: B.settings && B.settings.dryRun,
    ...extra,
  });
}

function progress(key, loaded, total, label) {
  B.progress[key] = { loaded, total, label, done: total > 0 && loaded >= total };
  send("progress", B.progress);
}

// ------------------------------------------------------------------ model download + cache

async function sha256(buf) {
  const h = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchCached(url, expect, key, label) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(url);
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    if (!expect.bytes || buf.byteLength === expect.bytes) {
      progress(key, buf.byteLength, buf.byteLength, `${label} (cached)`);
      return buf;
    }
    await cache.delete(url);
  }
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`download failed: HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get("content-length")) || expect.bytes || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.byteLength;
    progress(key, got, total, label);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.byteLength; }
  if (expect.sha256) {
    const h = await sha256(buf);
    if (h !== expect.sha256) throw new Error(`checksum mismatch for ${url} (got ${h.slice(0, 12)}…)`);
  }
  await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
  progress(key, got, got, label);
  return buf;
}

async function bundledModel(file) {
  try {
    const r = await fetch(URL_OF(`model/${file}`), { method: "HEAD" });
    return r.ok;
  } catch {
    return false;
  }
}

async function loadEngine(settings) {
  const card = await (await fetch(URL_OF("model/model.json"))).json();
  const tokJson = await (await fetch(URL_OF("model/tokenizer.json"))).json();
  const calibration = await (await fetch(URL_OF("model/calibration.json"))).json();
  const meta = await (await fetch(URL_OF("model/meta.json"))).json();
  B.gpu = B.gpu || await gpuInfo();
  const pref = settings.compute || "auto";
  let plans = [];
  if (pref !== "wasm" && B.gpu.webgpu) plans.push(B.gpu.f16 ? ["fp16", "webgpu"] : ["q8", "webgpu"]);
  if (pref !== "webgpu" || !B.gpu.webgpu) plans.push(["q8", "wasm"]);
  if (pref === "webgpu" && B.gpu.webgpu) plans.push(["q8", "wasm"]);
  let lastErr = null;
  for (const [variant, provider] of plans) {
    const v = card.variants[variant];
    try {
      let buf;
      if (await bundledModel(v.file)) buf = await fetchCached(URL_OF(`model/${v.file}`), {}, "model", `GenClass model (${variant}, bundled)`);
      else {
        const base = (settings.modelBaseUrl || card.default_base_url).replace(/\/?$/, "/");
        buf = await fetchCached(base + v.file, v, "model", `GenClass model (${variant})`);
      }
      const t0 = performance.now();
      const session = await ort.InferenceSession.create(buf, { executionProviders: [provider], graphOptimizationLevel: "all" });
      const engine = new Engine({ ort, session, tokenizer: new Tokenizer(tokJson), calibration, meta, provider, variant });
      // warm-up: compiles WebGPU pipelines / JITs the wasm kernels on a realistic request shape
      const q = buildQuestions(null, ["Gmail", "YouTube"], ["hello"], [], 60, { browser: true });
      await engine.evaluate({ screen: "Google Chrome", focused: "nothing", recent_actions: "none", pending: "none", transcript: "type hello" }, q);
      const warm = await timeIt(() => engine.evaluate({ screen: "Google Chrome", focused: "nothing", recent_actions: "none", pending: "none", transcript: "scroll down" }, q));
      B.engineInfo = { variant, provider, threads: provider === "wasm" ? ort.env.wasm.numThreads : null, loadMs: Math.round(performance.now() - t0), warmMs: Math.round(warm) };
      return engine;
    } catch (e) {
      lastErr = e;
      console.warn("[genclass] engine plan failed", variant, provider, e);
    }
  }
  throw lastErr || new Error("no way to run the model here");
}

async function timeIt(fn) {
  const t = performance.now();
  await fn();
  return performance.now() - t;
}

// ------------------------------------------------------------------ speech

async function loadSpeech(settings) {
  const engine = settings.speechEngine || "moonshine";
  B.gpu = B.gpu || await gpuInfo();
  if (!SPEECH_ENGINES[engine].local) { B.asr = null; B.asrKey = null; return null; }
  const plan = speechPlan(engine, B.gpu, settings.compute || "auto");
  const key = `${plan.model}|${plan.device}|${JSON.stringify(plan.dtype)}`;
  if (B.asr && B.asrKey === key) return B.asr;
  const files = {};
  const asr = await new LocalAsr(transformers, plan).load((p) => {
    if (p.status === "progress" || p.status === "done") {
      files[p.file] = { loaded: p.loaded || 0, total: p.total || 0 };
      const loaded = Object.values(files).reduce((a, f) => a + f.loaded, 0);
      const total = Math.max(plan.mb * 1e6 * 0.98, Object.values(files).reduce((a, f) => a + f.total, 0));
      progress("speech", loaded, total, `${SPEECH_ENGINES[engine].label} (${plan.device})`);
    }
  });
  progress("speech", 1, 1, `${SPEECH_ENGINES[engine].label} (${plan.device}) ready`);
  B.asr = asr;
  B.asrKey = key;
  B.asrPlan = plan;
  return asr;
}

function makeSource(onEvent) {
  const engine = B.settings.speechEngine || "moonshine";
  if (engine === "chrome") return new WebSpeechSource(onEvent, { lang: B.settings.lang || "en-US" });
  return new LocalSpeechSource(onEvent, B.asr, { stepMs: B.asrPlan.stepMs });
}

// ------------------------------------------------------------------ controller wiring

async function refreshTabs() {
  try { B.tabs = (await sw({ type: "tabs" })) || []; } catch { B.tabs = []; }
}

const observer = {
  async snapshot() {
    const now = performance.now();
    if (B.snapCache && now - B.snapCache.t < 1200) return B.snapCache.snap;
    const tail = B.controller && B.controller.stream.current(performance.now() / 1000);
    const snap = await sw({ type: "observe", tail: tail ? tail.text : "", max: B.settings.maxElements || 60, targetTabId: B.settings.targetTabId });
    if (snap && !snap.error) B.snapCache = { t: now, snap };
    return snap;
  },
  invalidate() { B.snapCache = null; },
  prefetch() { B.snapCache = null; refreshTabs(); this.snapshot().catch(() => {}); },
};

const executor = {
  async run(action, snap, said) {
    const dryRun = !!B.settings.dryRun;
    let resolved = null;
    if ((action.kind === "open_app" || action.kind === "quit_app") && B.catalog) resolved = B.catalog.resolve(action.app);
    const undoOf = action.kind === "undo" ? B.undo[B.undo.length - 1] : null;
    const res = await sw({ type: "execute", action, snap: snap && { id: snap.id, tabId: snap.tabId }, targetTabId: B.settings.targetTabId, said, dryRun, searchUrl: B.settings.searchUrl, resolved, undoOf });
    const r = res || { ok: false, detail: "no response" };
    if (action.kind === "undo" && r.ok && !r.dryRun) B.undo.pop();
    else if (r.ok && !r.dryRun && action.kind !== "undo") { B.undo.push({ action, undo: r.undo || null }); if (B.undo.length > 20) B.undo.shift(); }
    setTimeout(() => refreshTabs(), 300);
    return r;
  },
};

function appsFor(tailText) {
  B.catalog = appCatalog(B.tabs, tailText, B.settings.maxApps || 24);
  return B.catalog.names;
}

function buildController() {
  const cfg = defaultConfig();
  cfg.dryRun = !!B.settings.dryRun;
  cfg.maxElements = B.settings.maxElements || 60;
  cfg.searchUrl = B.settings.searchUrl || cfg.searchUrl;
  const c = new Controller({
    engine: B.engine, observer, executor, cfg, apps: appsFor, staged: B.settings.staged !== false, browser: true,
    ui: (event, payload) => {
      if (event === "picks") sw({ type: "badges", snapId: payload.snapId, eids: payload.eids }).catch(() => {});
      if (event === "pending" || event === "halted") setTimeout(() => status(), 0);
      ui(event, payload);
    },
    log: (kind, fields) => { B.log.push({ seq: ++B.logSeq, t: Date.now(), kind, ...jsonable(fields) }); if (B.log.length > 2000) B.log.splice(0, 500); },
  });
  return c;
}

function onSpeech(ev) {
  if (!B.controller) return;
  if (B.source && B.source.cadenceMs && B.settings.speechEngine !== "chrome") B.controller.cadenceMs = B.source.cadenceMs;
  B.controller.onEvent(ev);
}

async function startListening() {
  if (!B.engine) throw new Error("the model is not loaded yet");
  if (B.controller.halted) B.controller.resume();
  if (B.settings.speechEngine !== "chrome" && !B.asr) await loadSpeech(B.settings);
  B.source = makeSource(onSpeech);
  if (B.source instanceof LocalSpeechSource) await B.source.startMic();
  else B.source.start();
  B.listening = true;
  sw({ type: "listening", on: true }).catch(() => {});
  refreshTabs();
  status();
}

async function stopListening() {
  B.listening = false;
  if (B.source) await B.source.stop();
  B.source = null;
  sw({ type: "listening", on: false }).catch(() => {});
  status();
}

function kill(from) {
  if (B.controller) B.controller.halt();
  stopListening();
  B.textAbort = true;
  ui("killed", { from });
  status();
}

/** Type a command as if spoken: one partial per word at speaking pace, then a final. */
async function textCommand(text, { stream = true, wordMs = 230, finalAfterMs = 350, resume = false } = {}) {
  if (!B.controller) throw new Error("not ready");
  if (B.controller.halted) {
    if (!resume) return { ok: false, error: "stopped by the kill switch: press Start listening (or send again) to resume" };
    B.controller.resume();
  }
  B.textAbort = false;
  const uid = `tx${++B.textSeq}`;
  const words = text.trim().split(/\s+/);
  await refreshTabs();
  B.controller.onEvent({ kind: "speech_start", uid, text: "" });
  if (stream) {
    for (let i = 1; i <= words.length; i++) {
      if (B.textAbort) return;
      B.controller.onEvent({ kind: "partial", uid, text: words.slice(0, i).join(" ") });
      await new Promise((r) => setTimeout(r, wordMs));
    }
    await new Promise((r) => setTimeout(r, finalAfterMs));
  }
  if (B.textAbort) return;
  B.controller.onEvent({ kind: "final", uid, text: words.join(" ") });
  B.controller.onEvent({ kind: "speech_end", uid, text: "" });
  await new Promise((r) => setTimeout(r, 50));
  await B.controller.idle();
}

// ------------------------------------------------------------------ init + messages

async function init(settings) {
  B.settings = { ...B.settings, ...settings };
  status({ phase: "loading" });
  const jobs = [];
  if (!B.engine) {
    if (!B.enginePromise) B.enginePromise = loadEngine(B.settings).then((e) => { B.engine = e; }).finally(() => { B.enginePromise = null; });
    jobs.push(B.enginePromise);
  }
  const speechNeeded = B.settings.speechEngine !== "chrome" && B.settings.preloadSpeech !== false;
  if (speechNeeded) jobs.push((B.speechPromise ||= loadSpeech(B.settings).finally(() => { B.speechPromise = null; })).catch((e) => { ui("error", `speech model: ${e.message || e}`); }));
  const res = await Promise.allSettled(jobs);
  const failed = res.find((r) => r.status === "rejected");
  if (failed) {
    status({ phase: "error", error: String(failed.reason && failed.reason.message || failed.reason) });
    throw failed.reason;
  }
  if (!B.controller) B.controller = buildController();
  else applySettings();
  status({ phase: "ready" });
  return { ok: true, engine: B.engineInfo };
}

function applySettings() {
  if (!B.controller) return;
  B.controller.cfg.dryRun = !!B.settings.dryRun;
  B.controller.cfg.maxElements = B.settings.maxElements || 60;
  B.controller.cfg.searchUrl = B.settings.searchUrl || B.controller.cfg.searchUrl;
  B.controller.staged = B.settings.staged !== false;
}

async function benchSpeech({ url, engines }) {
  const resp = await fetch(url);
  const wav = new Uint8Array(await resp.arrayBuffer());
  const samples = decodeWav16k(wav);
  const out = {};
  for (const eng of engines) {
    try {
      const t0 = performance.now();
      const plan = speechPlan(eng, B.gpu || await gpuInfo(), B.settings.compute || "auto");
      const asr = await new LocalAsr(transformers, plan).load();
      const loadMs = performance.now() - t0;
      const events = [];
      const src = new LocalSpeechSource((e) => events.push({ ...e, t: performance.now() }), asr, { stepMs: plan.stepMs });
      src.start();
      const tStart = performance.now();
      await src.playClip(samples);
      out[eng] = { device: plan.device, dtype: plan.dtype, mb: plan.mb, loadMs: Math.round(loadMs), ...summarizeStats(src.stats),
        partials: events.filter((e) => e.kind === "partial").map((e) => [Math.round(e.t - tStart), e.text]),
        final: (events.find((e) => e.kind === "final") || {}).text || "" };
    } catch (e) {
      out[eng] = { error: String(e.message || e) };
    }
  }
  return out;
}

function decodeWav16k(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let off = 12;
  let fmt = null;
  while (off < bytes.length - 8) {
    const id = String.fromCharCode(...bytes.subarray(off, off + 4));
    const size = dv.getUint32(off + 4, true);
    if (id === "fmt ") fmt = { ch: dv.getUint16(off + 10, true), rate: dv.getUint32(off + 12, true), bits: dv.getUint16(off + 22, true) };
    if (id === "data") {
      const n = size / (fmt.bits / 8) / fmt.ch;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = dv.getInt16(off + 8 + i * 2 * fmt.ch, true) / 32768;
      if (fmt.rate !== 16000) throw new Error(`expected 16 kHz audio, got ${fmt.rate}`);
      return out;
    }
    off += 8 + size + (size % 2);
  }
  throw new Error("no data chunk in wav");
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.to !== "brain") return;
  const done = (p) => { Promise.resolve(p).then((v) => reply(jsonable(v ?? { ok: true })), (e) => reply({ ok: false, error: String(e && e.message || e) })); return true; };
  switch (msg.type) {
    case "init": return done(init(msg.settings || {}));
    case "settings": {
      const old = B.settings || {};
      B.settings = { ...old, ...msg.settings };
      applySettings();
      const speechChanged = old.speechEngine !== B.settings.speechEngine || old.compute !== B.settings.compute;
      if (speechChanged) {
        B.asr = null;
        B.asrKey = null;
        if (B.listening) stopListening();
        if (B.settings.speechEngine !== "chrome") loadSpeech(B.settings).then(() => status(), (e) => ui("error", `speech model: ${e.message || e}`));
      }
      status();
      return done({ ok: true });
    }
    case "start": return done(startListening());
    case "stop": return done(stopListening());
    case "toggle": return done(B.listening ? stopListening() : startListening());
    case "kill": kill(msg.from || "button"); return done({ ok: true });
    case "text": return done(textCommand(msg.text, msg.options || {}));
    case "confirm": return done(B.controller && B.controller.confirmFromUi());
    case "cancel": B.controller && B.controller.dropPending("cancelled from the panel"); return done({ ok: true });
    case "status": status(); return done({ ok: true });
    case "log": return done({ seq: B.logSeq, log: B.log.filter((x) => x.seq > (msg.since || 0)).slice(-(msg.n || 300)), stats: B.controller && B.controller.stats, decisions: B.controller && B.controller.decisions.slice(-50) });
    case "bench_speech": return done(benchSpeech(msg));
    case "bench_model": return done(benchModel(msg));
    case "play_audio": return done(playAudio(msg));
    case "clear_cache": return done(caches.delete(CACHE).then(async () => { for (const k of await caches.keys()) if (k.startsWith("transformers")) await caches.delete(k); return { ok: true }; }));
    default: return false;
  }
});

/** Feed a 16 kHz WAV through the selected local speech engine into the live decision loop (tests, demos). */
async function playAudio({ url }) {
  if (B.controller.halted) B.controller.resume();
  if (!B.asr) await loadSpeech(B.settings);
  const samples = decodeWav16k(new Uint8Array(await (await fetch(url)).arrayBuffer()));
  const events = [];
  const t0 = performance.now();
  const src = new LocalSpeechSource((ev) => { events.push({ kind: ev.kind, text: ev.text, ms: Math.round(performance.now() - t0) }); onSpeech(ev); }, B.asr, { stepMs: B.asrPlan.stepMs });
  B.source = src;
  src.start();
  await src.playClip(samples);
  await new Promise((r) => setTimeout(r, 300));
  await B.controller.idle();
  src.on = false;
  B.source = null;
  return { audioMs: Math.round((samples.length / 16000) * 1000), events, stats: summarizeStats(src.stats), plan: { device: B.asrPlan.device, model: B.asrPlan.model } };
}

/** Model-only latency on fixed requests: full fan-out vs staged pass 1. */
async function benchModel({ requests = [], repeat = 3 } = {}) {
  const e = B.engine;
  const out = { engine: B.engineInfo, full: [], stage1: [] };
  for (const r of requests) {
    const qs = {};
    for (const [k, q] of Object.entries(r.questions)) qs[k] = q.type === "choice" ? { ...q, criteria: new Map(q.criteria) } : q;
    const s1 = {};
    for (const k of ["intent", "complete", "is_command", "destructive"]) if (qs[k]) s1[k] = qs[k];
    for (let i = 0; i < repeat; i++) {
      out.full.push(await timeIt(() => e.evaluate(r.state, qs)));
      out.stage1.push(await timeIt(() => e.evaluate(r.state, s1)));
    }
  }
  const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
  return { engine: out.engine, n: requests.length, fullMsP50: med(out.full), stage1MsP50: med(out.stage1) };
}

const features = new Features(() => {
  if (!B.engine) throw new Error("model not loaded");
  return B.engine;
}, () => B.settings || {});

async function ensureEngine(settings) {
  if (!B.settings) B.settings = { ...settings };
  if (!B.engine) await init({ ...settings, preloadSpeech: false });
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.to !== "brain") return;
  const done = (p) => { Promise.resolve(p).then((v) => reply(jsonable(v ?? { ok: true })), (e) => reply({ ok: false, error: String(e && e.message || e) })); return true; };
  switch (msg.type) {
    case "filter_classify": return done(ensureEngine(msg.settings || {}).then(() => features.classifyBlocks(msg.blocks, msg.page)).then((r) => ({ results: r })));
    case "focus_config": return done(ensureEngine(msg.settings || {}).then(() => { B.settings = { ...B.settings, ...(msg.settings || {}) }; return features.focusConfigure(); }).then(() => features.focusState()));
    case "tab_event": return done(features.focus.on ? features.onTab(msg.event) : null);
    case "focus_keep": features.keep(msg.tabId); return done({ ok: true });
    case "focus_state": return done(features.focusState());
    case "relevance": return done(ensureEngine(msg.settings || {}).then(() => features.relevance(msg.task, msg.tabs)).then((m) => ({ scores: [...m] })));
    default: return false;
  }
});

status({ phase: "booted" });
