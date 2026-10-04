// GenClass side panel: a view onto the offscreen brain, plus settings.

import { SPEECH_ENGINES } from "../offscreen/speech.js";

const $ = (id) => document.getElementById(id);
const DEFAULTS = {
  speechEngine: "moonshine", dryRun: true, compute: "auto", searchUrl: "https://www.google.com/search?q={q}",
  maxElements: 60, staged: true, modelBaseUrl: "", lang: "en-US",
};
let settings = { ...DEFAULTS };
let st = {};
let pendingTimer = null;
const fired = new Map(); // uid -> [k, ...] word indices where actions fired
const lastWords = new Map(); // uid -> words heard
let current = { uid: null, words: [], consumed: 0 };

const brain = (type, extra = {}) => chrome.runtime.sendMessage({ to: "brain", type, ...extra });

async function loadSettings() {
  const s = await chrome.storage.local.get("settings");
  settings = { ...DEFAULTS, ...(s.settings || {}) };
  $("speechEngine").value = settings.speechEngine;
  $("live").checked = !settings.dryRun;
  $("compute").value = settings.compute;
  $("searchUrl").value = settings.searchUrl;
  $("maxElements").value = settings.maxElements;
  $("staged").checked = settings.staged;
  $("modelBaseUrl").value = settings.modelBaseUrl;
  renderMode();
  renderSpeechSize();
}

async function saveSettings(patch) {
  settings = { ...settings, ...patch };
  await chrome.storage.local.set({ settings });
  renderMode();
  renderSpeechSize();
  await brain("settings", { settings }).catch(() => {});
}

function renderMode() {
  const m = $("mode");
  m.textContent = settings.dryRun ? "Dry run" : "LIVE";
  m.className = `pill ${settings.dryRun ? "dry" : "live"}`;
  $("cloudNote").hidden = settings.speechEngine !== "chrome";
  $("privacy").textContent = settings.speechEngine === "chrome"
    ? "Chrome built-in speech: your audio is sent to Google. The decision model still runs on this device."
    : "Audio and decisions stay on this device. Models are downloaded once and cached.";
}

function renderSpeechSize() {
  const e = SPEECH_ENGINES[settings.speechEngine];
  const gpu = st.gpu || {};
  let t = "";
  if (!e.local) t = "No download. Needs internet; audio is processed by Google.";
  else {
    const v = gpu.webgpu && settings.compute !== "wasm" ? (gpu.f16 && e.webgpuF16 ? e.webgpuF16 : e.webgpu) : e.wasm;
    t = `Download: about ${v.mb} MB (${gpu.webgpu && settings.compute !== "wasm" ? "WebGPU" : "WASM"}), once.`;
    if (e.requiresWebGPU && !gpu.webgpu) t += " Not available: this browser has no WebGPU.";
  }
  $("speechSize").textContent = t;
}

// ------------------------------------------------------------------ status + progress

function renderStatus(s) {
  st = { ...st, ...s };
  const ready = !!st.engine;
  $("mic").disabled = !ready;
  $("mic").setAttribute("aria-pressed", st.listening ? "true" : "false");
  $("micLabel").textContent = st.listening ? "Listening… (click to stop)" : st.halted ? "Stopped. Start again" : "Start listening";
  if (st.engine) {
    const e = st.engine;
    $("engineLine").textContent = `Model: ${e.variant} on ${e.provider.toUpperCase()}${e.threads ? ` ×${e.threads}` : ""} · warm ${e.warmMs} ms · speech: ${SPEECH_ENGINES[st.speech || settings.speechEngine].label.split(" (")[0]}`;
  }
  if (st.phase === "error") {
    $("loading").hidden = false;
    $("loadTitle").textContent = "Could not load";
    $("loadDetail").textContent = st.error || "";
  }
  renderPending(st.pending);
  renderSpeechSize();
}

function renderProgress(p) {
  const items = Object.values(p || {});
  if (!items.length) return;
  const loaded = items.reduce((a, x) => a + (x.loaded || 0), 0);
  const total = items.reduce((a, x) => a + (x.total || 0), 0);
  const done = items.every((x) => x.done);
  $("loading").hidden = done && !!st.engine;
  const pct = total ? Math.min(100, Math.round((100 * loaded) / total)) : 0;
  $("loadBar").style.width = `${pct}%`;
  $("loadPct").textContent = total ? `${pct}% · ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB` : "";
  $("loadTitle").textContent = done ? "Starting the model…" : "Downloading models (first run only)";
  $("loadDetail").textContent = items.map((x) => `${x.label}: ${x.total ? Math.round((100 * x.loaded) / x.total) : 0}%`).join(" · ");
}

function renderPending(p) {
  clearInterval(pendingTimer);
  if (!p) { $("pending").hidden = true; return; }
  $("pending").hidden = false;
  $("pendingText").textContent = describe(p.action || p);
  let left = Math.round((p.timeoutMs || 8000) / 1000);
  $("pendingLeft").textContent = left;
  pendingTimer = setInterval(() => { left = Math.max(0, left - 1); $("pendingLeft").textContent = left; if (!left) clearInterval(pendingTimer); }, 1000);
}

function describe(a) {
  if (!a) return "";
  const arg = a.targetLabel || a.app || (a.text ? `"${a.text}"` : null) || a.key || a.url || "";
  return `${a.kind.replaceAll("_", " ")} ${arg}`.trim();
}

// ------------------------------------------------------------------ transcript + decisions

function renderTranscript() {
  const box = $("transcript");
  box.textContent = "";
  const marks = new Set(fired.get(current.uid) || []);
  current.words.forEach((w, i) => {
    const s = document.createElement("span");
    s.textContent = w + " ";
    if (i < current.consumed) s.className = "done";
    box.appendChild(s);
    if (marks.has(i + 1)) {
      const f = document.createElement("span");
      f.className = "fire";
      f.textContent = "⚡";
      f.title = `fired after word ${i + 1}`;
      box.appendChild(f);
    }
  });
}

function onTranscript(p) {
  if (current.uid !== p.uid) current = { uid: p.uid, words: [], consumed: 0 };
  current.words = p.text.split(/\s+/).filter(Boolean);
  current.consumed = Math.min(p.cursor ?? 0, current.words.length);
  lastWords.set(p.uid, current.words.length);
  renderTranscript();
  if (p.final) updateFiredLine(p.uid);
}

function updateFiredLine(uid) {
  const ks = fired.get(uid);
  const n = lastWords.get(uid);
  if (ks && ks.length && n) $("fired").textContent = `fired at word ${ks.join(", ")} of ${n}`;
}

function addDecision(d) {
  const li = document.createElement("li");
  const v = document.createElement("span");
  v.className = `v ${d.verdict}`;
  v.textContent = d.verdict;
  const lat = document.createElement("span");
  lat.className = "lat";
  const m = d.timings && d.timings.model;
  lat.textContent = m ? `${Math.round(m)} ms${d.passes === 2 ? " ·2" : ""}` : "";
  const txt = document.createElement("span");
  txt.textContent = d.action ? `${describe(d.action)}: ${d.reason}` : d.reason;
  const said = document.createElement("div");
  said.className = "said";
  said.textContent = `“${d.text}”${d.isFinal ? "" : " (partial)"}${d.stale ? " (stale)" : ""}`;
  li.append(v, lat, txt, said);
  const ol = $("decisions");
  ol.prepend(li);
  while (ol.children.length > 60) ol.lastChild.remove();
}

function addExec(e) {
  const li = document.createElement("li");
  const v = document.createElement("span");
  v.className = `v ${e.dryRun ? "clarify" : e.outcome === "ok" ? "act" : "deny"}`;
  v.textContent = e.dryRun ? "dry run" : e.outcome === "ok" ? "done" : "failed";
  const txt = document.createElement("span");
  txt.textContent = e.detail;
  li.append(v, txt);
  if (e.firedAtWord && e.uid) {
    const arr = fired.get(e.uid) || [];
    arr.push(e.firedAtWord);
    fired.set(e.uid, arr);
    const said = document.createElement("div");
    said.className = "said";
    said.textContent = `fired at word ${e.firedAtWord} of ${e.wordsHeard} heard so far`;
    li.append(said);
    renderTranscript();
    updateFiredLine(e.uid);
  }
  $("decisions").prepend(li);
}

function note(text, cls = "deny") {
  const li = document.createElement("li");
  const v = document.createElement("span");
  v.className = `v ${cls}`;
  v.textContent = cls === "deny" ? "note" : cls;
  const t = document.createElement("span");
  t.textContent = text;
  li.append(v, t);
  $("decisions").prepend(li);
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.to !== "panel") return;
  if (msg.type === "status") renderStatus(msg.payload);
  else if (msg.type === "progress") renderProgress(msg.payload);
  else if (msg.type === "ui") {
    const { event, payload } = msg.payload;
    if (event === "transcript") onTranscript(payload);
    else if (event === "decision") { if (payload.verdict !== "wait" || payload.isFinal) addDecision(payload); else addDecisionQuiet(payload); }
    else if (event === "executed") addExec(payload);
    else if (event === "pending") renderPending(payload);
    else if (event === "picks") note(`Which one? Say "number 1" to "number ${payload.eids.length}": ${payload.labels.join(" · ")}`, "clarify");
    else if (event === "error") note(payload);
    else if (event === "info") note(payload, "clarify");
    else if (event === "denied") note(payload.reason);
    else if (event === "killed") note(`Stopped (${payload.from}). Nothing more will run until you start again.`);
  }
});

// Waits are frequent mid-sentence; show the latest one in place instead of flooding the list.
function addDecisionQuiet(d) {
  const ol = $("decisions");
  const first = ol.firstElementChild;
  if (first && first.dataset.wait === "1") first.remove();
  addDecision(d);
  ol.firstElementChild.dataset.wait = "1";
}

// ------------------------------------------------------------------ controls

$("mic").addEventListener("click", async () => {
  const r = await brain(st.listening ? "stop" : "start");
  if (r && r.ok === false) {
    note(r.error);
    if (/Permission|NotAllowed|mic/i.test(r.error)) $("micHelp").hidden = false;
  }
});
$("kill").addEventListener("click", () => brain("kill", { from: "button" }));
$("confirmBtn").addEventListener("click", () => brain("confirm"));
$("cancelBtn").addEventListener("click", () => brain("cancel"));
$("grantMic").addEventListener("click", () => chrome.runtime.sendMessage({ to: "sw", type: "open_mic_page" }));
$("clearLog").addEventListener("click", () => { $("decisions").textContent = ""; });
$("textForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const t = $("textInput").value.trim();
  if (!t) return;
  $("textInput").value = "";
  const r = await brain("text", { text: t, options: { stream: true, resume: true } });
  if (r && r.ok === false) note(r.error);
});
$("save").addEventListener("click", async () => {
  const live = $("live").checked;
  if (live && settings.dryRun && !confirm("Live mode lets GenClass really click, type and navigate in your browser. Risky actions still need a spoken \"confirm\". Turn on live mode?")) {
    $("live").checked = false;
    return;
  }
  await saveSettings({
    speechEngine: $("speechEngine").value, dryRun: !live, compute: $("compute").value, searchUrl: $("searchUrl").value.trim() || DEFAULTS.searchUrl,
    maxElements: Math.max(10, Math.min(100, Number($("maxElements").value) || 60)), staged: $("staged").checked, modelBaseUrl: $("modelBaseUrl").value.trim(),
  });
  note("Settings saved.", "clarify");
});
$("speechEngine").addEventListener("change", () => { settings.speechEngine = $("speechEngine").value; renderSpeechSize(); renderMode(); });
$("clearCache").addEventListener("click", async () => { await brain("clear_cache"); note("Downloaded models cleared; they download again next time.", "clarify"); });

async function micPermission() {
  try {
    const p = await navigator.permissions.query({ name: "microphone" });
    $("micHelp").hidden = p.state === "granted" || settings.speechEngine === "chrome";
    p.onchange = () => { $("micHelp").hidden = p.state === "granted"; };
  } catch { /* permissions API unavailable */ }
}

// ------------------------------------------------------------------ features

const FEAT_DEFAULTS = {
  filter: { enabled: false, hide: { ad: true, sponsored: true, clickbait: false, off_topic: false }, sites: {}, model: false },
  focus: { enabled: false, task: "", action: "discard", graceMin: 5 },
  ram: { enabled: false, minFreePct: 15, idleMin: 10, maxPerRound: 3 },
};
let host = "";

function feat() {
  return {
    filter: { ...FEAT_DEFAULTS.filter, ...(settings.filter || {}), hide: { ...FEAT_DEFAULTS.filter.hide, ...((settings.filter || {}).hide || {}) } },
    focus: { ...FEAT_DEFAULTS.focus, ...(settings.focus || {}) },
    ram: { ...FEAT_DEFAULTS.ram, ...(settings.ram || {}) },
  };
}

async function renderFeatures() {
  const f = feat();
  host = await chrome.runtime.sendMessage({ to: "sw", type: "active_host" }).catch(() => "");
  $("fHost").textContent = host || "this page";
  $("fOn").checked = f.filter.enabled;
  $("fSite").checked = f.filter.sites[host] !== false;
  $("fAd").checked = f.filter.hide.ad; $("fSp").checked = f.filter.hide.sponsored;
  $("fCb").checked = f.filter.hide.clickbait; $("fOt").checked = f.filter.hide.off_topic;
  $("fModel").checked = f.filter.model;
  $("focusTask").value = f.focus.task; $("focusOn").checked = f.focus.enabled;
  $("focusAction").value = f.focus.action; $("focusGrace").value = f.focus.graceMin;
  $("ramOn").checked = f.ram.enabled; $("ramPct").value = f.ram.minFreePct; $("ramIdle").value = f.ram.idleMin;
  await renderLists();
}

function listItem(text, btnLabel, onClick, tag) {
  const li = document.createElement("li");
  const t = document.createElement("span");
  t.className = "t";
  t.textContent = text;
  t.title = text;
  if (tag) { const g = document.createElement("span"); g.className = `tag ${tag[1]}`; g.textContent = tag[0]; li.append(g); }
  li.append(t);
  if (btnLabel) { const b = document.createElement("button"); b.className = "link small"; b.textContent = btnLabel; b.onclick = onClick; li.append(b); }
  return li;
}

async function renderLists() {
  const { parked = [], ramFreed = [], ramStatus = null } = await chrome.storage.local.get(["parked", "ramFreed", "ramStatus"]);
  const restore = (e) => () => chrome.runtime.sendMessage({ to: "sw", type: "tab_op", op: "restore", entry: e });
  $("parkedList").replaceChildren(...parked.slice(0, 20).map((e) => listItem(e.title || e.url, "restore", restore(e), [e.op === "close" ? "closed" : "parked", "off"])));
  $("ramList").replaceChildren(...ramFreed.slice(0, 20).map((e) => listItem(e.title || e.url, "restore", restore(e), ["freed", "off"])));
  $("ramStatus").textContent = ramStatus ? `Free memory ${ramStatus.freePct}% of ${ramStatus.capacityGB} GB (${new Date(ramStatus.t).toLocaleTimeString()})` : "";
}

function renderFocus(st) {
  if (!st) return;
  $("focusTabs").replaceChildren(...(st.tabs || []).slice(0, 30).map((t) => listItem(`${t.title}`, t.onTask ? null : "keep",
    () => chrome.runtime.sendMessage({ to: "brain", type: "focus_keep", tabId: t.id }),
    t.onTask ? ["on task", "on"] : [t.scheduled ? "parking soon" : "off task", "off"])));
}

$("featSave").addEventListener("click", async () => {
  const f = feat();
  const sites = { ...f.filter.sites };
  if (host) sites[host] = $("fSite").checked;
  await saveSettings({
    filter: { ...f.filter, enabled: $("fOn").checked, sites, model: $("fModel").checked,
      hide: { ad: $("fAd").checked, sponsored: $("fSp").checked, clickbait: $("fCb").checked, off_topic: $("fOt").checked } },
    focus: { ...f.focus, enabled: $("focusOn").checked && !!$("focusTask").value.trim(), task: $("focusTask").value.trim(), action: $("focusAction").value,
      graceMin: Math.max(1, Number($("focusGrace").value) || 5) },
    ram: { ...f.ram, enabled: $("ramOn").checked, minFreePct: Math.max(5, Math.min(50, Number($("ramPct").value) || 15)), idleMin: Math.max(1, Number($("ramIdle").value) || 10) },
  });
  if ($("focusOn").checked && !$("focusTask").value.trim()) note("Focus mode needs a task description.");
  note("Features saved.", "clarify");
});
$("ramNow").addEventListener("click", async () => {
  const r = await chrome.runtime.sendMessage({ to: "sw", type: "ram_tick" });
  note(r ? `Free memory ${r.freePct}%${r.parked ? `; parked ${r.parked} tab(s)` : ""}` : "RAM manager is off.", "clarify");
  renderLists();
});
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.to !== "panel") return;
  if (msg.type === "lists_changed") renderLists();
  if (msg.type === "focus") renderFocus(msg.payload);
});
chrome.storage.onChanged.addListener((changes) => {
  if (changes.settings && JSON.stringify(changes.settings.newValue) !== JSON.stringify(settings)) {
    settings = { ...DEFAULTS, ...(changes.settings.newValue || {}) };
    $("live").checked = !settings.dryRun;
    $("speechEngine").value = settings.speechEngine;
    renderMode();
    renderSpeechSize();
    renderFeatures();
  }
  if (changes.parked || changes.ramFreed || changes.ramStatus) renderLists();
});
chrome.tabs && chrome.tabs.onActivated && chrome.tabs.onActivated.addListener(() => renderFeatures());

(async () => {
  await loadSettings();
  renderFeatures();
  chrome.runtime.sendMessage({ to: "brain", type: "focus_state" }).then(renderFocus).catch(() => {});
  await micPermission();
  $("loading").hidden = false;
  $("loadTitle").textContent = "Starting GenClass…";
  await chrome.runtime.sendMessage({ to: "sw", type: "ensure_brain" });
  const r = await brain("init", { settings });
  if (r && r.ok === false) renderStatus({ phase: "error", error: r.error });
  else { $("loading").hidden = true; brain("status"); }
})();
