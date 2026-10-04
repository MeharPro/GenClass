// GenClass service worker: opens the side panel, keeps the offscreen "brain" (speech + model + decision loop)
// alive, routes page observation and in-page actions to the content script, and performs tab-level actions.

import { discardOrder, filterRequest, protectedTab } from "../core/features.js";

const RESTRICTED_RE = /^(chrome|chrome-extension|chrome-untrusted|devtools|edge|brave|opera|vivaldi|about|view-source|file|data|javascript):|^https:\/\/(chromewebstore\.google\.com|chrome\.google\.com\/webstore)/i;
// Sites GenClass never acts inside (banking, payments, crypto, password managers). Navigation away is still allowed.
const DENY_SITE_RE = /(^|\.)(bank|.*bank|paypal|venmo|wise|revolut|stripe|coinbase|binance|kraken|robinhood|etrade|schwab|fidelity|vanguard|1password|bitwarden|lastpass|dashlane|keepersecurity|nordpass|proton)\.|(^|\.)(accounts\.google|myaccount\.google|appleid\.apple|login\.microsoftonline)\.com$/i;

chrome.runtime.onInstalled.addListener(async (d) => {
  await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
  if (d.reason === "install") chrome.tabs.create({ url: chrome.runtime.getURL("welcome.html") });
});
chrome.runtime.onStartup.addListener(() => chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {}));

let creating = null;
async function ensureBrain() {
  if (await chrome.offscreen.hasDocument()) return true;
  if (!creating) {
    creating = chrome.offscreen.createDocument({
      url: "offscreen.html",
      reasons: ["USER_MEDIA", "WORKERS"],
      justification: "Speech recognition from the microphone and the local decision model run here while the side panel is open.",
    }).finally(() => { creating = null; });
  }
  await creating;
  return true;
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab || null;
}

async function ensureContent(tabId) {
  try {
    const r = await chrome.tabs.sendMessage(tabId, { to: "content", type: "ping" });
    if (r === "pong") return true;
  } catch { /* not injected yet (tab opened before install) */ }
  await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
  return true;
}

function hostOf(url) {
  try { return new URL(url).hostname; } catch { return ""; }
}

async function targetTab(msg) {
  if (msg && msg.targetTabId) return chrome.tabs.get(msg.targetTabId).catch(() => null);
  return activeTab();
}

async function observe(msg) {
  const tab = await targetTab(msg);
  if (!tab) return null;
  const base = { appName: "Google Chrome", windowTitle: tab.title || "", url: tab.url || "", tabId: tab.id, browser: true, elements: [], focusedEid: null };
  if (!tab.url || RESTRICTED_RE.test(tab.url)) return { ...base, restricted: true, windowTitle: tab.title || "browser page" };
  const host = hostOf(tab.url);
  const deniedSite = DENY_SITE_RE.test(host) ? host : null;
  try {
    await ensureContent(tab.id);
    const snap = await chrome.tabs.sendMessage(tab.id, { to: "content", type: "observe", tail: msg.tail || "", max: msg.max || 60 });
    return { ...snap, tabId: tab.id, deniedSite };
  } catch (e) {
    return { ...base, deniedSite, error: String(e) };
  }
}

async function tabsList() {
  const tabs = await chrome.tabs.query({});
  return tabs.map((t) => ({ id: t.id, title: t.title, url: t.url, active: t.active }));
}

async function navigate(tab, url) {
  if (tab) await chrome.tabs.update(tab.id, { url });
  else await chrome.tabs.create({ url });
}

/** Tab-level actions here; in-page ones go to the content script of the snapshot's tab. */
async function execute({ action: a, snap, dryRun, searchUrl, resolved, undoOf, said }) {
  const tab = snap && snap.tabId ? await chrome.tabs.get(snap.tabId).catch(() => null) : await targetTab(arguments[0]);
  const dry = (detail) => ({ ok: true, changed: false, dryRun: true, detail: `dry run: would ${detail}` });
  switch (a.kind) {
    case "new_tab": {
      if (dryRun) return dry("open a new tab");
      const t = await chrome.tabs.create({});
      return { ok: true, changed: true, detail: "opened a new tab", undo: { closeTab: t.id } };
    }
    case "close_tab": {
      if (!tab) return { ok: false, detail: "no tab" };
      if (dryRun) return dry(`close the tab "${tab.title}"`);
      await chrome.tabs.remove(tab.id);
      return { ok: true, changed: true, detail: `closed "${tab.title}"` };
    }
    case "go_back":
    case "go_forward": {
      if (!tab) return { ok: false, detail: "no tab" };
      if (dryRun) return dry(a.kind === "go_back" ? "go back" : "go forward");
      await (a.kind === "go_back" ? chrome.tabs.goBack(tab.id) : chrome.tabs.goForward(tab.id)).catch(() => {});
      return { ok: true, changed: true, detail: a.kind === "go_back" ? "went back" : "went forward" };
    }
    case "open_url": {
      if (dryRun) return dry(`open ${a.url}`);
      await navigate(tab, a.url);
      return { ok: true, changed: true, detail: `opened ${a.url}`, undo: { back: tab && tab.id } };
    }
    case "search_web": {
      const url = (searchUrl || "https://www.google.com/search?q={q}").replace("{q}", encodeURIComponent(a.text));
      if (dryRun) return dry(`search the web for "${a.text}"`);
      await navigate(tab, url);
      return { ok: true, changed: true, detail: `searched for "${a.text}"`, undo: { back: tab && tab.id } };
    }
    case "open_app": {
      if (!resolved) return { ok: false, detail: `no tab or site called ${a.app}` };
      if (resolved.tabId !== undefined) {
        if (dryRun) return dry(`switch to the ${a.app} tab`);
        const t = await chrome.tabs.update(resolved.tabId, { active: true });
        await chrome.windows.update(t.windowId, { focused: true }).catch(() => {});
        return { ok: true, changed: true, detail: `switched to ${a.app}` };
      }
      if (dryRun) return dry(`open ${a.app} (${resolved.url})`);
      const reuse = tab && (!tab.url || /^chrome:\/\/newtab/.test(tab.url));
      if (reuse) await chrome.tabs.update(tab.id, { url: resolved.url });
      else await chrome.tabs.create({ url: resolved.url });
      return { ok: true, changed: true, detail: `opened ${a.app}` };
    }
    case "quit_app": {
      if (!resolved || resolved.tabId === undefined) return { ok: false, detail: `no open tab called ${a.app}` };
      if (dryRun) return dry(`close the ${a.app} tab`);
      await chrome.tabs.remove(resolved.tabId);
      return { ok: true, changed: true, detail: `closed the ${a.app} tab` };
    }
    case "open_folder":
      return { ok: false, detail: "GenClass cannot open folders from the browser" };
    case "undo": {
      if (!undoOf) return { ok: false, detail: "nothing to undo" };
      if (dryRun) return dry(`undo ${undoOf.action.kind}`);
      const u = undoOf.undo || {};
      if (u.closeTab) { await chrome.tabs.remove(u.closeTab).catch(() => {}); return { ok: true, changed: true, detail: "closed the tab it opened" }; }
      if (u.back !== undefined && u.back !== null) { await chrome.tabs.goBack(u.back).catch(() => {}); return { ok: true, changed: true, detail: "went back" }; }
      const k = undoOf.action.kind;
      if (k === "scroll_down" || k === "scroll_up") {
        return forward(tab, { kind: k === "scroll_down" ? "scroll_up" : "scroll_down", amount: undoOf.action.amount }, snap, false);
      }
      if (k === "type_text") return forward(tab, { kind: "undo_type" }, snap, false);
      return forward(tab, { kind: "press_key", key: "cmd+z" }, snap, false);
    }
    case "press_key":
      if (a.key === "cmd+r") {
        if (!tab) return { ok: false, detail: "no tab" };
        if (dryRun) return dry("reload the page");
        await chrome.tabs.reload(tab.id);
        return { ok: true, changed: true, detail: "reloaded" };
      }
      if (a.key === "cmd+n") {
        if (dryRun) return dry("open a new window");
        await chrome.windows.create({});
        return { ok: true, changed: true, detail: "opened a new window" };
      }
      return forward(tab, a, snap, dryRun);
    default:
      return forward(tab, a, snap, dryRun, said);
  }
}

async function forward(tab, action, snap, dryRun, said = "") {
  if (!tab) return { ok: false, detail: "no tab" };
  if (!tab.url || RESTRICTED_RE.test(tab.url)) return { ok: false, detail: "refused: browser-internal page" };
  await ensureContent(tab.id);
  return chrome.tabs.sendMessage(tab.id, { to: "content", type: "execute", action, snapId: snap && snap.id, dryRun, said });
}

async function broadcastListening(on) {
  const tabs = await chrome.tabs.query({});
  for (const t of tabs) {
    if (t.url && !RESTRICTED_RE.test(t.url)) chrome.tabs.sendMessage(t.id, { to: "content", type: "listening", on }).catch(() => {});
  }
}


// ------------------------------------------------------------------ settings + feature plumbing

const DEFAULT_FEATURES = {
  filter: { enabled: false, hide: { ad: true, sponsored: true, clickbait: false, off_topic: false }, sites: {}, model: false },
  focus: { enabled: false, task: "", action: "discard", graceMin: 5 },
  ram: { enabled: false, minFreePct: 15, idleMin: 10, maxPerRound: 3 },
};

async function getSettings() {
  const { settings } = await chrome.storage.local.get("settings");
  const s = settings || {};
  return { ...s, filter: { ...DEFAULT_FEATURES.filter, ...(s.filter || {}) }, focus: { ...DEFAULT_FEATURES.focus, ...(s.focus || {}) }, ram: { ...DEFAULT_FEATURES.ram, ...(s.ram || {}) } };
}

async function brainCall(msg) {
  await ensureBrain();
  const settings = await getSettings();
  return chrome.runtime.sendMessage({ to: "brain", settings, ...msg });
}

async function pushList(key, item, max = 50) {
  const cur = (await chrome.storage.local.get(key))[key] || [];
  cur.unshift(item);
  await chrome.storage.local.set({ [key]: cur.slice(0, max) });
  chrome.runtime.sendMessage({ to: "panel", type: "lists_changed" }).catch(() => {});
}

async function pageInfo(tabId) {
  try {
    const t = await chrome.tabs.get(tabId);
    if (!t.url || RESTRICTED_RE.test(t.url) || t.discarded) return { title: t.title, url: t.url, description: "", dirty: false };
    await ensureContent(tabId);
    return await chrome.tabs.sendMessage(tabId, { to: "content", type: "page_info" });
  } catch {
    return null;
  }
}

/** discard | close | restore, with an undo list ("parked" for focus mode, "ramFreed" for the RAM manager). */
async function tabOp({ op, tabId, reason, title, url, entry }) {
  if (op === "restore") {
    const e = entry;
    const live = e.tabId !== undefined ? await chrome.tabs.get(e.tabId).catch(() => null) : null;
    if (live) await chrome.tabs.update(live.id, { active: true });
    else await chrome.tabs.create({ url: e.url });
    const key = e.reason === "ram" ? "ramFreed" : "parked";
    const cur = (await chrome.storage.local.get(key))[key] || [];
    await chrome.storage.local.set({ [key]: cur.filter((x) => x.t !== e.t) });
    chrome.runtime.sendMessage({ to: "panel", type: "lists_changed" }).catch(() => {});
    return { ok: true };
  }
  const t = await chrome.tabs.get(tabId).catch(() => null);
  if (!t) return { ok: false, detail: "tab is gone" };
  const info = await pageInfo(tabId);
  const why = protectedTab(t, { dirty: info && info.dirty });
  if (why) return { ok: false, detail: `kept: ${why}` };
  if (op === "discard") {
    if (t.discarded) return { ok: false, detail: "already parked" };
    await chrome.tabs.discard(tabId);
  } else if (op === "close") {
    await chrome.tabs.remove(tabId);
  } else return { ok: false, detail: `unknown op ${op}` };
  await pushList(reason === "ram" ? "ramFreed" : "parked", { t: Date.now(), op, reason, tabId: op === "discard" ? tabId : undefined, title: title || t.title, url: url || t.url });
  return { ok: true };
}

// Content filter: deterministic signals here; optional model classification in the brain.
async function filterBlocks({ blocks, page }, sender) {
  const s = await getSettings();
  const f = s.filter;
  if (!f.enabled || f.sites[page.host] === false) return { hide: [] };
  const req = filterRequest(page, blocks, { task: s.focus.enabled ? s.focus.task : "" });
  const hide = [];
  for (const [id, kind] of req.preset) if (f.hide[kind]) hide.push([id, kind]);
  // Containers matched by known ad selectors (ins.adsbygoogle, div-gpt-ad, data-ad-slot, ...) are ads.
  for (const b of blocks) if (b.adSelector && !req.preset.has(b.id) && f.hide.ad) hide.push([b.id, "ad"]);
  if (f.model) {
    const rest = blocks.filter((b) => !req.preset.has(b.id) && !b.adSelector);
    if (rest.length) {
      const r = await brainCall({ type: "filter_classify", blocks: rest, page }).catch(() => null);
      for (const [id, kind, p] of (r && r.results) || []) if (kind !== "normal" && f.hide[kind] && p >= 0.6) hide.push([id, kind]);
    }
  }
  return { hide };
}

async function setFilterForAll() {
  const s = await getSettings();
  const tabs = await chrome.tabs.query({});
  for (const t of tabs) {
    if (!t.url || RESTRICTED_RE.test(t.url)) continue;
    const on = s.filter.enabled && s.filter.sites[hostOf(t.url)] !== false;
    chrome.tabs.sendMessage(t.id, { to: "content", type: "filter_set", on }).catch(() => {});
    if (!on) chrome.action.setBadgeText({ tabId: t.id, text: "" }).catch(() => {});
  }
}

// RAM manager: chrome.system.memory every minute; under pressure park idle, low-relevance tabs.
/** preview: report the tabs that would be parked without touching them. */
async function ramTick({ preview = false } = {}) {
  const s = await getSettings();
  if (!s.ram.enabled && !preview) return null;
  const mem = await chrome.system.memory.getInfo();
  const freePct = (100 * mem.availableCapacity) / mem.capacity;
  const status = { t: Date.now(), freePct: Math.round(freePct * 10) / 10, capacityGB: Math.round(mem.capacity / 1e8) / 10, source: "chrome.system.memory" };
  await chrome.storage.local.set({ ramStatus: status });
  chrome.runtime.sendMessage({ to: "panel", type: "lists_changed" }).catch(() => {});
  if (freePct >= s.ram.minFreePct && !preview) return status;
  const now = Date.now();
  const tabs = (await chrome.tabs.query({})).filter((t) => !t.discarded && !t.active && !t.pinned && !t.audible &&
    t.url && /^https?:/.test(t.url) && (!t.lastAccessed || now - t.lastAccessed > s.ram.idleMin * 60000));
  let relevance = new Map();
  if (s.focus.enabled && s.focus.task && tabs.length) {
    const r = await brainCall({ type: "relevance", task: s.focus.task, tabs: tabs.map((t) => ({ id: t.id, title: t.title, url: t.url })) }).catch(() => null);
    if (r && r.scores) relevance = new Map(r.scores);
  }
  let n = 0;
  if (preview) {
    const order = [];
    for (const t of discardOrder(tabs, relevance)) {
      const info = await pageInfo(t.id);
      if (!protectedTab(t, { dirty: info && info.dirty })) order.push({ id: t.id, title: t.title });
      if (order.length >= s.ram.maxPerRound) break;
    }
    return { ...status, wouldPark: order, pressure: freePct < s.ram.minFreePct };
  }
  for (const t of discardOrder(tabs, relevance)) {
    if (n >= s.ram.maxPerRound) break;
    const r = await tabOp({ op: "discard", tabId: t.id, reason: "ram", title: t.title, url: t.url });
    if (r.ok) n++;
  }
  return { ...status, parked: n };
}

async function configureAlarms() {
  const s = await getSettings();
  if (s.ram.enabled) chrome.alarms.create("genclass-ram", { periodInMinutes: 1 });
  else chrome.alarms.clear("genclass-ram");
}
chrome.alarms.onAlarm.addListener((a) => { if (a.name === "genclass-ram") ramTick(); });

chrome.storage.onChanged.addListener(async (changes) => {
  if (!changes.settings) return;
  const o = changes.settings.oldValue || {};
  const n = changes.settings.newValue || {};
  if (JSON.stringify(o.filter) !== JSON.stringify(n.filter)) setFilterForAll();
  if (JSON.stringify(o.ram) !== JSON.stringify(n.ram)) configureAlarms();
  if (JSON.stringify(o.focus) !== JSON.stringify(n.focus)) {
    if ((n.focus && n.focus.enabled) || (o.focus && o.focus.enabled)) brainCall({ type: "focus_config" }).catch(() => {});
  }
});

async function focusEvent(event) {
  const s = await getSettings();
  if (!s.focus.enabled) return;
  brainCall({ type: "tab_event", event }).catch(() => {});
}
chrome.tabs.onActivated.addListener(({ tabId }) => focusEvent({ kind: "activated", tabId }));
chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (info.status === "complete") focusEvent({ kind: "updated", tabId, tab: { id: tab.id, title: tab.title, url: tab.url, openerTabId: tab.openerTabId } });
});
chrome.tabs.onRemoved.addListener((tabId) => focusEvent({ kind: "removed", tabId }));
configureAlarms();

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.to !== "sw") return;
  const done = (p) => { Promise.resolve(p).then(reply, (e) => reply({ ok: false, error: String(e && e.message || e) })); return true; };
  switch (msg.type) {
    case "ensure_brain": return done(ensureBrain());
    case "observe": return done(observe(msg));
    case "execute": return done(execute(msg));
    case "tabs": return done(tabsList());
    case "badges": return done(activeTab().then((t) => t && chrome.tabs.sendMessage(t.id, { to: "content", type: "badges", snapId: msg.snapId, eids: msg.eids })));
    case "listening": return done(broadcastListening(msg.on));
    case "filter_enabled_for": return done(getSettings().then((s) => s.filter.enabled && s.filter.sites[msg.host] !== false));
    case "filter_blocks": return done(filterBlocks(msg, sender));
    case "filter_count": return done(sender.tab ? chrome.action.setBadgeText({ tabId: sender.tab.id, text: msg.count ? String(msg.count) : "" })
      .then(() => chrome.action.setBadgeBackgroundColor({ tabId: sender.tab.id, color: "#6b7280" })) : null);
    case "filter_site": return done(getSettings().then(async (s) => {
      const sites = { ...s.filter.sites, [msg.host]: !!msg.on };
      const { settings } = await chrome.storage.local.get("settings");
      await chrome.storage.local.set({ settings: { ...(settings || {}), filter: { ...s.filter, sites } } });
      return true;
    }));
    case "tab_list_full": return done(chrome.tabs.query({}).then((ts) => ts.map((t) => ({ id: t.id, title: t.title, url: t.url, active: t.active, pinned: t.pinned, audible: t.audible, discarded: t.discarded, lastAccessed: t.lastAccessed, openerTabId: t.openerTabId }))));
    case "tab_get": return done(chrome.tabs.get(msg.tabId));
    case "page_info": return done(pageInfo(msg.tabId));
    case "tab_op": return done(tabOp(msg));
    case "ram_tick": return done(ramTick({ preview: !!msg.preview }));
    case "active_host": return done(activeTab().then((t) => (t && t.url ? hostOf(t.url) : "")));
    case "open_mic_page": return done(chrome.tabs.create({ url: chrome.runtime.getURL("welcome.html#mic") }));
    default: return false;
  }
});

chrome.commands.onCommand.addListener((cmd) => {
  if (cmd === "kill-switch") chrome.runtime.sendMessage({ to: "brain", type: "kill", from: "shortcut" }).catch(() => {});
  if (cmd === "toggle-listening") ensureBrain().then(() => chrome.runtime.sendMessage({ to: "brain", type: "toggle" })).catch(() => {});
});
