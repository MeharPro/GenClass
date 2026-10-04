// Feature runtime in the brain: model-backed content-filter classification, focus mode, and tab relevance for
// the RAM manager. Tab operations go through the service worker.

import { filterRequest, filterResults, focusRequest, focusResults, focusVerdict, protectedTab } from "../core/features.js";

const sw = (msg) => chrome.runtime.sendMessage({ to: "sw", ...msg });
const toPanel = (type, payload) => chrome.runtime.sendMessage({ to: "panel", type, payload }).catch(() => {});

export class Features {
  constructor(getEngine, getSettings) {
    this.getEngine = getEngine;
    this.getSettings = getSettings;
    this.focus = { on: false, task: "", startedAt: 0, used: new Set(), onTask: new Set(), timers: new Map(), scores: new Map() };
    this.queue = Promise.resolve();
  }

  /** Serialize model work behind voice decisions (voice always wins: features are background work). */
  run(fn) {
    const p = this.queue.then(fn, fn);
    this.queue = p.catch(() => {});
    return p;
  }

  // ------------------------------------------------------------------ content filter (model part)

  async classifyBlocks(blocks, page) {
    const s = this.getSettings();
    const task = s.focus && s.focus.enabled ? s.focus.task : "";
    const out = [];
    for (let i = 0; i < blocks.length; i += 10) {
      const chunk = blocks.slice(i, i + 10);
      const req = filterRequest(page, chunk, { task });
      let resp = null;
      if (Object.keys(req.questions).length) resp = await this.run(() => this.getEngine().evaluate(req.state, req.questions));
      for (const [id, r] of filterResults(resp, req.preset)) out.push([id, r.kind, r.p]);
    }
    return out;
  }

  // ------------------------------------------------------------------ focus mode

  async relevance(task, tabs) {
    const scores = new Map();
    for (let i = 0; i < tabs.length; i += 16) {
      const chunk = tabs.slice(i, i + 16);
      const req = focusRequest(task, chunk);
      const resp = await this.run(() => this.getEngine().evaluate(req.state, req.questions));
      for (const [id, p] of focusResults(resp)) scores.set(id, p);
    }
    return scores;
  }

  /** Serialized: the settings listener and the panel may both ask at once. */
  async focusConfigure() {
    while (this.cfgP) await this.cfgP.catch(() => {});
    this.cfgP = this.focusConfigureOnce();
    try { return await this.cfgP; } finally { this.cfgP = null; }
  }

  async focusConfigureOnce() {
    const f = this.getSettings().focus || {};
    const was = this.focus.on;
    if (!f.enabled || !f.task) {
      this.focus.on = false;
      for (const t of this.focus.timers.values()) clearTimeout(t);
      this.focus.timers.clear();
      if (was) toPanel("focus", this.focusState());
      return;
    }
    if (!was || f.task !== this.focus.task) {
      this.focus = { ...this.focus, on: true, task: f.task, startedAt: Date.now(), used: new Set(), onTask: new Set(), scores: new Map() };
      const tabs = await sw({ type: "tab_list_full" });
      const active = tabs.find((t) => t.active);
      if (active) this.focus.used.add(active.id);
      await this.scoreTabs(tabs);
    }
  }

  async scoreTabs(tabs) {
    if (!this.focus.on) return;
    const cand = tabs.filter((t) => t.url && /^https?:/.test(t.url));
    const infos = await Promise.all(cand.map((t) => sw({ type: "page_info", tabId: t.id }).catch(() => null)));
    const withDesc = cand.map((t, i) => ({ ...t, description: (infos[i] && infos[i].description) || "" , dirty: !!(infos[i] && infos[i].dirty) }));
    const scores = await this.relevance(this.focus.task, withDesc);
    for (const t of withDesc) {
      const v = focusVerdict(this.focus.task, t, scores.get(t.id) ?? 0);
      this.focus.scores.set(t.id, { ...v, title: t.title, url: t.url });
      if (v.onTask) this.focus.onTask.add(t.id);
      else this.scheduleOffTask(t);
    }
    toPanel("focus", this.focusState());
  }

  scheduleOffTask(t) {
    if (this.focus.timers.has(t.id) || this.focus.onTask.has(t.id)) return;
    const graceMs = Math.max(0.25, Number(this.getSettings().focus.graceMin ?? 5)) * 60000;
    this.focus.timers.set(t.id, setTimeout(() => this.actOffTask(t.id), graceMs));
  }

  async actOffTask(tabId) {
    this.focus.timers.delete(tabId);
    if (!this.focus.on || this.focus.used.has(tabId) || this.focus.onTask.has(tabId)) return;
    const tab = await sw({ type: "tab_get", tabId }).catch(() => null);
    if (!tab) return;
    const info = await sw({ type: "page_info", tabId }).catch(() => null);
    const why = protectedTab(tab, { dirty: info && info.dirty });
    if (why) return;
    const op = this.getSettings().focus.action === "close" ? "close" : "discard";
    await sw({ type: "tab_op", op, tabId, reason: "focus", title: tab.title, url: tab.url });
    toPanel("focus", this.focusState());
  }

  /** Tab events from the service worker. */
  async onTab(ev) {
    if (!this.focus.on) return;
    if (ev.kind === "activated") {
      // Using a tab during the session marks it as on-task (the model misses semantic relevance often).
      this.focus.used.add(ev.tabId);
      const t = this.focus.timers.get(ev.tabId);
      if (t) { clearTimeout(t); this.focus.timers.delete(ev.tabId); }
      return;
    }
    if (ev.kind === "updated" && ev.tab) {
      if (ev.tab.openerTabId && (this.focus.onTask.has(ev.tab.openerTabId) || this.focus.used.has(ev.tab.openerTabId))) {
        this.focus.onTask.add(ev.tab.id); // opened from an on-task tab
        return;
      }
      await this.scoreTabs([ev.tab]);
    }
    if (ev.kind === "removed") {
      const t = this.focus.timers.get(ev.tabId);
      if (t) clearTimeout(t);
      this.focus.timers.delete(ev.tabId);
      this.focus.scores.delete(ev.tabId);
    }
  }

  focusState() {
    return {
      on: this.focus.on, task: this.focus.task,
      tabs: [...this.focus.scores.entries()].map(([id, s]) => ({ id, title: s.title, url: s.url, p: s.p, overlap: s.overlap,
        onTask: s.onTask || this.focus.onTask.has(id) || this.focus.used.has(id), scheduled: this.focus.timers.has(id) })),
    };
  }

  keep(tabId) {
    this.focus.onTask.add(tabId);
    const t = this.focus.timers.get(tabId);
    if (t) { clearTimeout(t); this.focus.timers.delete(tabId); }
    toPanel("focus", this.focusState());
  }
}

export { protectedTab };
