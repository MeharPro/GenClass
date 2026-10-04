// GenClass content script: observe the page (numbered actionable elements, in the Element/Snapshot format the
// model was trained on) and carry out in-page actions. Never types into password fields. Esc = kill switch.

import { SECURE_FIELD_RE } from "../core/catalog.js";

(() => {
  if (window.__genclassLoaded) return;
  window.__genclassLoaded = true;

  const MAX_LABEL = 60;
  const SELECTOR = [
    "a[href]", "button", "input:not([type=hidden])", "select", "textarea", "summary",
    "[role=button]", "[role=link]", "[role=checkbox]", "[role=radio]", "[role=tab]", "[role=menuitem]",
    "[role=menuitemcheckbox]", "[role=option]", "[role=switch]", "[role=combobox]", "[role=searchbox]",
    "[role=textbox]", "[role=slider]", "[contenteditable=''], [contenteditable=true]", "[onclick]",
  ].join(",");

  let snapSeq = 0;
  const snapshots = new Map(); // id -> Map(eid -> Element node)
  let listening = false;

  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const trunc = (s, n = MAX_LABEL) => (s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s);

  function isSecureInput(el) {
    if (!(el instanceof HTMLInputElement)) return false;
    const ac = (el.getAttribute("autocomplete") || "").toLowerCase();
    return el.type === "password" || /password|cc-number|cc-csc|cc-exp|one-time-code/.test(ac) ||
      SECURE_FIELD_RE.test(`${el.name} ${el.id} ${el.getAttribute("aria-label") || ""} ${el.placeholder || ""}`);
  }

  function roleOf(el) {
    const r = (el.getAttribute("role") || "").toLowerCase();
    const tag = el.tagName.toLowerCase();
    if (r === "searchbox") return "search field";
    if (r === "textbox") return el.getAttribute("aria-multiline") === "true" ? "text area" : "text field";
    if (r === "combobox") return "combo box";
    if (r === "checkbox" || r === "menuitemcheckbox") return "checkbox";
    if (r === "radio") return "radio button";
    if (r === "switch") return "switch";
    if (r === "tab") return "tab";
    if (r === "menuitem") return "menu item";
    if (r === "option") return "row";
    if (r === "slider") return "slider";
    if (r === "link") return "link";
    if (r === "button") return el.getAttribute("aria-haspopup") ? "menu button" : "button";
    if (tag === "a") return "link";
    if (tag === "button") return el.getAttribute("aria-haspopup") ? "menu button" : "button";
    if (tag === "select") return "pop up button";
    if (tag === "textarea") return "text area";
    if (tag === "summary") return "disclosure triangle";
    if (tag === "input") {
      const t = (el.type || "text").toLowerCase();
      if (["button", "submit", "reset", "image"].includes(t)) return "button";
      if (t === "checkbox") return "checkbox";
      if (t === "radio") return "radio button";
      if (t === "range") return "slider";
      if (t === "search") return "search field";
      if (["color", "file"].includes(t)) return "button";
      return "text field";
    }
    if (el.isContentEditable) return "text area";
    return "button";
  }

  function textOfIds(ids) {
    return ids.split(/\s+/).map((id) => document.getElementById(id)).filter(Boolean).map((n) => clean(n.innerText || n.textContent)).join(" ");
  }

  function labelOf(el) {
    const aria = el.getAttribute("aria-label");
    if (aria && clean(aria)) return clean(aria);
    const lb = el.getAttribute("aria-labelledby");
    if (lb) { const t = textOfIds(lb); if (t) return t; }
    if (el.labels && el.labels.length) {
      const t = clean([...el.labels].map((l) => l.innerText || l.textContent).join(" "));
      if (t) return t;
    }
    const tag = el.tagName.toLowerCase();
    if (tag === "input" && ["button", "submit", "reset"].includes((el.type || "").toLowerCase()) && el.value) return clean(el.value);
    if (el.placeholder) return clean(el.placeholder);
    const inner = ["input", "textarea", "select"].includes(tag) ? "" : clean(el.innerText || el.textContent);
    if (inner) return inner;
    if (el.title) return clean(el.title);
    const img = el.querySelector && el.querySelector("img[alt], svg[aria-label], [aria-label]");
    if (img) return clean(img.getAttribute("alt") || img.getAttribute("aria-label"));
    if (tag === "input" && (el.type || "").toLowerCase() === "image" && el.alt) return clean(el.alt);
    if (el.name) return clean(el.name);
    return "";
  }

  function valueOf(el, role) {
    if (el instanceof HTMLInputElement) {
      const t = (el.type || "").toLowerCase();
      if (t === "checkbox" || t === "radio") return el.checked ? "checked" : null;
      if (["button", "submit", "reset", "image", "file", "color"].includes(t)) return null;
      if (isSecureInput(el)) return null;
      return el.value ? trunc(clean(el.value)) : null;
    }
    if (el instanceof HTMLTextAreaElement) return el.value ? trunc(clean(el.value)) : null;
    if (el instanceof HTMLSelectElement) return el.selectedOptions[0] ? trunc(clean(el.selectedOptions[0].text)) : null;
    const ac = el.getAttribute("aria-checked");
    if (ac === "true") return "checked";
    if (role === "text area" && el.isContentEditable) { const t = clean(el.innerText); return t ? trunc(t) : null; }
    return null;
  }

  function contextOf(el) {
    for (let n = el.parentElement; n && n !== document.body; n = n.parentElement) {
      const r = (n.getAttribute("role") || "").toLowerCase();
      const tag = n.tagName.toLowerCase();
      if (tag === "dialog" || r === "dialog" || r === "alertdialog" || n.getAttribute("aria-modal") === "true") return "dialog";
      if (r === "search") return "search";
      if (tag === "nav" || r === "navigation") return "navigation";
      if (r === "toolbar") return "toolbar";
      if (r === "tablist") return "tab bar";
      if (r === "menu" || r === "menubar") return "menu";
      if (tag === "header" || r === "banner") return "header";
      if (tag === "footer" || r === "contentinfo") return "footer";
      if (tag === "aside" || r === "complementary") return "sidebar";
      if (tag === "form") return "form";
      if (tag === "article") return "post";
    }
    return null;
  }

  function formInfo(el) {
    const form = el.form || el.closest("form");
    if (!form) return { inForm: false, searchForm: false, submits: false };
    const sig = `${form.getAttribute("role") || ""} ${form.action || ""} ${form.id} ${form.className} ${form.getAttribute("aria-label") || ""}`;
    const searchForm = /search/i.test(sig) || !!form.querySelector("input[type=search], [role=searchbox], input[name=q], input[name=query], input[name=search_query]");
    const tag = el.tagName.toLowerCase();
    const t = (el.getAttribute("type") || "").toLowerCase();
    const submits = (tag === "button" && (t === "" || t === "submit")) || (tag === "input" && (t === "submit" || t === "image"));
    return { inForm: true, searchForm, submits };
  }

  function visible(el) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return null;
    if (r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) return null;
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden" || cs.display === "none" || Number(cs.opacity) === 0) return null;
    if (el.closest("[aria-hidden=true], [inert]")) return null;
    const cx = Math.min(innerWidth - 1, Math.max(0, r.left + r.width / 2));
    const cy = Math.min(innerHeight - 1, Math.max(0, r.top + r.height / 2));
    const hit = document.elementFromPoint(cx, cy);
    const occluded = hit && hit !== el && !el.contains(hit) && !hit.contains(el);
    return { r, occluded };
  }

  function deepActive() {
    let a = document.activeElement;
    while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
    return a;
  }

  /** -> Snapshot (without eids for the focused secure field's value) */
  function observe({ tail = "", max = 60 } = {}) {
    const t0 = performance.now();
    const active = deepActive();
    const seen = new Set();
    const raw = [];
    for (const el of document.querySelectorAll(SELECTOR)) {
      if (seen.has(el)) continue;
      seen.add(el);
      if (el.disabled && el.tagName === "INPUT" && el.type === "hidden") continue;
      const secure = isSecureInput(el);
      if (secure && el !== active) continue; // password fields are left out unless focused (then only to deny typing)
      const v = visible(el);
      if (!v) continue;
      if (v.occluded && el !== active) continue;
      // Skip a link/button wrapped inside another collected control (duplicates).
      const parentCtl = el.parentElement && el.parentElement.closest(SELECTOR);
      if (parentCtl && seen.has(parentCtl) && raw.some((x) => x.el === parentCtl) && !["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName)) continue;
      const role = roleOf(el);
      const label = trunc(labelOf(el));
      if (!label && !["text field", "search field", "text area"].includes(role)) continue;
      raw.push({ el, role, label, rect: v.r, secure });
    }
    // Over the cap: keep the ones whose words the speaker used, then the focused one, then screen order.
    let picked = raw;
    if (raw.length > max) {
      const words = new Set(tail.toLowerCase().split(/\W+/).filter((w) => w.length > 2));
      const score = (x) => (x.el === active ? 100 : 0) + x.label.toLowerCase().split(/\W+/).filter((w) => words.has(w)).length * 10 +
        (x.rect.top >= 0 && x.rect.top < innerHeight ? 1 : 0);
      picked = raw.map((x, i) => [x, score(x), i]).sort((a, b) => b[1] - a[1] || a[2] - b[2]).slice(0, max).map((x) => x[0]);
    }
    picked.sort((a, b) => (Math.abs(a.rect.top - b.rect.top) > 8 ? a.rect.top - b.rect.top : a.rect.left - b.rect.left));
    // " (k of n)" for repeated names, as the Mac observer does.
    const counts = new Map();
    for (const x of picked) { const k = `${x.role}|${x.label}`; counts.set(k, (counts.get(k) || 0) + 1); }
    const nth = new Map();
    const map = new Map();
    const elements = picked.map((x, i) => {
      const eid = `e${String(i + 1).padStart(2, "0")}`;
      map.set(eid, x.el);
      const k = `${x.role}|${x.label}`;
      let label = x.label;
      if (counts.get(k) > 1 && label) {
        const j = (nth.get(k) || 0) + 1;
        nth.set(k, j);
        label = trunc(label, MAX_LABEL - 9) + ` (${j} of ${counts.get(k)})`;
      }
      const fi = formInfo(x.el);
      return {
        eid, role: x.role, label, value: x.secure ? null : valueOf(x.el, x.role), context: contextOf(x.el),
        focused: x.el === active, enabled: !(x.el.disabled || x.el.getAttribute("aria-disabled") === "true"),
        secure: x.secure, ...fi,
      };
    });
    const id = ++snapSeq;
    snapshots.set(id, map);
    for (const k of [...snapshots.keys()]) if (k < id - 4) snapshots.delete(k);
    const f = elements.find((e) => e.focused);
    return {
      id, appName: "Google Chrome", windowTitle: document.title || location.hostname, url: location.href, browser: true,
      elements, focusedEid: f ? f.eid : null, walkMs: performance.now() - t0, takenAt: Date.now(),
    };
  }

  // ------------------------------------------------------------------ overlays

  let overlayRoot = null;
  function overlay() {
    if (overlayRoot && overlayRoot.isConnected) return overlayRoot;
    const host = document.createElement("div");
    host.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483647";
    const root = host.attachShadow({ mode: "open" });
    document.documentElement.appendChild(host);
    overlayRoot = root;
    return root;
  }

  function flash(el, text, color = "#2563eb", ms = 1600) {
    if (!el) return;
    const r = el.getBoundingClientRect();
    const box = document.createElement("div");
    box.style.cssText = `position:fixed;left:${r.left - 3}px;top:${r.top - 3}px;width:${r.width + 6}px;height:${r.height + 6}px;` +
      `border:3px solid ${color};border-radius:6px;box-shadow:0 0 0 4px ${color}33;transition:opacity .4s`;
    const tag = document.createElement("div");
    tag.textContent = text;
    tag.style.cssText = `position:absolute;left:0;top:-24px;background:${color};color:#fff;font:600 12px system-ui;padding:3px 7px;border-radius:5px;white-space:nowrap`;
    box.appendChild(tag);
    overlay().appendChild(box);
    setTimeout(() => { box.style.opacity = "0"; }, ms);
    setTimeout(() => box.remove(), ms + 500);
  }

  function badges(snapId, eids) {
    const map = snapshots.get(snapId);
    if (!map) return;
    const root = overlay();
    root.querySelectorAll(".gc-badge").forEach((n) => n.remove());
    eids.forEach((eid, i) => {
      const el = map.get(eid);
      if (!el) return;
      const r = el.getBoundingClientRect();
      const b = document.createElement("div");
      b.className = "gc-badge";
      b.textContent = String(i + 1);
      b.style.cssText = `position:fixed;left:${Math.max(0, r.left - 10)}px;top:${Math.max(0, r.top - 10)}px;background:#f59e0b;color:#111;` +
        "font:700 13px system-ui;padding:2px 6px;border-radius:10px;box-shadow:0 1px 3px #0005";
      root.appendChild(b);
    });
    setTimeout(() => root.querySelectorAll(".gc-badge").forEach((n) => n.remove()), 8000);
  }

  // ------------------------------------------------------------------ execution

  function resolve(snapId, eid) {
    const map = snapshots.get(snapId);
    const el = map && map.get(eid);
    return el && el.isConnected ? el : null;
  }

  function editableTarget() {
    const a = deepActive();
    if (a && (a.isContentEditable || a instanceof HTMLTextAreaElement ||
      (a instanceof HTMLInputElement && !["button", "submit", "reset", "checkbox", "radio", "image", "file", "range", "color"].includes(a.type)))) return a;
    return null;
  }

  /** No field is focused: pick the one the speaker named, else a search field, else the first visible one. */
  function bestTextField(said = "") {
    const words = new Set(said.toLowerCase().split(/[^a-z0-9]+/).filter((w) => w.length > 2 && !["type", "the", "into", "box", "field", "and", "then", "write", "enter"].includes(w)));
    const cands = [...document.querySelectorAll("input[type=search], [role=searchbox], input:not([type]), input[type=text], input[type=email], input[type=url], input[type=tel], input[type=number], textarea, [contenteditable=true], [role=textbox]")]
      .filter((el) => !el.disabled && !el.readOnly && !isSecureInput(el) && el.getClientRects().length && getComputedStyle(el).visibility !== "hidden");
    let best = null;
    let bestScore = -1;
    cands.forEach((el, i) => {
      const lab = `${labelOf(el)} ${el.name || ""} ${el.id || ""}`.toLowerCase();
      const named = [...words].filter((w) => lab.includes(w)).length;
      const search = el.type === "search" || el.getAttribute("role") === "searchbox" || /search|query|^q$/.test(`${el.name} ${el.id} ${el.placeholder || ""}`.toLowerCase());
      const wantsSearch = /\bsearch\b/.test(said.toLowerCase());
      const score = named * 10 + (search ? (wantsSearch ? 8 : 3) : 0) + (visible(el) ? 1 : 0) - i * 0.001;
      if (score > bestScore) { bestScore = score; best = el; }
    });
    if (best) best.scrollIntoView({ block: "nearest" });
    return best;
  }

  function insertText(el, text) {
    el.focus();
    if (document.execCommand && document.execCommand("insertText", false, text)) return true;
    if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
      const proto = el instanceof HTMLInputElement ? HTMLInputElement.prototype : HTMLTextAreaElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
      const s = el.selectionStart ?? el.value.length;
      const e = el.selectionEnd ?? el.value.length;
      setter.call(el, el.value.slice(0, s) + text + el.value.slice(e));
      el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      return true;
    }
    return false;
  }

  function key(el, k, opts = {}) {
    const init = { key: k, code: opts.code || k, bubbles: true, cancelable: true, composed: true, ...opts };
    const down = el.dispatchEvent(new KeyboardEvent("keydown", init));
    el.dispatchEvent(new KeyboardEvent("keypress", init));
    el.dispatchEvent(new KeyboardEvent("keyup", init));
    return down; // false when the page handled it (preventDefault)
  }

  function scroller() {
    const se = document.scrollingElement || document.documentElement;
    if (se.scrollHeight > innerHeight + 4) return se;
    let best = null;
    let area = 0;
    for (const el of document.querySelectorAll("body *")) {
      const cs = getComputedStyle(el);
      if (!/(auto|scroll)/.test(cs.overflowY) || el.scrollHeight <= el.clientHeight + 4) continue;
      const r = el.getBoundingClientRect();
      const a = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0)) * Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
      if (a > area) { area = a; best = el; }
    }
    return best || se;
  }

  function scroll(dir, amount) {
    const s = scroller();
    const before = s.scrollTop;
    const page = (s === document.scrollingElement ? innerHeight : s.clientHeight) * 0.85;
    const dy = amount === 0 ? 160 : amount === 2 ? (dir > 0 ? s.scrollHeight : -s.scrollHeight) : page;
    s.scrollBy({ top: amount === 2 ? dy : dir * dy, behavior: "instant" });
    return { changed: s.scrollTop !== before, detail: `scrolled ${dir > 0 ? "down" : "up"} ${["a little", "a page", "to the end"][amount ?? 1]}` };
  }

  function pressKey(k) {
    const el = deepActive() || document.body;
    const mac = /Mac/.test(navigator.platform);
    const mod = (extra = {}) => (mac ? { metaKey: true, ...extra } : { ctrlKey: true, ...extra });
    switch (k) {
      case "return": {
        const notHandled = key(el, "Enter", { code: "Enter", keyCode: 13, which: 13 });
        if (notHandled) {
          if ((el instanceof HTMLInputElement) && el.form) { el.form.requestSubmit ? el.form.requestSubmit() : el.form.submit(); return "submitted the form"; }
          if (el instanceof HTMLAnchorElement || el instanceof HTMLButtonElement || el.getAttribute("role") === "button") { el.click(); return "activated the focused control"; }
        }
        return "pressed Enter";
      }
      case "escape": key(el, "Escape", { keyCode: 27 }); return "pressed Escape";
      case "tab": {
        const f = [...document.querySelectorAll("a[href], button, input, select, textarea, [tabindex]:not([tabindex='-1'])")].filter((x) => !x.disabled && visible(x));
        const i = f.indexOf(el);
        const next = f[(i + 1) % f.length];
        if (next) next.focus();
        return "moved focus to the next control";
      }
      case "space": key(el, " ", { code: "Space", keyCode: 32 }); if (el.click && el !== document.body) el.click(); return "pressed Space";
      case "delete": key(el, "Backspace", { keyCode: 8 }); document.execCommand("delete"); return "pressed Backspace";
      case "up": case "down": case "left": case "right": {
        const n = { up: "ArrowUp", down: "ArrowDown", left: "ArrowLeft", right: "ArrowRight" }[k];
        if (key(el, n) && (k === "up" || k === "down") && !editableTarget()) scroll(k === "down" ? 1 : -1, 0);
        return `pressed ${n}`;
      }
      case "page_down": scroll(1, 1); return "paged down";
      case "page_up": scroll(-1, 1); return "paged up";
      case "cmd+a": {
        key(el, "a", mod());
        const ed = editableTarget();
        if (ed && ed.select) ed.select(); else document.execCommand("selectAll");
        return "selected all";
      }
      case "cmd+c": key(el, "c", mod()); document.execCommand("copy"); return "copied";
      case "cmd+x": key(el, "x", mod()); document.execCommand("cut"); return "cut";
      case "cmd+v": key(el, "v", mod()); document.execCommand("paste"); return "pasted (needs clipboard permission on some pages)";
      case "cmd+z": key(el, "z", mod()); document.execCommand("undo"); return "undid";
      case "cmd+shift+z": key(el, "z", mod({ shiftKey: true })); document.execCommand("redo"); return "redid";
      case "cmd+s": key(el, "s", mod()); return "sent the save shortcut to the page";
      case "cmd+f": key(el, "f", mod()); return "sent the find shortcut to the page (the browser find bar cannot be opened by extensions)";
      default: return null;
    }
  }

  async function execute(msg) {
    const a = msg.action;
    const dry = msg.dryRun;
    if (a.kind === "click") {
      const el = resolve(msg.snapId, a.targetEid);
      if (!el) return { ok: false, changed: false, detail: "the element is gone (page changed)" };
      if (isSecureInput(el)) return { ok: false, changed: false, detail: "refused: password field" };
      el.scrollIntoView({ block: "nearest", inline: "nearest" });
      if (dry) { flash(el, `would click ${a.targetLabel || ""}`, "#7c3aed"); return { ok: true, changed: false, dryRun: true, detail: `dry run: would click ${a.targetLabel}` }; }
      flash(el, "click", "#16a34a", 900);
      if (typeof el.focus === "function") el.focus({ preventScroll: true });
      el.click();
      return { ok: true, changed: true, detail: `clicked ${a.targetLabel}` };
    }
    if (a.kind === "type_text") {
      let el = editableTarget() || bestTextField(msg.said || "");
      if (!el) return { ok: false, changed: false, detail: "no text field to type into" };
      if (isSecureInput(el) || SECURE_FIELD_RE.test(labelOf(el))) return { ok: false, changed: false, detail: "refused: never types into password or card fields" };
      if (dry) { flash(el, `would type "${a.text}"`, "#7c3aed"); return { ok: true, changed: false, dryRun: true, detail: `dry run: would type "${a.text}"` }; }
      const before = el.value ?? el.innerText;
      const sep = before && !/\s$/.test(before) && el === deepActive() ? " " : "";
      insertText(el, sep + a.text);
      flash(el, "typed", "#16a34a", 700);
      return { ok: true, changed: true, detail: `typed "${a.text}"`, undo: { kind: "type" } };
    }
    if (a.kind === "press_key") {
      if (dry) return { ok: true, changed: false, dryRun: true, detail: `dry run: would press ${a.key}` };
      const d = pressKey(a.key);
      return d ? { ok: true, changed: true, detail: d } : { ok: false, changed: false, detail: `key ${a.key} not supported` };
    }
    if (a.kind === "scroll_down" || a.kind === "scroll_up") {
      if (dry) return { ok: true, changed: false, dryRun: true, detail: `dry run: would ${a.kind.replace("_", " ")}` };
      const r = scroll(a.kind === "scroll_down" ? 1 : -1, a.amount ?? 1);
      return { ok: true, ...r };
    }
    if (a.kind === "undo_type") {
      document.execCommand("undo");
      return { ok: true, changed: true, detail: "undid typing" };
    }
    return { ok: false, changed: false, detail: `unsupported in-page action ${a.kind}` };
  }


  // ------------------------------------------------------------------ page info, unsaved input, content blocks

  let dirty = false;
  document.addEventListener("input", (e) => {
    const t = e.target;
    if (t && (t.form || t.isContentEditable || t instanceof HTMLTextAreaElement) && e.isTrusted) dirty = true;
  }, true);
  document.addEventListener("submit", () => { dirty = false; }, true);

  function pageInfo() {
    const d = document.querySelector("meta[name=description], meta[property='og:description']");
    return { title: document.title, url: location.href, description: d ? clean(d.content).slice(0, 300) : "", dirty };
  }

  const AD_SEL = "ins.adsbygoogle, [id^=google_ads], [id*=div-gpt-ad], [data-ad-slot], [data-ad], [data-testid*=ad-], [class*=sponsored], [class*=Sponsored], [aria-label=Advertisement], [aria-label=Sponsored]";
  const BLOCK_SEL = "article, [role=article], aside, section, li, iframe, ins, div[class*=ad-], div[class*=promo], div[id*=ad-], " + AD_SEL;
  let blockSeq = 0;
  const blocks = new Map(); // id -> element
  const hidden = new Map(); // id -> {el, placeholder, kind}
  const classified = new Set(); // block ids already sent for classification

  function smallLabels(el) {
    const out = [];
    for (const n of el.querySelectorAll("span, div, a, p, small, b, strong, em, label")) {
      if (n.children.length) continue;
      const t = clean(n.textContent);
      if (t && t.length <= 24) out.push(t);
      if (out.length > 30) break;
    }
    const aria = el.getAttribute("aria-label");
    if (aria) out.push(aria);
    return out;
  }

  /** Visible, sizeable content blocks (outermost match wins), up to `max`. */
  function collectBlocks(max = 30) {
    const out = [];
    const taken = [];
    for (const el of document.querySelectorAll(BLOCK_SEL)) {
      if (el.dataset.genclassBlock && classified.has(el.dataset.genclassBlock)) { taken.push(el); continue; }
      if (taken.some((t) => t.contains(el))) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 120 || r.height < 30 || r.bottom < -innerHeight || r.top > 2 * innerHeight) continue;
      const isAdSel = el.matches(AD_SEL);
      const text = clean(el.innerText || "").slice(0, 400);
      const frame = el.tagName === "IFRAME" ? el.src : (el.querySelector("iframe") || {}).src || "";
      if (!isAdSel && !frame && (text.length < 25 || r.height > 1600)) continue;
      if (el.closest("nav, header, footer, [role=navigation], form")) continue;
      let id = el.dataset.genclassBlock;
      if (!id) { id = String(++blockSeq); el.dataset.genclassBlock = id; }
      blocks.set(id, el);
      taken.push(el);
      out.push({ id, text: text || "(ad frame)", labels: smallLabels(el), adFrame: frame, adSelector: isAdSel });
      if (out.length >= max) break;
    }
    return out;
  }

  function hideBlock(id, kind) {
    const el = blocks.get(id);
    if (!el || hidden.has(id) || !el.isConnected) return false;
    const ph = document.createElement("div");
    ph.style.cssText = "font:12px system-ui;color:#777;border:1px dashed #bbb;border-radius:6px;padding:4px 8px;margin:4px 0";
    ph.textContent = `Hidden by GenClass (${kind.replace("_", " ")}) · `;
    const show = document.createElement("a");
    show.href = "#";
    show.textContent = "show";
    show.onclick = (e) => { e.preventDefault(); unhide(id); };
    ph.appendChild(show);
    el.insertAdjacentElement("beforebegin", ph);
    el.dataset.genclassPrevDisplay = el.style.display || "";
    el.style.setProperty("display", "none", "important");
    hidden.set(id, { el, ph, kind });
    return true;
  }

  function unhide(id) {
    const h = hidden.get(id);
    if (!h) return;
    h.el.style.display = h.el.dataset.genclassPrevDisplay || "";
    h.ph.remove();
    hidden.delete(id);
    chrome.runtime.sendMessage({ to: "sw", type: "filter_count", count: hidden.size }).catch(() => {});
  }

  function unhideAll() { for (const id of [...hidden.keys()]) unhide(id); }

  let filterOn = false;
  let filterTimer = null;
  async function runFilter() {
    if (!filterOn) return;
    const bl = collectBlocks();
    if (!bl.length) return;
    for (const b of bl) classified.add(b.id);
    const res = await chrome.runtime.sendMessage({ to: "sw", type: "filter_blocks", blocks: bl, page: { host: location.hostname, title: document.title } }).catch(() => null);
    if (!res || !res.hide) return;
    for (const [id, kind] of res.hide) hideBlock(id, kind);
    chrome.runtime.sendMessage({ to: "sw", type: "filter_count", count: hidden.size }).catch(() => {});
  }
  function scheduleFilter(ms = 800) { clearTimeout(filterTimer); filterTimer = setTimeout(runFilter, ms); }
  window.addEventListener("scroll", () => filterOn && scheduleFilter(1200), { passive: true });
  new MutationObserver(() => filterOn && scheduleFilter(1500)).observe(document.documentElement, { childList: true, subtree: true });
  chrome.runtime.sendMessage({ to: "sw", type: "filter_enabled_for", host: location.hostname }).then((on) => {
    filterOn = !!on;
    if (filterOn) scheduleFilter(600);
  }).catch(() => {});

  chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
    if (msg.to !== "content") return;
    if (msg.type === "observe") { reply(observe(msg)); return; }
    if (msg.type === "execute") { execute(msg).then(reply, (e) => reply({ ok: false, changed: false, detail: String(e) })); return true; }
    if (msg.type === "badges") { badges(msg.snapId, msg.eids); reply(true); return; }
    if (msg.type === "listening") { listening = !!msg.on; reply(true); return; }
    if (msg.type === "ping") { reply("pong"); return; }
    if (msg.type === "page_info") { reply(pageInfo()); return; }
    if (msg.type === "filter_set") {
      filterOn = !!msg.on;
      if (filterOn) scheduleFilter(100); else { unhideAll(); }
      reply(true);
      return;
    }
    if (msg.type === "filter_unhide_all") { unhideAll(); reply(true); return; }
  });

  // Kill switch: Esc while GenClass is listening stops everything.
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && listening && e.isTrusted) chrome.runtime.sendMessage({ to: "brain", type: "kill", from: "esc" }).catch(() => {});
  }, true);
})();
