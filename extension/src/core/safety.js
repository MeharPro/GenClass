// Safety gating between policy and executor (port of jev_local/harness/safety.py) plus browser rules.
//
// Deterministic layers first, because a confident mis-hear passes every model gate:
// 1. hard deny (deny-listed apps/sites, typing or pasting into password fields, non-http(s) URLs);
// 2. risk lexicon + always-risky kinds -> HIGH -> spoken "confirm";
// 3. model gate: destructive >= T.destructive -> confirm; HIGH also needs intent p >= .85 (and target p >= .75);
// 4. confirmation only from a later utterance with an explicit yes-word;
// 5. rate limit: <= 3 executions per second.
// Browser additions (snap.browser): Return in a non-search form and clicking a non-search submit button submit
// a form -> HIGH; Return in a field of a search form is LOW-risk (it only searches).

import { DENY_APP_NAMES, DENY_APP_PATTERNS, FOLDER_PATHS, RISK_RE, SECURE_FIELD_RE } from "./catalog.js";
import { Q_DESTRUCTIVE, Q_INTENT, Q_TARGET } from "./questions.js";
import { Kind, Risk, describeAction } from "./types.js";
import { pyRepr } from "./serialize.js";

const reprOrNone = (v) => (v === null || v === undefined ? "None" : pyRepr(v));

export const EXTRA_RISK_RE = new RegExp(
  "\\b(don'?t save|do not save|revert|shut ?down|restart|force quit|deactivate|move to (?:the )?bin|" +
    "delete all|clear (?:all|history|everything)|leave (?:the )?(?:meeting|call|group|channel)|" +
    "clean(?:ing)? (?:up|out)|tidy(?: up)?|get rid of|throw (?:it |them |that |this )?away)\\b",
  "i",
);
export const TERMINAL_RE = /terminal|iterm|warp|ghostty|kitty|alacritty|wezterm|\bhyper\b|tabby|termius/i;
export const AFFIRM_RE = /\b(yes|confirm(?:ed)?|do it|go ahead|proceed|affirmative)\b/i;
export const MESSAGING_APP_RE =
  /\b(messages|slack|discord|whatsapp|telegram|signal|microsoft teams|teams|wechat|line|skype|zoom|messenger|webex|mattermost|element|beeper)\b/i;
export const COMPOSER_RE = /\b(i?message(?!\s+body)|reply|chat|comment|compose|tweet|post|send|write a|say something)\b/i;
const DIALOG_RE = /dialog|sheet|alert/i;
export const NEGATE_RE = /\b(no|nope|nah|don'?t|do not|not|cancel|stop|never ?mind|abort|wait|hold on)\b/i;
const HOST_RE = /^(?:localhost|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}|\d{1,3}(?:\.\d{1,3}){3})$/i;

export const CONFIRM_REACTION_S = 0.8;

const IN_APP_KINDS = new Set([Kind.CLICK, Kind.TYPE_TEXT, Kind.PRESS_KEY, Kind.SCROLL_DOWN, Kind.SCROLL_UP,
  Kind.GO_BACK, Kind.GO_FORWARD, Kind.NEW_TAB, Kind.CLOSE_TAB, Kind.UNDO]);
const SCREEN_KINDS = new Set([Kind.CLICK, Kind.TYPE_TEXT, Kind.PRESS_KEY]);
const ALWAYS_HIGH = new Set([Kind.QUIT_APP, Kind.CLOSE_TAB]);
const MEDIUM_KINDS = new Set([Kind.TYPE_TEXT, Kind.SEARCH_WEB, Kind.OPEN_URL, Kind.PRESS_KEY]);
export const PASSTHROUGH = new Set([Kind.CONFIRM, Kind.CANCEL]);

export const isRiskyText = (s) => !!s && (RISK_RE.test(s) || EXTRA_RISK_RE.test(s));

export function isDeniedApp(name, cfg) {
  if (!name) return false;
  const n = name.toLowerCase().trim();
  if ((cfg.allowApps || []).some((a) => n === a.toLowerCase())) return false;
  return DENY_APP_NAMES.has(n) || DENY_APP_PATTERNS.some((p) => n.includes(p));
}

const elementOf = (snap, eid) => (snap && eid ? snap.elements.find((e) => e.eid === eid) || null : null);

export function focusedElement(snap) {
  if (!snap) return null;
  return elementOf(snap, snap.focusedEid) || snap.elements.find((e) => e.focused) || null;
}

function isSecure(el) {
  if (!el) return false;
  return !!el.secure || SECURE_FIELD_RE.test(el.label || "") || SECURE_FIELD_RE.test(el.role || "");
}

/** urllib.parse.urlsplit-like check: http(s) scheme, no userinfo, a plain host name. */
export function urlOk(url) {
  if (!url) return false;
  const m = /^([A-Za-z][A-Za-z0-9+.-]*):(.*)$/s.exec(url);
  if (!m) return false;
  const scheme = m[1].toLowerCase();
  if (scheme !== "http" && scheme !== "https") return false;
  const rest = m[2];
  if (!rest.startsWith("//")) return false;
  const netloc = rest.slice(2).split(/[/?#]/, 1)[0];
  if (netloc.includes("@")) return false;
  let host = netloc;
  if (host.startsWith("[")) return false;
  const c = host.lastIndexOf(":");
  if (c >= 0) host = host.slice(0, c);
  return HOST_RE.test(host.toLowerCase());
}

export function denyReason(action, snap, cfg) {
  const k = action.kind;
  if ((k === Kind.OPEN_APP || k === Kind.QUIT_APP) && isDeniedApp(action.app, cfg)) {
    return `${action.app} is on the deny list (add it to allow_apps to permit)`;
  }
  if (IN_APP_KINDS.has(k) && snap && isDeniedApp(snap.appName, cfg)) {
    return `acting inside ${snap.appName} is denied (add it to allow_apps to permit)`;
  }
  const focused = focusedElement(snap);
  if (k === Kind.TYPE_TEXT && isSecure(focused)) return "never types into a password or secure field";
  if (k === Kind.PRESS_KEY && action.key === "cmd+v" && isSecure(focused)) return "never pastes into a password or secure field";
  if (k === Kind.PRESS_KEY && action.key === "return" && snap && TERMINAL_RE.test(snap.appName)) return "never presses Return in a terminal";
  if (k === Kind.OPEN_URL && !urlOk(action.url)) return `only plain http(s) URLs are opened, not ${reprOrNone(action.url)}`;
  if (k === Kind.OPEN_FOLDER && !(action.folder in FOLDER_PATHS)) return `unknown folder ${reprOrNone(action.folder)}`;
  if (snap && snap.browser) {
    if (k === Kind.OPEN_FOLDER) return "a browser extension cannot open folders";
    if (IN_APP_KINDS.has(k) && snap.restricted) return "GenClass never acts on browser-internal pages (chrome://, the Web Store)";
    if (IN_APP_KINDS.has(k) && snap.deniedSite) return `acting on ${snap.deniedSite} is denied (banking, password manager or similar site)`;
    if (k === Kind.CLICK) {
      const el = elementOf(snap, action.targetEid);
      if (el && el.secure) return "never acts on a password field";
    }
  }
  return null;
}

export function classifyRisk(action, snap, cfg, said = null) {
  if (PASSTHROUGH.has(action.kind)) return Risk.LOW;
  if (denyReason(action, snap, cfg) !== null) return Risk.DENY;
  if (ALWAYS_HIGH.has(action.kind)) return Risk.HIGH;
  const el = elementOf(snap, action.targetEid);
  // Browser history navigation ("go forward") has no target or payload; "forward" in the words is not an e-mail forward.
  const nav = snap && snap.browser && (action.kind === Kind.GO_FORWARD || action.kind === Kind.GO_BACK);
  const texts = [action.targetLabel, el ? el.label : null, action.text, action.key, nav ? null : said];
  if (texts.some(isRiskyText)) return Risk.HIGH;
  if (action.kind === Kind.PRESS_KEY && (action.key === "return" || action.key === "space")) {
    const f = focusedElement(snap);
    if (f && isRiskyText(f.label)) return Risk.HIGH;
  }
  if (action.kind === Kind.PRESS_KEY && action.key === "return" && snap && returnIsRisky(snap)) return Risk.HIGH;
  if (snap && snap.browser) {
    // A submit button or Return in a form that is not a search form sends what was entered.
    if (action.kind === Kind.CLICK && el && el.submits && !el.searchForm) return Risk.HIGH;
    if (action.kind === Kind.PRESS_KEY && action.key === "return") {
      const f = focusedElement(snap);
      if (f && f.inForm && !f.searchForm && f.role !== "button" && f.role !== "link") return Risk.HIGH;
    }
  }
  if (MEDIUM_KINDS.has(action.kind)) return Risk.MEDIUM;
  return Risk.LOW;
}

export function returnIsRisky(snap) {
  const f = focusedElement(snap);
  if (MESSAGING_APP_RE.test(snap.appName || "")) return true;
  if (f && COMPOSER_RE.test(`${f.label || ""} ${f.context || ""}`)) return true;
  for (const e of snap.elements) {
    if (e.role === "button" && DIALOG_RE.test(e.context || "") && isRiskyText(e.label)) return true;
  }
  return !!snap.windowTitle && snap.windowTitle.includes("?") && isRiskyText(snap.windowTitle);
}

function topP(resp, qid, label = null) {
  const a = resp ? resp.answers[qid] : null;
  if (!a || a.type !== "choice") return 0;
  return a.probabilities.get(label !== null ? label : a.choice) ?? 0;
}

function noulOf(resp, qid) {
  const a = resp ? resp.answers[qid] : null;
  return a && a.type === "noul" ? a.noul : null;
}

/** Upgrade an `act` to confirm / clarify / deny when it is risky. Other verdicts pass through. */
export function gate(decision, snap, cfg, said = null) {
  const a = decision.action;
  if (decision.verdict !== "act" || !a || PASSTHROUGH.has(a.kind)) return decision;
  const T = cfg.thresholds;
  if (!snap && SCREEN_KINDS.has(a.kind)) {
    return { ...decision, verdict: "clarify", reason: `${describeAction(a)}: the screen could not be read` };
  }
  const risk = classifyRisk(a, snap, cfg, said);
  if (risk === Risk.DENY) return { ...decision, verdict: "deny", risk, reason: `denied: ${denyReason(a, snap, cfg)}` };
  const resp = decision.answers;
  if (risk === Risk.HIGH) {
    const ip = topP(resp, Q_INTENT);
    const tp = a.kind === Kind.CLICK ? topP(resp, Q_TARGET, a.targetEid) : 1.0;
    if (ip < T.high_risk_intent_p || tp < T.high_risk_target_p) {
      return { ...decision, verdict: "clarify", risk,
        reason: `risky ${describeAction(a)} but not sure enough (intent p=${ip.toFixed(2)}, target p=${tp.toFixed(2)})` };
    }
    return { ...decision, verdict: "confirm", risk, reason: `risky: confirm ${describeAction(a)}?` };
  }
  const destructive = noulOf(resp, Q_DESTRUCTIVE);
  if (destructive !== null && destructive >= T.destructive) {
    return { ...decision, verdict: "confirm", risk: Math.max(risk, Risk.MEDIUM),
      reason: `looks destructive (${destructive.toFixed(2)}): confirm ${describeAction(a)}?` };
  }
  return { ...decision, risk };
}

/** A "yes" counts only from a different virtual utterance, spoken in reaction to the prompt. */
export function confirmationAllowed(pending, tail, mark, heardAt = null) {
  if (tail.vid === pending.sourceVid) return false;
  if (tail.joined) return false;
  if (mark && tail.uid === mark[0]) {
    if (tail.cursor < mark[1]) return false;
    if (mark.length > 2 && (heardAt === null || heardAt === undefined || heardAt - mark[2] < CONFIRM_REACTION_S)) return false;
  }
  return true;
}

export class RateLimiter {
  constructor(maxActions = 3, windowS = 1.0) {
    this.maxActions = maxActions;
    this.windowS = windowS;
    this.times = [];
  }
  waitS(now) {
    while (this.times.length && now - this.times[0] >= this.windowS) this.times.shift();
    if (this.times.length < this.maxActions) return 0;
    return this.windowS - (now - this.times[0]);
  }
  record(now) {
    this.times.push(now);
  }
}
