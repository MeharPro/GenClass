// Shared shapes (port of jev_local/harness/types.py), as plain objects.
//
// Element:  {eid, role, label, value?, context?, focused?, enabled?, secure?, submits?, ...}
// Snapshot: {appName, windowTitle, url, elements[], focusedEid?, takenAt}
// Tail:     {vid, text, words[], cursor, isFinal, silentMs, uid, joined}
// Action:   {kind, sourceVid, confidence, targetEid?, targetLabel?, app?, text?, key?, url?, folder?, amount?, consumedWords}
// Decision: {verdict, action, reason, seq, retryInMs, answers, latencyMs, risk}

import { pyRepr } from "./serialize.js";

export const Kind = Object.freeze({
  OPEN_APP: "open_app",
  QUIT_APP: "quit_app",
  CLICK: "click",
  TYPE_TEXT: "type_text",
  SEARCH_WEB: "search_web",
  OPEN_URL: "open_url",
  PRESS_KEY: "press_key",
  SCROLL_DOWN: "scroll_down",
  SCROLL_UP: "scroll_up",
  OPEN_FOLDER: "open_folder",
  GO_BACK: "go_back",
  GO_FORWARD: "go_forward", // GenClass (browser) extension
  NEW_TAB: "new_tab",
  CLOSE_TAB: "close_tab",
  UNDO: "undo",
  CONFIRM: "confirm",
  CANCEL: "cancel",
});
export const KINDS = new Set(Object.values(Kind));

export const Risk = Object.freeze({ LOW: 0, MEDIUM: 1, HIGH: 2, DENY: 3 });

export function makeAction(kind, sourceVid, confidence, kw = {}) {
  return {
    kind, sourceVid, confidence,
    targetEid: null, targetLabel: null, app: null, text: null, key: null, url: null, folder: null, amount: null,
    consumedWords: 0, ...kw,
  };
}

export function describeAction(a) {
  const arg = a.targetLabel || a.app || a.text || a.key || a.url || a.folder;
  return arg ? `${a.kind}(${pyRepr(arg)})` : a.kind;
}

export const DEFAULT_THRESHOLDS = Object.freeze({
  is_command: 0.5,
  intent_conf: 0.45,
  intent_top_p: 0.55,
  complete: 0.65,
  stable_complete: 0.85,
  target_conf: 0.45,
  target_top_p: 0.35,
  span_top_p: 0.35,
  app_top_p: 0.5,
  key_top_p: 0.6,
  folder_top_p: 0.5,
  destructive: 0.5,
  high_risk_intent_p: 0.85,
  high_risk_target_p: 0.75,
  confirm_p: 0.8,
  silence_complete_ms: 900,
  payload_silence_ms: 600,
  debounce_ms: 120,
  confirm_timeout_ms: 8000,
});

export function defaultConfig() {
  return {
    dryRun: true,
    searchUrl: "https://www.google.com/search?q={q}",
    allowApps: [],
    maxElements: 60,
    maxApps: 24,
    thresholds: { ...DEFAULT_THRESHOLDS },
  };
}
