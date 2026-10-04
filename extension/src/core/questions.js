// The fixed question schema sent on every transcript update (port of jev_local/harness/questions.py).

import { BROWSER_EXTRA_INTENTS, FOLDERS, INTENTS, KEYS, SCROLL_LEVELS } from "./catalog.js";
import { extract } from "./fuzz.js";

export const Q_INTENT = "intent";
export const Q_COMPLETE = "complete";
export const Q_IS_COMMAND = "is_command";
export const Q_DESTRUCTIVE = "destructive";
export const Q_TARGET = "target";
export const Q_APP = "app";
export const Q_KEY = "key";
export const Q_FOLDER = "folder";
export const Q_TEXT = "text_span";
export const Q_URL = "url_span";
export const Q_SCROLL = "scroll_amount";
export const NONE = "none";

export const INSTR = {
  [Q_INTENT]:
    "Which action does `transcript` ask the computer to do? If several actions are chained, pick the " +
    "first one. If the sentence is unfinished, pick the action its words already commit to, or wait " +
    "if they do not commit to one yet.",
  [Q_COMPLETE]:
    "Does `transcript` already contain a complete first command: the action plus everything it needs, " +
    "such as which app, which element, which key, or what text?",
  [Q_IS_COMMAND]: "Is `transcript` an instruction for this computer, rather than speech meant for someone else?",
  [Q_DESTRUCTIVE]:
    "Would doing what `transcript` asks delete, send, submit, buy, quit, close, or otherwise do " +
    "something hard to undo?",
  [Q_TARGET]: "Which on-screen element does `transcript` tell the computer to act on?",
  [Q_APP]: "Which application does `transcript` name?",
  [Q_KEY]: "Which key or shortcut does `transcript` ask to press?",
  [Q_FOLDER]: "Which folder does `transcript` ask to open?",
  [Q_TEXT]: "Which exact words from `transcript` should be typed or searched for?",
  [Q_URL]: "Which website address does `transcript` mention?",
  [Q_SCROLL]: "How far does `transcript` ask to scroll?",
};

export const COMPLETE_CRITERIA = {
  true: "complete, for example: open safari, click send, type hello there, press escape, scroll down",
  false: "unfinished, for example: open, click the, type, search for, go to, press",
};
export const COMMAND_CRITERIA = {
  true: "a request for the computer, for example: open mail, scroll down, click the blue button",
  false: "talking to a person or thinking aloud, for example: can you pass the salt, I think it's fine",
};
export const TARGET_NONE = "no on-screen element is mentioned";
export const APP_NONE = "no application is mentioned";
export const TEXT_NONE = "nothing should be typed or searched";
export const URL_NONE = "no website is mentioned";

/** How an element is shown to the model: `button "Send"`, `text field "To" = "bob" (focused)`. */
export function elementLine(el) {
  let s = el.label ? `${el.role} "${el.label}"` : el.role;
  if (el.value) s += ` = "${el.value}"`;
  if (el.context) s += ` in ${el.context}`;
  if (el.focused) s += " (focused)";
  if (el.enabled === false) s += " (disabled)";
  return s;
}

/** Pick <= maxN names: fuzzy matches to the transcript first, then running ones, then others. */
export function rankApps(tailText, names, running = [], maxN = 24) {
  names = [...new Set(names.filter((n) => n))];
  const out = [];
  if (tailText.trim() && names.length) {
    for (const [n, score] of extract(tailText, names, maxN)) if (score >= 60) out.push(n);
  }
  for (const n of [...running, ...names]) {
    if (out.length >= maxN) break;
    if (!out.includes(n)) out.push(n);
  }
  return out.slice(0, maxN);
}

/**
 * opts.browser: GenClass mode. Appends BROWSER_EXTRA_INTENTS to the intent options and leaves out the folder
 * question (no file system in a browser). Every other question is byte-identical to the Mac harness.
 */
export function buildQuestions(snap, apps, textCands, urlCands, maxElements = 60, opts = {}) {
  const q = {};
  const intents = new Map(Object.entries(INTENTS));
  if (opts.browser) for (const [k, v] of Object.entries(BROWSER_EXTRA_INTENTS)) intents.set(k, v);
  q[Q_INTENT] = { type: "choice", instructions: INSTR[Q_INTENT], criteria: intents };
  q[Q_COMPLETE] = { type: "noul", instructions: INSTR[Q_COMPLETE], criteria: COMPLETE_CRITERIA };
  q[Q_IS_COMMAND] = { type: "noul", instructions: INSTR[Q_IS_COMMAND], criteria: COMMAND_CRITERIA };
  q[Q_DESTRUCTIVE] = { type: "noul", instructions: INSTR[Q_DESTRUCTIVE], criteria: null };
  if (snap && snap.elements && snap.elements.length) {
    const crit = new Map();
    for (const el of snap.elements.slice(0, maxElements)) crit.set(el.eid, elementLine(el));
    crit.set(NONE, TARGET_NONE);
    q[Q_TARGET] = { type: "choice", instructions: INSTR[Q_TARGET], criteria: crit };
  }
  const appCrit = new Map();
  for (const a of apps) if (a && a !== NONE && !appCrit.has(a)) appCrit.set(a, null);
  appCrit.set(NONE, APP_NONE);
  q[Q_APP] = { type: "choice", instructions: INSTR[Q_APP], criteria: appCrit };
  q[Q_KEY] = { type: "choice", instructions: INSTR[Q_KEY], criteria: new Map(Object.entries(KEYS)) };
  if (!opts.browser) q[Q_FOLDER] = { type: "choice", instructions: INSTR[Q_FOLDER], criteria: new Map(Object.entries(FOLDERS)) };
  if (textCands.length) {
    const tc = new Map();
    for (const c of textCands) if (c.toLowerCase() !== NONE && !tc.has(c)) tc.set(c, null);
    tc.set(NONE, TEXT_NONE);
    q[Q_TEXT] = { type: "choice", instructions: INSTR[Q_TEXT], criteria: tc };
  }
  if (urlCands.length) {
    const uc = new Map();
    for (const u of urlCands) if (u.toLowerCase() !== NONE && !uc.has(u)) uc.set(u, null);
    uc.set(NONE, URL_NONE);
    q[Q_URL] = { type: "choice", instructions: INSTR[Q_URL], criteria: uc };
  }
  q[Q_SCROLL] = { type: "score", instructions: INSTR[Q_SCROLL], criteria: [...SCROLL_LEVELS] };
  return q;
}

/** Questions as plain JSON (criteria Maps -> objects), for logs and the side panel. */
export function questionsToJson(qs) {
  const out = {};
  for (const [k, q] of Object.entries(qs)) {
    out[k] = { ...q, criteria: q.criteria instanceof Map ? Object.fromEntries(q.criteria) : q.criteria };
  }
  return out;
}
