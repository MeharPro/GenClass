// One response -> a verdict for the current tail (port of jev_local/harness/policy.py). Pure, no I/O.
// Gate order: pending confirm/cancel; is_command; intent none/wait/low; complete (model, 900 ms silence, or
// final); payload (type/search wait for final or 600 ms silence; open_url too unless the address is closed);
// argument gates; stability (a closed-set act on a partial needs complete >= .85 or the same pick twice).

import { FOLDERS, KEYS, PAYLOAD_INTENTS } from "./catalog.js";
import { NONE, Q_APP, Q_COMPLETE, Q_FOLDER, Q_INTENT, Q_IS_COMMAND, Q_KEY, Q_SCROLL, Q_TARGET, Q_TEXT, Q_URL } from "./questions.js";
import { AFFIRM_RE, NEGATE_RE, confirmationAllowed } from "./safety.js";
import { TLDS, consumedFor } from "./spans.js";
import { KINDS, Kind, describeAction, makeAction } from "./types.js";
import { pyRound } from "./pyutil.js";
import { pyRepr } from "./serialize.js";

const URL_CLOSED_RE = new RegExp("(?:\\bdot\\s+|\\.)(?:" + TLDS + ")\\b[^\\s]*\\s+(?:and|then)\\b", "i");
const BARE_DOMAIN_RE = /^(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d{1,5})?(?:\/\S*)?$/i;
export const CANCEL_TOP_P = 0.5;

/** The argument question each intent reads (null: none). */
export const ARG_QUESTIONS = {
  open_app: Q_APP, quit_app: Q_APP, click: Q_TARGET, type_text: Q_TEXT, search_web: Q_TEXT, open_url: Q_URL,
  press_key: Q_KEY, open_folder: Q_FOLDER, scroll_down: Q_SCROLL, scroll_up: Q_SCROLL,
};

export function newPolicyContext(kw = {}) {
  return { prevPick: null, pending: null, pendingMark: null, tailHeardAt: null, stale: false, history: [], ...kw };
}

export function choice(resp, qid) {
  const a = resp ? resp.answers[qid] : null;
  if (!a || a.type !== "choice") return null;
  return { label: a.choice, p: a.probabilities.get(a.choice) ?? 0, conf: a.confidence };
}

export function noul(resp, qid, dflt) {
  const a = resp ? resp.answers[qid] : null;
  return a && a.type === "noul" ? a.noul : dflt;
}

export function scoreLevel(resp, qid, dflt) {
  const a = resp ? resp.answers[qid] : null;
  if (!a || a.type !== "score" || !a.probabilities || !Object.keys(a.probabilities).length) return dflt;
  let best = null;
  for (const [k, p] of Object.entries(a.probabilities)) {
    const i = parseInt(k, 10);
    if (best === null || p > best[1] || (p === best[1] && i < best[0])) best = [i, p];
  }
  return best[0];
}

/** (vid, intent, argument) identity of an action, for the stability rule. */
export function pickOf(action) {
  if (!action) return null;
  let arg = action.targetEid || action.app || action.text || action.key || action.url || action.folder || null;
  if (arg === null && action.amount !== null && action.amount !== undefined) arg = String(action.amount);
  return [action.sourceVid, action.kind, arg];
}

const samePick = (a, b) => (a === null || b === null ? a === b : a[0] === b[0] && a[1] === b[1] && a[2] === b[2]);

const d = (verdict, reason, resp, action = null, retry = null) =>
  ({ verdict, action, reason, seq: 0, retryInMs: retry, answers: resp, latencyMs: 0, risk: 0 });

const f2 = (x) => x.toFixed(2);

export function evaluatePolicy(resp, tail, snap, ctx, T) {
  const isFinal = tail.isFinal && !ctx.stale;
  const silent = ctx.stale ? 0 : tail.silentMs;
  const settled = isFinal || silent >= T.payload_silence_ms;

  const intent = choice(resp, Q_INTENT);
  if (!intent) return d("wait", "no intent answer", resp);

  if (ctx.pending) {
    const r = pendingDecision(resp, tail, intent, ctx, T, settled);
    if (r) return r;
  }
  if (noul(resp, Q_IS_COMMAND, 1.0) < T.is_command) return d("ignore", "not a command (is_command low)", resp);
  if (intent.label === NONE) return d("ignore", "intent none", resp);
  if (intent.label === "wait") return d("wait", "intent wait: words do not commit to an action yet", resp);
  if (intent.conf < T.intent_conf || intent.p < T.intent_top_p) {
    return d("wait", `intent ${intent.label} low (p=${f2(intent.p)}, conf=${f2(intent.conf)})`, resp);
  }
  if (intent.label === Kind.CONFIRM || intent.label === Kind.CANCEL) return d("ignore", `${intent.label} with nothing pending`, resp);
  if (!KINDS.has(intent.label)) return d("ignore", `unknown intent ${pyRepr(intent.label)}`, resp);
  const kind = intent.label;

  const complete = noul(resp, Q_COMPLETE, 0.0);
  if (complete < T.complete && !(isFinal || silent >= T.silence_complete_ms)) {
    return d("wait", `incomplete (complete=${f2(complete)})`, resp, null, Math.max(0, T.silence_complete_ms - silent));
  }
  if (PAYLOAD_INTENTS.has(intent.label) && !(isFinal || silent >= T.payload_silence_ms)) {
    return d("wait", `${intent.label} waits for the final transcript or silence`, resp, null, Math.max(0, T.payload_silence_ms - silent));
  }
  if (intent.label === Kind.OPEN_URL && !(isFinal || silent >= T.payload_silence_ms) && !URL_CLOSED_RE.test(tail.text)) {
    return d("wait", "open_url waits for the address to finish (final, silence, or 'and')", resp, null,
      Math.max(0, T.payload_silence_ms - silent));
  }
  // Staged evaluation (GenClass): pass 1 has no argument answers yet; say which pass 2 needs. Exact, because
  // questions are block-isolated: an answer does not depend on which other questions share its pass.
  if (ctx.stage1) return d("need_args", `${kind} needs ${ARG_QUESTIONS[kind] || "nothing"}`, resp);
  const built = build(kind, intent, resp, tail, snap, T);
  if (typeof built === "string") {
    if (isFinal || silent >= T.silence_complete_ms) return d("clarify", built, resp);
    return d("wait", `argument not resolved yet: ${built}`, resp);
  }
  const action = built;
  if (!settled && complete < T.stable_complete && !samePick(ctx.prevPick, pickOf(action))) {
    return d("wait", `unstable: ${describeAction(action)} needs a second agreeing partial (complete=${f2(complete)})`, resp, action);
  }
  return d("act", `${describeAction(action)} (conf=${f2(action.confidence)})`, resp, action);
}

function pendingDecision(resp, tail, intent, ctx, T, settled) {
  const pending = ctx.pending;
  const text = tail.text;
  const negated = NEGATE_RE.test(text);
  if ((intent.label === Kind.CANCEL && intent.p >= CANCEL_TOP_P) ||
      (negated && ["cancel", "confirm", "none", "wait"].includes(intent.label))) {
    const n = Math.min(tail.words.length, consumedFor(tail.words, "cancel"));
    const a = makeAction(Kind.CANCEL, tail.vid, intent.conf, { consumedWords: n });
    return d("act", `cancel pending ${describeAction(pending)}`, resp, a);
  }
  if (intent.label !== Kind.CONFIRM) return null;
  if (!confirmationAllowed(pending, tail, ctx.pendingMark, ctx.tailHeardAt)) {
    return d("wait", "a confirmation must come from a new utterance, after the prompt", resp);
  }
  if (!AFFIRM_RE.test(text)) return d("clarify", 'say "confirm" (or yes) to confirm, or "cancel"', resp);
  if (intent.p < T.confirm_p) return d("wait", `confirm too weak (p=${f2(intent.p)})`, resp);
  if (!settled && noul(resp, Q_COMPLETE, 0.0) < T.stable_complete) return d("wait", "confirm waits for a finished phrase", resp);
  const n = Math.min(tail.words.length, consumedFor(tail.words, "confirm"));
  const a = makeAction(Kind.CONFIRM, tail.vid, intent.conf, { consumedWords: n });
  return d("act", `confirm ${describeAction(pending)}`, resp, a);
}

function gateChoice(resp, qid, topP) {
  const c = choice(resp, qid);
  if (!c || c.label === NONE || c.p < topP) return null;
  return c;
}

function build(kind, intent, resp, tail, snap, T) {
  let conf = intent.conf;
  let span = null;
  const kw = {};
  if (kind === Kind.OPEN_APP || kind === Kind.QUIT_APP) {
    const a = gateChoice(resp, Q_APP, T.app_top_p);
    if (!a) return "which app?";
    kw.app = a.label;
    conf = Math.min(conf, a.conf);
  } else if (kind === Kind.CLICK) {
    if (!snap || !snap.elements.length) return "no screen elements to click";
    const t = gateChoice(resp, Q_TARGET, T.target_top_p);
    if (!t || t.conf < T.target_conf) return "which element?";
    const el = snap.elements.find((e) => e.eid === t.label);
    if (!el) return `element ${t.label} is not on screen`;
    if (el.enabled === false) return `${el.role} "${el.label}" is disabled`;
    kw.targetEid = el.eid;
    kw.targetLabel = el.label ? `${el.role} "${el.label}"` : el.role;
    conf = Math.min(conf, t.conf);
  } else if (PAYLOAD_INTENTS.has(kind)) {
    const s = gateChoice(resp, Q_TEXT, T.span_top_p);
    if (!s) return "what text?";
    if (!tail.text.includes(s.label)) return "text span is not verbatim from the transcript";
    kw.text = span = s.label;
    conf = Math.min(conf, s.conf);
  } else if (kind === Kind.OPEN_URL) {
    const u = gateChoice(resp, Q_URL, T.span_top_p);
    if (!u) return "which website?";
    kw.url = BARE_DOMAIN_RE.test(u.label) ? `https://${u.label}` : u.label;
    conf = Math.min(conf, u.conf);
  } else if (kind === Kind.PRESS_KEY) {
    const k = gateChoice(resp, Q_KEY, T.key_top_p);
    if (!k || !(k.label in KEYS)) return "which key?";
    kw.key = k.label;
    conf = Math.min(conf, k.conf);
  } else if (kind === Kind.OPEN_FOLDER) {
    const f = gateChoice(resp, Q_FOLDER, T.folder_top_p);
    if (!f || !(f.label in FOLDERS)) return "which folder?";
    kw.folder = f.label;
    conf = Math.min(conf, f.conf);
  } else if (kind === Kind.SCROLL_DOWN || kind === Kind.SCROLL_UP) {
    kw.amount = scoreLevel(resp, Q_SCROLL, 1);
  }
  const n = Math.min(tail.words.length, consumedFor(tail.words, kind, span));
  return makeAction(kind, tail.vid, pyRound(conf, 4), { consumedWords: n, ...kw });
}
