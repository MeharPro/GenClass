// The incremental decision loop (port of jev_local/harness/controller.py): act mid-sentence, never twice,
// risky things only on a spoken "confirm".
//
//   event -> Stream.update -> debounce (120 ms partial, 0 final) -> decide
//   decide = snapshot -> state + questions -> engine (staged) -> stale check -> policy -> safety.gate
//          -> execute -> consume -> re-decide the rest (up to MAX_LOOPS per trigger)
//
// One decision in flight; newer triggers coalesce into one "latest" request. A result computed on a tail that
// has since grown is stale: it may commit a closed-set command (if its words are still there) but never a
// payload or silence-gated one. halt() is the kill switch: nothing is decided or executed afterwards.

import { CHAIN_WORDS, PAYLOAD_INTENTS } from "./catalog.js";
import { ARG_QUESTIONS, choice, evaluatePolicy, newPolicyContext, pickOf } from "./policy.js";
import { NONE, Q_COMPLETE, Q_DESTRUCTIVE, Q_INTENT, Q_IS_COMMAND, buildQuestions, rankApps } from "./questions.js";
import { RateLimiter, gate } from "./safety.js";
import { consumedFor, extractTextCandidates, extractUrlCandidates, parseCandidatePick } from "./spans.js";
import { buildState } from "./state.js";
import { Stream, coreWords, normWord } from "./stream.js";
import { Kind, describeAction, makeAction } from "./types.js";

export const MAX_LOOPS = 4;
export const MAX_DEBOUNCE_MS = 300;
const SILENCE_MARGIN_MS = 15;
const RATE_MAX = 3;
const RATE_WINDOW_S = 1.0;
const STAGE1 = [Q_INTENT, Q_COMPLETE, Q_IS_COMMAND, Q_DESTRUCTIVE];

/** (payload_silence_ms, silence_complete_ms) for a source that updates every cadenceMs. */
export function silenceGatesFor(cadenceMs) {
  const payload = Math.round(2 * cadenceMs + 150);
  return [payload, Math.max(payload, 900)];
}

const CHAIN_SINGLE = new Set(CHAIN_WORDS.filter((w) => !w.includes(" ")));
const commandOf = (a) => JSON.stringify([a.kind, a.targetEid, a.app, a.text, a.key, a.url, a.folder]);

const POLITE = new Set("please now right away for me thanks thank you real quick quickly again".split(" "));
const SCROLL_MODS = new Set("a little bit lot few lines more some all the way to top bottom of further".split(" "));
const ROLE_NOUNS = new Set("button link icon item option box field checkbox row tab menu".split(" "));
const LEFTOVER_WORDS = {
  open_app: new Set(["app", "application", "program"]),
  quit_app: new Set(["app", "application", "program"]),
  open_folder: new Set(["folder", "directory", "window"]),
  click: ROLE_NOUNS,
  press_key: new Set(["key", "keys", "button"]),
  scroll_down: SCROLL_MODS,
  scroll_up: SCROLL_MODS,
  go_back: new Set(["a", "to", "the", "previous", "one"]),
  go_forward: new Set(["a", "to", "the", "next", "one"]),
  new_tab: new Set(["tab", "window"]),
  close_tab: new Set(["tab", "window"]),
  open_url: new Set(["website", "site", "dot", "com", "org", "net", "io"]),
  undo: new Set(["that", "it", "this", "last", "the", "thing"]),
};

export function leftoverWords(a) {
  if (PAYLOAD_INTENTS.has(a.kind)) return new Set();
  const words = new Set([...(LEFTOVER_WORDS[a.kind] || []), ...POLITE]);
  if (a.targetLabel) for (const w of coreWords(a.targetLabel.replaceAll('"', " ").split(/\s+/).filter(Boolean))) words.add(w);
  return words;
}

function endsWithChain(words) {
  if (!words.length) return false;
  const last = normWord(words[words.length - 1]);
  const two = words.slice(-2).map(normWord).join(" ");
  return CHAIN_SINGLE.has(last) || CHAIN_WORDS.includes(two);
}

const eqArr = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

export class Controller {
  /**
   * engine:   {evaluate(state, questions) -> resp}
   * observer: {snapshot() -> Snapshot|null, invalidate?(), prefetch?()}
   * executor: {run(action, snap) -> {ok, changed, detail, dryRun}}
   * apps:     (tailText) -> names for the `app` question
   * ui:       (event, payload) => void
   */
  constructor({ engine, observer, executor, cfg, ui = () => {}, apps = null, clock = null, staged = true, browser = true, log = null }) {
    this.engine = engine;
    this.observer = observer;
    this.executor = executor;
    this.cfg = cfg;
    this.ui = ui;
    this.appsFn = apps;
    this.clock = clock || (() => performance.now() / 1000);
    this.staged = staged;
    this.browser = browser;
    this.logFn = log;
    this.stream = new Stream((k, f) => this.log(k, f));
    this.history = [];
    this.decisions = [];
    this.pending = null;
    this.pendingSnap = null;
    this.pendingMark = null;
    this.pendingTimer = null;
    this.pendingId = 0;
    this.seq = 0;
    this.appliedSeq = 0;
    this.prevPick = null;
    this.pickable = null;
    this.lastKey = null;
    this.lastAct = null;
    this.want = null;
    this.running = null;
    this.debounce = null;
    this.debounceSince = 0;
    this.silence = [];
    this.retry = null;
    this.rate = new RateLimiter(RATE_MAX, RATE_WINDOW_S);
    this.closed = false;
    this.halted = false;
    this.cadenceMs = null;
    this.stats = { events: 0, decisions: 0, engine_calls: 0, stale_dropped: 0, stale_applied: 0, same_text_skipped: 0, executed: 0, rate_limited: 0, errors: 0 };
  }

  get T() {
    const T = this.cfg.thresholds;
    if (!this.cadenceMs) return T;
    const [payload, complete] = silenceGatesFor(this.cadenceMs);
    return { ...T, payload_silence_ms: Math.max(T.payload_silence_ms, payload), silence_complete_ms: Math.max(T.silence_complete_ms, complete), debounce_ms: 0 };
  }

  // ------------------------------------------------------------------ events

  /** ev: {kind: speech_start|partial|final|speech_end|error, uid, text, joined?} */
  onEvent(ev) {
    const now = this.clock();
    this.stats.events++;
    if (this.halted) return;
    this.log("transcript", { event: ev.kind, uid: ev.uid, text: ev.text });
    if (ev.kind === "speech_start") {
      this.observer.prefetch?.();
      return;
    }
    if (ev.kind === "error") {
      this.ui("error", `speech: ${ev.error || ev.text}`);
      return;
    }
    if (ev.kind === "speech_end") return;
    const tail = this.stream.update(ev, now);
    this.ui("transcript", { uid: ev.uid, text: ev.text, final: ev.kind === "final", tail, consumed: this.stream.consumedText, cursor: this.stream.cursor, words: this.stream.words.length });
    if (!tail) return;
    if (this.stream.changed) this.armSilence();
    if (ev.kind === "final") {
      this.clearDebounce();
      this.request("final");
    } else {
      this.debounceRequest("partial", this.T.debounce_ms);
    }
  }

  cancelAll() {
    this.want = null;
    this.clearDebounce();
    this.silence.forEach(clearTimeout);
    this.silence = [];
    this.dropPending("cancel_all");
    this.stream.skipToEnd();
    this.lastAct = null;
    this.log("cancel_all", {});
  }

  halt() {
    this.halted = true;
    this.cancelAll();
    clearTimeout(this.retry);
    clearTimeout(this.pendingTimer);
    this.log("halt", {});
    this.ui("halted", {});
  }

  resume() {
    this.halted = false;
    this.stream.reset();
    this.lastKey = null;
  }

  async idle() {
    while (this.running) await this.running;
  }

  // ------------------------------------------------------------------ scheduling

  request(reason) {
    if (this.closed || this.halted) return;
    this.want = reason;
    if (!this.running) this.running = this.drainWants().finally(() => { this.running = null; });
  }

  async drainWants() {
    while (this.want !== null && !this.closed) {
      const reason = this.want;
      this.want = null;
      try {
        await this.decide(reason);
      } catch (e) {
        this.stats.errors++;
        this.log("error", { stage: "decide", error: String(e && e.stack || e) });
        this.ui("error", `decide failed: ${e && e.message || e}`);
      }
    }
  }

  debounceRequest(reason, delayMs) {
    const now = this.clock();
    if (this.debounce === null) this.debounceSince = now;
    else clearTimeout(this.debounce);
    const fireAt = Math.min(now + delayMs / 1000, this.debounceSince + MAX_DEBOUNCE_MS / 1000);
    this.debounce = setTimeout(() => { this.debounce = null; this.request(reason); }, Math.max(0, (fireAt - now) * 1000));
  }

  clearDebounce() {
    clearTimeout(this.debounce);
    this.debounce = null;
  }

  armSilence() {
    this.silence.forEach(clearTimeout);
    const T = this.T;
    this.silence = [...new Set([T.payload_silence_ms, T.silence_complete_ms])].sort((a, b) => a - b)
      .map((ms) => setTimeout(() => this.request("silence"), ms + SILENCE_MARGIN_MS));
  }

  async decide(reason = "partial") {
    let first = null;
    let r = reason;
    for (let i = 0; i < MAX_LOOPS; i++) {
      const [d, again] = await this.decideOnce(r);
      if (first === null) first = d;
      if (!again) break;
      r = "post_action";
    }
    return first;
  }

  silenceBucket(tail) {
    if (tail.isFinal) return 2;
    return (tail.silentMs >= this.T.payload_silence_ms ? 1 : 0) + (tail.silentMs >= this.T.silence_complete_ms ? 1 : 0);
  }

  // ------------------------------------------------------------------ one decision

  async ask(state, questions, tail, snap, ctx) {
    if (!this.staged) {
      this.stats.engine_calls++;
      return { resp: await this.engine.evaluate(state, questions), passes: 1 };
    }
    const q1 = {};
    for (const k of STAGE1) if (questions[k]) q1[k] = questions[k];
    this.stats.engine_calls++;
    const r1 = await this.engine.evaluate(state, q1);
    const pre = evaluatePolicy(r1, tail, snap, { ...ctx, stage1: true }, this.T);
    if (pre.verdict !== "need_args") return { resp: r1, passes: 1 };
    const intent = choice(r1, Q_INTENT);
    const argQ = ARG_QUESTIONS[intent.label];
    if (!argQ || !questions[argQ]) return { resp: r1, passes: 1 };
    this.stats.engine_calls++;
    const r2 = await this.engine.evaluate(state, { [argQ]: questions[argQ] });
    return {
      resp: { ...r1, answers: { ...r1.answers, ...r2.answers }, usage: { input_tokens: r1.usage.input_tokens + r2.usage.input_tokens },
        timings: { forward: r1.timings.forward + r2.timings.forward, pack: r1.timings.pack + r2.timings.pack } },
      passes: 2,
    };
  }

  async decideOnce(reason) {
    const now = this.clock();
    let tail = this.stream.current(now);
    if (tail) {
      this.syncLastAct(tail);
      if (this.absorbLeftovers(tail)) tail = this.stream.current(now);
    }
    if (!tail) return [null, false];
    const key = JSON.stringify([tail.vid, tail.cursor, tail.text, tail.isFinal, this.silenceBucket(tail), this.pendingId]);
    if (key === this.lastKey) {
      this.stats.same_text_skipped++;
      return [null, false];
    }
    this.lastKey = key;
    const seq = ++this.seq;

    const t0 = performance.now();
    let snap = null;
    try {
      snap = await this.observer.snapshot();
    } catch (e) {
      this.log("error", { stage: "snapshot", error: String(e) });
    }
    const tSnap = performance.now();
    const apps = this.appsFn ? this.appsFn(tail.text) : [];
    const state = buildState(tail.text, snap, this.history, this.pending);
    const questions = buildQuestions(snap, apps, extractTextCandidates(tail.text), extractUrlCandidates(tail.text),
      this.cfg.maxElements, { browser: this.browser });
    const tBuild = performance.now();
    const picked = this.numberedPick(tail, snap, seq);
    if (picked) return picked;
    const ctx0 = newPolicyContext({ prevPick: this.prevPick, pending: this.pending, pendingMark: this.pendingMark,
      tailHeardAt: this.stream.heardAt(tail.cursor), stale: false, history: this.history.slice(-3) });
    let resp;
    let passes;
    try {
      ({ resp, passes } = await this.ask(state, questions, tail, snap, ctx0));
    } catch (e) {
      this.stats.errors++;
      this.lastKey = null;
      this.log("error", { stage: "engine", seq, error: String(e && e.stack || e) });
      this.ui("error", `decision failed: ${e && e.message || e}`);
      return [null, false];
    }
    const tModel = performance.now();
    if (seq <= this.appliedSeq || this.halted) return [null, false];
    const cur = this.stream.current(this.clock());
    if (!cur || cur.vid !== tail.vid || cur.cursor !== tail.cursor) {
      this.stats.stale_dropped++;
      this.log("stale_dropped", { seq, vid: tail.vid, text: tail.text, current: cur && cur.text });
      return [null, false];
    }
    const stale = cur.text !== tail.text;
    const ctx = { ...ctx0, stale };
    const rescue = rescueTarget(resp, tail, snap, this.T);
    let decision = evaluatePolicy(resp, tail, snap, ctx, this.T);
    const a = decision.action;
    if (!a || (a.kind !== Kind.CONFIRM && a.kind !== Kind.CANCEL)) this.prevPick = pickOf(a);
    const said = a && a.consumedWords ? tail.words.slice(0, a.consumedWords).join(" ") : tail.text;
    decision = gate(decision, snap, this.cfg, said);
    const tPolicy = performance.now();
    decision = { ...decision, seq, latencyMs: Math.round((tModel - tBuild) * 100) / 100 };
    this.appliedSeq = seq;
    this.stats.decisions++;
    if (stale) this.stats.stale_applied++;
    const info = {
      seq, trigger: reason, vid: tail.vid, text: tail.text, isFinal: tail.isFinal, silentMs: tail.silentMs, stale,
      verdict: decision.verdict, reason: decision.reason, action: decision.action, risk: decision.risk,
      passes, tokens: resp.usage && resp.usage.input_tokens, rescue,
      timings: { snapshot: tSnap - t0, build: tBuild - tSnap, model: tModel - tBuild, policy: tPolicy - tModel },
      words: { heard: this.stream.words.length, consumedBefore: tail.cursor, actionWords: a ? a.consumedWords : 0 },
      answers: summarize(resp),
    };
    this.decisions.push(info);
    if (this.decisions.length > 300) this.decisions.shift();
    this.log("decision", info);
    this.ui("decision", info);
    return [decision, await this.apply(decision, tail, snap, said)];
  }

  /**
   * GenClass: after "which element?", the top candidates get numbered badges on the page; "number two" /
   * "the second one" within 10 s clicks that one (parsed in code, no model call), still through safety.gate.
   */
  async numberedPickApply(tail, snap, seq, idx) {
    const P = this.pickable;
    this.pickable = null;
    const eid = P.eids[idx];
    const el = snap && snap.elements.find((e) => e.eid === eid && e.label === P.labels[idx]);
    if (!el) return null;
    const n = tail.words.length;
    const a = makeAction(Kind.CLICK, tail.vid, 1.0, { targetEid: el.eid, targetLabel: el.label ? `${el.role} "${el.label}"` : el.role, consumedWords: n });
    let decision = { verdict: "act", action: a, reason: `numbered pick ${idx + 1}: ${describeAction(a)}`, seq, retryInMs: null,
      answers: { answers: { intent: { type: "choice", choice: "click", confidence: 1, probabilities: new Map([["click", 1]]) },
        target: { type: "choice", choice: el.eid, confidence: 1, probabilities: new Map([[el.eid, 1]]) } } }, latencyMs: 0, risk: 0 };
    decision = gate(decision, snap, this.cfg, tail.text);
    this.appliedSeq = seq;
    const info = { seq, trigger: "pick", vid: tail.vid, text: tail.text, isFinal: tail.isFinal, verdict: decision.verdict, reason: decision.reason,
      action: a, risk: decision.risk, passes: 0, timings: { model: 0 }, words: { heard: this.stream.words.length, consumedBefore: tail.cursor, actionWords: n }, answers: {} };
    this.decisions.push(info);
    this.ui("decision", info);
    return [decision, await this.apply(decision, tail, snap, tail.text)];
  }

  numberedPick(tail, snap, seq) {
    const P = this.pickable;
    if (!P || this.clock() - P.t > 10 || tail.words.length > 5) return null;
    const idx = parseCandidatePick(tail.text, P.eids.length);
    if (idx === null) return null;
    return this.numberedPickApply(tail, snap, seq, idx);
  }

  offerPicks(decision, snap) {
    const a = decision.answers && decision.answers.answers && decision.answers.answers.target;
    if (!a || !snap || a.type !== "choice") return;
    const top = [...a.probabilities].filter(([l]) => l !== NONE).sort((x, y) => y[1] - x[1]).slice(0, 3).map(([l]) => l);
    const els = top.map((eid) => snap.elements.find((e) => e.eid === eid)).filter(Boolean);
    if (!els.length) return;
    this.pickable = { eids: els.map((e) => e.eid), labels: els.map((e) => e.label), t: this.clock() };
    this.ui("picks", { snapId: snap.id, eids: this.pickable.eids, labels: els.map((e) => `${e.role} "${e.label}"`) });
  }

  async apply(decision, tail, snap, said) {
    const a = decision.action;
    const v = decision.verdict;
    this.syncLastAct(tail);
    if (v === "ignore" || v === "wait") return this.skipLeftoverHead(decision, tail);
    if (v === "clarify" && /which element|not sure enough/.test(decision.reason)) this.offerPicks(decision, snap);
    if (!a || !["act", "confirm", "deny"].includes(v)) return false;
    const n = a.consumedWords;
    const key = [tail.vid, tail.cursor, n];
    if (this.stream.alreadyFired(key) || !this.stream.matches(tail, n)) {
      this.log("not_fired", { seq: decision.seq, vid: tail.vid });
      return false;
    }
    if ((v === "act" || v === "confirm") && a.kind !== Kind.CONFIRM && a.kind !== Kind.CANCEL) {
      const why = this.repeatOfLast(a, tail);
      if (why) {
        this.stream.markFired(key);
        this.stream.consume(n);
        this.log(why, { seq: decision.seq, vid: tail.vid, action: a });
        this.remember(tail, n, { extend: true });
        return true;
      }
    }
    if (v === "deny") {
      this.stream.markFired(key);
      this.stream.dropUtterance();
      this.ui("denied", { reason: decision.reason, action: a });
      return false;
    }
    if (v === "confirm") {
      this.stream.markFired(key);
      this.propose(a, snap, tail, decision.reason);
      this.stream.dropUtterance();
      return false;
    }
    if (a.kind === Kind.CANCEL) {
      this.stream.markFired(key);
      this.stream.consume(n);
      this.lastAct = null;
      this.dropPending("cancelled by user");
      return true;
    }
    if (a.kind === Kind.CONFIRM) {
      if (!this.pending) return false;
      if (this.rateLimited(decision)) return false;
      this.stream.markFired(key);
      this.stream.consume(n);
      this.lastAct = null;
      const pending = this.pending;
      const psnap = this.pendingSnap;
      this.clearPending();
      this.log("confirmed", { action: pending, by: tail.vid, said });
      await this.execute(pending, psnap, `(confirmed by: ${said})`, tail);
      return true;
    }
    if (this.rateLimited(decision)) return false;
    this.stream.markFired(key);
    this.stream.consume(n);
    this.remember(tail, n, { action: a });
    if (this.pending) this.dropPending(`superseded by ${describeAction(a)}`);
    await this.execute(a, snap, said, tail);
    return true;
  }

  // ------------------------------------------------------------------ leftovers and repeats

  syncLastAct(tail) {
    const L = this.lastAct;
    if (!L || tail.uid === L.uid) return;
    if (this.stream.gen === 0 && this.stream.consumedText && L.vidAfter.startsWith(`${L.uid}+`)) {
      L.uid = tail.uid;
      L.vidAfter = tail.vid;
    } else {
      this.lastAct = null;
    }
  }

  remember(tail, n, { action = null, extend = false } = {}) {
    const words = tail.words.slice(0, n);
    let L = this.lastAct;
    if (extend && L) L.consumedCore = [...L.consumedCore, ...coreWords(words)];
    else if (action) {
      L = this.lastAct = { uid: tail.uid, vidAfter: "", command: commandOf(action), consumedCore: coreWords(words), remainderCore: [], midPhrase: false, leftovers: leftoverWords(action) };
    } else return;
    L.vidAfter = this.stream.vid;
    L.remainderCore = coreWords(tail.words.slice(n));
    L.midPhrase = !endsWithChain(words);
  }

  absorbLeftovers(tail) {
    const L = this.lastAct;
    if (!L || !L.midPhrase || tail.vid !== L.vidAfter || !L.leftovers.size) return false;
    let k = 0;
    while (k < tail.words.length && L.leftovers.has(normWord(tail.words[k]))) k++;
    if (k === 0 || k === tail.words.length) return false;
    this.stream.consume(k);
    this.log("leftover_absorbed", { vid: tail.vid, consumed: tail.words.slice(0, k).join(" "), rule: "words" });
    L.consumedCore = [...L.consumedCore, ...coreWords(tail.words.slice(0, k))];
    L.vidAfter = this.stream.vid;
    L.remainderCore = [];
    return true;
  }

  repeatOfLast(a, tail) {
    const L = this.lastAct;
    if (!L || tail.vid !== L.vidAfter || commandOf(a) !== L.command) return null;
    if (L.midPhrase) return "leftover_absorbed";
    const seen = L.remainderCore.length > 0 && eqArr(coreWords(tail.words).slice(0, L.remainderCore.length), L.remainderCore);
    const mine = coreWords(tail.words.slice(0, a.consumedWords));
    const consumed = new Set(L.consumedCore);
    if (seen && !mine.every((w) => consumed.has(w))) return "duplicate_skipped";
    return null;
  }

  skipLeftoverHead(decision, tail) {
    const L = this.lastAct;
    if (!L || !L.midPhrase || tail.vid !== L.vidAfter) return false;
    if (decision.verdict === "wait") {
      const intent = choice(decision.answers, Q_INTENT);
      if (!intent || intent.label !== "wait") return false;
    }
    const n = consumedFor(tail.words, NONE);
    if (n >= tail.words.length) return false;
    const key = [tail.vid, tail.cursor, n];
    if (this.stream.alreadyFired(key) || !this.stream.matches(tail, n)) return false;
    this.stream.markFired(key);
    this.stream.consume(n);
    this.log("leftover_skipped", { seq: decision.seq, vid: tail.vid, consumed: tail.words.slice(0, n).join(" ") });
    L.vidAfter = this.stream.vid;
    L.remainderCore = [];
    L.midPhrase = false;
    return true;
  }

  rateLimited(decision) {
    const waitS = this.rate.waitS(this.clock());
    if (waitS <= 0) return false;
    this.stats.rate_limited++;
    this.lastKey = null;
    this.log("rate_limited", { seq: decision.seq, retry_in_ms: Math.round(waitS * 1000) });
    clearTimeout(this.retry);
    this.retry = setTimeout(() => this.request("rate_limit"), waitS * 1000 + 5);
    return true;
  }

  // ------------------------------------------------------------------ execution

  async execute(action, snap, said, tail) {
    if (this.halted) return { ok: false, detail: "halted" };
    this.rate.record(this.clock());
    let result;
    try {
      result = await this.executor.run(action, snap, said);
    } catch (e) {
      this.stats.errors++;
      result = { ok: false, changed: false, detail: `error: ${e && e.message || e}` };
    }
    const outcome = !result.ok ? `failed: ${result.detail}` : result.dryRun ? "dry-run" : "ok";
    this.history.push({ said, action, outcome, t: this.clock() });
    if (this.history.length > 500) this.history.splice(0, 100);
    this.stats.executed++;
    this.observer.invalidate?.();
    const heard = this.stream.words.length;
    const firedAt = tail ? tail.cursor + action.consumedWords : null;
    this.log("exec", { action, said, outcome, detail: result.detail });
    this.ui("executed", { action, said, outcome, detail: result.detail, dryRun: !!result.dryRun, firedAtWord: firedAt, wordsHeard: heard, uid: tail && tail.uid });
    return result;
  }

  // ------------------------------------------------------------------ pending confirmation

  propose(action, snap, tail, reason) {
    if (this.pending) this.dropPending(`replaced by ${describeAction(action)}`);
    this.pending = action;
    this.pendingId++;
    this.pendingSnap = snap;
    this.pendingMark = [tail.uid, this.stream.heardWords, this.clock()];
    clearTimeout(this.pendingTimer);
    this.pendingTimer = setTimeout(() => { if (this.pending === action) this.dropPending("timeout"); }, this.T.confirm_timeout_ms);
    this.log("pending", { action, mark: this.pendingMark });
    this.ui("pending", { action, describe: describeAction(action), reason, timeoutMs: this.T.confirm_timeout_ms });
  }

  /** Button confirm from the UI (a deliberate click counts as the explicit yes). */
  async confirmFromUi() {
    if (!this.pending || this.halted) return false;
    const pending = this.pending;
    const psnap = this.pendingSnap;
    this.clearPending();
    this.log("confirmed", { action: pending, by: "ui" });
    await this.execute(pending, psnap, "(confirmed by button)", null);
    return true;
  }

  clearPending() {
    clearTimeout(this.pendingTimer);
    this.pending = this.pendingSnap = this.pendingMark = this.pendingTimer = null;
    this.pendingId++;
    this.ui("pending", null);
  }

  dropPending(why) {
    if (!this.pending) return;
    this.log("pending_dropped", { action: this.pending, why });
    this.ui("info", `not done: ${describeAction(this.pending)} (${why})`);
    this.clearPending();
  }

  log(kind, fields) {
    if (this.logFn) this.logFn(kind, fields);
  }
}

const ORD = { first: 1, second: 2, third: 3, fourth: 4, fifth: 5, last: -1 };
const normLabel = (s) => s.toLowerCase().replace(/\s*\(\d+ of \d+\)$/, "").replace(/[^\p{L}\p{N}]+/gu, " ").trim();

/**
 * GenClass addition (logged as `rescue`): when the intent is click but the target answer is "none" or below the
 * threshold, and exactly one on-screen label occurs verbatim (whole words) in the transcript, answer the target
 * question with that element. An ordinal ("the second add to cart") picks among "(k of n)" duplicates. The
 * model's own answer is kept whenever it already clears the gates. Safety gates still apply afterwards.
 */
export function rescueTarget(resp, tail, snap, T) {
  const a = resp && resp.answers && resp.answers.target;
  const intent = resp && resp.answers && resp.answers.intent;
  if (!a || !snap || !intent || intent.choice !== "click" || a.type !== "choice") return null;
  const pTop = a.probabilities.get(a.choice) ?? 0;
  const said = ` ${normLabel(tail.text)} `;
  const ordM = /\b(first|second|third|fourth|fifth|last)\b/i.exec(tail.text);
  const groups = new Map();
  for (const e of snap.elements) {
    const l = normLabel(e.label || "");
    if (l.length < 3 || !said.includes(` ${l} `)) continue;
    if (!groups.has(l)) groups.set(l, []);
    groups.get(l).push(e);
  }
  // Longest matching label wins ("add to cart" over "cart"); it must be unique.
  const labels = [...groups.keys()].sort((x, y) => y.length - x.length);
  if (!labels.length) return null;
  if (labels.length > 1 && labels[0].length === labels[1].length) return null;
  const group = groups.get(labels[0]);
  let el = null;
  if (group.length === 1) el = group[0];
  else if (ordM) {
    const k = ORD[ordM[1].toLowerCase()];
    el = k === -1 ? group[group.length - 1] : group[k - 1] || null;
  }
  if (!el) return null;
  const modelOk = a.choice !== NONE && pTop >= T.target_top_p && a.confidence >= T.target_conf;
  if (modelOk && (a.choice === el.eid || group.length === 1)) return null; // the model already resolved it
  if (modelOk && !ordM) return null;
  const n = a.probabilities.size;
  a.modelChoice = a.choice;
  a.modelP = pTop;
  a.choice = el.eid;
  a.probabilities = new Map([...a.probabilities].map(([l]) => [l, l === el.eid ? 0.9 : 0.1 / (n - 1)]));
  a.confidence = Math.round(((n * 0.9 - 1) / (n - 1)) * 100) / 100;
  return { eid: el.eid, label: el.label, model: a.modelChoice, modelP: pTop, rule: group.length > 1 ? "ordinal" : "verbatim label" };
}

/** Compact answers for the UI/log. */
export function summarize(resp) {
  const out = {};
  for (const [qid, a] of Object.entries(resp.answers || {})) {
    if (a.type === "noul") out[qid] = a.noul;
    else if (a.type === "choice") {
      const top = [...a.probabilities].sort((x, y) => y[1] - x[1]).slice(0, 3);
      out[qid] = { choice: a.choice, p: a.probabilities.get(a.choice), confidence: a.confidence, top };
    } else out[qid] = { score: a.score, confidence: a.confidence };
  }
  return out;
}
