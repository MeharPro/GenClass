// JS ports vs the Python harness (jev_local/harness) on fixtures from scripts/make_py_fixtures.py.
import assert from "node:assert/strict";
import { test } from "node:test";
import { join } from "node:path";
import * as spans from "../../src/core/spans.js";
import { buildQuestions, elementLine, rankApps } from "../../src/core/questions.js";
import { buildState } from "../../src/core/state.js";
import { evaluatePolicy, newPolicyContext, pickOf } from "../../src/core/policy.js";
import { gate } from "../../src/core/safety.js";
import { Stream } from "../../src/core/stream.js";
import { DEFAULT_THRESHOLDS, defaultConfig } from "../../src/core/types.js";
import { FIX, readJson } from "./helpers.mjs";

const split = (t) => (t.trim() ? t.trim().split(/\s+/) : []);

test("spans: text/url candidates, picks, consumption, sentence ends", () => {
  const fx = readJson(join(FIX, "spans_py.json"));
  for (const f of fx) {
    const t = f.text;
    const w = split(t);
    assert.deepEqual(spans.extractTextCandidates(t), f.text_cands, `text cands: ${t}`);
    assert.deepEqual(spans.extractUrlCandidates(t), f.url_cands, `url cands: ${t}`);
    assert.equal(spans.normalizeSpokenUrl(t), f.spoken_url, `spoken url: ${t}`);
    assert.deepEqual([1, 3, 5, 12].map((n) => spans.parseCandidatePick(t, n)), f.pick, `pick: ${t}`);
    for (const [intent, n] of Object.entries(f.consumed)) assert.equal(spans.consumedFor(w, intent), n, `consumed ${intent}: ${t}`);
    for (const [c, n] of f.consumed_span) assert.equal(spans.consumedFor(w, "type_text", c), n, `consumed span ${c}: ${t}`);
    assert.equal(spans.stripLeadingChain(w), f.strip_chain, `strip chain: ${t}`);
    assert.deepEqual(w.map(spans.endsSentence), f.ends, `ends: ${t}`);
  }
  assert.ok(fx.length > 100);
});

export function snapFromPy(s) {
  if (!s) return null;
  const elements = s.elements.map((e) => ({ eid: e.eid, role: e.role, label: e.label || "", value: e.value ?? null, context: e.context ?? null,
    focused: !!e.focused, enabled: e.enabled !== false, secure: !!e.secure }));
  const f = elements.find((e) => e.focused);
  return { appName: s.app_name, windowTitle: s.window_title ?? null, elements, focusedEid: f ? f.eid : null };
}

const actionFromPy = (a) => a && ({
  kind: a.kind, sourceVid: a.source_vid, confidence: a.confidence, targetEid: a.target_eid ?? null, targetLabel: a.target_label ?? null,
  app: a.app ?? null, text: a.text ?? null, key: a.key ?? null, url: a.url ?? null, folder: a.folder ?? null, amount: a.amount ?? null,
  consumedWords: a.consumed_words ?? 0,
});

const qToPy = (qs) => {
  const out = {};
  for (const [k, q] of Object.entries(qs)) {
    out[k] = { type: q.type, instructions: q.instructions,
      criteria: q.criteria instanceof Map ? [...q.criteria] : q.criteria };
  }
  return out;
};

test("questions + state: identical requests to the Mac harness", () => {
  const fx = readJson(join(FIX, "questions_py.json"));
  for (const c of fx.cases) {
    const snap = snapFromPy(c.screen);
    assert.deepEqual(snap.elements.map(elementLine), c.element_lines);
    const apps = rankApps(c.transcript, fx.apps, c.screen.running || [], 24);
    assert.deepEqual(apps, c.apps, `rank_apps: ${c.transcript}`);
    const qs = buildQuestions(snap, apps, spans.extractTextCandidates(c.transcript), spans.extractUrlCandidates(c.transcript), 60);
    const py = c.questions;
    assert.deepEqual(Object.keys(qs), Object.keys(py));
    const js = qToPy(qs);
    for (const k of Object.keys(py)) {
      assert.equal(js[k].type, py[k].type);
      assert.equal(js[k].instructions, py[k].instructions);
      if (py[k].type === "noul") assert.deepEqual(js[k].criteria ?? null, py[k].criteria ?? null, k);
      else assert.deepEqual(js[k].criteria, py[k].criteria, k);
    }
    const hist = c.history.map((h) => ({ action: { kind: h.kind, targetLabel: h.target_label, app: h.app, text: h.text, key: h.key, url: null, folder: null } }));
    const pending = c.pending && { kind: c.pending.kind, app: c.pending.app, targetLabel: c.pending.target_label, targetEid: c.pending.target_eid, text: null, key: null, url: null, folder: null };
    assert.deepEqual(buildState(c.transcript, snap, hist, pending), c.state);
  }
  for (const r of fx.rank) assert.deepEqual(rankApps(r.text, fx.apps, r.running, 24), r.out, `rank: ${r.text}`);
});

test("policy + safety gate: 700 synthetic responses give identical verdicts, reasons and actions", () => {
  const fx = readJson(join(FIX, "policy_py.json"));
  const cfg = defaultConfig();
  let n = 0;
  for (const f of fx) {
    const snap = snapFromPy(f.screen);
    const t = f.tail;
    const tail = { vid: t.vid, text: t.text, words: t.words, cursor: t.cursor, isFinal: t.is_final, silentMs: t.silent_ms, uid: t.uid, joined: t.joined };
    const answers = {};
    for (const [k, a] of Object.entries(f.answers)) {
      if (a.type === "choice") answers[k] = { ...a, probabilities: new Map(a.probabilities) };
      else if (a.type === "score") answers[k] = { ...a, probabilities: Object.fromEntries(a.probabilities) };
      else answers[k] = a;
    }
    const resp = { answers };
    const ctx = newPolicyContext({ prevPick: f.ctx.prev_pick, pending: actionFromPy(f.ctx.pending), pendingMark: f.ctx.pending_mark,
      tailHeardAt: f.ctx.tail_heard_at, stale: f.ctx.stale });
    const d = evaluatePolicy(resp, tail, snap, ctx, DEFAULT_THRESHOLDS);
    const where = `case ${n}: "${t.text}"`;
    assert.equal(d.verdict, f.policy.verdict, where);
    assert.equal(d.reason, f.policy.reason, where);
    assert.equal(d.retryInMs, f.policy.retry_in_ms, where);
    assert.deepEqual(d.action, actionFromPy(f.policy.action), where);
    assert.deepEqual(pickOf(d.action), f.policy.pick, where);
    const a = d.action;
    const said = a && a.consumedWords ? tail.words.slice(0, a.consumedWords).join(" ") : tail.text;
    const g = gate(d, snap, cfg, said);
    assert.equal(g.verdict, f.gate.verdict, where);
    assert.equal(g.reason, f.gate.reason, where);
    assert.equal(g.risk, f.gate.risk, where);
    n++;
  }
  assert.equal(n, 700);
});

test("stream: consumed prefix, revisions, uid echo, drop, skip", () => {
  const fx = readJson(join(FIX, "stream_py.json"));
  for (const rec of fx) {
    const s = new Stream();
    for (const r of rec) {
      if (r.op === "partial" || r.op === "final") {
        const tl = s.update({ kind: r.op, uid: r.uid, text: r.text }, r.t);
        const exp = r.tail && { vid: r.tail.vid, text: r.tail.text, words: r.tail.words, cursor: r.tail.cursor, isFinal: r.tail.is_final,
          silentMs: r.tail.silent_ms, uid: r.tail.uid, joined: r.tail.joined };
        assert.deepEqual(tl, exp, `${r.op} ${r.text}`);
        assert.equal(s.frozen, r.frozen);
        assert.equal(s.dropped, r.dropped);
        assert.equal(s.vid, r.vid);
        assert.equal(s.consumedText, r.consumed);
      } else if (r.op === "consume") {
        s.consume(r.n);
        assert.equal(s.vid, r.vid);
        assert.equal(s.consumedText, r.consumed);
        assert.equal(s.cursor, r.cursor);
      } else if (r.op === "drop") s.dropUtterance();
      else if (r.op === "skip") s.skipToEnd();
    }
  }
});
