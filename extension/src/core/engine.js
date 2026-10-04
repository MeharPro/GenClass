// GenClass decision engine: packer + ONNX model + calibration -> Jev-shaped answers.
// Calibration mirrors jev_local/engine/encoder/calibrate.py; answers mirror jev_local/confidence.py with the
// server default precision "round2" (every number rounded to 2 dp; choice = argmax of the unrounded p).

import { Packer, planInputs, unpackLogits } from "./packer.js";
import { clip01, pyRound } from "./pyutil.js";
import { criteriaMap } from "./serialize.js";

const K_BUCKETS = [[2, 2], [3, 5], [6, 10], [11, 30], [31, 100], [101, 255]];
const BUCKET_CLAMP = [0.5, 5.0];

export async function sha1Hex(s) {
  const buf = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const headerKeyCache = new Map();
export async function headerKey(header) {
  let k = headerKeyCache.get(header);
  if (!k) {
    k = (await sha1Hex(header)).slice(0, 12);
    headerKeyCache.set(header, k);
  }
  return k;
}

function kBucket(k) {
  for (const [lo, hi] of K_BUCKETS) if (k <= hi) return lo === hi ? String(lo) : `${lo}-${hi}`;
  const [lo, hi] = K_BUCKETS[K_BUCKETS.length - 1];
  return `${lo}-${hi}`;
}

export function tauFor(calib, kind, hk, k = null) {
  const bh = calib.by_header || {};
  if (hk in bh) return Number(bh[hk]);
  if (k !== null && kind !== "noul") {
    const [lo, hi] = calib.bucket_clamp || BUCKET_CLAMP;
    const bb = (calib.by_bucket || {})[`${kind}:${kBucket(k)}`];
    if (bb !== undefined) return Number(bb);
    const tk = (calib.tau_k || {})[kind];
    if (tk) return Math.min(Math.max(tk[0] + tk[1] * Math.log(Math.max(k, 2)), lo), hi);
  }
  return Number(calib[kind] ?? 1.0);
}

export function noulAffine(calib, hk) {
  const bh = calib.by_header || {};
  if (hk in bh) return [1 / Number(bh[hk]), 0];
  if (calib.noul_platt) return [Number(calib.noul_platt.a), Number(calib.noul_platt.b)];
  return [1 / Number(calib.noul ?? 1.0), 0];
}

/** Raw (tau=1) logits of one question -> calibrated probabilities (float64). */
export function calibrateLogits(kind, hk, logits, calib) {
  if (kind === "noul") {
    const [a, b] = noulAffine(calib, hk);
    return [1 / (1 + Math.exp(-(a * logits[0] + b)))];
  }
  const tau = tauFor(calib, kind, hk, logits.length);
  const z = logits.map((x) => x / tau);
  const m = Math.max(...z);
  const e = z.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

function normalize(p) {
  const clean = p.map((x) => (Number.isFinite(x) && x > 0 ? x : 0));
  const t = clean.reduce((a, b) => a + b, 0);
  if (t <= 0) return clean.map(() => 1 / clean.length);
  return clean.map((x) => x / t);
}

export const choiceConfidence = (p) => (p.length <= 1 ? 1 : clip01((p.length * Math.max(...p) - 1) / (p.length - 1)));

function madUniform(k) {
  const c = (k - 1) / 2;
  let s = 0;
  for (let i = 0; i < k; i++) s += Math.abs(i - c);
  return s / k;
}

function scoreConfidence(p) {
  const k = p.length;
  if (k <= 1) return 1;
  let c = 0;
  for (let i = 1; i < k; i++) if (p[i] > p[c]) c = i;
  let spread = 0;
  p.forEach((pi, i) => { spread += pi * Math.abs(i - c); });
  return Math.max(0, 1 - spread / madUniform(k));
}

/** Calibrated distribution -> Jev answer (round2). Choice probabilities are a Map label -> p. */
export function buildAnswer(q, probs, digits = 2) {
  if (q.type === "noul") {
    let p = probs[0];
    p = clip01(Number.isFinite(p) ? p : 0.5);
    return { type: "noul", noul: pyRound(p, digits) };
  }
  if (q.type === "choice") {
    const labels = [...criteriaMap(q.criteria).keys()];
    const p = normalize(probs);
    let best = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    return {
      type: "choice",
      choice: labels[best],
      confidence: pyRound(choiceConfidence(p), digits),
      probabilities: new Map(labels.map((l, i) => [l, pyRound(clip01(p[i]), digits)])),
      raw: p,
    };
  }
  if (q.type === "score") {
    const p = normalize(probs);
    const probsObj = {};
    p.forEach((x, i) => { probsObj[String(i)] = pyRound(clip01(x), digits); });
    return {
      type: "score",
      score: pyRound(p.reduce((a, x, i) => a + i * x, 0), digits),
      confidence: pyRound(scoreConfidence(p), digits),
      probabilities: probsObj,
    };
  }
  throw new TypeError(`unsupported question type ${q.type}`);
}

/**
 * Engine around an ORT InferenceSession. `ort` is the onnxruntime-web (or -node) module.
 * evaluate(state, questions) -> {model, answers: {qid: answer}, usage, timings}
 */
export class Engine {
  constructor({ ort, session, tokenizer, calibration, meta, provider = "wasm", variant = "int8" }) {
    this.ort = ort;
    this.session = session;
    this.calib = calibration;
    this.meta = meta || {};
    this.provider = provider;
    this.variant = variant;
    this.packer = new Packer(tokenizer, { maxPositions: Number(this.meta.max_len || 1536) });
    this.name = `genclass-${variant}`;
    this.busy = Promise.resolve();
  }

  feeds(p) {
    const { Tensor } = this.ort;
    const n = p.inputIds.length;
    const i64 = (arr) => BigInt64Array.from(arr, (x) => BigInt(x));
    const plan = planInputs(p);
    return {
      input_ids: new Tensor("int64", i64(p.inputIds), [1, n]),
      position_ids: new Tensor("int64", i64(p.positionIds), [1, n]),
      q_group: new Tensor("int64", i64(p.qGroup), [1, n]),
      i_group: new Tensor("int64", i64(p.iGroup), [1, n]),
      choice_q: new Tensor("int64", i64(plan.choice.q), [plan.choice.q.length]),
      choice_items: new Tensor("int64", i64(plan.choice.items.flat()), [plan.choice.q.length, plan.choice.k]),
      score_q: new Tensor("int64", i64(plan.score.q), [plan.score.q.length]),
      score_items: new Tensor("int64", i64(plan.score.items.flat()), [plan.score.q.length, plan.score.k]),
      noul_q: new Tensor("int64", i64(plan.noul.q), [plan.noul.q.length]),
      noul_t: new Tensor("int64", i64(plan.noul.t), [plan.noul.t.length]),
      noul_f: new Tensor("int64", i64(plan.noul.f), [plan.noul.f.length]),
    };
  }

  /** Raw logits per qid (Map) plus the packed sequence; serialised so calls never overlap on one session. */
  async logits(state, questions) {
    const run = async () => {
      const t0 = performance.now();
      const { packed } = this.packer.pack(state, questions);
      const t1 = performance.now();
      const out = await this.session.run(this.feeds(packed));
      const t2 = performance.now();
      const res = unpackLogits(packed, {
        choice: { data: out.choice_logits.data, dims: out.choice_logits.dims },
        score: { data: out.score_logits.data, dims: out.score_logits.dims },
        noul: { data: out.noul_logits.data, dims: out.noul_logits.dims },
      });
      return { logits: res, packed, timings: { pack: t1 - t0, forward: t2 - t1 } };
    };
    const p = this.busy.then(run, run);
    this.busy = p.catch(() => {});
    return p;
  }

  async evaluate(state, questions) {
    const t0 = performance.now();
    const { logits, packed, timings } = await this.logits(state, questions);
    const answers = {};
    for (const [qid, q] of Object.entries(questions)) {
      const qi = packed.qIndex.get(qid);
      const hk = await headerKey(qi.header);
      answers[qid] = buildAnswer(q, calibrateLogits(qi.kind, hk, logits.get(qid), this.calib));
    }
    return {
      model: this.name,
      answers,
      usage: { input_tokens: packed.inputIds.length },
      timings: { ...timings, total: performance.now() - t0 },
    };
  }
}
