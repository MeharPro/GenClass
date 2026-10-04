// End to end: JS packer -> ONNX (onnxruntime-web, wasm, in Node) -> JS calibration, vs PyTorch on 50 requests.
// Needs release-assets/genclass-*.onnx (built by scripts/genclass_export.py on the VM); skipped when absent.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { join } from "node:path";
import * as ort from "onnxruntime-web";
import { Tokenizer } from "../../src/core/tokenizer.js";
import { Engine, calibrateLogits, headerKey } from "../../src/core/engine.js";
import { ASSETS, FIX, questionsFromJson, readJson } from "./helpers.mjs";

const VARIANTS = (process.env.GENCLASS_VARIANTS || "q8").split(",");
const reqs = readJson(join(FIX, "requests50.json"));
const packs = readJson(join(FIX, "pack_fixtures.json"));
const torch = readJson(join(FIX, "torch_fixtures.json"));
const tok = new Tokenizer(readJson(join(ASSETS, "tokenizer.json")));
const calib = readJson(join(ASSETS, "calibration.json"));
const meta = readJson(join(ASSETS, "meta.json"));
ort.env.wasm.numThreads = Number(process.env.GENCLASS_THREADS || 1);
export const results = {};

test("header keys match Python sha1(header)[:12]", async () => {
  for (const tf of torch.slice(0, 3)) {
    const p = packs.find((x) => x.id === tf.id);
    for (const [qid, qi] of Object.entries(p.q_index)) assert.equal(await headerKey(qi.header), tf.header_key[qid]);
  }
});

test("JS calibration of PyTorch logits reproduces PyTorch probabilities", () => {
  let mx = 0;
  for (const tf of torch) {
    const p = packs.find((x) => x.id === tf.id);
    for (const [qid, qi] of Object.entries(p.q_index)) {
      const got = calibrateLogits(qi.kind, tf.header_key[qid], tf.logits[qid], calib);
      got.forEach((x, i) => { mx = Math.max(mx, Math.abs(x - tf.probs[qid][i])); });
    }
  }
  assert.ok(mx < 1e-9, `max |dp| ${mx}`);
});

for (const v of VARIANTS) {
  const path = join(ASSETS, `genclass-${v}.onnx`);
  test(`ONNX ${v} via onnxruntime-web matches PyTorch`, { skip: !existsSync(path) && "model not built" }, async () => {
    const session = await ort.InferenceSession.create(readFileSync(path), { executionProviders: ["wasm"] });
    const eng = new Engine({ ort, session, tokenizer: tok, calibration: calib, meta, variant: v });
    let maxLogit = 0;
    let maxProb = 0;
    let agree = 0;
    let total = 0;
    let decisionsSame = 0;
    const ms = [];
    for (let i = 0; i < reqs.length; i++) {
      const r = reqs[i];
      const tf = torch[i];
      const labels = Object.fromEntries(Object.entries(packs[i].q_index).map(([q, qi]) => [q, qi.labels]));
      const qs = questionsFromJson(r.questions, labels);
      const t0 = performance.now();
      const res = await eng.evaluate(r.state, qs);
      ms.push(performance.now() - t0);
      const { logits } = await eng.logits(r.state, qs);
      let allSame = true;
      for (const [qid, z] of logits) {
        z.forEach((x, j) => { maxLogit = Math.max(maxLogit, Math.abs(x - tf.logits[qid][j])); });
        const ref = tf.probs[qid];
        const ans = res.answers[qid];
        if (ans.type === "noul") {
          maxProb = Math.max(maxProb, Math.abs(ans.noul - ref[0]));
          const same = (ans.noul >= 0.5) === (ref[0] >= 0.5);
          agree += same; allSame &&= same;
        } else {
          const raw = ans.type === "choice" ? ans.raw : null;
          const refArg = ref.indexOf(Math.max(...ref));
          const gotArg = raw ? raw.indexOf(Math.max(...raw)) : Object.values(ans.probabilities).indexOf(Math.max(...Object.values(ans.probabilities)));
          if (raw) raw.forEach((x, j) => { maxProb = Math.max(maxProb, Math.abs(x - ref[j])); });
          const same = gotArg === refArg;
          agree += same; allSame &&= same;
        }
        total++;
      }
      decisionsSame += allSame;
    }
    ms.sort((a, b) => a - b);
    results[v] = { maxLogit, maxProb, agree, total, requestsAllSame: decisionsSame, p50ms: ms[ms.length >> 1] };
    console.log(`[parity ${v}]`, JSON.stringify(results[v]));
    await session.release();
    if (v === "fp32") assert.ok(maxLogit < 1e-3);
    if (v === "fp16") assert.ok(agree / total >= 0.99);
    if (v === "q8") assert.ok(agree / total >= 0.99);
  });
}
