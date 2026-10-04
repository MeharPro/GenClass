// Probe question phrasings for the zero-shot features (dev tool).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-web";
import { Tokenizer } from "../../src/core/tokenizer.js";
import { Engine } from "../../src/core/engine.js";
const ROOT = join(import.meta.dirname, "..", ".."); const A = join(ROOT, "release-assets");
ort.env.wasm.numThreads = 4;
const rj = (p) => JSON.parse(readFileSync(p, "utf8"));
const eng = new Engine({ ort, session: await ort.InferenceSession.create(readFileSync(join(A, "genclass-q8.onnx"))), tokenizer: new Tokenizer(rj(join(A, "tokenizer.json"))), calibration: rj(join(A, "calibration.json")), meta: rj(join(A, "meta.json")) });
const src = readFileSync(join(import.meta.dirname, "features_eval.mjs"), "utf8");
const BLOCKS = eval(src.match(/const BLOCKS = (\[[\s\S]*?\n\]);/)[1]);
const isAd = (g) => g === "ad" || g === "sponsored";
const auc = (ps) => { let a = 0, n = 0; for (const [g1, p1] of ps) for (const [g2, p2] of ps) if (g1 && !g2) { n++; a += p1 > p2 ? 1 : p1 === p2 ? .5 : 0; } return +(a / n).toFixed(3); };
async function noulVariant(name, mk, stateFn, goldFn) {
  const qs = {}; BLOCKS.forEach(([t], i) => { qs["b" + i] = mk(t); });
  const r = await eng.evaluate(stateFn(), qs);
  const ps = BLOCKS.map(([, g], i) => [goldFn(g), r.answers["b" + i].noul]);
  const acc = ps.filter(([g, p]) => (p >= .5) === g).length;
  console.log(name, "acc", acc + "/" + ps.length, "auc", auc(ps), "p", ps.map(([g, p]) => (g ? "+" : "-") + p).join(" "));
}
const S = () => ({ site: "news.example.com", page: "Example News", task: "none" });
await noulVariant("A-ad-noul", (t) => ({ type: "noul", instructions: `Is this page block an advertisement or sponsored content? Block: "${t}"`, criteria: { true: "an ad or sponsored post, for example: shop now, 50% off, sponsored, promoted", false: "normal content, for example: news text, navigation, comments, recipes" } }), S, isAd);
await noulVariant("B-ad-noul-state", (t) => ({ type: "noul", instructions: `Is \`block\` an advertisement or sponsored content?`, criteria: { true: "an ad or sponsored post", false: "normal page content" } }), S, isAd);
await noulVariant("C-clickbait", (t) => ({ type: "noul", instructions: `Is this headline clickbait, written mainly to make people click? Headline: "${t}"`, criteria: { true: "clickbait, for example: you won't believe, one weird trick, shocking", false: "a plain, informative headline or text" } }), S, (g) => g === "clickbait");
await noulVariant("D-is-command-style", (t) => ({ type: "noul", instructions: `Is \`transcript\` trying to sell something or get a click, rather than inform?`, criteria: { true: "selling or click bait, for example: buy now, 50% off, you won't believe", false: "informing, for example: the council voted, step 3 preheat the oven" } }), S, (g) => g !== "normal");
// B and D put the block in the state (one request per block)
for (const [nm, key, ins, gold] of [["B'-ad-state", "block", "Is `block` an advertisement or sponsored content?", isAd], ["D'-sell-transcript", "transcript", "Is `transcript` trying to sell something or get a click, rather than inform?", (g) => g !== "normal"]]) {
  const ps = [];
  for (const [t, g] of BLOCKS) {
    const r = await eng.evaluate({ site: "news.example.com", [key]: t }, { q: { type: "noul", instructions: ins, criteria: { true: "yes, for example: buy now, 50% off, sponsored, you won't believe", false: "no, plain information, for example: the council voted, preheat the oven" } } });
    ps.push([gold(g), r.answers.q.noul]);
  }
  console.log(nm, "acc", ps.filter(([g, p]) => (p >= .5) === g).length + "/" + ps.length, "auc", auc(ps));
}
