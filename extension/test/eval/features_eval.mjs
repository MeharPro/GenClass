// Zero-shot check of the feature questions (content filter, focus relevance) with the shipped model.
// node test/eval/features_eval.mjs [q8|fp16]   (CPU, onnxruntime-web wasm)
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-web";
import { Tokenizer } from "../../src/core/tokenizer.js";
import { Engine } from "../../src/core/engine.js";
import { filterRequest, filterResults, focusRequest, focusResults } from "../../src/core/features.js";

const ROOT = join(import.meta.dirname, "..", "..");
const A = join(ROOT, "release-assets");
const variant = process.argv[2] || "q8";
ort.env.wasm.numThreads = 4;
const rj = (p) => JSON.parse(readFileSync(p, "utf8"));
const session = await ort.InferenceSession.create(readFileSync(join(A, `genclass-${variant}.onnx`)));
const eng = new Engine({ ort, session, tokenizer: new Tokenizer(rj(join(A, "tokenizer.json"))), calibration: rj(join(A, "calibration.json")), meta: rj(join(A, "meta.json")), variant });

const BLOCKS = [
  ["Get 50% off premium noise-cancelling headphones today only. Shop now at SoundMax", "ad"],
  ["Buy one get one free on all pizzas this weekend. Order online.", "ad"],
  ["Refinance your mortgage at record low rates. Check your rate in 2 minutes.", "ad"],
  ["Try our new cloud hosting free for 30 days. No credit card required. Sign up", "ad"],
  ["Promoted · Acme CRM: the sales tool 10,000 teams trust", "sponsored"],
  ["Paid partnership with GlowSkin: my honest morning routine with their serum", "sponsored"],
  ["Sponsored content: 7 reasons travelers love this credit card", "sponsored"],
  ["You won't BELIEVE what this celebrity looks like now! Number 7 will shock you", "clickbait"],
  ["Doctors hate him: one weird trick to lose belly fat overnight", "clickbait"],
  ["This simple mistake is costing you thousands. Click to find out what it is", "clickbait"],
  ["She opened the box and couldn't believe what was inside...", "clickbait"],
  ["The city council voted 7-2 on Tuesday to expand the bike lane network along Main Street, with construction starting in spring.", "normal"],
  ["Python's asyncio module provides infrastructure for writing single-threaded concurrent code using coroutines.", "normal"],
  ["Home · News · Sport · Weather · Contact us", "normal"],
  ["Comments (24): I tried this recipe last night and it worked perfectly with less sugar.", "normal"],
  ["Researchers at the university published a study on sleep and memory consolidation in adults.", "normal"],
  ["Step 3: Preheat the oven to 180°C and grease a 20cm cake tin.", "normal"],
  ["Copyright 2026 Example Inc. All rights reserved. Privacy policy · Terms", "normal"],
];
const TASK = "writing my machine learning thesis about transformer attention";
const TABS = [
  ["Attention Is All You Need - arXiv", "https://arxiv.org/abs/1706.03762", "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms", true],
  ["thesis_draft.docx - Google Docs", "https://docs.google.com/document/d/abc", "", true],
  ["torch.nn.MultiheadAttention — PyTorch documentation", "https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html", "Allows the model to jointly attend to information from different representation subspaces", true],
  ["The Illustrated Transformer – Jay Alammar", "https://jalammar.github.io/illustrated-transformer/", "Visualizing machine learning one concept at a time", true],
  ["Overleaf - thesis.tex", "https://www.overleaf.com/project/123", "LaTeX editor", true],
  ["Google Scholar: self-attention efficiency", "https://scholar.google.com/scholar?q=self-attention+efficiency", "", true],
  ["Funny cat compilation 2026 - YouTube", "https://www.youtube.com/watch?v=xyz", "Try not to laugh", false],
  ["r/nba - Lakers vs Celtics game thread", "https://www.reddit.com/r/nba/comments/1", "", false],
  ["Amazon.com: running shoes men", "https://www.amazon.com/s?k=running+shoes", "", false],
  ["Netflix - Stranger Things", "https://www.netflix.com/watch/1", "", false],
  ["Cheap flights to Cancun | Expedia", "https://www.expedia.com/Flights", "", false],
  ["Instagram", "https://www.instagram.com/", "", false],
  ["Celebrity gossip: who wore it best at the Met Gala", "https://www.tmz.com/2026/05/met-gala", "", false],
  ["Fantasy football rankings week 5", "https://www.espn.com/fantasy/football", "", false],
];

const t0 = performance.now();
const blocks = BLOCKS.map(([text], i) => ({ id: String(i), text, labels: [] }));
const fr = filterRequest({ host: "news.example.com", title: "Example News" }, blocks, {});
const r1 = await eng.evaluate(fr.state, fr.questions);
const res = filterResults(r1, fr.preset, { minP: 0 });
let ok = 0;
let okBin = 0;
const conf = {};
BLOCKS.forEach(([, gold], i) => {
  const got = res.get(String(i)).top;
  ok += got === gold;
  okBin += (got === "normal") === (gold === "normal");
  conf[`${gold}->${got}`] = (conf[`${gold}->${got}`] || 0) + 1;
});
const tabs = TABS.map(([title, url, description], i) => ({ id: i, title, url, description }));
const fq = focusRequest(TASK, tabs);
const r2 = await eng.evaluate(fq.state, fq.questions);
const rel = focusResults(r2);
let fok = 0;
const ps = [];
TABS.forEach(([, , , gold], i) => { const p = rel.get(i); ps.push([gold, Number(p.toFixed(2))]); fok += (p >= 0.5) === gold; });
// AUC
let auc = 0;
let pairs = 0;
for (const [g1, p1] of ps) for (const [g2, p2] of ps) if (g1 && !g2) { pairs++; auc += p1 > p2 ? 1 : p1 === p2 ? 0.5 : 0; }
const out = {
  variant, ms: Math.round(performance.now() - t0),
  filter: { acc5: `${ok}/${BLOCKS.length}`, hideVsKeep: `${okBin}/${BLOCKS.length}`, confusion: conf, tokens: r1.usage.input_tokens },
  focus: { acc: `${fok}/${TABS.length}`, auc: Number((auc / pairs).toFixed(3)), p: ps, tokens: r2.usage.input_tokens },
};
console.log(JSON.stringify(out, null, 1));
writeFileSync(join(ROOT, "test", "eval", `features_eval_${variant}.json`), JSON.stringify(out, null, 1));
