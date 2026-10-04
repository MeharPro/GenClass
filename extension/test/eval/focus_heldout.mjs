// Held-out check of focusVerdict (model relevance + keyword overlap): tasks and tabs written after the rule was fixed.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-web";
import { Tokenizer } from "../../src/core/tokenizer.js";
import { Engine } from "../../src/core/engine.js";
import { focusRequest, focusResults, focusVerdict } from "../../src/core/features.js";
const ROOT = join(import.meta.dirname, "..", ".."); const A = join(ROOT, "release-assets");
ort.env.wasm.numThreads = 4;
const rj = (p) => JSON.parse(readFileSync(p, "utf8"));
const eng = new Engine({ ort, session: await ort.InferenceSession.create(readFileSync(join(A, "genclass-q8.onnx"))), tokenizer: new Tokenizer(rj(join(A, "tokenizer.json"))), calibration: rj(join(A, "calibration.json")), meta: rj(join(A, "meta.json")) });
const SETS = {
  "planning a family trip to Japan in April": [
    ["Kyoto cherry blossom forecast 2027", "https://www.japan-guide.com/sakura/", 1], ["JR Pass prices and options", "https://www.jrpass.com/", 1],
    ["Flights Toronto to Tokyo | Air Canada", "https://www.aircanada.com/flights", 1], ["Booking.com: Hotels in Osaka", "https://www.booking.com/city/jp/osaka.html", 1],
    ["Family-friendly things to do in Tokyo - Lonely Planet", "https://www.lonelyplanet.com/japan/tokyo", 1], ["Yen to CAD exchange rate", "https://www.xe.com/currencyconverter/", 1],
    ["Gmail - Inbox (3)", "https://mail.google.com/mail/u/0/", 0], ["Lakers highlights - YouTube", "https://www.youtube.com/watch?v=1", 0],
    ["Q3 budget.xlsx - Excel", "https://office.com/excel/q3", 0], ["How to fix a leaky faucet", "https://www.familyhandyman.com/faucet", 0],
    ["Twitch - Just Chatting", "https://www.twitch.tv/directory", 0], ["Zillow: homes for sale in Austin", "https://www.zillow.com/austin-tx/", 0]],
  "fixing a memory leak in our React dashboard": [
    ["React Profiler – React docs", "https://react.dev/reference/react/Profiler", 1], ["useEffect cleanup memory leak - Stack Overflow", "https://stackoverflow.com/questions/1", 1],
    ["dashboard/src/Chart.tsx · GitHub", "https://github.com/acme/dashboard/blob/main/src/Chart.tsx", 1], ["Chrome DevTools: Fix memory problems", "https://developer.chrome.com/docs/devtools/memory-problems", 1],
    ["Issue #482: heap grows after navigation · acme/dashboard", "https://github.com/acme/dashboard/issues/482", 1], ["localhost:3000 - Acme Dashboard", "http://localhost:3000/", 1],
    ["Best pasta recipes", "https://www.allrecipes.com/pasta", 0], ["Spotify – Discover Weekly", "https://open.spotify.com/playlist/1", 0],
    ["Reddit - r/funny", "https://www.reddit.com/r/funny", 0], ["Amazon.com: standing desk", "https://www.amazon.com/s?k=standing+desk", 0],
    ["Weather Toronto - The Weather Network", "https://www.theweathernetwork.com/ca/weather/ontario/toronto", 0], ["Netflix", "https://www.netflix.com/browse", 0]],
  "studying for my organic chemistry midterm": [
    ["Khan Academy: Alkene reactions", "https://www.khanacademy.org/science/organic-chemistry/alkenes", 1], ["Master Organic Chemistry: SN1 vs SN2", "https://www.masterorganicchemistry.com/sn1-sn2", 1],
    ["CHEM 266 Midterm practice problems.pdf", "https://learn.uwaterloo.ca/d2l/chem266/practice.pdf", 1], ["Quizlet - Orgo functional groups flashcards", "https://quizlet.com/orgo-groups", 1],
    ["Chirality and stereoisomers - LibreTexts", "https://chem.libretexts.org/stereo", 1], ["Instagram", "https://www.instagram.com/", 0],
    ["NBA trade rumors", "https://www.espn.com/nba/rumors", 0], ["Steam Store - Summer Sale", "https://store.steampowered.com/", 0],
    ["Shein - Women's dresses", "https://www.shein.com/dresses", 0], ["TikTok - For You", "https://www.tiktok.com/foryou", 0],
    ["Uber Eats - Sushi near you", "https://www.ubereats.com/", 0], ["LinkedIn Feed", "https://www.linkedin.com/feed/", 0]],
};
const res = { per: {}, tp: 0, fp: 0, tn: 0, fn: 0, modelOnly: { correct: 0 }, ms: 0 };
const t0 = performance.now();
for (const [task, rows] of Object.entries(SETS)) {
  const tabs = rows.map(([title, url], i) => ({ id: i, title, url }));
  const fq = focusRequest(task, tabs);
  const rel = focusResults(await eng.evaluate(fq.state, fq.questions));
  res.per[task] = rows.map(([title, url, g], i) => {
    const v = focusVerdict(task, tabs[i], rel.get(i));
    if (g && v.onTask) res.tp++; else if (g) res.fn++; else if (v.onTask) res.fp++; else res.tn++;
    res.modelOnly.correct += (rel.get(i) >= 0.04) === !!g;
    return { title, gold: g, p: Number(rel.get(i).toFixed(3)), overlap: v.overlap, onTask: v.onTask };
  });
}
res.ms = Math.round(performance.now() - t0);
res.summary = `on-task kept ${res.tp}/${res.tp + res.fn}; off-task flagged ${res.tn}/${res.tn + res.fp}; model-only at 0.04: ${res.modelOnly.correct}/36`;
console.log(res.summary, res.ms + "ms");
for (const [t, r] of Object.entries(res.per)) console.log(t, r.map((x) => `${x.gold ? "+" : "-"}${x.p}${x.overlap ? "k" : ""}${x.onTask === !!x.gold ? "" : "✗"}`).join(" "));
writeFileSync(join(ROOT, "test", "eval", "focus_heldout.json"), JSON.stringify(res, null, 1));
