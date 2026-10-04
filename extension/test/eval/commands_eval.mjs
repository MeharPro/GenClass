// Which spoken commands resolve to the right action with the shipped model (q8, Node). Streamed word by word.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-web";
import { Tokenizer } from "../../src/core/tokenizer.js";
import { Engine } from "../../src/core/engine.js";
import { Controller } from "../../src/core/controller.js";
import { appCatalog } from "../../src/core/sites.js";
import { defaultConfig } from "../../src/core/types.js";
const ROOT = join(import.meta.dirname, "..", ".."); const A = join(ROOT, "release-assets");
ort.env.wasm.numThreads = 4;
const rj = (p) => JSON.parse(readFileSync(p, "utf8"));
const engine = new Engine({ ort, session: await ort.InferenceSession.create(readFileSync(join(A, "genclass-q8.onnx"))), tokenizer: new Tokenizer(rj(join(A, "tokenizer.json"))), calibration: rj(join(A, "calibration.json")), meta: rj(join(A, "meta.json")) });
const TABS = [{ id: 1, title: "Google", url: "https://www.google.com/" }, { id: 2, title: "Lo-fi beats - YouTube", url: "https://www.youtube.com/watch?v=1" }, { id: 3, title: "Inbox - Gmail", url: "https://mail.google.com/" }];
const PAGE = { id: 1, appName: "Google Chrome", windowTitle: "weather toronto - Google Search", browser: true, elements: [
  ["search field", "Search", "search", true], ["button", "Google Search", "search"], ["link", "Images", "navigation"], ["link", "News", "navigation"],
  ["link", "Toronto Weather Forecast - The Weather Network", "results"], ["link", "Toronto, ON 10-Day Weather - weather.com", "results"], ["link", "Sign in", "header"],
].map(([role, label, context, focused], i) => ({ eid: `e0${i + 1}`, role, label, context, focused: !!focused, enabled: true, inForm: context === "search", searchForm: context === "search" })), focusedEid: "e01" };
const CASES = [
  ["search for weather in toronto", "search_web", "weather in toronto"], ["open github", "open_app", "GitHub"], ["switch to youtube", "open_app", "YouTube"],
  ["go to wikipedia dot org", "open_url", "https://wikipedia.org"], ["press enter", "press_key", "return"], ["go back", "go_back", null],
  ["go forward", "go_forward", null], ["open a new tab", "new_tab", null], ["close this tab", "close_tab", null], ["scroll down", "scroll_down", null],
  ["scroll up a little", "scroll_up", null], ["click sign in", "click", "e07"], ["click the images link", "click", "e03"], ["click the first result", "click", "e05"],
  ["type hello world", "type_text", "hello world"], ["undo that", "undo", null], ["copy that", "press_key", "cmd+c"], ["reload the page", "press_key", "cmd+r"],
];
const rows = [];
for (const [said, kind, arg] of CASES) {
  const acts = []; const pend = [];
  const c = new Controller({ engine, cfg: defaultConfig(), browser: true, observer: { snapshot: async () => PAGE },
    executor: { run: async (a) => { acts.push(a); return { ok: true }; } }, apps: (t) => appCatalog(TABS, t).names, ui: (e, p) => { if (e === "pending" && p) pend.push(p.action); } });
  const w = said.split(" ");
  for (let i = 1; i <= w.length; i++) { c.onEvent({ kind: "partial", uid: "u", text: w.slice(0, i).join(" ") }); await new Promise((r) => setTimeout(r, 200)); }
  c.onEvent({ kind: "final", uid: "u", text: said });
  await new Promise((r) => setTimeout(r, 1000)); await c.idle();
  const a = acts[0] || pend[0];
  const got = a ? [a.kind, a.targetEid || a.app || a.text || a.key || a.url || null] : ["-", null];
  const ok = got[0] === kind && (arg === null || got[1] === arg);
  rows.push({ said, want: [kind, arg], got, viaConfirm: !acts[0] && !!pend[0], ok, last: c.decisions.at(-1)?.reason });
  console.log(ok ? "ok  " : "MISS", said.padEnd(32), JSON.stringify(got), !acts[0] && pend[0] ? "(confirm)" : "", ok ? "" : c.decisions.at(-1)?.reason);
}
const n = rows.filter((r) => r.ok).length;
console.log(`${n}/${rows.length}`);
writeFileSync(join(ROOT, "test", "eval", "commands_eval.json"), JSON.stringify({ score: `${n}/${rows.length}`, rows }, null, 1));
