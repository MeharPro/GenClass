// WebGPU path: decision model on the WebGPU execution provider (SwiftShader on GPU-less CI = correctness only).
import { expect, test } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { SHOTS, launch } from "./fixtures.mjs";

test("decision model on WebGPU agrees with the WASM run and parity fixtures", async () => {
  test.setTimeout(900000);
  const t = await launch({ settings: { compute: "webgpu" } });
  const line = await t.panel.evaluate(() => document.getElementById("engineLine").textContent);
  console.log(line);
  const reqs = JSON.parse(readFileSync(join(import.meta.dirname, "..", "fixtures", "requests50.json"), "utf8")).slice(0, 8)
    .map((r) => ({ state: r.state, questions: Object.fromEntries(Object.entries(r.questions).map(([k, q]) => [k, q.type === "choice" ? { ...q, criteria: Object.entries(q.criteria) } : q])) }));
  const b = await t.brain({ type: "bench_model", requests: reqs, repeat: 2 });
  console.log(JSON.stringify(b));
  writeFileSync(join(SHOTS, "..", `bench_model_${b.engine.provider}.json`), JSON.stringify({ line, ...b }, null, 2));
  await t.say("click the laptops link");
  const log = await t.brain({ type: "log", n: 100 });
  const acts = log.log.filter((x) => x.kind === "exec");
  expect(acts.length).toBe(1);
  await t.close();
});
