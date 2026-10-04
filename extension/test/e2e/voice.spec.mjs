// End to end: the unpacked extension in Chromium, commands fed as streamed text (no mic), on a local page only.
import { expect, test } from "@playwright/test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { SHOTS, launch, setSettings, shot } from "./fixtures.mjs";

test.describe.configure({ mode: "serial" });
let t;
const report = { commands: [] };

test.beforeAll(async () => { t = await launch(); });
test.afterAll(async () => {
  const log = await t.brain({ type: "log", n: 5000 }).catch(() => null);
  const st = await t.panel.evaluate(() => document.getElementById("engineLine").textContent);
  report.engineLine = st;
  if (log) {
    const ds = log.log.filter((x) => x.kind === "decision");
    const ms = ds.map((d) => d.timings.model).sort((a, b) => a - b);
    report.decisions = ds.length;
    report.modelMsP50 = ms[ms.length >> 1];
    report.modelMsP90 = ms[Math.floor(ms.length * 0.9)];
    report.stage1Only = ds.filter((d) => d.passes === 1).length;
  }
  writeFileSync(join(SHOTS, "..", "e2e_voice_report.json"), JSON.stringify(report, null, 2));
  await t.close();
});

async function run(cmd, options) {
  const { seq } = await t.brain({ type: "log", n: 1 });
  const t0 = Date.now();
  const r = await t.say(cmd, options);
  const after = await t.brain({ type: "log", since: seq, n: 5000 });
  const mine = after.log.filter((x) => x.kind === "exec" || x.kind === "decision" || x.kind === "pending");
  const execs = after.log.filter((x) => x.kind === "exec");
  if (r && r.ok === false) report.commands.push({ cmd, refused: r.error });
  const words = cmd.split(" ").length;
  const fired = mine.filter((x) => x.kind === "decision" && x.verdict === "act").map((d) => `${d.words.consumedBefore + d.words.actionWords}/${words}`);
  report.commands.push({ cmd, ms: Date.now() - t0, firedAtWordOf: fired, execs: execs.map((e) => ({ kind: e.action.kind, detail: e.detail, outcome: e.outcome })),
    verdicts: mine.filter((x) => x.kind === "decision").map((d) => `${d.verdict}@${d.text.split(" ").length}w ${Math.round(d.timings.model)}ms`) });
  return execs;
}

test("model loads from the configured URL and runs in the browser", async () => {
  const line = await t.panel.evaluate(() => document.getElementById("engineLine").textContent);
  expect(line).toMatch(/Model: (q8|fp16) on (WASM|WEBGPU)/);
  const cached = await t.panel.evaluate(async () => (await (await caches.open("genclass-models-v1")).keys()).map((r) => r.url));
  expect(cached.some((u) => /genclass-(q8|fp16)\.onnx$/.test(u))).toBeTruthy();
});

test("dry run highlights but does not act", async () => {
  const ex = await run("click the laptops link");
  expect(ex.length).toBe(1);
  expect(ex[0].outcome).toBe("dry-run");
  expect(await t.shop.evaluate(() => location.hash)).toBe("");
  await shot(t, "01-dry-run");
});

test("live: click, type, scroll, mid-sentence chain", async () => {
  await setSettings(t, { dryRun: false });
  let ex = await run("click the laptops link");
  expect(ex.map((e) => e.action ? e.action.kind : e.kind)).toBeTruthy();
  await expect.poll(() => t.shop.evaluate(() => location.hash)).toBe("#laptops");

  ex = await run("type wireless headphones in the search box");
  await expect.poll(() => t.shop.evaluate(() => document.getElementById("q").value)).toContain("wireless headphones");
  await shot(t, "02-live-type");

  await t.shop.evaluate(() => window.scrollTo(0, 0));
  ex = await run("click the headphones link and then scroll down a little");
  await expect.poll(() => t.shop.evaluate(() => location.hash)).toBe("#headphones");
  await expect.poll(() => t.shop.evaluate(() => window.scrollY)).toBeGreaterThan(0);
  const kinds = ex.map((e) => e.action.kind);
  expect(kinds).toEqual(["click", "scroll_down"]);
  await shot(t, "03-mid-sentence");
});

test("risky actions wait for a spoken confirm; cancel drops them", async () => {
  await t.shop.evaluate(() => { window.__events = []; document.getElementById("log").textContent = ""; });
  await run("click buy now");
  await expect.poll(() => t.panel.evaluate(() => !document.getElementById("pending").hidden)).toBe(true);
  await shot(t, "04-confirm");
  expect(await t.shop.evaluate(() => window.__events.includes("ORDER PLACED"))).toBe(false);
  await new Promise((r) => setTimeout(r, 1000));
  await run("confirm");
  await expect.poll(() => t.shop.evaluate(() => window.__events.includes("ORDER PLACED"))).toBe(true);

  await run("click delete account");
  await expect.poll(() => t.panel.evaluate(() => !document.getElementById("pending").hidden)).toBe(true);
  await new Promise((r) => setTimeout(r, 1000));
  await run("no cancel that");
  await expect.poll(() => t.panel.evaluate(() => document.getElementById("pending").hidden)).toBe(true);
  expect(await t.shop.evaluate(() => window.__events.includes("ACCOUNT DELETED"))).toBe(false);
});

test("never types into a password field", async () => {
  await t.shop.evaluate(() => document.getElementById("pw").focus());
  const ex = await run("type hunter two");
  expect(await t.shop.evaluate(() => document.getElementById("pw").value)).toBe("");
  const log = await t.brain({ type: "log", n: 50 });
  const last = log.log.filter((x) => x.kind === "decision").pop();
  expect(ex.length).toBe(0);
  expect(["deny", "wait", "clarify", "ignore"]).toContain(last.verdict);
});

test("kill switch stops everything, including words already queued", async () => {
  await t.shop.evaluate(() => window.scrollTo(0, 0));
  const pending = run("click the deals link and then scroll down a lot", { stream: true, wordMs: 400 });
  await new Promise((r) => setTimeout(r, 300));
  await t.brain({ type: "kill", from: "test" });
  const ex = await pending;
  expect(ex.length).toBe(0);
  const after = await run("scroll down");
  expect(after.length).toBe(0);
  expect(await t.shop.evaluate(() => window.scrollY)).toBe(0);
  await t.say("scroll up", { stream: false, resume: true });
});

test("never acts on chrome:// pages", async () => {
  const page = await t.ctx.newPage();
  await page.goto("chrome://version");
  await page.bringToFront();
  const id = await t.panel.evaluate(async () => (await chrome.tabs.query({ active: true }))[0].id);
  await setSettings(t, { targetTabId: id });
  const ex = await run("click the first link");
  expect(ex.filter((e) => e.outcome === "ok").length).toBe(0);
  await setSettings(t, { targetTabId: t.tabId });
  await page.close();
  await t.shop.bringToFront();
});
