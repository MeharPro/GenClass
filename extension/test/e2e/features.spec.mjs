// End to end for the opt-in features: content filter, focus mode, RAM manager (local pages only).
import { expect, test } from "@playwright/test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { SHOTS, launch, setSettings, shot } from "./fixtures.mjs";

test.describe.configure({ mode: "serial" });
let t;
const report = {};
test.beforeAll(async () => { t = await launch(); });
test.afterAll(async () => { writeFileSync(join(SHOTS, "..", "e2e_features_report.json"), JSON.stringify(report, null, 2)); await t.close(); });

const tabIdOf = (u) => t.panel.evaluate(async (url) => (await chrome.tabs.query({})).find((x) => x.url && x.url.startsWith(url)).id, u);
const openFeatures = () => t.panel.evaluate(() => { document.getElementById("featuresBox").open = true; document.getElementById("featuresBox").scrollIntoView(); });

test("content filter hides marked ads and counts them on the badge; off by default", async () => {
  expect(await t.shop.evaluate(() => document.querySelectorAll("[data-ad-slot], aside.ad")[0].style.display)).toBe("");
  await setSettings(t, { filter: { enabled: true, hide: { ad: true, sponsored: true, clickbait: false, off_topic: false }, sites: {}, model: false } });
  await t.shop.reload();
  await expect.poll(() => t.shop.evaluate(() => [...document.querySelectorAll(".ad")].filter((e) => e.style.display === "none").length), { timeout: 15000 }).toBe(2);
  const badge = await t.panel.evaluate((id) => chrome.action.getBadgeText({ tabId: id }), t.tabId);
  expect(badge).toBe("2");
  report.filter = { hidden: 2, badge };
  await t.panel.reload();
  await t.panel.waitForFunction(() => /Model:/.test(document.getElementById("engineLine").textContent), null, { timeout: 120000 });
  await openFeatures();
  await shot(t, "05-content-filter");
  // per-site toggle off -> blocks come back
  await setSettings(t, { filter: { ...t.settings.filter, sites: { "127.0.0.1": false } } });
  await expect.poll(() => t.shop.evaluate(() => [...document.querySelectorAll(".ad")].filter((e) => e.style.display === "none").length), { timeout: 15000 }).toBe(0);
  await setSettings(t, { filter: { ...t.settings.filter, enabled: false, sites: {} } });
});

test("focus mode parks an off-task tab after the grace period and can restore it", async () => {
  const notes = await t.ctx.newPage();
  await notes.goto(`${t.base}/site/notes.html`);
  const recipes = await t.ctx.newPage();
  await recipes.goto(`${t.base}/site/recipes.html`);
  await t.shop.bringToFront();
  const notesId = await tabIdOf(`${t.base}/site/notes.html`);
  const recId = await tabIdOf(`${t.base}/site/recipes.html`);
  // "close" here: chrome.tabs.discard under a CDP-attached Playwright browser tears the context down (harness limit);
  // discard itself was exercised live in desktop Chrome.
  await setSettings(t, { focus: { enabled: true, task: "choosing a new laptop with good battery life", action: "close", graceMin: 0.25 } });
  const st = await t.brain({ type: "focus_config" });
  report.focus = { scores: st.tabs };
  const byId = new Map(st.tabs.map((x) => [x.id, x]));
  expect(byId.get(recId).onTask).toBe(false);
  expect(byId.get(notesId).onTask).toBe(true);
  await openFeatures();
  await shot(t, "06-focus-mode");
  await expect.poll(() => t.panel.evaluate((id) => chrome.tabs.get(id).then(() => "open", () => "closed"), recId), { timeout: 40000, intervals: [2000] }).toBe("closed");
  expect(await t.panel.evaluate((id) => chrome.tabs.get(id).then(() => "open", () => "closed"), notesId)).toBe("open");
  const { parked } = await t.panel.evaluate(() => chrome.storage.local.get("parked"));
  expect(parked[0].title).toMatch(/pasta/i);
  await t.panel.evaluate(() => document.getElementById("featuresBox").scrollIntoView());
  await shot(t, "07-focus-parked");
  await t.panel.evaluate((e) => chrome.runtime.sendMessage({ to: "sw", type: "tab_op", op: "restore", entry: e }), parked[0]);
  await expect.poll(() => t.panel.evaluate(async () => (await chrome.tabs.query({})).some((x) => /recipes/.test(x.url || ""))), { timeout: 15000 }).toBe(true);
  expect((await t.panel.evaluate(() => chrome.storage.local.get("parked"))).parked.length).toBe(0);
  report.focus.parked = parked[0].title;
  await setSettings(t, { focus: { ...t.settings.focus, enabled: false } });
  await t.brain({ type: "focus_config" });
  await notes.close().catch(() => {});
  for (const p of t.ctx.pages()) if (/recipes/.test(p.url())) await p.close().catch(() => {});
});

test("RAM manager parks idle unpinned tabs under memory pressure, never pinned ones", async () => {
  const a = await t.ctx.newPage();
  await a.goto(`${t.base}/site/recipes.html?ram=a`);
  const b = await t.ctx.newPage();
  await b.goto(`${t.base}/site/notes.html?ram=b`);
  await t.shop.bringToFront();
  const aId = await tabIdOf(`${t.base}/site/recipes.html?ram=a`);
  const bId = await tabIdOf(`${t.base}/site/notes.html?ram=b`);
  await t.panel.evaluate((id) => chrome.tabs.update(id, { pinned: true }), bId);
  // minFreePct 101 forces "pressure" so the policy can be checked on any machine.
  await setSettings(t, { ram: { enabled: true, minFreePct: 101, idleMin: 0, maxPerRound: 5 } });
  // preview: which tabs it would park (real discards are a harness limit under Playwright; see the focus test)
  const r = await t.panel.evaluate(() => chrome.runtime.sendMessage({ to: "sw", type: "ram_tick", preview: true }));
  report.ram = r;
  const ids = r.wouldPark.map((x) => x.id);
  expect(r.pressure).toBe(true);
  expect(r.source).toBe("chrome.system.memory");
  expect(ids).toContain(aId);
  expect(ids).not.toContain(bId); // pinned
  expect(ids).not.toContain(t.tabId); // active
  await openFeatures();
  await t.panel.evaluate(() => document.getElementById("ramStatus").scrollIntoView());
  await shot(t, "08-ram-manager");
  await setSettings(t, { ram: { ...t.settings.ram, enabled: false } });
});
