// Speech engines in the browser without a microphone: WAV clips through the real VAD/partials pipeline.
// Downloads the speech models from Hugging Face (network needed). Results -> dist/store/e2e_speech_report.json
import { expect, test } from "@playwright/test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { SHOTS, launch, setSettings, shot } from "./fixtures.mjs";

test.describe.configure({ mode: "serial" });
let t;
const report = {};
test.beforeAll(async () => { t = await launch({ settings: { speechEngine: "moonshine" } }); });
test.afterAll(async () => { writeFileSync(join(SHOTS, "..", "e2e_speech_report.json"), JSON.stringify(report, null, 2)); await t.close(); });

test("benchmark local speech engines on a 3 s clip (first partial, RTF)", async () => {
  test.setTimeout(1800000);
  const engines = (process.env.ENGINES || "moonshine,whisper-base,whisper-turbo").split(",");
  const r = await t.brain({ type: "bench_speech", url: `${t.base}/audio/scroll.wav`, engines });
  report.bench = r;
  console.log(JSON.stringify(r, null, 1));
  expect(r.moonshine.final.toLowerCase()).toContain("scroll down");
});

test("Moonshine audio drives the live loop: spoken 'click the laptops link' clicks it", async () => {
  test.setTimeout(600000);
  await setSettings(t, { dryRun: false, speechEngine: "moonshine" });
  await t.shop.evaluate(() => { location.hash = ""; window.scrollTo(0, 0); });
  const r = await t.brain({ type: "play_audio", url: `${t.base}/audio/click.wav` });
  report.clickClip = r;
  console.log(JSON.stringify(r));
  await expect.poll(() => t.shop.evaluate(() => location.hash), { timeout: 10000 }).toBe("#laptops");
  await shot(t, "09-moonshine-voice");
});
