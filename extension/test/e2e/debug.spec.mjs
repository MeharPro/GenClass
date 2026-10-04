// Dev helper: run commands and print the brain log.
import { test } from "@playwright/test";
import { launch, setSettings } from "./fixtures.mjs";
test("debug commands", async () => {
  const t = await launch();
  if (process.env.LIVE) await setSettings(t, { dryRun: false });
  let i = 0;
  for (const cmd of (process.env.CMD || "click the laptops link").split("|")) {
    if (process.env.LIVE_AFTER && i++ === 1) await setSettings(t, { dryRun: false });
    const { seq } = await t.brain({ type: "log", n: 1 });
    const r = await t.say(cmd);
    console.log("say", cmd, "->", JSON.stringify(r));
    const log = await t.brain({ type: "log", since: seq, n: 500 });
    for (const x of log.log) {
      if (x.kind === "transcript") continue;
      if (x.kind === "decision") console.log("  decision", x.text, x.verdict, x.reason, x.rescue ? "RESCUE " + JSON.stringify(x.rescue) : "");
      else console.log(" ", x.kind, JSON.stringify({ ...x, seq: undefined, t: undefined }).slice(0, 400));
    }
    console.log("  page:", await t.shop.evaluate(() => JSON.stringify({ hash: location.hash, q: document.getElementById("q").value, y: scrollY, ev: window.__events || [] })));
  }
  await t.close();
});
