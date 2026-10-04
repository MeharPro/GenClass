// The `state` object for a decision (port of jev_local/harness/state.py). Stable context first, transcript last.

import { elementLine } from "./questions.js";
import { describeAction } from "./types.js";

export function describeRecord(r) {
  const a = r.action;
  const arg = a.targetLabel || a.app || (a.text ? `"${a.text}"` : null) || a.key || a.url || a.folder;
  const s = a.kind.replaceAll("_", " ");
  return arg ? `${s} ${arg}` : s;
}

export function buildState(transcript, snap, history = [], pending = null) {
  let screen;
  let focused;
  if (snap) {
    screen = snap.appName + (snap.windowTitle ? `, window "${snap.windowTitle}"` : "");
    const f = snap.elements.find((e) => e.focused);
    focused = f ? elementLine(f) : "nothing";
  } else {
    screen = "unknown";
    focused = "nothing";
  }
  const recent = history.slice(-3).map(describeRecord);
  return {
    screen,
    focused,
    recent_actions: recent.length ? recent : "none",
    pending: pending ? `${describeAction(pending)} is waiting for the user to confirm` : "none",
    transcript,
  };
}
