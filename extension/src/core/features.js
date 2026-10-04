// Typed questions for GenClass's page and tab features. Same model, no extra training: each feature is a set of
// choice / noul questions over a small state. Questions are block-isolated, so one request can carry many
// blocks or tabs and every answer is the same as if it were asked alone.

const clip = (s, n) => {
  const t = (s || "").replace(/\s+/g, " ").trim();
  return t.length > n ? t.slice(0, n - 1).trimEnd() + "…" : t;
};

// ------------------------------------------------------------------ content filter

export const BLOCK_KINDS = {
  ad: "an advertisement for a product or service",
  sponsored: "sponsored, promoted or paid content mixed in with the page",
  clickbait: "a sensational headline written mainly to make people click",
  off_topic: "content unrelated to `task`",
  normal: "ordinary page content, navigation or text",
};

// Deterministic ad signals: an ad iframe or an explicit "Sponsored"/"Ad" label decides without the model.
const AD_HOST_RE = /doubleclick|googlesyndication|googleadservices|adservice|amazon-adsystem|taboola|outbrain|criteo|adnxs|pubmatic|rubiconproject|moatads|media\.net/i;
const LABEL_RE = /^(sponsored|promoted|advertisement|ad|ads|paid partnership|anzeige|publicité)$/i;

/** blocks: [{id, text, labels[], adFrame?}] -> {state, questions, preset: Map id -> kind} */
export function filterRequest(page, blocks, { task = "" } = {}) {
  const preset = new Map();
  const questions = {};
  for (const b of blocks) {
    if (b.adFrame && AD_HOST_RE.test(b.adFrame)) { preset.set(b.id, "ad"); continue; }
    if ((b.labels || []).some((l) => LABEL_RE.test(l.trim()))) { preset.set(b.id, "sponsored"); continue; }
    const crit = new Map(Object.entries(BLOCK_KINDS));
    if (!task) crit.delete("off_topic");
    questions[`b${b.id}`] = {
      type: "choice",
      instructions: `What kind of page block is this? Block: "${clip(b.text, 220)}"`,
      criteria: crit,
    };
  }
  const state = { site: page.host || "", page: clip(page.title, 80), task: task || "none" };
  return { state, questions, preset };
}

/** Answers -> Map id -> {kind, p}. Hide only confident non-normal picks. */
export function filterResults(resp, preset, { minP = 0.6 } = {}) {
  const out = new Map();
  for (const [id, kind] of preset) out.set(id, { kind, p: 1, rule: true });
  for (const [qid, a] of Object.entries(resp ? resp.answers : {})) {
    const id = qid.slice(1);
    const p = a.probabilities.get(a.choice) ?? 0;
    out.set(id, { kind: a.choice !== "normal" && p >= minP ? a.choice : "normal", p, top: a.choice });
  }
  return out;
}

// ------------------------------------------------------------------ focus mode

export const RELEVANT = {
  true: "relevant: needed for the task or directly about it",
  false: "off-task: unrelated to the task, a distraction",
};

/** tabs: [{id, title, url, description}] -> request with one noul per tab. */
export function focusRequest(task, tabs) {
  const questions = {};
  for (const t of tabs) {
    let host = "";
    try { host = new URL(t.url).hostname.replace(/^www\./, ""); } catch { /* chrome:// etc. */ }
    const desc = t.description ? ` - ${clip(t.description, 140)}` : "";
    questions[`t${t.id}`] = {
      type: "noul",
      instructions: `Does this browser tab help with \`task\`? Tab: "${clip(t.title, 90)}" (${host})${desc}`,
      criteria: RELEVANT,
    };
  }
  return { state: { task: clip(task, 200) }, questions };
}

/** -> Map tabId -> relevance p(true) */
export function focusResults(resp) {
  const out = new Map();
  for (const [qid, a] of Object.entries(resp ? resp.answers : {})) out.set(Number(qid.slice(1)), a.noul);
  return out;
}

/** Tabs focus mode and the RAM manager may never touch. */
export function protectedTab(t, info = {}) {
  if (t.pinned) return "pinned";
  if (t.audible) return "playing audio";
  if (t.active) return "active";
  if (info.dirty) return "has unsaved form input";
  if (t.url && /^(chrome|chrome-extension|devtools|edge):/i.test(t.url)) return "browser page";
  return null;
}

/** RAM manager order: lowest relevance first (when known), then least recently used. */
export function discardOrder(tabs, relevance = new Map()) {
  return [...tabs].sort((a, b) => {
    const ra = relevance.has(a.id) ? relevance.get(a.id) : 0.5;
    const rb = relevance.has(b.id) ? relevance.get(b.id) : 0.5;
    if (Math.abs(ra - rb) > 0.15) return ra - rb;
    return (a.lastAccessed || 0) - (b.lastAccessed || 0);
  });
}

// ------------------------------------------------------------------ focus verdict
// The model ranks tabs well but its relevance probabilities are low in absolute terms (it was trained on
// computer-use questions, not tab relevance). A tab counts as on-task if the model clears a low bar OR it shares
// a content word with the task. Validated on held-out tasks in test/eval/focus_heldout.mjs.

const STOP = new Set("the and for with that this from your about into what when where which have will just more than then them they their there our out are was were how why who its it's my me i a an of to in on at by is be or as up".split(" "));
export function contentWords(s) {
  return (s || "").toLowerCase().split(/[^a-z0-9]+/).filter((w) => w.length >= 4 && !STOP.has(w)).map((w) => w.slice(0, 6));
}

export function keywordOverlap(task, tab) {
  const tw = new Set(contentWords(task));
  let host = "";
  try { host = new URL(tab.url).hostname; } catch { /* ignore */ }
  const words = contentWords(`${tab.title} ${host} ${tab.url || ""} ${tab.description || ""}`);
  return words.filter((w) => tw.has(w)).length;
}

export const FOCUS_MODEL_MIN = 0.04;

/** -> {onTask, p, overlap} */
export function focusVerdict(task, tab, p, minP = FOCUS_MODEL_MIN) {
  const overlap = keywordOverlap(task, tab);
  return { onTask: p >= minP || overlap > 0, p, overlap };
}
