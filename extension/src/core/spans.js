// Verbatim span candidates from a transcript (port of jev_local/harness/spans.py).
// Select, don't generate: every text candidate is an exact substring of the input.

import { CHAIN_WORDS, FILLERS, FILLERS_LIST, reEscape } from "./catalog.js";
import { isAlnum, rstrip, split, split1, strip } from "./pyutil.js";

const QUOTE_RE = /["“”']([^"“”']{1,120})["“”']/g;

const PAYLOAD_VERB_RE = new RegExp(
  "\\b(?:" +
    "search\\s+(?:the\\s+web|google|online|the\\s+internet)\\s+for|" +
    "search\\s+(?<site>\\w+(?:\\s+\\w+)?)\\s+for|" +
    "search\\s+for|look\\s+up|google|find\\s+me|find|" +
    "type\\s+(?:in|out)|type|enter|write\\s+down|write|put\\s+in|put|fill\\s+in|dictate|say|search" +
    ")\\b[\\s,:]+(?<rest>.+)$",
  "i",
);

const DEST_SRC =
  "\\s+(?:in|into|on|inside|to|in\\s+to)\\s+(?:the\\s+|this\\s+|that\\s+|a\\s+)?(?:[\\w-]+\\s+){0,3}?" +
  "(?:box|field|input|bar|search\\s*bar|search\\s*box|search\\s*field|search|text\\s*box|textbox|" +
  "document|doc|note|notes|window|message|email|chat|address\\s*bar|url\\s*bar|terminal)\\s*$";
export const DEST_RE = new RegExp(DEST_SRC, "i");
const DEST_RE_AT0 = new RegExp(DEST_SRC, "iy"); // re.match: anchored at the start

const CHAIN_RE = new RegExp("\\s+(?:" + CHAIN_WORDS.map(reEscape).join("|") + ")\\s+", "gi");

const SENT_END_RE = new RegExp(
  "[.!?]+[\"')\\]]*(?=\\s+(?:(?:and|then|um|uh|ok|okay|so)[\\s,]+)*(?:open|launch|start|switch|quit|close|click|tap|" +
    "press|hit|select|choose|type|write|enter|dictate|search|google|look|go|visit|navigate|scroll|page|undo|" +
    "new|copy|paste|save|refresh|reload|cancel|confirm)\\b)",
  "gi",
);
export const EDGE_PUNCT = " \t,.;:!?";
const TRAILING_POLITE_RE = /(?:[\s,]+(?:please|thanks|thank\s+you|for\s+me|now|right\s+now))+\s*[.!?]*$/i;
const LEADING_FILLER_RE = new RegExp(
  "^(?:(?:" + [...FILLERS_LIST].sort((a, b) => b.length - a.length).join("|") + ")[\\s,]+)+",
  "i",
);
const CLEAN_STRIP = EDGE_PUNCT + "\"'“”‘’";

function clean(s) {
  s = strip(s, CLEAN_STRIP);
  s = s.replace(TRAILING_POLITE_RE, "");
  s = s.replace(LEADING_FILLER_RE, "");
  return strip(s, CLEAN_STRIP);
}

function variants(rest) {
  const cuts = new Set();
  for (const m of rest.matchAll(CHAIN_RE)) cuts.add(m.index);
  for (const m of rest.matchAll(SENT_END_RE)) cuts.add(m.index);
  const sorted = [...cuts].sort((a, b) => a - b);
  const prefixes = [rest, ...sorted.map((c) => rest.slice(0, c))];
  const out = [];
  for (const p of prefixes) {
    const m = DEST_RE.exec(p);
    if (m) out.push(p.slice(0, m.index));
    out.push(p);
  }
  return out;
}

export function extractTextCandidates(tail, maxN = 8, maxChars = 120) {
  const text = tail.trim();
  if (!text) return [];
  const raw = [];
  for (const m of text.matchAll(QUOTE_RE)) raw.push(m[1]);
  const m = PAYLOAD_VERB_RE.exec(text);
  if (m) raw.push(...variants(m.groups.rest));
  const idx = text.toLowerCase().indexOf(" for ");
  if (idx >= 0) raw.push(...variants(text.slice(idx + 5)));
  const parts = split1(text);
  if (parts.length === 2) raw.push(...variants(parts[1]));
  raw.push(text);

  const seen = new Set();
  const out = [];
  for (const r of raw) {
    const c = clean(r);
    const key = c.toLowerCase();
    if (!c || seen.has(key) || [...c].length > maxChars || key === "none") continue;
    if (!text.includes(c)) continue;
    seen.add(key);
    out.push(c);
    if (out.length >= maxN) break;
  }
  return out;
}

// ---------------------------------------------------------------- URLs

export const TLDS =
  "com|org|net|io|ai|dev|app|edu|gov|co|uk|ca|de|fr|es|it|nl|me|tv|gg|xyz|info|so|sh|ly|us|news|" +
  "fm|to|in|jp|au|ch|se|no|fi|be|at|cc|biz|page|site|online|tech|blog|wiki";
const URL_RE = new RegExp("\\b((?:https?://)?(?:[a-z0-9-]+\\.)+(?:" + TLDS + ")(?:/[^\\s,]*)?)", "gi");
const GOTO_RE =
  /\b(?:go\s+to|goto|visit|navigate\s+to|open\s+up|open|head\s+to|pull\s+up|load)\s+(?:the\s+)?([a-z0-9][a-z0-9-]{1,40})(?:\s+(?:website|site|page|dot\s+com))?\b/gi;
const WEB_CUE_RE = /\b(website|site|web\s*page|dot\s+com|\.com|go\s+to|visit|navigate)\b/i;

export function normalizeSpokenUrl(text) {
  let t = text.replace(/\s+dot\s+/gi, ".");
  t = t.replace(/\s+slash\s+/gi, "/");
  t = t.replace(/\s*colon\s*\/\s*\/\s*/gi, "://");
  t = t.replace(/\bw\s*w\s*w\b\.?/gi, "www.");
  return t;
}

export function extractUrlCandidates(tail, maxN = 6) {
  const out = [];
  const seen = new Set();
  const add = (u) => {
    u = strip(u, EDGE_PUNCT).toLowerCase();
    if (u && !seen.has(u)) {
      seen.add(u);
      out.push(u);
    }
  };
  for (const src of [tail, normalizeSpokenUrl(tail)]) {
    for (const m of src.matchAll(URL_RE)) add(m[1]);
  }
  if (WEB_CUE_RE.test(tail)) {
    for (const m of tail.matchAll(GOTO_RE)) {
      const w = m[1].toLowerCase();
      if (!["the", "a", "my", "new", "tab", "page", "website", "site"].includes(w)) add(`${w}.com`);
    }
  }
  return out.slice(0, maxN);
}

// ---------------------------------------------------------------- numbered picks

const ORDINALS = { first: 1, second: 2, third: 3, fourth: 4, fifth: 5, sixth: 6, seventh: 7, eighth: 8, ninth: 9, tenth: 10, last: -1 };
const CARDINALS = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, to: 2, too: 2, for: 4 };
const CARD_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];

/** 'the second one' / 'number 3' / 'option two' / '3' -> 0-based index, else null. */
export function parseCandidatePick(tail, nCandidates) {
  const t = strip(tail.toLowerCase(), EDGE_PUNCT);
  let m = /\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b(?:\s+one)?/.exec(t);
  if (m) {
    const k = ORDINALS[m[1]];
    const idx = k === -1 ? nCandidates - 1 : k - 1;
    return idx >= 0 && idx < nCandidates ? idx : null;
  }
  m = /\b(?:number|option|choice|item|pick)\s+(\d+|\w+)\b/.exec(t);
  if (m) {
    const v = m[1];
    const k = /^\d+$/.test(v) ? parseInt(v, 10) : (CARDINALS[v] ?? null);
    if (k !== null && k >= 1 && k <= nCandidates) return k - 1;
  }
  if (/^\d{1,2}$/.test(t)) {
    const k = parseInt(t, 10);
    return k >= 1 && k <= nCandidates ? k - 1 : null;
  }
  if (CARD_WORDS.includes(t)) {
    const k = CARDINALS[t];
    return k <= nCandidates ? k - 1 : null;
  }
  return null;
}

// ---------------------------------------------------------------- consumption

const CHAIN_SET = new Set(CHAIN_WORDS);
const CHAIN_SINGLE = new Set(CHAIN_WORDS.filter((w) => !w.includes(" ")));
const low = (w) => strip(w.toLowerCase(), EDGE_PUNCT);

/** Number of leading words that are chain/filler words ('and', 'then', 'um', ...). */
export function stripLeadingChain(words) {
  let i = 0;
  const lowered = words.map(low);
  while (i < lowered.length) {
    const two = lowered.slice(i, i + 2).join(" ");
    if (CHAIN_SET.has(two)) i += 2;
    else if (CHAIN_SINGLE.has(lowered[i]) || FILLERS.has(lowered[i])) i += 1;
    else break;
  }
  return i;
}

export const NO_ARG_INTENTS = new Set(["undo", "confirm", "cancel", "go_back", "go_forward", "new_tab", "close_tab", "scroll_down", "scroll_up"]);

/** How many tail words the executed action accounts for. */
export function consumedFor(tailWords, intent, span = null) {
  const words = [...tailWords];
  if (!words.length) return 0;
  const lowered = words.map(low);
  let end = words.length;
  let startSearch = 1;
  if (span) {
    const sw = split(span).map(low);
    const n = sw.length;
    let found = false;
    for (let i = 0; i < lowered.length - n + 1; i++) {
      let eq = true;
      for (let j = 0; j < n; j++) if (lowered[i + j] !== sw[j]) { eq = false; break; }
      if (eq) {
        end = i + n;
        startSearch = end;
        found = true;
        break;
      }
    }
    if (!found) return words.length;
    const rest = lowered.slice(end);
    let cut = rest.length;
    for (let j = 0; j < rest.length; j++) {
      if (CHAIN_SET.has(rest[j]) || CHAIN_SET.has(rest.slice(j, j + 2).join(" "))) {
        cut = j;
        break;
      }
    }
    const seg = words.slice(end, end + cut).join(" ");
    if (seg) {
      DEST_RE_AT0.lastIndex = 0;
      if (DEST_RE_AT0.test(" " + seg)) end += cut;
    }
    return end + stripLeadingChain(words.slice(end));
  }
  for (let i = 0; i < lowered.length; i++) {
    if (i >= startSearch) {
      const two = lowered.slice(i, i + 2).join(" ");
      if (CHAIN_SET.has(two)) return i + 2;
      if (CHAIN_SET.has(lowered[i])) return i + 1;
    }
    if (i + 1 < words.length && endsSentence(words[i]) && (i >= 1 || NO_ARG_INTENTS_PY.has(intent))) {
      return i + 1 + stripLeadingChain(words.slice(i + 1));
    }
  }
  return end;
}

// The exact Python set (consumed_for parity); go_forward is a GenClass intent that behaves the same.
const NO_ARG_INTENTS_PY = new Set([...NO_ARG_INTENTS]);

/** 'notes.' / 'down!' / 'it?"' end a sentence; 'wikipedia.org' and '...' alone do not. */
export function endsSentence(word) {
  const w = rstrip(word, "\"')]");
  return w.length > 1 && ".!?".includes(w[w.length - 1]) && [...w].some(isAlnum);
}
