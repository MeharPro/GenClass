// The transcript stream: a consumed-prefix cursor that makes mid-sentence acting safe
// (port of jev_local/harness/stream.py). Times are in seconds (same clock as the controller).
//
// 1. Executed words are consumed; later partials must still start with that prefix (case, punctuation, fillers
//    and chain words ignored), else the recognizer rewrote acted-on words: `revised_after_act`, frozen.
// 2. Each consume starts a new virtual utterance "<uid>+<gen>" whose tail holds only the unconsumed words.
// 3. A fired set of (vid, cursor, n) keys stops a decision computed twice from running twice.
// A new uid that echoes the consumed prefix of the previous one within a short window inherits it (uid_echo).

import { CHAIN_WORDS, FILLERS } from "./catalog.js";
import { stripLeadingChain } from "./spans.js";

const NON_WORD_RE = /[^\p{L}\p{N}]+/gu;
const CHAIN_SINGLE = new Set(CHAIN_WORDS.filter((w) => !w.includes(" ")));
const CHAIN_PAIRS = new Set(CHAIN_WORDS.filter((w) => w.includes(" ")));
const IGNORABLE = new Set([...FILLERS, ...CHAIN_SINGLE]);
const SHORT = -1;

export function normWord(w) {
  return w.toLowerCase().replace(NON_WORD_RE, "");
}

export function coreWords(words) {
  return words.map(normWord).filter((n) => n && !IGNORABLE.has(n));
}

function required(core) {
  const r = [...core];
  while (r.length >= 2 && CHAIN_PAIRS.has(`${r[r.length - 2]} ${r[r.length - 1]}`)) r.splice(-2, 2);
  return r;
}

/** Raw index in `words` just past `req` (ignorable words skipped). null = mismatch, -1 = too short. */
export function matchPrefix(words, req) {
  let j = 0;
  for (let i = 0; i < words.length; i++) {
    if (j === req.length) return i;
    const n = normWord(words[i]);
    if (!n || IGNORABLE.has(n)) continue;
    if (n !== req[j]) return null;
    j++;
  }
  return j === req.length ? words.length : SHORT;
}

function skipLeading(words) {
  let i = 0;
  while (i < words.length) {
    const k = stripLeadingChain(words.slice(i));
    if (k) i += k;
    else if (!normWord(words[i])) i += 1;
    else break;
  }
  return Math.min(i, words.length);
}

const eqArr = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const splitWords = (t) => { const s = t.trim(); return s ? s.split(/\s+/) : []; };

export class Stream {
  constructor(log = null, echoWindowS = 2.0, dupWindowS = 0.35) {
    this.echoWindowS = echoWindowS;
    this.dupWindowS = dupWindowS;
    this.logFn = log;
    this.revisions = [];
    this.reset();
  }

  reset() {
    this.uid = "";
    this.gen = 0;
    this.words = [];
    this.wordT = [];
    this.cursor = 0;
    this.final = false;
    this.joined = false;
    this.frozen = false;
    this.dropped = false;
    this.lastChangeT = 0;
    this.lastEventT = 0;
    this.changed = false;
    this.consumedCore = [];
    this.required = [];
    this.carried = false;
    this.carriedDropped = false;
    this.fired = new Set();
    this.closed = [];
  }

  get vid() { return `${this.uid}+${this.gen}`; }
  get heardWords() { return this.words.length; }
  get consumedText() { return this.consumedCore.join(" "); }

  /** ev: {kind, uid, text, joined?}. Returns the unconsumed tail or null. */
  update(ev, now) {
    this.changed = false;
    if (ev.kind !== "partial" && ev.kind !== "final") return null;
    if (ev.uid !== this.uid) {
      if (this.closed.includes(ev.uid)) {
        this.log("late_event", { uid: ev.uid, current: this.uid, event: ev.kind, text: ev.text });
        return null;
      }
      this.begin(ev.uid, splitWords(ev.text), now);
    }
    this.joined = this.joined || !!ev.joined;
    this.lastEventT = now;
    if (this.dropped) return null;

    const words = splitWords(ev.text);
    let pos = matchPrefix(words, this.required);
    if (pos === null && this.carried && this.gen === 0) {
      this.log("echo_rejected", { uid: this.uid, consumed: this.consumedText, text: ev.text });
      this.consumedCore = [];
      this.required = [];
      this.carried = false;
      pos = 0;
    }
    if (pos === null) {
      if (!this.frozen) {
        const rec = { uid: this.uid, consumed: this.consumedText, text: ev.text, t: now };
        this.revisions.push(rec);
        this.log("revised_after_act", rec);
      }
      this.frozen = true;
      return null;
    }
    if (pos === SHORT) return null;
    if (this.carried && this.required.length) {
      this.carried = false;
      this.log("uid_echo", { uid: this.uid, consumed: this.consumedText, text: ev.text });
      if (this.carriedDropped) {
        this.dropped = true;
        return null;
      }
    }
    this.frozen = false;
    if (!eqArr(words, this.words)) {
      this.changed = true;
      this.lastChangeT = now;
      this.wordT = words.map((w, i) =>
        i < this.wordT.length && normWord(this.words[i]) === normWord(w) ? this.wordT[i] : now);
    }
    this.words = words;
    this.cursor = Math.max(pos, 0);
    if (ev.kind === "final") this.final = true;
    return this.current(now);
  }

  begin(uid, words, now) {
    let carry = false;
    if (this.uid && this.required.length) {
      const gap = now - this.lastEventT;
      const echoOk = (!this.final && gap <= this.echoWindowS) || gap <= this.dupWindowS;
      carry = echoOk && matchPrefix(words, this.required) !== null;
    }
    if (this.uid) {
      this.closed.push(this.uid);
      if (this.closed.length > 64) this.closed.shift();
    }
    const consumed = carry ? this.consumedCore : [];
    const req = carry ? this.required : [];
    this.carriedDropped = carry && this.dropped;
    this.uid = uid;
    this.gen = 0;
    this.words = [];
    this.wordT = [];
    this.cursor = 0;
    this.final = this.frozen = this.dropped = this.joined = false;
    this.consumedCore = consumed;
    this.required = req;
    this.carried = carry;
    this.lastChangeT = now;
  }

  tailStart() {
    return this.cursor + skipLeading(this.words.slice(this.cursor));
  }

  current(now) {
    if (!this.uid || this.frozen || this.dropped) return null;
    const start = this.tailStart();
    const tw = this.words.slice(start);
    if (!tw.length) return null;
    return {
      vid: this.vid, text: tw.join(" "), words: tw, cursor: start, isFinal: this.final,
      silentMs: Math.max(0, Math.round((now - this.lastChangeT) * 1000)), uid: this.uid, joined: this.joined,
    };
  }

  heardAt(index) {
    return index >= 0 && index < this.wordT.length ? this.wordT[index] : null;
  }

  matches(tail, nWords) {
    if (this.frozen || this.dropped || tail.vid !== this.vid || tail.cursor !== this.tailStart()) return false;
    const nowWords = this.words.slice(tail.cursor, tail.cursor + nWords);
    return eqArr(coreWords(nowWords), coreWords(tail.words.slice(0, nWords))) && nowWords.length === nWords;
  }

  consume(nWords) {
    const start = this.tailStart();
    const end = Math.min(this.words.length, start + Math.max(0, nWords));
    this.consumeTo(end);
  }

  skipToEnd() {
    this.consumeTo(this.words.length);
  }

  consumeTo(end) {
    this.consumedCore = coreWords(this.words.slice(0, end));
    this.required = required(this.consumedCore);
    this.carried = false;
    this.cursor = end;
    this.gen += 1;
    this.log("consume", { uid: this.uid, vid: this.vid, cursor: end, consumed: this.consumedText });
  }

  dropUtterance() {
    this.skipToEnd();
    this.dropped = true;
  }

  markFired(key) { this.fired.add(JSON.stringify(key)); }
  alreadyFired(key) { return this.fired.has(JSON.stringify(key)); }

  log(kind, fields) {
    if (this.logFn) this.logFn(kind, fields);
  }
}
