// Minimal byte-level BPE for the ModernBERT/ettin tokenizer.json (HF `tokenizers` semantics), as used by
// jev_local Packer: encode(text, add_special_tokens=False) with `encode_special_tokens = True`, i.e.
// - special added tokens ([CLS], [Q], ...) are NEVER matched in user text (so text cannot forge a marker);
// - non-special added tokens (|||IP_ADDRESS|||, [unusedN], runs of 2-24 spaces) are split out first,
//   leftmost-longest, and get their own ids;
// - the rest: NFC normalizer -> GPT-2 ByteLevel regex pre-tokenizer -> byte-to-unicode -> BPE merges.

const PRETOK_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const m = new Array(256);
  bs.forEach((b, i) => { m[b] = String.fromCodePoint(cs[i]); });
  return m;
}

export class Tokenizer {
  /** json: parsed tokenizer.json */
  constructor(json) {
    const model = json.model;
    if (model.type !== "BPE") throw new Error(`unsupported tokenizer model ${model.type}`);
    this.vocab = new Map(Object.entries(model.vocab));
    this.ranks = new Map();
    model.merges.forEach((m, i) => {
      const [a, b] = Array.isArray(m) ? m : m.split(" ");
      this.ranks.set(a + " " + b, i);
    });
    this.added = new Map(); // content -> id, for every added token
    this.matchable = new Map(); // non-special added tokens (matched in text)
    for (const t of json.added_tokens || []) {
      this.added.set(t.content, t.id);
      this.vocab.set(t.content, t.id);
      if (!t.special) this.matchable.set(t.content, t.id);
    }
    this.matchLens = [...new Set([...this.matchable.keys()].map((s) => s.length))].sort((a, b) => b - a);
    this.matchFirst = new Set([...this.matchable.keys()].map((s) => s[0]));
    this.normalize = json.normalizer && json.normalizer.type === "NFC" ? (s) => s.normalize("NFC") : (s) => s;
    this.byteMap = bytesToUnicode();
    this.encoder = new TextEncoder();
    this.cache = new Map();
  }

  tokenId(s) {
    return this.vocab.get(s);
  }

  /** Split around non-special added tokens (leftmost-longest). -> [{text}|{id}] */
  splitAdded(text) {
    const out = [];
    let start = 0;
    let i = 0;
    while (i < text.length) {
      let hit = null;
      if (this.matchFirst.has(text[i])) {
        for (const L of this.matchLens) {
          if (i + L > text.length) continue;
          const s = text.substr(i, L);
          const id = this.matchable.get(s);
          if (id !== undefined) { hit = [s, id]; break; }
        }
      }
      if (hit) {
        if (i > start) out.push({ text: text.slice(start, i) });
        out.push({ id: hit[1] });
        i += hit[0].length;
        start = i;
      } else {
        i++;
      }
    }
    if (start < text.length) out.push({ text: text.slice(start) });
    return out;
  }

  bpe(word) {
    const c = this.cache.get(word);
    if (c) return c;
    let parts = [...word];
    while (parts.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = parts[best];
      const b = parts[best + 1];
      const merged = [];
      for (let i = 0; i < parts.length; ) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i += 2; }
        else { merged.push(parts[i]); i += 1; }
      }
      parts = merged;
    }
    const ids = parts.map((p) => {
      const id = this.vocab.get(p);
      if (id === undefined) throw new Error(`BPE produced unknown token ${JSON.stringify(p)}`);
      return id;
    });
    if (this.cache.size > 50000) this.cache.clear();
    this.cache.set(word, ids);
    return ids;
  }

  /** Token ids of `text` with no special tokens added. */
  encode(text) {
    const ids = [];
    for (const seg of this.splitAdded(this.normalize(text))) {
      if (seg.id !== undefined) { ids.push(seg.id); continue; }
      for (const m of seg.text.matchAll(PRETOK_RE)) {
        const bytes = this.encoder.encode(m[0]);
        let w = "";
        for (const b of bytes) w += this.byteMap[b];
        for (const id of this.bpe(w)) ids.push(id);
      }
    }
    return ids;
  }
}
