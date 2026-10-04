// Pack (state, questions) into one encoder sequence (port of tokenize_pack.Packer) and build the ONNX feeds.
//
//   [CLS] seg_0 [SEP] seg_1 [SEP] ... | [Q] header_0 | [O] item_00 | [O] item_01 | ... | [Q] header_1 | ...
//
// q_group: -1 for state tokens, else the question ordinal. i_group: -1 for state and header tokens, else the
// item ordinal. Positions restart per branch: header at S.., each item at S+len(header)... The ONNX graph builds
// the block attention mask from these (a token sees the state, plus its own header and its own item).

import { questionBlock, stateSegments } from "./serialize.js";

export const MARKERS = ["[Q]", "[O]", "[L]", "[T]", "[F]"];
export const STATE = -1;

export class Packer {
  constructor(tokenizer, { maxPositions = 1536, maxTotal = 8192, cacheSize = 8192 } = {}) {
    this.tok = tokenizer;
    this.maxPositions = maxPositions;
    this.maxTotal = maxTotal;
    this.cacheSize = cacheSize;
    this.cache = new Map();
    this.marker = {};
    for (const m of MARKERS) {
      const id = tokenizer.tokenId(m);
      if (id === undefined) throw new Error(`tokenizer lacks marker token ${m}`);
      this.marker[m] = id;
    }
    this.clsId = tokenizer.tokenId("[CLS]");
    this.sepId = tokenizer.tokenId("[SEP]");
  }

  encode(text) {
    let ids = this.cache.get(text);
    if (!ids) {
      ids = this.tok.encode(text);
      if (this.cache.size >= this.cacheSize) this.cache.clear();
      this.cache.set(text, ids);
    }
    return ids;
  }

  static segmentText(s) {
    return s.key ? `${s.key}: ${s.text}` : s.text;
  }

  stateIds(segments) {
    const ids = [this.clsId];
    for (const s of segments) {
      ids.push(...this.encode(Packer.segmentText(s)));
      ids.push(this.sepId);
    }
    return ids;
  }

  itemMarkers(b) {
    if (b.kind === "choice") return b.items.map(() => this.marker["[O]"]);
    if (b.kind === "score") return b.items.map(() => this.marker["[L]"]);
    if (b.kind === "noul") return [this.marker["[T]"], this.marker["[F]"]];
    throw new Error(`unsupported block kind ${b.kind}`);
  }

  blockParts(blocks) {
    return blocks.map((b) => {
      const header = [this.marker["[Q]"], ...this.encode(b.header)];
      const marks = this.itemMarkers(b);
      const items = b.items.map((it, j) => [marks[j], ...this.encode(it)]);
      return { header, items };
    });
  }

  /** -> {inputIds, positionIds, qGroup, iGroup, nState, qIndex: Map qid -> {kind, header, labels, qPos, itemPos}} */
  layout(state, blocks, parts) {
    const s = state.length;
    const ids = [...state];
    const pos = Array.from({ length: s }, (_, i) => i);
    const qg = new Array(s).fill(STATE);
    const ig = new Array(s).fill(STATE);
    const qIndex = new Map();
    blocks.forEach((b, qn) => {
      const { header, items } = parts[qn];
      const qPos = ids.length;
      header.forEach((t, j) => { ids.push(t); pos.push(s + j); qg.push(qn); ig.push(STATE); });
      const base = s + header.length;
      const itemPos = [];
      items.forEach((item, j) => {
        itemPos.push(ids.length);
        item.forEach((t, k) => { ids.push(t); pos.push(base + k); qg.push(qn); ig.push(j); });
      });
      qIndex.set(b.qid, { kind: b.kind, header: b.header, labels: b.labels, qPos, itemPos });
    });
    return { inputIds: ids, positionIds: pos, qGroup: qg, iGroup: ig, nState: s, qIndex };
  }

  /** Full pipeline from a wire request. Throws {detail:'max_tokens_exceeded'} when a branch is too long. */
  pack(state, questions) {
    const segs = stateSegments(state);
    const blocks = Object.entries(questions).map(([qid, q]) => questionBlock(qid, q));
    const st = this.stateIds(segs);
    const parts = this.blockParts(blocks);
    let need = 0;
    let total = st.length;
    for (const p of parts) {
      const longest = Math.max(0, ...p.items.map((x) => x.length));
      need = Math.max(need, p.header.length + longest);
      total += p.header.length + p.items.reduce((a, x) => a + x.length, 0);
    }
    if (st.length + need > this.maxPositions || total > this.maxTotal) {
      const err = new Error("max_tokens_exceeded");
      err.detail = { detail: "max_tokens_exceeded", tokens: st.length + need, total, max_tokens: this.maxPositions };
      throw err;
    }
    return { packed: this.layout(st, blocks, parts), blocks };
  }
}

/** ONNX feeds for one packed sequence (mirrors scripts/genclass_export.py plan_inputs). Plain arrays. */
export function planInputs(p) {
  const by = { choice: [], score: [], noul: [] };
  for (const qi of p.qIndex.values()) by[qi.kind].push(qi);
  const grp = (lst) => {
    if (!lst.length) return { q: [0], items: [[0]], k: 1 };
    const k = Math.max(...lst.map((x) => x.itemPos.length));
    return { q: lst.map((x) => x.qPos), items: lst.map((x) => [...x.itemPos, ...new Array(k - x.itemPos.length).fill(0)]), k };
  };
  const nl = by.noul;
  return {
    choice: grp(by.choice),
    score: grp(by.score),
    noul: { q: nl.length ? nl.map((x) => x.qPos) : [0], t: nl.length ? nl.map((x) => x.itemPos[0]) : [0], f: nl.length ? nl.map((x) => x.itemPos[1]) : [0] },
  };
}

/** Graph outputs (flat Float32Arrays + dims) -> raw logits per qid, in request order. */
export function unpackLogits(p, out) {
  const idx = { choice: 0, score: 0, noul: 0 };
  const res = new Map();
  for (const [qid, qi] of p.qIndex) {
    const g = idx[qi.kind]++;
    if (qi.kind === "noul") res.set(qid, [out.noul.data[g]]);
    else {
      const t = qi.kind === "choice" ? out.choice : out.score;
      const K = t.dims[1];
      res.set(qid, Array.from(t.data.subarray(g * K, g * K + qi.labels.length)));
    }
  }
  return res;
}
