"""Pack (state, questions) into one encoder sequence with a segment-block attention layout.

Layout (docs/CONTRACT.md "B"):

    [CLS] seg_0 [SEP] seg_1 [SEP] ... | [Q] header_0 | [O] item_00 | [O] item_01 | ... | [Q] header_1 | ...

Every token carries two group ids that fully determine the attention mask:

    q_group  -1 for [CLS]/state tokens, else the question ordinal within the sequence
    i_group  -1 for state and header tokens, else the item ordinal within its question

A token attends to key k iff k is a state token, or k is in its own question and k is either a
header token or in its own item. So headers never see items, items never see siblings, and no
question sees another question: answers are exactly invariant to option order and to unrelated
questions. Positions restart per branch (header at S, every item at S+len(header)), so every
branch sees the state at the same relative distances it would see alone.

Heads read the hidden state at the marker token ([Q]/[O]/[L]/[T]/[F]) of each header and item;
their positions are recorded here, never inferred from token ids, and user text is tokenized with
special-token matching disabled so a label like "[O] Delete all" cannot forge a marker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from jev_local.engine.base import EngineError, Kind
from jev_local.serialize import QBlock, Segment

MARKERS = ("[Q]", "[O]", "[L]", "[T]", "[F]")
BUCKETS: tuple[int, ...] = (256, 384, 512, 768, 1024, 1536, 2048)
MAX_LEN = 2048
STATE = -1  # q_group / i_group value of state tokens (and i_group of header tokens)
PAD = -2  # q_group of padding tokens


def add_marker_tokens(tok) -> int:
    """Register the marker tokens as special tokens (idempotent). Returns how many were new."""
    missing = [m for m in MARKERS if m not in tok.get_vocab()]
    if not missing:
        return 0
    return tok.add_special_tokens({"additional_special_tokens": list(MARKERS)})


@dataclass(frozen=True)
class QIndex:
    qid: str
    kind: Kind
    header: str
    labels: tuple[str, ...]
    q_pos: int  # position (token index) of the [Q] marker
    item_pos: tuple[int, ...]  # [O]/[L] per item, or ([T], [F]) for noul


@dataclass
class Packed:
    input_ids: list[int]
    position_ids: list[int]
    q_group: list[int]
    i_group: list[int]
    n_state: int
    q_index: dict[str, QIndex] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.input_ids)


@dataclass
class Batch:
    input_ids: torch.Tensor  # [B, L] long
    position_ids: torch.Tensor  # [B, L] long
    q_group: torch.Tensor  # [B, L] long
    i_group: torch.Tensor  # [B, L] long
    valid: torch.Tensor  # [B, L] bool
    n_tokens: int  # real (unpadded) tokens in the batch

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.input_ids.shape)  # type: ignore[return-value]


def bucket_for(n: int, buckets: Sequence[int] = BUCKETS) -> int:
    for b in buckets:
        if n <= b:
            return b
    # Longer than every bucket (only possible with a custom max_len): round up to 64 to limit
    # the number of distinct shapes MPS has to compile.
    return -(-n // 64) * 64


class Packer:
    """Tokenizes serialized segments/blocks and lays them out. Keeps a small token cache because the
    harness re-sends the same option texts (intents, keys, folders, headers) on every partial."""

    def __init__(self, tok, max_len: int = MAX_LEN, buckets: Sequence[int] = BUCKETS, cache_size: int = 8192):
        self.tok = tok
        self.max_len = max_len
        self.buckets = tuple(buckets)
        self.cache_size = cache_size
        self._cache: dict[str, tuple[int, ...]] = {}
        vocab = tok.get_vocab()
        missing = [m for m in MARKERS if m not in vocab]
        if missing:
            raise ValueError(f"tokenizer lacks marker tokens {missing}; call add_marker_tokens first")
        self.marker = {m: vocab[m] for m in MARKERS}
        self.cls_id = tok.cls_token_id
        self.sep_id = tok.sep_token_id
        self.pad_id = tok.pad_token_id
        self._backend = tok.backend_tokenizer
        # User text must never produce marker/[SEP]/[CLS] ids: tokenize special-token strings as text.
        self._backend.encode_special_tokens = True

    # ------------------------------------------------------------------ tokenization

    def encode(self, texts: Sequence[str]) -> list[tuple[int, ...]]:
        misses = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if misses:
            if len(self._cache) + len(misses) > self.cache_size:
                self._cache.clear()
                misses = list(dict.fromkeys(texts))  # this call's cache hits were just evicted too
            for t, enc in zip(misses, self._backend.encode_batch(misses, add_special_tokens=False)):
                self._cache[t] = tuple(enc.ids)
        return [self._cache[t] for t in texts]

    # ------------------------------------------------------------------ layout

    @staticmethod
    def segment_text(s: Segment) -> str:
        return f"{s.key}: {s.text}" if s.key else s.text

    def _state_ids(self, segments: Sequence[Segment]) -> list[int]:
        ids = [self.cls_id]
        for toks in self.encode([self.segment_text(s) for s in segments]):
            ids.extend(toks)
            ids.append(self.sep_id)
        return ids

    def _item_markers(self, b: QBlock) -> list[int]:
        if b.kind == "choice":
            return [self.marker["[O]"]] * len(b.items)
        if b.kind == "score":
            return [self.marker["[L]"]] * len(b.items)
        if b.kind == "noul":
            return [self.marker["[T]"], self.marker["[F]"]]
        raise ValueError(f"unsupported block kind {b.kind!r}")

    def _block_parts(self, blocks: Sequence[QBlock]) -> list[tuple[list[int], list[list[int]]]]:
        texts: list[str] = []
        for b in blocks:
            texts.append(b.header)
            texts.extend(b.items)
        enc = iter(self.encode(texts))
        parts = []
        for b in blocks:
            header = [self.marker["[Q]"], *next(enc)]
            items = [[m, *next(enc)] for m in self._item_markers(b)]
            parts.append((header, items))
        return parts

    @staticmethod
    def _parts_len(part: tuple[list[int], list[list[int]]]) -> int:
        header, items = part
        return len(header) + sum(len(x) for x in items)

    def _layout(
        self, state: list[int], blocks: Sequence[QBlock], parts: Sequence[tuple[list[int], list[list[int]]]]
    ) -> Packed:
        s = len(state)
        ids = list(state)
        pos = list(range(s))
        qg = [STATE] * s
        ig = [STATE] * s
        qidx: dict[str, QIndex] = {}
        for qn, (b, (header, items)) in enumerate(zip(blocks, parts)):
            q_pos = len(ids)
            ids += header
            pos += range(s, s + len(header))
            qg += [qn] * len(header)
            ig += [STATE] * len(header)
            base = s + len(header)
            item_pos = []
            for j, item in enumerate(items):
                item_pos.append(len(ids))
                ids += item
                pos += range(base, base + len(item))
                qg += [qn] * len(item)
                ig += [j] * len(item)
            qidx[b.qid] = QIndex(b.qid, b.kind, b.header, b.labels, q_pos, tuple(item_pos))
        return Packed(ids, pos, qg, ig, s, qidx)

    def count(self, segments: Sequence[Segment], blocks: Sequence[QBlock]) -> int:
        return len(self._state_ids(segments)) + sum(self._parts_len(p) for p in self._block_parts(blocks))

    def pack(self, segments: Sequence[Segment], blocks: Sequence[QBlock]) -> Packed:
        """One sequence holding every question; EngineError 400 beyond max_len."""
        state = self._state_ids(segments)
        parts = self._block_parts(blocks)
        n = len(state) + sum(self._parts_len(p) for p in parts)
        if n > self.max_len:
            raise EngineError(
                {"detail": "max_tokens_exceeded", "tokens": n, "max_tokens": self.max_len}, status=400
            )
        return self._layout(state, blocks, parts)

    def pack_split(
        self, segments: Sequence[Segment], blocks: Sequence[QBlock], max_len: int | None = None
    ) -> tuple[list[Packed], list[str]]:
        """Training helper: questions are isolated, so an over-long example can be split into several
        sequences that each repeat the state and carry a subset of the questions, with identical
        per-question results. Returns (packs, qids dropped because state + that block alone is too long)."""
        max_len = max_len or self.max_len
        state = self._state_ids(segments)
        parts = self._block_parts(blocks)
        packs: list[Packed] = []
        dropped: list[str] = []
        cur_b: list[QBlock] = []
        cur_p: list[tuple[list[int], list[list[int]]]] = []
        used = len(state)
        for b, p in zip(blocks, parts):
            n = self._parts_len(p)
            if len(state) + n > max_len:
                dropped.append(b.qid)
                continue
            if used + n > max_len and cur_b:
                packs.append(self._layout(state, cur_b, cur_p))
                cur_b, cur_p, used = [], [], len(state)
            cur_b.append(b)
            cur_p.append(p)
            used += n
        if cur_b:
            packs.append(self._layout(state, cur_b, cur_p))
        return packs, dropped

    # ------------------------------------------------------------------ batching

    def collate(self, packs: Sequence[Packed], device: torch.device | str = "cpu", pad_to: int | None = None) -> Batch:
        n_max = max(p.length for p in packs)
        L = pad_to if pad_to is not None else bucket_for(n_max, self.buckets)
        if L < n_max:
            raise ValueError(f"pad_to={L} < longest sequence {n_max}")
        B = len(packs)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        pos = torch.zeros((B, L), dtype=torch.long)
        qg = torch.full((B, L), PAD, dtype=torch.long)
        ig = torch.full((B, L), STATE, dtype=torch.long)
        for r, p in enumerate(packs):
            n = p.length
            ids[r, :n] = torch.tensor(p.input_ids)
            pos[r, :n] = torch.tensor(p.position_ids)
            qg[r, :n] = torch.tensor(p.q_group)
            ig[r, :n] = torch.tensor(p.i_group)
        valid = qg != PAD
        dev = torch.device(device)
        return Batch(
            ids.to(dev, non_blocking=True),
            pos.to(dev, non_blocking=True),
            qg.to(dev, non_blocking=True),
            ig.to(dev, non_blocking=True),
            valid.to(dev, non_blocking=True),
            sum(p.length for p in packs),
        )


def build_masks(batch: Batch, window: int) -> dict[str, torch.Tensor]:
    """Boolean [B,1,L,L] masks (True = may attend) for global and local layers.

    Padding queries attend only to themselves so their rows stay finite: a NaN there would leak
    into real rows through 0 * NaN in the attention-weighted sum."""
    qg, ig, pos, valid = batch.q_group, batch.i_group, batch.position_ids, batch.valid
    q_q, q_k = qg[:, :, None], qg[:, None, :]
    i_q, i_k = ig[:, :, None], ig[:, None, :]
    key_state = (q_k == STATE) & valid[:, None, :]
    same_q = (q_q == q_k) & (q_q >= 0)
    same_branch = same_q & ((i_k == STATE) | (i_k == i_q))
    full = key_state | same_branch
    eye = torch.eye(qg.shape[1], dtype=torch.bool, device=qg.device)[None]
    full = full | (eye & ~valid[:, :, None])
    local = full & ((pos[:, :, None] - pos[:, None, :]).abs() <= window)
    return {"full_attention": full[:, None], "sliding_attention": local[:, None]}


# ---------------------------------------------------------------------------- tree layout
#
# The dense [B,1,L,L] mask above is the reference semantics, but most of it is False: every
# question token only needs the state plus its own branch. The tree layout computes exactly the same
# attention with three small attention problems per layer:
#
#   trunk  [B,  Sp]  state tokens            keys: own row's state
#   head   [NH, Hp]  one question header     keys: its row's state + itself
#   item   [NI, C ]  whole items of ONE question, packed into chunks of <= C tokens
#                                           keys: its row's state + its header + same-item tokens
#
# Linear layers run on the flat, unpadded token list; only attention uses these padded layouts.
# Cost per layer drops from L^2 to about L * (S + H + C), and no L x L tensor is ever built.

ITEM_CHUNK = 64


def _round_up(n: int, m: int) -> int:
    return max(m, -(-n // m) * m)


@dataclass
class TreeBatch:
    input_ids: torch.Tensor  # [N] flat tokens of every row, padded at the end
    position_ids: torch.Tensor  # [N]
    trunk_idx: torch.Tensor  # [B, Sp] flat index per slot; N = padding (a zero row)
    head_idx: torch.Tensor  # [NH, Hp]
    head_row: torch.Tensor  # [NH] trunk row of each header
    item_idx: torch.Tensor  # [NI, C]
    item_row: torch.Tensor  # [NI]
    item_head: torch.Tensor  # [NI] header chunk (index into head_idx) of each item chunk
    inv: torch.Tensor  # [N] slot of each flat token in cat(trunk, head, item) outputs
    masks: dict[str, dict[str, torch.Tensor]]  # layout -> layer type -> bool [n, 1, q, k]
    row_offsets: list[int]  # flat offset of each row (heads index = offset + position in the row)
    n_tokens: int  # real tokens

    @property
    def shape_key(self) -> tuple[int, ...]:
        return (self.input_ids.shape[0], *self.trunk_idx.shape, *self.head_idx.shape, *self.item_idx.shape)


def _branches(p: Packed) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """-> per question: (header start, header end, [(item start, item end), ...]) in row positions."""
    qs = sorted(p.q_index.values(), key=lambda q: q.q_pos)
    out = []
    for n, qi in enumerate(qs):
        end = qs[n + 1].q_pos if n + 1 < len(qs) else p.length
        bounds = list(qi.item_pos) + [end]
        items = [(bounds[j], bounds[j + 1]) for j in range(len(qi.item_pos))]
        out.append((qi.q_pos, qi.item_pos[0] if qi.item_pos else end, items))
    return out


def collate_tree(
    packs: Sequence[Packed],
    pad_id: int,
    window: int,
    device: torch.device | str = "cpu",
    buckets: Sequence[int] = BUCKETS,
    chunk: int = ITEM_CHUNK,
) -> TreeBatch:
    import numpy as np

    B = len(packs)
    lens = [p.length for p in packs]
    offsets = [0]
    for n in lens[:-1]:
        offsets.append(offsets[-1] + n)
    n_real = sum(lens)
    # Flat length: contract buckets while they fit, else multiples of 256 (training batches).
    N = bucket_for(n_real, buckets) if n_real <= buckets[-1] else _round_up(n_real, 256)
    ids = np.full(N, pad_id, dtype=np.int64)
    pos = np.zeros(N, dtype=np.int64)
    for p, o in zip(packs, offsets):
        ids[o : o + p.length] = p.input_ids
        pos[o : o + p.length] = p.position_ids

    Sp = _round_up(max(p.n_state for p in packs), 32)
    trunk = np.full((B, Sp), N, dtype=np.int64)
    for r, (p, o) in enumerate(zip(packs, offsets)):
        trunk[r, : p.n_state] = np.arange(o, o + p.n_state)

    heads: list[tuple[int, int, int]] = []  # (row, flat start, flat end)
    chunks: list[tuple[int, int, list[tuple[int, int]]]] = []  # (row, head index, [(flat start, flat end)])
    for r, (p, o) in enumerate(zip(packs, offsets)):
        for hs, he, items in _branches(p):
            hi = len(heads)
            heads.append((r, o + hs, o + he))
            cur: list[tuple[int, int]] = []
            used = 0
            for s, e in items:
                if cur and used + (e - s) > chunk:
                    chunks.append((r, hi, cur))
                    cur, used = [], 0
                cur.append((o + s, o + e))
                used += e - s
            if cur:
                chunks.append((r, hi, cur))
    NH = _round_up(max(len(heads), 1), 4)
    Hp = _round_up(max((e - s for _, s, e in heads), default=1), 16)
    NI = _round_up(max(len(chunks), 1), 8)
    C = max(chunk, _round_up(max((sum(e - s for s, e in c) for _, _, c in chunks), default=1), 32))

    head_idx = np.full((NH, Hp), N, dtype=np.int64)
    head_row = np.zeros(NH, dtype=np.int64)
    for h, (r, s, e) in enumerate(heads):
        head_idx[h, : e - s] = np.arange(s, e)
        head_row[h] = r
    item_idx = np.full((NI, C), N, dtype=np.int64)
    item_iid = np.full((NI, C), -1, dtype=np.int64)  # item id within the chunk (sibling isolation)
    item_row = np.zeros(NI, dtype=np.int64)
    item_head = np.zeros(NI, dtype=np.int64)
    for c, (r, hi, spans) in enumerate(chunks):
        k = 0
        for j, (s, e) in enumerate(spans):
            item_idx[c, k : k + e - s] = np.arange(s, e)
            item_iid[c, k : k + e - s] = j
            k += e - s
        item_row[c], item_head[c] = r, hi

    inv = np.zeros(N, dtype=np.int64)
    base_h, base_i = B * Sp, B * Sp + NH * Hp
    for arr, base in ((trunk, 0), (head_idx, base_h), (item_idx, base_i)):
        flat = arr.reshape(-1)
        real = np.nonzero(flat < N)[0]
        inv[flat[real]] = base + real

    dev = torch.device(device)
    t = lambda a: torch.from_numpy(a).to(dev, non_blocking=True)  # noqa: E731
    tb = TreeBatch(
        t(ids), t(pos), t(trunk), t(head_idx), t(head_row), t(item_idx), t(item_row), t(item_head), t(inv),
        {}, offsets, n_real,
    )
    tb.masks = _tree_masks(tb, t(item_iid), window)
    return tb


def _only_first_key(m: torch.Tensor, q_valid: torch.Tensor) -> torch.Tensor:
    """Padding query rows attend only to key 0 (the row's [CLS]): no all-False rows, so softmax never
    produces NaN, whose gradient would poison real rows through 0 * NaN."""
    first = torch.zeros_like(m)
    first[..., 0] = True
    return torch.where(q_valid[..., None], m, first)


def _tree_masks(tb: TreeBatch, item_iid: torch.Tensor, window: int) -> dict[str, dict[str, torch.Tensor]]:
    N = tb.input_ids.shape[0]
    pos_pad = torch.cat([tb.position_ids, tb.position_ids.new_zeros(1)])

    def slot_pos(idx: torch.Tensor) -> torch.Tensor:
        return pos_pad[idx]

    t_valid, h_valid, i_valid = tb.trunk_idx < N, tb.head_idx < N, tb.item_idx < N
    t_pos, h_pos, i_pos = slot_pos(tb.trunk_idx), slot_pos(tb.head_idx), slot_pos(tb.item_idx)

    def finish(m: torch.Tensor, q_valid: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor) -> dict[str, torch.Tensor]:
        local = m & ((q_pos[:, :, None] - k_pos[:, None, :]).abs() <= window)
        return {
            "full_attention": _only_first_key(m, q_valid)[:, None],
            "sliding_attention": _only_first_key(local, q_valid)[:, None],
        }

    out: dict[str, dict[str, torch.Tensor]] = {}
    m = t_valid[:, None, :] & t_valid[:, :, None]
    out["trunk"] = finish(m, t_valid, t_pos, t_pos)

    kv = torch.cat([t_valid[tb.head_row], h_valid], dim=1)
    kp = torch.cat([t_pos[tb.head_row], h_pos], dim=1)
    m = h_valid[:, :, None] & kv[:, None, :]
    out["head"] = finish(m, h_valid, h_pos, kp)

    C = tb.item_idx.shape[1]
    trunk_part = t_valid[tb.item_row][:, None, :].expand(-1, C, -1)
    head_part = h_valid[tb.item_head][:, None, :].expand(-1, C, -1)
    own = (item_iid[:, :, None] == item_iid[:, None, :]) & i_valid[:, None, :]
    m = torch.cat([trunk_part, head_part, own], dim=2) & i_valid[:, :, None]
    kp = torch.cat([t_pos[tb.item_row], h_pos[tb.item_head], i_pos], dim=1)
    out["item"] = finish(m, i_valid, i_pos, kp)
    return out
