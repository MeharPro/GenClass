"""Streaming mixture reader for the v2 training data (PLAN §2.1 mixing rules, §4.1 E3).

    corpus = Corpus.open(["data/v2"], cache_dir="runs/stream_cache")        # index once, reuse
    mix = Mixture(corpus, MixConfig(seed=0, batch_tokens=16384, ...))
    for pass_k, cursor, rows in mix.iterate(rank=r, world=W):                 # rows: list[dict] examples
        ...

Inputs
- `<root>/raw/<bucket>/<source>.jsonl.zst` (+ `<source>.stats.json`) written by the data agents, or any
  tree of `*.jsonl.zst` / `*.jsonl` shards (phase-2 assembled shards). Rows are v1 CONTRACT "D" examples
  plus the v2 provenance fields of `jev_local.data.v2.render.make_example` (`bucket`, `source`,
  `license_use`, `split`, `variant`, `group_id`); missing fields fall back to the directory/file names.
- Each `.zst` file is decompressed ONCE into the cache dir and indexed by byte offset, so rows are read
  with one `pread` each (random access: exact shuffles, exact caps, O(1) resume). Plain `.jsonl` files are
  indexed in place. The index stores per row: offset, bytes, group, split, licence, #supervised questions,
  class key and an estimated token length; nothing else is held in RAM.

Mixing (one "pass" = one draw of the mixture; pass k uses seeds derived from (seed, k))
1. Rows are filtered by split (default `train`), licence (`--license commercial` = D5 clean variant) and bucket.
2. Group = (bucket, source, subsource); subsource = meta.cap_key | meta.subsource | meta.task | "" so a file
   that bundles many original datasets (tasksource-jev) can still be capped and sqrt-sampled per dataset.
3. Per group, rows are permuted and capped: at most `class_cap` rows per (single-question) class and at
   most `source_cap` supervised decisions per group. A fresh permutation per pass means a capped group shows
   a different subset each pass.
4. Bucket token budget = bucket weight x pass tokens (weights are token shares, renormalised over the buckets
   present; default pass tokens = total capped tokens, i.e. one capped epoch). Within a bucket, groups get
   tokens proportional to n_decisions^alpha (alpha = 0.5: sqrt(n) sampling); a group is repeated at most
   `max_repeat` times per pass, and the excess is water-filled to the other groups of its bucket.
5. All rows of the pass are globally shuffled, then cut into micro-batches: pools of `pool` rows are sorted
   by length and cut by a token budget (`batch_tokens`, padded estimate rows x longest) or a row count
   (`batch_rows`); the micro-batches of a pool are shuffled. Optional length curriculum: before progress p
   only rows up to `cap(p)` tokens are batched; longer rows are deferred to later pools (never dropped).

Distributed: every rank builds the same micro-batch order (index-only arithmetic, no I/O); rank r of W takes
global micro-batches c = base + r for base = cursor, cursor + W, ... so the cursor is independent of the world
size and a run can resume on a different number of nodes. Token-budget batches keep ranks balanced.

This module is torch-free (numpy only) so it can be tested anywhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

INDEX_VERSION = 2  # v2: per-row layout sizes (state / header / option bytes, #questions, #options)
SPLITS = ("train", "dev_mix", "dev_family", "dev", "test")
LICENSES = ("commercial", "research", "unknown")
DEFAULT_TOKENS_PER_BYTE = 0.27  # packed ettin tokens per JSON byte; re-estimated per file when a packer is given

# PLAN §2.1 token shares. s0_families (programmatic S0) stands in for b6 until synthetic data exists.
DEFAULT_BUCKET_WEIGHTS: dict[str, float] = {
    "b1_tasksource_jev": 0.36,
    "b2_label_semantics": 0.10,
    "b3_nli": 0.05,
    "b4_procedural": 0.08,
    "b5_open_jev": 0.04,
    "b6_synthetic": 0.14,
    "b7_laurer": 0.06,
    "b8_extractive": 0.04,
    "b9_cu": 0.12,
    "s0_families": 0.14,
}
OTHER_BUCKET_WEIGHT = 0.04  # buckets not in the table (e.g. v1 data/cu, data/gen dirs)

# ---------------------------------------------------------------------------- zstd


def _open_zst(path: Path):
    """Binary stream of the decompressed file: python-zstandard if installed, else the zstd CLI."""
    try:
        import zstandard  # type: ignore

        fh = path.open("rb")
        return zstandard.ZstdDecompressor().stream_reader(fh, read_size=1 << 20), fh
    except ImportError:
        if not shutil.which("zstd"):
            raise RuntimeError(f"cannot read {path}: install `zstandard` or the zstd CLI") from None
        proc = subprocess.Popen(["zstd", "-dcq", str(path)], stdout=subprocess.PIPE)
        return proc.stdout, proc


def _lines(stream, chunk: int = 1 << 20) -> Iterator[bytes]:
    """Exact lines (with their trailing newline) of a binary stream that only needs .read()."""
    buf = b""
    while True:
        data = stream.read(chunk)
        if not data:
            break
        buf += data
        start = 0
        while True:
            j = buf.find(b"\n", start)
            if j < 0:
                break
            yield buf[start : j + 1]
            start = j + 1
        buf = buf[start:]
    if buf:
        yield buf


def _close(h) -> None:
    if isinstance(h, subprocess.Popen):
        if h.wait() != 0:
            raise RuntimeError(f"zstd exited with {h.returncode}")
    else:
        h.close()


# ---------------------------------------------------------------------------- per-file index


@dataclass
class FileIndex:
    src: str  # original file
    data: str  # file the offsets point into (decompressed cache or the .jsonl itself)
    offset: np.ndarray  # int64
    nbytes: np.ndarray  # int32
    group: np.ndarray  # int32 into groups
    split: np.ndarray  # int8 into SPLITS (+ len(SPLITS) = other)
    license: np.ndarray  # int8 into LICENSES
    n_dec: np.ndarray  # int16 supervised questions
    cls: np.ndarray  # int64 class key hash (0 = none)
    st: np.ndarray  # int32 state JSON bytes
    hd: np.ndarray  # int32 longest question header (instructions) bytes
    it: np.ndarray  # int32 longest option / level / true-false text bytes
    nq: np.ndarray  # int16 questions
    no: np.ndarray  # int32 options over all questions (noul = 2)
    groups: list[tuple[str, str, str]]  # (bucket, source, subsource)
    tokens_per_byte: float = DEFAULT_TOKENS_PER_BYTE

    def __len__(self) -> int:
        return len(self.offset)


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True) or 1


def _class_key(obj: dict) -> int:
    """Class of a single-question row (hard label), for per-class caps; 0 when undefined."""
    labels = obj.get("labels") or {}
    if len(obj.get("questions") or {}) != 1 or len(labels) != 1:
        return 0
    (qid, lab), = labels.items()
    if not isinstance(lab, dict):
        return 0
    for k in ("label", "level"):
        if k in lab:
            return _h64(f"{qid}\x1f{lab[k]}")
    if "p" in lab:
        return _h64(f"{qid}\x1fp{round(float(lab['p']))}")
    return 0


def _layout_sizes(obj: dict) -> tuple[int, int, int, int, int]:
    """(state bytes, longest header bytes, longest option bytes, #questions, #options): what the tree layout
    pads to (trunk = longest state, header rows = longest header, option chunks = max(64, longest option))."""
    st = len(json.dumps(obj.get("state"), ensure_ascii=False).encode())
    hd = it = no = 0
    qs = obj.get("questions") or {}
    for q in qs.values():
        if not isinstance(q, dict):
            continue
        hd = max(hd, len(str(q.get("instructions") or "").encode()))
        crit = q.get("criteria")
        if isinstance(crit, dict):
            items = [f"{k}: {v}" if v not in (None, "") else str(k) for k, v in crit.items()]
        elif isinstance(crit, list):
            items = [json.dumps(c, ensure_ascii=False) if not isinstance(c, str) else c for c in crit]
        else:
            items = []
        if q.get("type") == "noul":
            items = [json.dumps(c, ensure_ascii=False) if not isinstance(c, str) else c
                     for c in ((crit or {}).values() if isinstance(crit, dict) else [])] or ["yes", "no"]
            no += 2
        else:
            no += len(items)
        if items:
            it = max(it, max(len(x.encode()) for x in items))
    return min(st, 2**31 - 1), hd, it, min(len(qs), 32767), no


def _row_fields(obj: dict, default_bucket: str, default_source: str) -> tuple[tuple[str, str, str], int, int, int, int]:
    meta = obj.get("meta") or {}
    sub = meta.get("cap_key") or meta.get("subsource") or meta.get("task") or ""
    g = (str(obj.get("bucket") or default_bucket), str(obj.get("source") or default_source), str(sub))
    sp = obj.get("split", "train")
    split = SPLITS.index(sp) if sp in SPLITS else len(SPLITS)
    lic = obj.get("license_use", "unknown")
    license = LICENSES.index(lic) if lic in LICENSES else LICENSES.index("unknown")
    n_dec = max(0, min(len(obj.get("labels") or {}), 32767))
    return g, split, license, n_dec, _class_key(obj)


def _names(src: Path) -> tuple[str, str]:
    name = src.name
    for suf in (".jsonl.zst", ".jsonl"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return src.parent.name, name  # (bucket dir, source)


def _manifest_key(src: Path) -> dict:
    st = src.stat()
    return {"src": str(src.resolve()), "size": st.st_size, "mtime": int(st.st_mtime), "version": INDEX_VERSION}


def _cache_paths(src: Path, cache_dir: Path) -> tuple[Path, Path, Path]:
    h = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:10]
    stem = f"{src.parent.name}__{_names(src)[1]}__{h}"
    return cache_dir / f"{stem}.jsonl", cache_dir / f"{stem}.idx.npz", cache_dir / f"{stem}.json"


def build_file_index(src: str | Path, cache_dir: str | Path) -> FileIndex:
    """Decompress (if needed) and index one file; reuses a valid cached index."""
    src, cache_dir = Path(src), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_p, idx_p, man_p = _cache_paths(src, cache_dir)
    key = _manifest_key(src)
    if man_p.is_file() and idx_p.is_file():
        try:
            man = json.loads(man_p.read_text())
            if man.get("key") == key:
                return _load_index(man, idx_p)
        except (OSError, ValueError, KeyError):
            pass
    compressed = src.name.endswith(".zst")
    data_path = data_p if compressed else src.resolve()
    d_bucket, d_source = _names(src)
    groups: dict[tuple[str, str, str], int] = {}
    cols: dict[str, list] = {k: [] for k in ("offset", "nbytes", "group", "split", "license", "n_dec", "cls",
                                             "st", "hd", "it", "nq", "no")}
    tmp = data_p.with_name(data_p.name + f".tmp-{os.getpid()}")

    def consume(lines: Iterable[bytes], out) -> None:
        off = 0
        for line in lines:
            n = len(line)
            if out is not None:
                out.write(line)
            if line.strip():
                try:
                    obj = json.loads(line)
                except ValueError:
                    off += n
                    continue
                g, split, lic, n_dec, cls = _row_fields(obj, d_bucket, d_source)
                cols["offset"].append(off)
                cols["nbytes"].append(n)
                cols["group"].append(groups.setdefault(g, len(groups)))
                cols["split"].append(split)
                cols["license"].append(lic)
                cols["n_dec"].append(n_dec)
                cols["cls"].append(cls)
                for k, v in zip(("st", "hd", "it", "nq", "no"), _layout_sizes(obj)):
                    cols[k].append(v)
            off += n

    if compressed:
        stream, handle = _open_zst(src)
        with tmp.open("wb") as out:
            consume(_lines(stream), out)
        _close(handle)
        os.replace(tmp, data_p)
    else:
        with src.open("rb") as f:
            consume(f, None)
    tpb = DEFAULT_TOKENS_PER_BYTE
    stats_p = src.with_name(_names(src)[1] + ".stats.json")
    total_bytes = sum(cols["nbytes"])
    if stats_p.is_file() and total_bytes:
        try:
            te = float(json.loads(stats_p.read_text()).get("tokens_est") or 0)
            if te > 0:
                tpb = te / total_bytes
        except (OSError, ValueError):
            pass
    arrays = {
        "offset": np.asarray(cols["offset"], dtype=np.int64),
        "nbytes": np.asarray(cols["nbytes"], dtype=np.int32),
        "group": np.asarray(cols["group"], dtype=np.int32),
        "split": np.asarray(cols["split"], dtype=np.int8),
        "license": np.asarray(cols["license"], dtype=np.int8),
        "n_dec": np.asarray(cols["n_dec"], dtype=np.int16),
        "cls": np.asarray(cols["cls"], dtype=np.int64),
        "st": np.asarray(cols["st"], dtype=np.int32),
        "hd": np.asarray(cols["hd"], dtype=np.int32),
        "it": np.asarray(cols["it"], dtype=np.int32),
        "nq": np.asarray(cols["nq"], dtype=np.int16),
        "no": np.asarray(cols["no"], dtype=np.int32),
    }
    tmp_idx = idx_p.with_name(idx_p.name + f".tmp-{os.getpid()}.npz")
    np.savez(tmp_idx, **arrays)
    os.replace(tmp_idx, idx_p)
    man = {"key": key, "data": str(data_path), "groups": [list(g) for g in groups], "tokens_per_byte": tpb,
           "rows": len(cols["offset"])}
    tmp_man = man_p.with_name(man_p.name + f".tmp-{os.getpid()}")
    tmp_man.write_text(json.dumps(man))
    os.replace(tmp_man, man_p)
    return _load_index(man, idx_p)


def _load_index(man: dict, idx_p: Path) -> FileIndex:
    z = np.load(idx_p)
    return FileIndex(
        src=man["key"]["src"], data=man["data"], offset=z["offset"], nbytes=z["nbytes"], group=z["group"],
        split=z["split"], license=z["license"], n_dec=z["n_dec"], cls=z["cls"], st=z["st"], hd=z["hd"], it=z["it"],
        nq=z["nq"], no=z["no"],
        groups=[tuple(g) for g in man["groups"]], tokens_per_byte=float(man.get("tokens_per_byte", DEFAULT_TOKENS_PER_BYTE)),
    )


def _build_one(args: tuple[str, str]) -> str:
    build_file_index(*args)
    return args[0]


def discover(roots: Sequence[str | Path]) -> list[Path]:
    """Data files under each root: `<root>/raw/**` when present (the data agents' layout), else `<root>/**`.
    A root may also be a single file. Sorted for determinism; *.stats.json and hidden dirs are skipped."""
    out: list[Path] = []
    for r in roots:
        r = Path(r)
        if r.is_file():
            out.append(r)
            continue
        base = r / "raw" if (r / "raw").is_dir() else r
        for p in sorted(base.rglob("*")):
            if any(part.startswith(".") for part in p.relative_to(base).parts):
                continue
            if p.is_file() and (p.name.endswith(".jsonl.zst") or p.name.endswith(".jsonl")):
                out.append(p)
    # a source written both compressed and plain: keep the compressed one
    names = {str(p) for p in out}
    return [p for p in out if not (p.name.endswith(".jsonl") and f"{p}.zst" in names)]


# ---------------------------------------------------------------------------- corpus


class Corpus:
    """All indexed rows of a set of files, with global row ids (file-major)."""

    def __init__(self, files: list[FileIndex]):
        self.files = files
        self.groups: list[tuple[str, str, str]] = []
        gmap: dict[tuple[str, str, str], int] = {}
        parts: dict[str, list[np.ndarray]] = {k: [] for k in ("file", "offset", "nbytes", "group", "split", "license",
                                                              "n_dec", "cls", "est", "st", "hd", "it", "nq", "no", "tpb")}
        for fi, f in enumerate(files):
            remap = np.asarray([gmap.setdefault(g, len(gmap)) for g in f.groups] or [0], dtype=np.int32)
            n = len(f)
            parts["file"].append(np.full(n, fi, dtype=np.int32))
            for k in ("offset", "nbytes", "split", "license", "n_dec", "cls", "st", "hd", "it", "nq", "no"):
                parts[k].append(getattr(f, k))
            parts["tpb"].append(np.full(n, f.tokens_per_byte, dtype=np.float32))
            parts["group"].append(remap[f.group] if n else f.group)
            parts["est"].append(np.maximum(1, np.round(f.nbytes * f.tokens_per_byte)).astype(np.int32))
        self.groups = list(gmap)
        cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in parts.items()}
        self.file = cat["file"].astype(np.int32)
        self.offset = cat["offset"].astype(np.int64)
        self.nbytes = cat["nbytes"].astype(np.int32)
        self.group = cat["group"].astype(np.int32)
        self.split = cat["split"].astype(np.int8)
        self.license = cat["license"].astype(np.int8)
        self.n_dec = cat["n_dec"].astype(np.int16)
        self.cls = cat["cls"].astype(np.int64)
        self.st = cat["st"].astype(np.int32)
        self.hd = cat["hd"].astype(np.int32)
        self.it = cat["it"].astype(np.int32)
        self.nq = cat["nq"].astype(np.int16)
        self.no = cat["no"].astype(np.int32)
        self.tpb = cat["tpb"].astype(np.float32)  # tokens per byte of each row's file
        self.est = cat["est"].astype(np.int32)  # estimated packed tokens
        self._fds: dict[int, int] = {}

    @classmethod
    def open(cls, roots: Sequence[str | Path], cache_dir: str | Path = "runs/stream_cache", workers: int = 4,
             build: bool = True) -> "Corpus":
        """Index (or load cached indices of) every data file under `roots`. With build=False the indices must
        already exist (DDP: local rank 0 builds, the other ranks wait at a barrier and then load)."""
        files = discover(roots)
        if not files:
            raise FileNotFoundError(f"no *.jsonl(.zst) under {list(map(str, roots))}")
        cache_dir = Path(cache_dir)
        if build and workers > 1 and len(files) > 1:
            import multiprocessing as mp

            # spawn, not fork: the trainer process already runs torch / tokenizer threads
            with ProcessPoolExecutor(max_workers=min(workers, len(files)), mp_context=mp.get_context("spawn")) as ex:
                list(ex.map(_build_one, [(str(f), str(cache_dir)) for f in files]))
        return cls([build_file_index(f, cache_dir) for f in files])

    def __len__(self) -> int:
        return len(self.offset)

    def row_bytes(self, i: int) -> bytes:
        fi = int(self.file[i])
        fd = self._fds.get(fi)
        if fd is None:
            fd = self._fds[fi] = os.open(self.files[fi].data, os.O_RDONLY)
        return os.pread(fd, int(self.nbytes[i]), int(self.offset[i]))

    def row(self, i: int) -> dict:
        return json.loads(self.row_bytes(i))

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __getstate__(self) -> dict:  # file descriptors are per process
        d = dict(self.__dict__)
        d["_fds"] = {}
        return d

    def calibrate_lengths(self, count_tokens, per_file: int = 64, seed: int = 0) -> dict[str, float]:
        """Re-estimate tokens-per-byte per file from `count_tokens(example) -> packed tokens` on a sample,
        and refresh `est`. Deterministic sample; returns {file: ratio}."""
        out = {}
        rng = np.random.default_rng(seed)
        for fi, f in enumerate(self.files):
            ids = np.nonzero(self.file == fi)[0]
            if not len(ids):
                continue
            pick = rng.choice(ids, size=min(per_file, len(ids)), replace=False)
            tok = sum(count_tokens(self.row(int(i))) for i in pick)
            byt = int(self.nbytes[pick].sum())
            if tok > 0 and byt > 0:
                f.tokens_per_byte = tok / byt
                m = self.file == fi
                self.est[m] = np.maximum(1, np.round(self.nbytes[m] * f.tokens_per_byte)).astype(np.int32)
                self.tpb[m] = f.tokens_per_byte
                out[f.src] = round(f.tokens_per_byte, 4)
        return out

    def summary(self) -> dict:
        rows_by_bucket: dict[str, int] = {}
        for g, n in zip(self.groups, np.bincount(self.group, minlength=len(self.groups))):
            rows_by_bucket[g[0]] = rows_by_bucket.get(g[0], 0) + int(n)
        return {"files": len(self.files), "rows": len(self), "groups": len(self.groups),
                "rows_by_bucket": rows_by_bucket, "est_tokens": int(self.est.astype(np.int64).sum()),
                "by_split": {s: int((self.split == i).sum()) for i, s in enumerate(SPLITS)}}


# ---------------------------------------------------------------------------- mixture


@dataclass
class MixConfig:
    seed: int = 0
    splits: tuple[str, ...] = ("train",)
    licenses: tuple[str, ...] | None = None  # None = all; ("commercial",) = D5 clean variant
    buckets: tuple[str, ...] | None = None  # None = all present
    bucket_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BUCKET_WEIGHTS))
    weight_unit: str = "tokens"  # "tokens" (PLAN shares) | "rows"
    alpha: float = 0.5  # group sampling ∝ n_decisions^alpha within a bucket
    source_cap: int | dict[str, int | None] | None = None  # max supervised decisions per group per pass (PLAN:
    #                     ~20k); a dict gives per-bucket caps with a "*" default, like max_repeat (phase-2 assemble)
    class_cap: int | dict[str, int | None] | None = None  # max rows per (group, class) per pass (PLAN: 500-2000)
    max_repeat: float | dict[str, float] = 2.0  # per bucket: times a group's capped rows may repeat in a pass
    pass_tokens: int | None = None  # tokens per pass; default = all capped tokens (one capped epoch)
    passes: float = 1.0  # fractional: 1.5 = one full pass + half a pass
    pool: int = 4096  # rows sorted together for length bucketing
    batch_tokens: int | None = 16384  # budget per micro-batch: estimated padded slots (layout="tree") or rows x longest
    layout: str = "tree"  # "tree": group rows by option/header size buckets + state length, cut by tree-layout
    #                         padded-slot cost (trunk B x S, headers Q x H, option chunks x C); "dense": v1 rule
    batch_rows: int | None = None  # fixed rows per micro-batch instead (v1 style)
    max_len: int = 8192  # packed rows are at most this long (longer examples are split by the trainer)
    curriculum: tuple[tuple[float, int], ...] = ()  # ((progress, max tokens), ...) piecewise linear; () = off
    balance: bool = False  # DDP: deal each step's world x grad_accum micro-batches to ranks by estimated cost (equal work per rank)
    #                        (opt-in: on real v2 data it did not reduce the measured straggler wait)

    def weight(self, bucket: str) -> float:
        return float(self.bucket_weights.get(bucket, OTHER_BUCKET_WEIGHT))

    def caps(self, bucket: str) -> tuple[int | None, int | None]:
        """(source_cap, class_cap) for a bucket; dict values are per bucket with a "*" default."""
        def one(v):
            return v.get(bucket, v.get("*")) if isinstance(v, dict) else v
        return one(self.source_cap), one(self.class_cap)

    def repeat(self, bucket: str) -> float:
        if isinstance(self.max_repeat, dict):
            return float(self.max_repeat.get(bucket, self.max_repeat.get("*", 2.0)))
        return float(self.max_repeat)

    def cap_at(self, progress: float) -> int | None:
        c = self.curriculum
        if not c:
            return None
        if progress <= c[0][0]:
            return int(c[0][1])
        for (p0, v0), (p1, v1) in zip(c, c[1:]):
            if progress <= p1:
                return int(v0 + (v1 - v0) * (progress - p0) / max(p1 - p0, 1e-9))
        return int(c[-1][1])

    @classmethod
    def from_json(cls, path_or_obj: str | Path | dict, **over: Any) -> "MixConfig":
        obj = path_or_obj if isinstance(path_or_obj, dict) else json.loads(Path(path_or_obj).read_text())
        obj = {**obj, **{k: v for k, v in over.items() if v is not None}}
        for k in ("splits", "licenses", "buckets"):
            if obj.get(k) is not None:
                obj[k] = tuple(obj[k])
        if obj.get("curriculum") is not None:
            obj["curriculum"] = tuple((float(p), int(v)) for p, v in obj["curriculum"])
        if obj.get("bucket_weights") is not None:
            obj["bucket_weights"] = {**DEFAULT_BUCKET_WEIGHTS, **obj["bucket_weights"]}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(obj) - known
        if unknown:
            raise ValueError(f"unknown mixture keys: {sorted(unknown)}")
        return cls(**obj)

    def to_json(self) -> dict:
        d = asdict(self)
        d["curriculum"] = [list(x) for x in self.curriculum]
        return d


def _rng(*parts: int) -> np.random.Generator:
    words = []
    for p in parts:
        p = int(p)
        words += [p & 0xFFFFFFFF, (p >> 32) & 0xFFFFFFFF]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


@dataclass
class PassPlan:
    rows: np.ndarray  # int64 global row ids in pass order (after the global shuffle)
    order: np.ndarray  # int64 row ids in micro-batch order
    starts: np.ndarray  # int64 CSR starts of each micro-batch in `order` (len = n_micro + 1)
    groups: list[dict]  # per-group accounting
    tokens: int
    costs: np.ndarray | None = None  # estimated cost per micro-batch (NaN until a balanced step needs it)

    @property
    def n_micro(self) -> int:
        return len(self.starts) - 1

    def micro(self, c: int) -> np.ndarray:
        return self.order[self.starts[c] : self.starts[c + 1]]


class Mixture:
    def __init__(self, corpus: Corpus, cfg: MixConfig):
        self.corpus = corpus
        self.cfg = cfg
        ok = np.isin(corpus.split, [SPLITS.index(s) if s in SPLITS else len(SPLITS) for s in cfg.splits])
        if cfg.licenses is not None:
            ok &= np.isin(corpus.license, [LICENSES.index(x) for x in cfg.licenses])
        if cfg.buckets is not None:
            keep = np.asarray([g[0] in cfg.buckets for g in corpus.groups] or [False])
            ok &= keep[corpus.group]
        self.eligible = np.nonzero(ok)[0].astype(np.int64)
        if not len(self.eligible):
            raise ValueError("mixture is empty after split/licence/bucket filters")
        order = np.argsort(corpus.group[self.eligible], kind="stable")
        e = self.eligible[order]
        g = corpus.group[e]
        bounds = np.flatnonzero(np.diff(g)) + 1
        self._group_rows = {int(gr[0]): rows for gr, rows in zip(np.split(g, bounds), np.split(e, bounds))}
        self._gseed = {gid: _h64("\x1f".join(corpus.groups[gid])) for gid in self._group_rows}  # stable across corpora
        self._plans: dict[int, PassPlan] = {}

    # ------------------------------------------------------------------ pass construction

    def _capped(self, gid: int, k: int) -> np.ndarray:
        cfg, c = self.cfg, self.corpus
        rows = self._group_rows[gid]
        rows = rows[_rng(cfg.seed, k, self._gseed[gid], 0).permutation(len(rows))]
        source_cap, class_cap = cfg.caps(c.groups[gid][0])
        if class_cap:
            cls = c.cls[rows]
            keep = np.ones(len(rows), dtype=bool)
            has = cls != 0
            if has.any():
                # rank of each row within its class, in permuted order
                order = np.argsort(cls, kind="stable")
                sc = cls[order]
                first = np.r_[0, np.flatnonzero(np.diff(sc)) + 1]
                rank = np.arange(len(sc)) - np.repeat(first, np.diff(np.r_[first, len(sc)]))
                r = np.empty(len(rows), dtype=np.int64)
                r[order] = rank
                keep = ~has | (r < class_cap)
            rows = rows[keep]
        if source_cap:
            dec = np.maximum(c.n_dec[rows].astype(np.int64), 1)
            n = int(np.searchsorted(np.cumsum(dec), source_cap, side="right"))
            rows = rows[: max(1, n)]
        return rows

    def plan(self, k: int) -> PassPlan:
        if k not in self._plans:
            self._plans[k] = self._build(k)
            for old in [x for x in self._plans if x < k - 1]:
                del self._plans[old]
        return self._plans[k]

    def _build(self, k: int) -> PassPlan:
        cfg, c = self.cfg, self.corpus
        capped = {gid: self._capped(gid, k) for gid in sorted(self._group_rows)}
        info = {}
        for gid, rows in capped.items():
            est = c.est[rows].astype(np.int64)
            info[gid] = {"rows": len(rows), "dec": int(np.maximum(c.n_dec[rows], 1).sum()), "tok": int(est.sum())}
        unit = "tok" if cfg.weight_unit == "tokens" else "rows"
        total = sum(v[unit] for v in info.values())
        budget = (cfg.pass_tokens if (cfg.pass_tokens and unit == "tok") else total)
        budget = budget * self._frac(k)
        buckets: dict[str, list[int]] = {}
        for gid in capped:
            buckets.setdefault(c.groups[gid][0], []).append(gid)
        wsum = sum(cfg.weight(b) for b in buckets)
        quota: dict[int, float] = {}
        for b, gids in buckets.items():
            want = budget * cfg.weight(b) / wsum if wsum > 0 else 0.0
            limit = {g: cfg.repeat(b) * info[g][unit] for g in gids}
            share = {g: float(info[g]["dec"]) ** cfg.alpha for g in gids}
            alloc = {g: 0.0 for g in gids}
            free = [g for g in gids if limit[g] > 0]
            left = want
            while free and left > 1e-9:  # water-filling under the repeat limits
                s = sum(share[g] for g in free)
                if s <= 0:
                    break
                give = {g: left * share[g] / s for g in free}
                over = [g for g in free if alloc[g] + give[g] >= limit[g]]
                if not over:
                    for g in free:
                        alloc[g] += give[g]
                    left = 0.0
                    break
                for g in over:
                    left -= limit[g] - alloc[g]
                    alloc[g] = limit[g]
                free = [g for g in free if g not in over]
            quota.update(alloc)
        seq: list[np.ndarray] = []
        groups_out = []
        for gid, rows in capped.items():
            n = len(rows)
            if n == 0:
                continue
            per_row = info[gid][unit] / n
            take = int(round(quota[gid] / per_row)) if per_row > 0 else 0
            parts, r = [], 0
            while sum(len(p) for p in parts) < take:
                parts.append(rows if r == 0 else rows[_rng(cfg.seed, k, self._gseed[gid], r + 1).permutation(n)])
                r += 1
            sel = np.concatenate(parts)[:take] if parts else rows[:0]
            seq.append(sel)
            b, s, sub = c.groups[gid]
            groups_out.append({"bucket": b, "source": s, "subsource": sub, "rows_avail": int(len(self._group_rows[gid])),
                               "rows_capped": n, "decisions_capped": info[gid]["dec"], "rows_taken": int(take),
                               "repeat": round(take / n, 3), "tokens": int(c.est[sel].astype(np.int64).sum())})
        rows = np.concatenate(seq) if seq else np.zeros(0, dtype=np.int64)
        rows = rows[_rng(cfg.seed, k, 0xB00).permutation(len(rows))]
        order, starts = self._batch(rows, k)
        return PassPlan(rows, order, starts, groups_out, int(c.est[rows].astype(np.int64).sum()))

    def _batch(self, rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        est = np.minimum(self.corpus.est, cfg.max_len)
        out: list[np.ndarray] = []
        sizes: list[int] = []
        deferred = np.zeros(0, dtype=np.int64)  # rows waiting for the curriculum cap, oldest first
        n = len(rows)
        passes = max(cfg.passes, 1e-9)
        pi = 0
        for s in range(0, max(n, 1), cfg.pool):
            pool = rows[s : s + cfg.pool]
            last = s + cfg.pool >= n
            cap = cfg.cap_at(min(1.0, (k + (s + len(pool)) / max(n, 1)) / passes))
            if last or cap is None:
                pool, deferred = np.concatenate([deferred, pool]), deferred[:0]
            else:
                if len(deferred):  # release rows the cap now allows, at most half a pool at a time
                    ok = np.flatnonzero(self.corpus.est[deferred] <= cap)[: cfg.pool // 2]
                    keep = np.ones(len(deferred), dtype=bool)
                    keep[ok] = False
                    pool, deferred = np.concatenate([deferred[ok], pool]), deferred[keep]
                long = self.corpus.est[pool] > cap
                deferred, pool = np.concatenate([deferred, pool[long]]), pool[~long]
            if not len(pool):
                continue
            pool = self._order_pool(pool, est)
            batches = self._cut(pool, est)
            perm = _rng(cfg.seed, k, 0xBA7C, pi).permutation(len(batches))
            pi += 1
            for j in perm:
                out.append(batches[j])
                sizes.append(len(batches[j]))
        order = np.concatenate(out) if out else np.zeros(0, dtype=np.int64)
        starts = np.zeros(len(sizes) + 1, dtype=np.int64)
        np.cumsum(sizes, out=starts[1:])
        return order, starts

    def _sizes(self, ids: np.ndarray) -> dict[str, np.ndarray]:
        """Estimated packed-token sizes of the tree-layout parts of each row."""
        c = self.corpus
        tpb = c.tpb[ids].astype(np.float64)
        tot = np.minimum(c.est[ids], self.cfg.max_len).astype(np.float64)
        st = np.minimum(np.ceil(c.st[ids] * tpb) + 2, tot)
        hd = np.ceil(c.hd[ids] * tpb) + 2
        it = np.ceil(c.it[ids] * tpb) + 1
        nq = np.maximum(c.nq[ids].astype(np.float64), 1)
        itok = np.maximum(tot - st - nq * hd, 2 * c.no[ids])
        return {"tot": tot, "st": st, "hd": hd, "cw": np.maximum(64, np.ceil(it / 32) * 32), "nq": nq, "itok": itok}

    def batch_cost(self, ids: np.ndarray) -> float:
        """Estimated padded slots of one micro-batch in the tree layout."""
        z = self._sizes(np.asarray(ids))
        C = z["cw"].max()
        chunks = max(z["nq"].sum(), np.ceil(1.3 * z["itok"].sum() / C))
        return float(len(ids) * z["st"].max() + z["nq"].sum() * z["hd"].max() + chunks * C)

    def _order_pool(self, pool: np.ndarray, est: np.ndarray) -> np.ndarray:
        if self.cfg.layout != "tree" or self.cfg.batch_rows:
            return pool[np.argsort(est[pool], kind="stable")]
        z = self._sizes(pool)
        cb = np.searchsorted([64, 128, 256, 512], z["cw"], side="left")  # option-chunk width class
        hb = np.searchsorted([32, 64, 128], z["hd"], side="left")  # header length class
        return pool[np.lexsort((z["st"], hb, cb))]

    def _cut(self, pool: np.ndarray, est: np.ndarray) -> list[np.ndarray]:
        cfg = self.cfg
        if cfg.batch_rows:
            return [pool[j : j + cfg.batch_rows] for j in range(0, len(pool), cfg.batch_rows)]
        budget = cfg.batch_tokens or 16384
        out, j = [], 0
        if cfg.layout != "tree":
            lens = est[pool]  # ascending
            while j < len(pool):
                # rows j..m-1 share the padded length lens[m-1] (sorted ascending)
                m = j + 1
                while m < len(pool) and (m + 1 - j) * lens[m] <= budget:
                    m += 1
                out.append(pool[j:m])
                j = m
            return out
        z = {k: v.tolist() for k, v in self._sizes(pool).items()}
        st, hd, cw, nq, itok = z["st"], z["hd"], z["cw"], z["nq"], z["itok"]
        n = len(pool)
        while j < n:
            S, H, C, Q, IT = st[j], hd[j], cw[j], nq[j], itok[j]
            m = j + 1
            while m < n:
                S2, H2, C2, Q2, IT2 = max(S, st[m]), max(H, hd[m]), max(C, cw[m]), Q + nq[m], IT + itok[m]
                cost = (m + 1 - j) * S2 + Q2 * H2 + max(Q2, math.ceil(1.3 * IT2 / C2)) * C2
                if cost > budget:
                    break
                S, H, C, Q, IT = S2, H2, C2, Q2, IT2
                m += 1
            out.append(pool[j:m])
            j = m
        return out

    # ------------------------------------------------------------------ iteration

    def n_passes(self) -> int:
        return max(1, math.ceil(self.cfg.passes - 1e-9))

    def _frac(self, k: int) -> float:
        """Budget fraction of pass k: 1.5 passes = a full pass then half a pass; passes beyond the
        configured number (cycling to reach --max-steps) are full."""
        if k >= self.n_passes():
            return 1.0
        return max(0.0, min(1.0, self.cfg.passes - k))

    def steps_in_pass(self, k: int, world: int, grad_accum: int) -> int:
        return self.plan(k).n_micro // max(1, world * grad_accum)

    def total_steps(self, world: int, grad_accum: int) -> int:
        return sum(self.steps_in_pass(k, world, grad_accum) for k in range(self.n_passes()))

    def iterate(self, rank: int = 0, world: int = 1, grad_accum: int = 1, start_pass: int = 0,
                start_cursor: int = 0, cycle: bool = False) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield (pass, global cursor after this micro-step, row ids of this rank's micro-batch).
        Each optimizer step consumes P = world * grad_accum consecutive global micro-batches; the tail of a pass
        that cannot fill a whole step is skipped on every rank alike. Without balancing, micro-step j of a step
        gives rank r global micro-batch c0 + j*world + r; with `balance` (and world > 1) the P micro-batches of
        the step are dealt to ranks by estimated cost (longest first to the least-loaded rank, exactly
        grad_accum each), so ranks finish a step together. cycle=True keeps drawing passes past `passes`
        (the trainer stops at --max-steps)."""
        per_step = world * grad_accum
        k = start_pass
        while cycle or k < self.n_passes():
            plan = self.plan(k)
            usable = (plan.n_micro // per_step) * per_step
            if usable == 0 and (cycle or k >= self.n_passes() - 1):
                return  # nothing to train on (and cycling would spin forever)
            c = start_cursor if k == start_pass else 0
            c -= c % (per_step if self.cfg.balance and world > 1 else world)  # start on a (micro-)step boundary
            while c + world <= usable:
                if self.cfg.balance and world > 1 and c % per_step == 0:
                    mine = self._deal(plan, c, world, grad_accum)[rank]
                    for j, m in enumerate(mine):
                        yield k, c + (j + 1) * world, plan.micro(m)
                    c += per_step
                    continue
                yield k, c + world, plan.micro(c + rank)
                c += world
            k += 1

    def _deal(self, plan: PassPlan, c0: int, world: int, grad_accum: int) -> list[list[int]]:
        """Deal the step's world*grad_accum micro-batches to ranks so every rank gets (close to) the same estimated
        work and exactly grad_accum micro-batches: LPT (largest cost first to the least-loaded rank with room),
        then pairwise swaps between the most- and least-loaded ranks while they lower the maximum load. Each
        rank's list is ordered by cost, largest first, so micro-step j costs about the same on every rank too
        (matters only if something still synchronises per micro-batch). Costs are estimated per step (cheap),
        not for the whole pass up front."""
        ids = list(range(c0, c0 + world * grad_accum))
        if plan.costs is None:
            plan.costs = np.full(plan.n_micro, np.nan)
        cost = plan.costs
        for i in ids:
            if cost[i] != cost[i]:  # NaN: not estimated yet
                cost[i] = self.batch_cost(plan.micro(i))
        ids.sort(key=lambda i: (-cost[i], i))
        load = [0.0] * world
        out: list[list[int]] = [[] for _ in range(world)]
        for i in ids:
            r = min((r for r in range(world) if len(out[r]) < grad_accum), key=lambda r: (load[r], r))
            out[r].append(i)
            load[r] += cost[i]
        if grad_accum > 1:
            for _ in range(4 * world):
                a = max(range(world), key=lambda r: (load[r], -r))
                b = min(range(world), key=lambda r: (load[r], r))
                gap = load[a] - load[b]
                if gap <= 0:
                    break
                best, pair = gap, None
                for x in out[a]:
                    for y in out[b]:
                        d = cost[x] - cost[y]
                        if 0 < d < gap and abs(gap - 2 * d) < best:
                            best, pair = abs(gap - 2 * d), (x, y, d)
                if pair is None:
                    break
                x, y, d = pair
                out[a][out[a].index(x)] = y
                out[b][out[b].index(y)] = x
                load[a] -= d
                load[b] += d
        return [sorted(x, key=lambda i: (-cost[i], i)) for x in out]

    def plan_summary(self, k: int = 0) -> dict:
        p = self.plan(k)
        lens = np.minimum(self.corpus.est[p.rows], self.cfg.max_len)
        hist = np.histogram(lens, bins=[0, 1024, 4096, 8192, 1 << 30])[0]
        by_bucket: dict[str, dict] = {}
        for g in p.groups:
            b = by_bucket.setdefault(g["bucket"], {"rows": 0, "tokens": 0, "groups": 0})
            b["rows"] += g["rows_taken"]
            b["tokens"] += g["tokens"]
            b["groups"] += 1
        tot = max(1, sum(b["tokens"] for b in by_bucket.values()))
        for b in by_bucket.values():
            b["token_share"] = round(b["tokens"] / tot, 4)
        sizes = np.diff(p.starts)
        sample = range(0, p.n_micro, max(1, p.n_micro // 500))
        pad = [self.batch_cost(p.micro(c)) / max(1.0, float(np.minimum(self.corpus.est[p.micro(c)], self.cfg.max_len).sum()))
               for c in sample]
        return {"pass": k, "rows": int(len(p.rows)), "tokens": p.tokens, "n_micro": p.n_micro,
                "rows_per_micro_mean": float(sizes.mean()) if len(sizes) else 0.0,
                "est_slots_per_token": round(float(np.mean(pad)), 3) if pad else None,
                "len_hist": {"<=1k": int(hist[0]), "1-4k": int(hist[1]), "4-8k": int(hist[2]), ">8k": int(hist[3])},
                "by_bucket": by_bucket, "groups": p.groups, "config": self.cfg.to_json()}


def main(argv: Sequence[str] | None = None) -> None:
    """Index data and print the mixture plan: python -m jev_local.train.stream data/v2 [--mixture m.json]"""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m jev_local.train.stream")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--cache", default="runs/stream_cache")
    ap.add_argument("--mixture", default=None, help="MixConfig JSON")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="write the plan summary JSON here")
    args = ap.parse_args(argv)
    t0 = time.time()
    corpus = Corpus.open(args.roots, args.cache, workers=args.workers)
    t1 = time.time()
    cfg = MixConfig.from_json(args.mixture) if args.mixture else MixConfig()
    mix = Mixture(corpus, cfg)
    s = {"corpus": corpus.summary(), "index_s": round(t1 - t0, 1), "plan": mix.plan_summary(0),
         "plan_s": round(time.time() - t1, 2)}
    text = json.dumps(s, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text if len(text) < 20000 else json.dumps({k: v for k, v in s.items() if k != "plan"} |
                                                   {"plan": {k: v for k, v in s["plan"].items() if k != "groups"}}, indent=2),
          file=sys.stdout)


if __name__ == "__main__":
    main()
