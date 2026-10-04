"""“A First Glance at Jev for Network Traffic Classification” (arXiv 2610.00376; targets T490 Jev-0 .098, T491 Jev-40 .2842).

Published protocol (§3): Lystea/CESNET-QUICEXT25-PARQUET (fixed archived revision; daily files D2024MMDD.parquet);
ten classes; records with >= 10 packets; input = the first ten (inter-packet time ms, direction -1/+1, payload size)
triples, values unscaled, as packet-major JSON with field descriptions, no label / domain / address / port / time;
weeks = 7-day daily-file partitions from 2024-06-01 (weeks 1-4 train, 5-30 test; 2024-10-31 and 12-11..14 missing);
per week a seeded hash of the source ID (seed 2026092305) ranks eligible records, at most 8,192 candidates, the
first 2,000 kept after excluding previously selected IDs and exact float32-prefix fingerprints; one Choice question
over the ten names; the 40-example state = the lowest-ranked training record of each week/class cell.
NOT published: the hash function, the request wording, the 52,000 selected IDs. This adapter is INDICATIVE: our hash
is sha256(f"{seed}:{flow_id}"), the wording is ours, and the sample differs from the paper's. Nothing here trains on
or tunes against the test weeks; the 40 examples come from weeks 1-4 only.
Metrics: accuracy over the requested records (invalid/failed = wrong, 100% valid coverage upstream), per-week
accuracy, macro-F1, per-class recall, prediction histogram.
Self-check split "validation": the weeks 1-4 selection (the paper's RF/ET training pool, 8,000 records).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import struct
from typing import Any, Iterable, Sequence

from jev_local.bench.benchmax.adapters_b.common import (AdapterB, Item, accuracy, answers_of, by_task, choice_pick,
                                                        coverage, macro_f1)

SPEC_ID = "study:jev_for_network_traffic_classification_arxiv_261"  # exactly as bench/public/targets.json spells it
TARGETS = ("T490", "T491")
UPSTREAM = {"paper": "arXiv 2610.00376", "dataset": "Lystea/CESNET-QUICEXT25-PARQUET", "licence": "CC-BY-4.0 data; protocol from the paper",
            "indicative": True}
DATASET = "Lystea/CESNET-QUICEXT25-PARQUET"
REVISION = "4180d7d97e440e7fdf9e43693e052509074999a4"
CLASSES = ("apple-privaterelay", "facebook-graph", "google-ads", "google-gstatic", "google-play", "google-services",
           "google-www", "instagram", "snapchat", "youtube")
SEED = 2026092305
PREFIX, PER_WEEK_CANDIDATES, PER_WEEK_KEEP = 10, 8192, 2000
WEEK1_START = dt.date(2024, 6, 1)
TRAIN_WEEKS, TEST_WEEKS = tuple(range(1, 5)), tuple(range(5, 31))
MISSING_DATES = {dt.date(2024, 10, 31), dt.date(2024, 12, 11), dt.date(2024, 12, 12), dt.date(2024, 12, 13), dt.date(2024, 12, 14)}
COLUMNS = ("flow_id", "label_service", "ppi_len", "ppi_ipt", "ppi_dir", "ppi_size")
FIELDS = {"ipt_ms": "inter-packet time in milliseconds between this packet and the previous one (0 for the first packet)",
          "direction": "+1 for a packet from the client to the server, -1 for a packet from the server to the client",
          "size": "transport payload size of the packet in bytes"}
INSTRUCTIONS = ("The state describes the first ten packets of one QUIC connection: `packets` lists them in order with "
                "the fields explained in `fields`. Decide which application generated the connection. "
                "`examples`, when present, are labelled connections of the same kind.")
QUESTION = "Which of the candidate applications generated this connection?"
TASKS = ("k0", "k40")
TARGET_OF = {"k0": "T490", "k40": "T491"}


def week_dates(week: int) -> list[dt.date]:
    start = WEEK1_START + dt.timedelta(days=7 * (week - 1))
    return [d for d in (start + dt.timedelta(days=i) for i in range(7)) if d not in MISSING_DATES]


def file_name(d: dt.date) -> str:
    return f"D{d:%Y%m%d}.parquet"


def priority(flow_id: str, seed: int = SEED) -> int:
    """Our seeded hash (the paper's is unpublished): lower = selected first."""
    return int(hashlib.sha256(f"{seed}:{flow_id}".encode()).hexdigest()[:16], 16)


def prefix_fingerprint(ipt: Sequence[float], direction: Sequence[int], size: Sequence[int]) -> str:
    b = b"".join(struct.pack("<fbi", float(ipt[i]), int(direction[i]), int(size[i])) for i in range(PREFIX))
    return hashlib.sha256(b).hexdigest()


def packets_json(ipt: Sequence[float], direction: Sequence[int], size: Sequence[int]) -> list[dict[str, Any]]:
    out = []
    for i in range(PREFIX):
        v = float(ipt[i])
        out.append({"ipt_ms": int(v) if v.is_integer() else v, "direction": int(direction[i]), "size": int(size[i])})
    return out


def build_request(packets: list[dict[str, Any]], examples: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    state: dict[str, Any] = {"instructions": INSTRUCTIONS, "fields": dict(FIELDS), "packets": packets}
    if examples:
        state["examples"] = [{"packets": e["packets"], "application": e["label"]} for e in examples]
    return {"state": state, "questions": {"application": {"type": "choice", "instructions": QUESTION,
                                                          "criteria": {c: None for c in CLASSES}}}}


def select_week(records: Iterable[dict[str, Any]], seen_ids: set[str], seen_fps: set[str],
                candidates: int = PER_WEEK_CANDIDATES, keep: int = PER_WEEK_KEEP) -> list[dict[str, Any]]:
    """Paper §3 selection on already-eligible records (10 classes, >= 10 packets): rank by `priority`, cap the
    candidate pool, then keep the first `keep` whose id and prefix fingerprint are new. Mutates the seen sets."""
    ranked = sorted(records, key=lambda r: (r["priority"], r["flow_id"]))[:candidates]
    out = []
    for r in ranked:
        if r["flow_id"] in seen_ids or r["fingerprint"] in seen_fps:
            continue
        seen_ids.add(r["flow_id"])
        seen_fps.add(r["fingerprint"])
        out.append(r)
        if len(out) == keep:
            break
    if len(out) < keep:
        raise RuntimeError(f"insufficient pool: {len(out)} < {keep} after {len(ranked)} candidates")
    return out


def pick_examples(train: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """40 fixed examples: the lowest-ranked (smallest priority) training record per week/class cell, week-major."""
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for r in train:
        k = (r["week"], r["label"])
        if k not in best or r["priority"] < best[k]["priority"]:
            best[k] = r
    out = [best[(w, c)] for w in TRAIN_WEEKS for c in CLASSES if (w, c) in best]
    if len(out) != len(TRAIN_WEEKS) * len(CLASSES):
        raise RuntimeError(f"examples: {len(out)} week/class cells filled, expected {len(TRAIN_WEEKS) * len(CLASSES)}")
    return out


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")
    TASKS = TASKS

    # ------------------------------------------------------------------ data (VM; pyarrow + fsspec over HTTP)
    def _eligible(self, d: dt.date, week: int) -> list[dict[str, Any]]:
        cache = self.raw_dir("cesnet") / "eligible" / f"{file_name(d)}.jsonl"
        if cache.exists():
            return [json.loads(l) for l in cache.read_text().split("\n") if l]
        import fsspec  # VM only
        import pyarrow.parquet as pq

        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{file_name(d)}"
        out = []
        with fsspec.open(url, "rb") as f:
            pf = pq.ParquetFile(f)
            for g in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(g, columns=list(COLUMNS)).to_pylist()
                for r in t:
                    if r["label_service"] not in CLASSES or (r["ppi_len"] or 0) < PREFIX or not r["ppi_ipt"]:
                        continue
                    ipt, dr, sz = r["ppi_ipt"][:PREFIX], r["ppi_dir"][:PREFIX], r["ppi_size"][:PREFIX]
                    out.append({"flow_id": str(r["flow_id"]), "label": r["label_service"], "week": week, "date": d.isoformat(),
                                "priority": priority(str(r["flow_id"])), "fingerprint": prefix_fingerprint(ipt, dr, sz),
                                "packets": packets_json(ipt, dr, sz)})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("".join(json.dumps(r) + "\n" for r in out))
        return out

    def _week(self, week: int, seen_ids: set[str], seen_fps: set[str]) -> list[dict[str, Any]]:
        cache = self.raw_dir("cesnet") / "selected" / f"week{week:02d}.jsonl"
        if cache.exists():
            rows = [json.loads(l) for l in cache.read_text().split("\n") if l]
        else:
            pool = [r for d in week_dates(week) for r in self._eligible(d, week)]
            rows = select_week(pool, set(seen_ids), set(seen_fps))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
        seen_ids.update(r["flow_id"] for r in rows)
        seen_fps.update(r["fingerprint"] for r in rows)
        return rows

    def _selection(self, weeks: Sequence[int], stop_after: int | None = None) -> list[dict[str, Any]]:
        """Weeks are processed in order (the exclusion sets depend on every earlier week, training weeks first)."""
        seen_ids: set[str] = set()
        seen_fps: set[str] = set()
        out: list[dict[str, Any]] = []
        for w in TRAIN_WEEKS + TEST_WEEKS:
            if w > max(weeks):
                break
            rows = self._week(w, seen_ids, seen_fps)
            if w in weeks:
                out.extend(rows)
                if stop_after is not None and len(out) >= stop_after:
                    break
        return out

    def prepare(self, split: str) -> dict[str, Any]:
        weeks = TRAIN_WEEKS if split == "validation" else TEST_WEEKS
        rows = self._selection(list(weeks))
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        return {"spec": SPEC_ID, "upstream": UPSTREAM, "revision": REVISION, "seed": SEED, "weeks": list(weeks), "n": len(rows),
                "by_class": counts, "examples": len(pick_examples(self._selection(list(TRAIN_WEEKS)))),
                "note": "indicative sample: the paper's hash, wording and selected IDs are unpublished"}

    def expected_counts(self, split: str) -> dict[str, int]:
        n = PER_WEEK_KEEP * (len(TEST_WEEKS) if split == "test" else len(TRAIN_WEEKS))
        return {t: n for t in TASKS}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        weeks = list(TRAIN_WEEKS if split == "validation" else TEST_WEEKS)
        rows = self._selection(weeks, stop_after=limit)
        if limit is not None:
            rows = rows[:limit]
        tasks = self._select_tasks(tasks)
        examples = pick_examples(self._selection(list(TRAIN_WEEKS))) if "k40" in tasks else []
        out = []
        for task in tasks:
            for r in rows:
                req = build_request(r["packets"], examples if task == "k40" else ())
                out.append(Item(f"{task}/{r['flow_id']}", task, req, r["label"],
                                meta={"week": r["week"], "date": r["date"], "in_examples": any(e["flow_id"] == r["flow_id"] for e in examples)}))
        return out

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "indicative": True, "tasks": {}}
        for task, its in by_task(items).items():
            gold, pred, weeks, failed = [], [], [], 0
            hist: dict[str, int] = {}
            for it in its:
                a = answers_of(answers.get(it.id))
                pick = choice_pick(a.get("application"), CLASSES) if a else None
                if pick is None:
                    failed += 1
                gold.append(it.gold)
                pred.append(pick)
                weeks.append(it.meta["week"])
                hist[str(pick)] = hist.get(str(pick), 0) + 1
            r: dict[str, Any] = {**coverage(its, answers), "failed": failed, "target": TARGET_OF[task], "n": len(its)}
            if its:
                r["accuracy"] = accuracy(pred, gold)  # failed = wrong
                r["macro_f1"] = macro_f1([p if p is not None else "__none__" for p in pred], gold, list(CLASSES))
                r["recall_by_class"] = {c: sum(1 for p, g in zip(pred, gold) if g == c and p == g) / max(1, sum(1 for g in gold if g == c))
                                        for c in CLASSES if c in gold}
                r["accuracy_by_week"] = {str(w): sum(1 for p, g, ww in zip(pred, gold, weeks) if ww == w and p == g) / weeks.count(w)
                                         for w in sorted(set(weeks))}
                r["prediction_histogram"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
            res["tasks"][task] = r
        return res
