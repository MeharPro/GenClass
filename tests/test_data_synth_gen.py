"""The generic System-One corpus: valid shapes, varied states, and labels re-derived independently."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from jev_local.data import synth_gen as SG
from jev_local.data.stats import iter_corpus, validate_example
from jev_local.schema import ChoiceQuestion, question_from_json
from jev_local.serialize import entry_text

N = 2500


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[Path, dict, list[dict]]:
    d = tmp_path_factory.mktemp("gen")
    man = SG.generate(N, d, seed=11)
    return d, man, list(iter_corpus(d))


def _instr(ex, qid) -> str:
    return entry_text(ex["questions"][qid].get("instructions"))


def test_shapes_and_validation(corpus):
    d, man, exs = corpus
    assert len(exs) == N == man["n_written"]
    assert len({e["id"] for e in exs}) == N
    for e in exs:
        assert not validate_example(e), (e["id"], validate_example(e))
        assert 1 <= len(e["questions"]) <= 4
        assert set(e["labels"]) <= set(e["questions"])
        for qid, q in e["questions"].items():
            question_from_json(q)


def test_task_and_state_variety(corpus):
    _, man, exs = corpus
    tasks = Counter(e["meta"]["task"] for e in exs)
    assert set(tasks) == {"route", "fact", "priority", "sentiment", "line", "json", "entity"}
    assert min(tasks.values()) > 0.05 * N
    kinds = Counter(type(e["state"]).__name__ for e in exs)
    assert {"str", "dict", "list"} <= set(kinds)
    qkinds = Counter(q["type"] for e in exs for q in e["questions"].values())
    assert min(qkinds[k] for k in ("noul", "choice", "score")) > 100
    instr_kinds = Counter(type(q.get("instructions")).__name__ for e in exs for q in e["questions"].values())
    assert {"str", "dict", "list"} <= set(instr_kinds)
    null_desc = sum(1 for e in exs for q in e["questions"].values() if q["type"] == "choice" and None in q["criteria"].values())
    assert null_desc > 100


def test_noul_labels_are_balanced_enough(corpus):
    _, _, exs = corpus
    by_q: dict[str, Counter] = defaultdict(Counter)
    for e in exs:
        for qid, lab in e["labels"].items():
            if lab["type"] == "noul":
                by_q[re.sub(r"_\d+$", "", qid)][lab["p"]] += 1
    tot = Counter()
    for c in by_q.values():
        tot.update(c)
    assert 0.25 < tot[1.0] / (tot[0.0] + tot[1.0]) < 0.75
    for q in ("mentions_money", "mentions_order"):
        assert by_q[q][1.0] > 0.1 * sum(by_q[q].values()), (q, by_q[q])


def test_heldout_variants_only_in_test(corpus):
    _, _, exs = corpus
    fams = defaultdict(set)
    for e in exs:
        fams[e["split"]].add(e["family"])
    assert fams["test"] and not fams["test"] & (fams["train"] | fams["dev"])
    assert any("heldout-claims" in f for f in fams["test"])
    assert not any("heldout-claims" in f for f in fams["train"] | fams["dev"])
    splits = Counter(e["split"] for e in exs)
    assert 0.06 < splits["test"] / N < 0.2 and 0.01 < splits["dev"] / N < 0.06


# ----------------------------------------------------------------------------- labels re-derived


def test_route_labels(corpus):
    _, _, exs = corpus
    n = 0
    for e in exs:
        if e["meta"]["task"] != "route":
            continue
        text = json.dumps(e["state"], ensure_ascii=False)
        qid = next(q for q in e["questions"] if q in ("department", "route", "team", "queue"))
        gold = e["labels"][qid]["label"]
        pat = re.compile("|".join(re.sub(r"\\\{\w+\\\}", ".+?", re.escape(t)) for t in SG.DEPARTMENTS[gold][1]))
        assert pat.search(text), (e["id"], gold, text)
        for other in e["questions"][qid]["criteria"]:
            if other != gold:
                # the ticket sentence belongs to the gold department only
                pat_o = re.compile("|".join(re.sub(r"\\\{\w+\\\}", ".+?", re.escape(t)) for t in SG.DEPARTMENTS[other][1]))
                assert not pat_o.fullmatch(text)
        st = e["state"]
        msg = (st if isinstance(st, str) else st[0]["text"] if isinstance(st, list) else st.get("body")
               or (st["ticket"] if isinstance(st["ticket"], str) else st["ticket"]["message"]))
        for q2 in ("mentions_money", "mentions_order"):
            if q2 in e["labels"]:
                want = "$" in msg if q2 == "mentions_money" else "#" in msg
                assert e["labels"][q2]["p"] == float(want), e["id"]
        n += 1
    assert n > 200


def test_line_labels(corpus):
    _, _, exs = corpus
    n = 0
    for e in exs:
        if e["meta"]["task"] != "line":
            continue
        st = e["state"]
        if isinstance(st, list):
            lines = {x["id"]: x["text"] for x in st}
        elif "lines" in st:
            lines = dict(l.split(": ", 1) for l in st["lines"])
        else:
            lines = dict(st)
        for qid, lab in e["labels"].items():
            t = qid.removesuffix("_line")
            pat = re.compile("|".join(re.sub(r"\\\{\w+\\\}", ".+?", re.escape(s)) + "$" for s in SG.LINE_TYPES[t][0]))
            matching = [i for i, s in lines.items() if pat.match(s)]
            if lab["label"] == "none":
                assert not matching and "none" in e["questions"][qid]["criteria"]
            else:
                assert matching == [lab["label"]], (e["id"], qid, lines)
            n += 1
    assert n > 200


def test_json_labels(corpus):
    _, _, exs = corpus
    n = 0
    for e in exs:
        if e["meta"]["task"] != "json":
            continue
        s = e["state"]
        for qid, lab in e["labels"].items():
            ins = _instr(e, qid)
            num = lambda: float(re.search(r"(\d+(?:\.\d+)?)", ins).group(1))  # noqa: E731
            if "items" in s:
                total = sum(i["qty"] * i["price"] for i in s["items"])
                want = {
                    "is_paid": lambda: s["status"] == "paid",
                    "over_threshold": lambda: total > num(),
                    "has_category": lambda: any(i["category"] in ins for i in s["items"]),
                    "ships_express": lambda: s["shipping"]["method"] == "express",
                    "is_vip": lambda: s["customer"]["vip"],
                    "many_items": lambda: sum(i["qty"] for i in s["items"]) > num(),
                    "international": lambda: s["shipping"]["country"] != "US",
                }[qid]()
            elif "user" in s:
                u = s["user"]
                want = {
                    "is_admin": lambda: bool({"admin", "owner"} & set(u["roles"])),
                    "mfa": lambda: u["mfa_enabled"],
                    "old_account": lambda: int(u["created"][:4]) < num(),
                    "inactive": lambda: u["last_login_days_ago"] > num(),
                    "on_team": lambda: re.search(r"the (\w+) team", ins).group(1) in u["teams"],
                    "can_edit": lambda: bool({"editor", "admin", "owner"} & set(u["roles"])),
                }[qid]()
            else:
                want = {
                    "delayed_long": lambda: s["delay_minutes"] > num() and s["status"] != "cancelled",
                    "cancelled": lambda: s["status"] == "cancelled",
                    "seats": lambda: s["seats_left"] > 0 and s["status"] != "cancelled",
                    "morning": lambda: int(s["scheduled"][:2]) < 12,
                    "gate_a": lambda: s["gate"].startswith("A"),
                }[qid]()
            assert lab["p"] == float(want), (e["id"], qid, ins, s)
            n += 1
    assert n > 300


def test_entity_labels(corpus):
    _, _, exs = corpus
    n = 0
    for e in exs:
        if e["meta"]["task"] != "entity":
            continue
        (qid, lab), = e["labels"].items()
        ins, s, gold = _instr(e, qid), e["state"], lab["label"]
        assert gold in e["questions"][qid]["criteria"]
        if qid == "manager":
            team = re.search(r"the (\w+) team", ins).group(1)
            assert [p["name"] for p in s["employees"] if p["team"] == team and p["role"] == "manager"] == [gold]
        elif qid == "pick":
            price = {x["product"]: int(x["price"][1:]) for x in s}
            if "cheapest" in ins:
                assert gold == min(price, key=price.get)
            elif "expensive" in ins:
                assert gold == max(price, key=price.get)
            else:
                best = max(x["rating"] for x in s)
                assert [x["product"] for x in s if x["rating"] == best] == [gold]
        elif qid == "city":
            temps = {c: int(t.removesuffix("°C")) for c, t in s["forecast"].items()}
            assert gold == (max(temps, key=temps.get) if "warmest" in ins else min(temps, key=temps.get))
        elif qid == "author":
            years = {b["author"]: b["year"] for b in s["books"]}
            assert gold == (min(years, key=years.get) if "oldest" in ins else max(years, key=years.get))
        elif qid == "runner":
            times = {m.group(1): int(m.group(2)) * 60 + int(m.group(3)) for m in re.finditer(r"([^:;]+?) finished in (\d+):(\d+)", s.removeprefix("Race results: "))}
            times = {k.strip(): v for k, v in times.items()}
            assert gold == (min(times, key=times.get) if "first" in ins else max(times, key=times.get))
        n += 1
    assert n > 150


def test_priority_and_sentiment_labels(corpus):
    _, _, exs = corpus
    cue_sev = {c: i for i, cues in enumerate(SG.SEVERITY_CUES) for c in cues}
    n = 0
    for e in exs:
        task = e["meta"]["task"]
        text = json.dumps(e["state"], ensure_ascii=False)
        if task == "priority":
            sev = next(v for c, v in cue_sev.items() if c in text)
            k = len(e["questions"]["priority"]["criteria"])
            want = sev if k == 4 else {0: 0, 1: 1, 2: 1, 3: 2}[sev]
            assert e["labels"]["priority"]["level"] == want
            n += 1
        elif task == "sentiment":
            pos = sum(p in text for p in SG.POS)
            neg = sum(p in text for p in SG.NEG)
            kind = "mixed" if pos and neg else "positive" if pos else "negative" if neg else "neutral"
            if "stars" in e["labels"]:
                lvl = e["labels"]["stars"]["level"]
                assert {"positive": lvl >= 3, "negative": lvl <= 1, "neutral": lvl == 2, "mixed": lvl == 2}[kind]
            else:
                lab = e["labels"]["sentiment"]["label"]
                crit = e["questions"]["sentiment"]["criteria"]
                want = {"positive": "good", "negative": "bad", "neutral": "meh", "mixed": "meh"}[kind] if "good" in crit else kind
                assert lab == want, (e["id"], text)
            n += 1
    assert n > 300


def test_generate_is_deterministic(tmp_path):
    SG.generate(200, tmp_path / "a", seed=5)
    SG.generate(200, tmp_path / "b", seed=5)
    for s in ("train", "dev", "test"):
        assert (tmp_path / "a" / f"{s}.jsonl").read_bytes() == (tmp_path / "b" / f"{s}.jsonl").read_bytes()
