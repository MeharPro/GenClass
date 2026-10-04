"""bm-integrate: group-B adapters through the shared runner (`bridge_b`), the CLI for every spec, run.json fields.

Pure Python (no torch / transformers / datasets): a fake group-B adapter and a fake engine stand in for the real ones.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from jev_local.bench.benchmax import Spec, adapter_class, load_specs, runner_adapter_class
from jev_local.bench.benchmax import bridge_b as B
from jev_local.bench.benchmax import runner as R
from jev_local.bench.benchmax.adapters_b.common import AdapterB, Item
from jev_local.engine.base import EngineResult, RawDist
from jev_local.schema import ChoiceQuestion, NoulQuestion

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "transformers", "datasets")


def _load_cli():
    spec = importlib.util.spec_from_file_location("benchmax_cli_under_test", ROOT / "scripts" / "benchmax.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ fakes


class _FakeEngine:
    name = "fake-engine-0"
    max_tokens = 4096

    def evaluate(self, state, qs):
        dists = {}
        for qid, q in qs.items():
            if isinstance(q, NoulQuestion):
                dists[qid] = RawDist("noul", (0.7,))
            elif isinstance(q, ChoiceQuestion):
                labels = tuple(q.criteria.keys())
                k = len(labels)
                rest = 0.4 / (k - 1)
                dists[qid] = RawDist("choice", (0.6,) + (rest,) * (k - 1), labels)
            else:
                k = len(q.criteria)
                dists[qid] = RawDist("score", tuple(1.0 / k for _ in range(k)))
        return EngineResult(dists, 7, 3, self.name)


CHOICE = {"type": "choice", "instructions": "Which emotion?", "criteria": {"joy": "", "sad": "", "fear": ""}}
STAGE2 = {"type": "choice", "instructions": "Which one, again?", "criteria": {"a": None, "b": None}}
NOUL = {"type": "noul", "instructions": "Is it?"}


class _FakeB(AdapterB):
    """Two tasks, two stages (like zhuyansen BANKING77), one target row, a tuned variant."""

    SPEC_ID = "fake_b"
    TARGETS = ("T243",)
    SPLITS = ("validation", "test")
    TASKS = ("emo", "two")
    EVAL_SPLIT = "test"
    STAGES = 2

    def __init__(self, spec=None, args=None):
        super().__init__(spec, args)
        self.prepared: list[str] = []

    def prepare(self, split):
        self.prepared.append(split)
        return {"spec": self.SPEC_ID, "datasets": {"x": {"revision": "abcdef1234567890", "n": 3, "licence": "MIT"}},
                "repo": {"commit": "0123456789abcdef0123456789abcdef01234567"}, "note": "nothing downloaded"}

    def expected_counts(self, split):
        return {"emo": 3, "two": 1} if split == "test" else {}

    def items(self, split, limit=None, tasks=None):
        ts = self._select_tasks(tasks)
        out: list[Item] = []
        if "emo" in ts:
            for i in range(3):
                out.append(Item(f"emo/{split}/{i}", "emo", {"state": {"text": f"text {i}"}, "questions": {"q": CHOICE}},
                                gold="joy" if i < 2 else "sad", meta={"stage": 1}))
        if "two" in ts:
            out.append(Item(f"two/{split}/0", "two", {"state": "plain string state", "questions": {"n": NOUL}}, gold=True, meta={"stage": 1}))
        return out[:limit] if limit else out

    def stage_items(self, stage, items, answers):
        if stage != 2:
            return []
        return [Item(it.id + "/stage2", it.task, {"state": it.request["state"], "questions": {"q2": STAGE2}}, gold=it.gold, meta={"stage": 2})
                for it in items if it.task == "emo" and it.meta.get("stage") == 1 and answers.get(it.id, {}).get("ok")]

    def score(self, items, answers, split, thresholds=None):
        tasks = {}
        for task in sorted({it.task for it in items}):
            its = [it for it in items if it.task == task]
            ok = sum(1 for it in its if answers.get(it.id, {}).get("ok"))
            tasks[task] = {"n_items": len(its), "n_answered": ok, "target": "T243" if task == "emo" else None, "thresholds": thresholds}
        return {"spec": self.SPEC_ID, "split": split, "tasks": tasks, "per_item": [1, 2, 3]}

    def fit(self, items, answers):
        return {"emo": 0.42}


class _EvalOnlyB(_FakeB):
    SPEC_ID = "fake_eval_only"
    SPLITS = ("test",)
    STAGES = 1


def _spec(sid="fake_b"):
    return Spec(id=sid, suite="fake suite", adapter="tests.test_bm_integrate:_FakeB", harness={"repo": "fake"}, owner="b")


def _args(**kw):
    base = dict(split=None, limit=None, tasks=None, work=None, thresholds=None, fit=False, allow_test=False, skip_prepare=False, resume=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _ctx(tmp_path, args, sid="fake_b"):
    return R.RunContext(_spec(sid), tmp_path / "run", "meharsjev-68m", vars(args), precision="unrounded")


def _session(ctx, keep_first=5):
    return R.Session(R.InProcessClient(_FakeEngine(), "meharsjev-68m", overflow="truncate"), ctx, keep_first=keep_first)


# ------------------------------------------------------------------ registry / CLI


def test_runner_adapter_class_wraps_group_b_specs_only():
    specs = load_specs()
    assert len(specs) == 19
    for sid, s in specs.items():
        inner = adapter_class(s)
        cls = runner_adapter_class(s)
        for m in ("add_arguments", "verify", "run"):
            assert callable(getattr(cls, m)), (sid, m)
        if s.owner == "b":
            assert B.is_group_b_class(inner) and not B.is_group_b_class(cls)
            assert issubclass(cls, B.BridgedAdapter) and cls.inner_cls is inner
            assert cls.SPEC_ID == sid and cls.TARGETS == tuple(inner.TARGETS) and cls.SPLITS == tuple(inner.SPLITS)
            assert cls.default_overflow == "truncate" and cls.allow_truncate
            assert runner_adapter_class(s) is cls, "bridge classes are cached per inner class"
        else:
            assert cls is inner and not B.is_group_b_class(inner)
    # import hygiene of the bridge + CLI module, in a fresh interpreter (a full-suite process already holds torch)
    code = ("import sys, importlib.util\nfrom jev_local.bench.benchmax import load_specs, runner_adapter_class\n"
            "for s in load_specs().values():\n    runner_adapter_class(s)\n"
            f"print(','.join(h for h in {HEAVY!r} if h in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    assert out == "", f"imported at module level: {out}"


def test_cli_builds_help_for_every_spec_and_exposes_bridge_options(capsys):
    cli = _load_cli()
    for sid in load_specs():
        for cmd in ("run", "verify"):
            with pytest.raises(SystemExit) as e:
                cli.main([cmd, "--spec", sid, "--help"])
            assert e.value.code == 0, (cmd, sid)
    capsys.readouterr()
    with pytest.raises(SystemExit):
        cli.main(["run", "--spec", "zhuyansen_batch20", "--help"])
    out = capsys.readouterr().out
    for opt in ("--split", "--limit", "--tasks", "--work", "--thresholds", "--fit", "--allow-test", "--skip-prepare", "--resume"):
        assert opt in out, opt
    assert "zhuyansen_batch20 options" in out
    assert cli.main(["list"]) == 0
    assert "elcronos_plain" in capsys.readouterr().out


# ------------------------------------------------------------------ bridge helpers


def test_collect_revisions_finds_named_hex_leaves():
    man = {"datasets": {"x": {"revision": "abcdef1234567890", "n": 3, "licence": "MIT"}, "y": {"hub_sha": "deadbeefcafe", "name": "very long prose " * 10}},
           "repo": {"commit": "0123456789abcdef0123456789abcdef01234567"}, "files": [{"sha256": "ab" * 32}, {"path": "no"}],
           "branch_revision": "refs/convert/parquet"}
    revs = B.collect_revisions(man)
    assert revs["datasets.x.revision"] == "abcdef1234567890"
    assert revs["datasets.y.hub_sha"] == "deadbeefcafe"
    assert revs["repo.commit"].startswith("0123456789abcdef")
    assert revs["files[0].sha256"] == "ab" * 32
    assert revs["branch_revision"] == "refs/convert/parquet"
    assert "datasets.x.licence" not in revs and "datasets.x.n" not in revs and "files[1].path" not in revs


def test_answer_record_shapes_match_the_driver():
    ok = R.Outcome("ok", {"model": "meharsjev-68m", "answers": {"q": {"type": "noul", "noul": 0.7}}, "usage": {"input_tokens": 7}}, None, 200, 0.0123, True, 40)
    rec = B.answer_record("i1", ok)
    assert rec["ok"] and rec["answers"]["q"]["noul"] == 0.7 and rec["truncated"] is True and rec["tokens_cut"] == 40 and rec["ms"] == 12.3
    refused = R.Outcome("refused", None, R.refusal_body("context", model_id="meharsjev-68m", need=9000, limit=8192), 400, 0.001)
    rec2 = B.answer_record("i2", refused)
    assert not rec2["ok"] and rec2["permanent"] and rec2["refused"] and rec2["status"] == 400 and R.CONTEXT_MARKER in rec2["error"]
    err = R.Outcome("error", None, {"detail": "boom"}, 500, 0.001)
    rec3 = B.answer_record("i3", err)
    assert not rec3["ok"] and not rec3["refused"]
    assert B.answer_counts({"a": rec, "b": rec2, "c": rec3}) == {"requests": 3, "ok": 1, "truncated": 1, "failed": 2, "refused": 1}


# ------------------------------------------------------------------ bridge verify / run


def test_bridge_run_end_to_end_writes_run_json_items_answers_scores_and_fits(tmp_path):
    cls = B.bridge_class(_FakeB)
    args = _args(work=str(tmp_path / "w"), fit=True)
    ad = cls(_spec(), args)
    assert ad.inner.work == tmp_path / "w"
    ctx = _ctx(tmp_path, args)
    ver = ad.verify(ctx)
    assert ver["ok"] and ver["split"] == "validation" and ver["eval_split"] == "test" and ver["stages"] == 2
    assert ad.inner.prepared == ["validation"]
    assert ctx.dataset_revisions["datasets.x.revision"] == "abcdef1234567890" and ctx.dataset_revisions["repo.commit"].startswith("0123")
    ctx.harness = ver
    session = _session(ctx)
    res = ad.run(session, ctx)
    session.determinism_check()
    session.close()
    ctx.finish("complete")

    d = json.loads(ctx.path.read_text())
    assert d["schema"] == R.RUN_SCHEMA and d["status"] == "complete" and d["model_id"] == "meharsjev-68m"
    assert d["requests"] == {"sent": 7, "ok": 7, "refused": 0, "errors": 0, "truncated": 0}  # 3 emo + 1 two + 3 stage-2
    assert d["determinism"] == {"n_resent": 5, "mismatches": 0, "identical": True, "engine": "inprocess"}
    assert d["harness"]["prepare"]["note"] == "nothing downloaded" and d["dataset_revisions"]["datasets.x.revision"]
    assert d["results"]["latency"]["n"] == 7

    assert res["split"] == "validation" and res["comparable"] is False and "note" in res
    assert res["n_items"] == 7 and res["stages"] == 2 and res["resumed"] == 0
    assert res["coverage"] == {"requests": 7, "ok": 7, "truncated": 0, "failed": 0, "refused": 0}
    assert "per_item" not in res["scores"] and res["scores"]["tasks"]["emo"] == {"n_items": 6, "n_answered": 6, "target": "T243", "thresholds": None}
    (t,) = res["targets"]["emo"]
    assert t["target"] == "T243" and t["dataset"] and isinstance(t["jev_score"], float) and t["metric"] and "bar_z" in t and "two" not in res["targets"]
    multi = ad.target_table({"tasks": {"s7": {"target": ["T110", "T111"]}, "x": {"target": "T999"}, "none": {"target": None}}})
    assert [r["target"] for r in multi["s7"]] == ["T110", "T111"] and all("jev_score" in r for r in multi["s7"])
    assert multi["x"] == [{"target": "T999", "error": "not a targets.json id"}] and "none" not in multi
    assert res["thresholds_fitted"] and res["thresholds_fitted"].endswith("thresholds.json")

    run_dir = tmp_path / "run"
    items = [json.loads(l) for l in (run_dir / "items.jsonl").read_text().splitlines() if l.strip()]
    assert len(items) == 7 and sum(1 for it in items if it["meta"].get("stage") == 2) == 3 and all("gold" in it for it in items)
    answers = [json.loads(l) for l in (run_dir / "answers.jsonl").read_text().splitlines() if l.strip()]
    assert len(answers) == 7 and all(a["ok"] and a["answers"] for a in answers)
    emo = next(a for a in answers if a["id"] == "emo/validation/0")
    assert emo["answers"]["q"]["choice"] == "joy" and emo["answers"]["q"]["probabilities"]["joy"] == pytest.approx(0.6)
    assert json.loads((run_dir / "scores.json").read_text())["per_item"] == [1, 2, 3]
    assert json.loads((run_dir / "thresholds.json").read_text()) == {"emo": 0.42}

    # the driver's readers accept what the bridge wrote
    from jev_local.bench.benchmax.adapters_b.engine_client import load_answers, truncation_counts

    recs = load_answers(run_dir / "answers.jsonl")
    assert len(recs) == 7 and truncation_counts(recs.values()) == {"requests": 7, "truncated": 0, "failed": 0}


def test_bridge_refuses_the_evaluated_split_without_allow_test(tmp_path):
    cls = B.bridge_class(_FakeB)
    args = _args(split="test", work=str(tmp_path))
    ad = cls(_spec(), args)
    ctx = _ctx(tmp_path, args)
    with pytest.raises(SystemExit, match="evaluated split"):
        ad.run(_session(ctx), ctx)
    assert ctx.requests["sent"] == 0 and not (tmp_path / "run" / "answers.jsonl").exists()
    # --fit on the evaluated split is refused before anything is sent
    args2 = _args(split="test", allow_test=True, fit=True, work=str(tmp_path))
    ctx2 = _ctx(tmp_path, args2)
    with pytest.raises(SystemExit, match="fitted on non-evaluated"):
        cls(_spec(), args2).run(_session(ctx2), ctx2)
    assert ctx2.requests["sent"] == 0
    # with --allow-test and no subset the run is the comparable one
    args3 = _args(split="test", allow_test=True, work=str(tmp_path))
    ctx3 = _ctx(tmp_path, args3)
    res = cls(_spec(), args3).run(_session(ctx3), ctx3)
    assert res["comparable"] is True and "note" not in res and ctx3.requests["sent"] == 7


def test_bridge_default_split_and_eval_only_suites(tmp_path):
    cls = B.bridge_class(_FakeB)
    assert cls(_spec(), _args(work=str(tmp_path))).split() == "validation"
    with pytest.raises(SystemExit, match="unknown split"):
        cls(_spec(), _args(split="dev", work=str(tmp_path))).split()
    eo = B.bridge_class(_EvalOnlyB)
    ad = eo(_spec("fake_eval_only"), _args(work=str(tmp_path)))
    assert ad.split() == "test"
    ctx = _ctx(tmp_path, _args(work=str(tmp_path)), "fake_eval_only")
    ver = ad.verify(ctx)  # prepare on the evaluated split only verifies sources: allowed
    assert ver["ok"] and ver["expected_counts"] == {"emo": 3, "two": 1}
    with pytest.raises(SystemExit, match="evaluated split"):
        ad.run(_session(ctx), ctx)


def test_bridge_limit_tasks_and_resume(tmp_path):
    cls = B.bridge_class(_FakeB)
    args = _args(work=str(tmp_path), limit=2, tasks="emo")
    ad = cls(_spec(), args)
    ctx = _ctx(tmp_path, args)
    res = ad.run(_session(ctx), ctx)
    assert res["tasks"] == ["emo"] and res["limit"] == 2 and res["n_items"] == 4 and ctx.requests["sent"] == 4
    assert "two" not in res["scores"]["tasks"]
    # resume: the ok answers already on disk are not re-sent
    args2 = _args(work=str(tmp_path), limit=2, tasks="emo", resume=True)
    ctx2 = _ctx(tmp_path, args2)
    res2 = cls(_spec(), args2).run(_session(ctx2), ctx2)
    assert res2["resumed"] == 4 and ctx2.requests["sent"] == 0 and res2["coverage"]["ok"] == 4
    # without --resume the answers file starts fresh
    ctx3 = _ctx(tmp_path, args)
    cls(_spec(), args).run(_session(ctx3), ctx3)
    assert ctx3.requests["sent"] == 4
    assert len((tmp_path / "run" / "answers.jsonl").read_text().splitlines()) == 4


def test_bridge_records_refusals_and_failures_for_the_scorer(tmp_path):
    class _Refusing(_FakeEngine):
        def evaluate(self, state, qs):
            from jev_local.engine.base import EngineError

            if isinstance(state, dict) and state.get("text") == "text 1":
                raise EngineError({"detail": [{"type": R.MAX_TOKENS_EXCEEDED, "msg": "too long"}], "tokens": 9000, "max_tokens": 8192}, status=400)
            return super().evaluate(state, qs)

    cls = B.bridge_class(_FakeB)
    args = _args(work=str(tmp_path))
    ctx = _ctx(tmp_path, args)
    session = R.Session(R.InProcessClient(_Refusing(), "meharsjev-68m", overflow="refuse"), ctx)
    res = cls(_spec(), args).run(session, ctx)
    # text 1 refused at stage 1 -> no stage-2 item for it: 3 + 1 + 2 requests
    assert ctx.requests == {"sent": 6, "ok": 5, "refused": 1, "errors": 0, "truncated": 0}
    assert ctx.overflow["refused"] == 1
    assert res["coverage"] == {"requests": 6, "ok": 5, "truncated": 0, "failed": 1, "refused": 1}
    assert res["scores"]["tasks"]["emo"] == {"n_items": 5, "n_answered": 4, "target": "T243", "thresholds": None}
