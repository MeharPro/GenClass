"""Trainer smoke test (CONTRACT "B": 20 steps on 16 generated examples, the loss decreases), resume
after an interruption, init-only export, and the eval/calibrate pipeline on the result."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from jev_local.train import fixture
from jev_local.train.train import main as train_main

pytestmark = [pytest.mark.model, pytest.mark.slow]

COMMON = ["--batch", "4", "--grad-accum", "1", "--max-len", "1024", "--lr", "1e-4", "--head-lr", "3e-3",
          "--device", "cpu", "--log-every", "1", "--warmup", "0.05"]


def _log(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "log.jsonl").read_text().splitlines() if x.strip()]


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> Path:
    return fixture.write_splits(tmp_path_factory.mktemp("data") / "cu", n_train=16, n_dev=16, seed=0, n_elements=8)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory, data) -> dict:
    root = tmp_path_factory.mktemp("smoke")
    out, runs = root / "model", root / "runs"
    train_main(["--data", str(data), "--out", str(out), "--runs-dir", str(runs), "--max-steps", "20",
                "--ckpt-every", "10", *COMMON])
    return {"out": out, "run": runs / "model", "log": _log(runs / "model")}


def test_smoke_loss_decreases(smoke):
    log = smoke["log"]
    assert [r["step"] for r in log] == list(range(1, 21))
    first = sum(r["loss"] for r in log[:4]) / 4  # one pass over the 16 examples
    last = sum(r["loss"] for r in log[-4:]) / 4
    assert last < 0.8 * first, (first, last)
    for key in ("loss_choice", "loss_noul", "steps_per_s", "tokens_per_s", "examples_per_s", "rss_peak_mb", "lr"):
        assert key in log[-1], key
    assert log[-1]["dropped_qids"] == 0 and log[-1]["bad_labels"] == 0


def test_smoke_writes_servable_checkpoint(smoke):
    from jev_local.engine.encoder.engine import FastEngine

    out = smoke["out"]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["trained"] is True and meta["step"] == 20 and meta["final"] is True
    trainer = json.loads((smoke["run"] / "ckpt" / "trainer.json").read_text())
    assert trainer["step"] == 20
    eng = FastEngine(out, device="cpu", dtype=torch.float32)
    ex = fixture.make_examples(1, seed=5, n_elements=8)[0]
    from jev_local.schema import question_from_json

    res = eng.evaluate(ex["state"], {k: question_from_json(v) for k, v in ex["questions"].items()})
    assert set(res.dists) == set(ex["questions"])


def test_resume_after_interrupt_matches_uninterrupted(tmp_path, data, smoke):
    out, runs = tmp_path / "model", tmp_path / "runs"
    args = ["--data", str(data), "--out", str(out), "--runs-dir", str(runs), "--max-steps", "20",
            "--ckpt-every", "10", *COMMON]
    with pytest.raises(SystemExit) as ei:
        train_main([*args, "--stop-at", "7"])  # interrupted mid-epoch, between checkpoints
    assert ei.value.code == 130
    info = json.loads((runs / "model" / "ckpt" / "trainer.json").read_text())
    assert info["step"] == 7
    assert json.loads((out / "meta.json").read_text())["final"] is False
    train_main([*args, "--resume"])
    log = _log(runs / "model")
    assert [r["step"] for r in log] == list(range(1, 21))
    # The resumed run continues the same data order, schedule and optimizer state: same weights.
    a = load_file(str(out / "heads.safetensors"))
    b = load_file(str(smoke["out"] / "heads.safetensors"))
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-4, rtol=1e-3), k
    # and the resumed losses retrace the uninterrupted run
    for r, s in zip(log[7:], smoke["log"][7:]):
        assert r["loss"] == pytest.approx(s["loss"], rel=1e-2, abs=1e-3)


def test_init_only(tmp_path):
    out = tmp_path / "init"
    train_main(["--init-only", "--out", str(out), "--runs-dir", str(tmp_path / "runs")])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["trained"] is False
    assert (out / "backbone" / "model.safetensors").is_file() and (out / "heads.safetensors").is_file()
    assert not (tmp_path / "runs").exists()  # nothing to log for an init-only export


def test_eval_and_calibrate_cli(smoke, data, tmp_path):
    from jev_local.engine.encoder import calibrate
    from jev_local.train import eval as ev

    ckpt = tmp_path / "ckpt"
    import shutil

    shutil.copytree(smoke["out"], ckpt)
    out = tmp_path / "eval_dev.json"
    ev.main(["--ckpt", str(ckpt), "--data", str(data), "--split", "dev", "--device", "cpu", "--out", str(out)])
    m = json.loads(out.read_text())
    assert m["n_examples"] == 16
    for qid in ("intent", "complete", "is_command", "destructive", "target", "app", "key", "folder"):
        assert qid in m["by_qid"], qid
    assert 0.0 <= m["by_qid"]["intent"]["acc"] <= 1.0
    assert {"intent_acc_complete", "intent_acc_prefix", "target_top1_nonnone"} <= set(m["harness"])
    for kind in ("choice", "noul"):
        assert m["by_kind"][kind]["ece"] >= 0.0

    calibrate.main(["--ckpt", str(ckpt), "--data", str(data), "--split", "dev", "--device", "cpu",
                    "--min-per-header", "4"])
    cal = json.loads((ckpt / "calibration.json").read_text())
    assert all(calibrate.TAU_MIN <= cal[k] <= calibrate.TAU_MAX for k in ("noul", "choice", "score"))
    assert len(cal["by_header"]) >= 4  # intent, target, complete, is_command, ... each seen 16 times
    assert cal["_fit"]["choice"]["nll_after"] <= cal["_fit"]["choice"]["nll_before"] + 1e-9
