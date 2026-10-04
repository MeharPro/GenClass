"""Trainer v2: stream mode (mixture over raw/<bucket>/<source>.jsonl.zst), v2 losses with consistency pairs,
resume, --init-from, and DDP over torchrun (2 local gloo ranks): equivalence with single-process gradient
accumulation, graceful SIGTERM on all ranks, and resume after it."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from jev_local.train import fixture
from jev_local.train.train import main as train_main

pytestmark = [pytest.mark.model, pytest.mark.slow]

COMMON = ["--lr", "1e-4", "--head-lr", "3e-3", "--device", "cpu", "--log-every", "1", "--warmup", "0.05",
          "--threads", "4"]


def _log(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "log.jsonl").read_text().splitlines() if x.strip()]


def _v2_rows(n: int, seed: int, bucket: str, source: str, split: str = "train") -> list[dict]:
    rows = []
    for ex in fixture.make_examples(n, seed=seed, n_elements=8):
        ex = dict(ex)
        ex["questions"] = dict(ex["questions"])
        ex["questions"]["incomplete"] = {"type": "noul", "instructions": "Is the command still incomplete?"}
        ex["labels"] = dict(ex["labels"])
        ex["labels"]["incomplete"] = {"type": "noul", "p": 1.0 - ex["labels"]["complete"]["p"]}
        ex["meta"] = {**ex["meta"], "pairs": [{"kind": "neg", "a": "complete", "b": "incomplete"}]}
        ex.update(split=split, bucket=bucket, source=source, license_use="commercial", variant="", group_id="")
        rows.append(ex)
    return rows


def _write_zst(path: Path, rows: list[dict]) -> None:
    import zstandard

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstandard.ZstdCompressor().compress("".join(json.dumps(r) + "\n" for r in rows).encode()))


@pytest.fixture(scope="module")
def v2root(tmp_path_factory) -> Path:
    pytest.importorskip("zstandard")
    root = tmp_path_factory.mktemp("v2")
    _write_zst(root / "raw" / "b9_cu" / "fx_a.jsonl.zst", _v2_rows(12, 0, "b9_cu", "fx_a")
               + _v2_rows(4, 50, "b9_cu", "fx_a", split="dev_mix"))
    _write_zst(root / "raw" / "s0_families" / "fx_b.jsonl.zst", _v2_rows(8, 100, "s0_families", "fx_b"))
    return root


def _stream_args(root: Path, tmp: Path, name: str, *extra: str) -> list[str]:
    return ["--stream", str(root), "--stream-cache", str(tmp / "cache"), "--out", str(tmp / name),
            "--runs-dir", str(tmp / "runs"), "--max-len", "1024", *COMMON, *extra]


@pytest.fixture(scope="module")
def stream_smoke(tmp_path_factory, v2root) -> dict:
    tmp = tmp_path_factory.mktemp("smoke")
    train_main(_stream_args(v2root, tmp, "model", "--batch-rows", "4", "--grad-accum", "1", "--max-steps", "20",
                            "--ckpt-every", "10", "--loss", "v2", "--prefetch", "2"))
    return {"tmp": tmp, "out": tmp / "model", "run": tmp / "runs" / "model", "log": _log(tmp / "runs" / "model")}


def test_stream_smoke_v2_losses(stream_smoke):
    log = stream_smoke["log"]
    assert [r["step"] for r in log] == list(range(1, 21))
    first = sum(r["loss"] for r in log[:5]) / 5
    last = sum(r["loss"] for r in log[-5:]) / 5
    assert last < 0.8 * first, (first, last)
    assert "loss_x_cons" in log[-1] and log[-1]["n_x_cons"] > 0  # consistency pairs reached the loss
    plan = json.loads((stream_smoke["run"] / "mixture_plan.json").read_text())
    assert plan["corpus"]["rows"] == 24 and plan["plan"]["config"]["batch_rows"] == 4
    assert set(plan["plan"]["by_bucket"]) == {"b9_cu", "s0_families"}
    meta = json.loads((stream_smoke["out"] / "meta.json").read_text())
    assert meta["trained"] and meta["step"] == 20 and meta["final"]
    info = json.loads((stream_smoke["run"] / "ckpt" / "trainer.json").read_text())
    assert info["data_mode"] == "stream" and info["world"] == 1


def test_stream_resume_matches_uninterrupted(stream_smoke, v2root, tmp_path):
    args = _stream_args(v2root, tmp_path, "model", "--batch-rows", "4", "--grad-accum", "1", "--max-steps", "20",
                        "--ckpt-every", "10", "--loss", "v2", "--prefetch", "2")
    with pytest.raises(SystemExit) as ei:
        train_main([*args, "--stop-at", "7"])
    assert ei.value.code == 130
    train_main([*args, "--resume"])
    log = _log(tmp_path / "runs" / "model")
    assert [r["step"] for r in log] == list(range(1, 21))
    a = load_file(str(tmp_path / "model" / "heads.safetensors"))
    b = load_file(str(stream_smoke["out"] / "heads.safetensors"))
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-4, rtol=1e-3), k
    for r, s in zip(log[7:], stream_smoke["log"][7:]):
        assert r["loss"] == pytest.approx(s["loss"], rel=1e-2, abs=1e-3)


def test_stream_token_budget_and_init_from(stream_smoke, v2root, tmp_path):
    train_main(_stream_args(v2root, tmp_path, "tok", "--batch-tokens", "2048", "--grad-accum", "2", "--max-steps", "3",
                            "--init-from", str(stream_smoke["out"]), "--max-len", "2048"))
    log = _log(tmp_path / "runs" / "tok")
    assert len(log) == 3 and log[0]["tokens_per_s"] > 0
    # init-from starts from trained weights: the first loss is already low
    assert log[0]["loss"] < stream_smoke["log"][0]["loss"]


# ---------------------------------------------------------------------------- DDP (torchrun, 2 local ranks)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _torchrun(args: list[str], nproc: int = 2, **popen) -> subprocess.Popen:
    env = {**os.environ, "OMP_NUM_THREADS": "2", "GLOO_SOCKET_IFNAME": "lo"}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc-per-node", str(nproc),
           "--master-addr", "127.0.0.1", "--master-port", str(_free_port()), "-m", "jev_local.train.train", "--ddp", *args]
    return subprocess.Popen(cmd, env=env, cwd=Path(__file__).resolve().parents[1], start_new_session=True, **popen)


@pytest.fixture(scope="module")
def cu_data(tmp_path_factory) -> Path:
    return fixture.write_splits(tmp_path_factory.mktemp("cu") / "cu", n_train=32, n_dev=8, seed=0, n_elements=8)


def _index_args(data: Path, tmp: Path, name: str, *extra: str) -> list[str]:
    return ["--data", str(data), "--out", str(tmp / name), "--runs-dir", str(tmp / "runs"), "--max-len", "1024",
            "--batch", "2", "--lr", "1e-4", "--head-lr", "3e-3", "--device", "cpu", "--log-every", "1",
            "--warmup", "0.05", "--threads", "2", *extra]


def test_ddp_matches_single_process_accumulation(cu_data, tmp_path):
    # 2 ranks x grad_accum 1 consume global micro-batches (0,1), (2,3), ... exactly like 1 rank x grad_accum 2
    p = _torchrun(_index_args(cu_data, tmp_path, "ddp", "--grad-accum", "1", "--max-steps", "6", "--epochs", "1"),
                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = p.communicate(timeout=600)
    assert p.returncode == 0, out.decode()[-3000:]
    train_main(_index_args(cu_data, tmp_path, "single", "--grad-accum", "2", "--max-steps", "6", "--epochs", "1"))
    a = load_file(str(tmp_path / "ddp" / "heads.safetensors"))
    b = load_file(str(tmp_path / "single" / "heads.safetensors"))
    for k in a:
        assert torch.allclose(a[k], b[k], atol=2e-5, rtol=1e-4), (k, (a[k] - b[k]).abs().max())
    la, lb = _log(tmp_path / "runs" / "ddp"), _log(tmp_path / "runs" / "single")
    assert [r["step"] for r in la] == [r["step"] for r in lb] == list(range(1, 7))
    assert la[-1]["world"] == 2
    for r, s in zip(la, lb):
        assert r["loss"] == pytest.approx(s["loss"], rel=1e-4)
        assert r["examples_per_s"] > 0 and r["n_choice"] == s["n_choice"]
    info = json.loads((tmp_path / "runs" / "ddp" / "ckpt" / "trainer.json").read_text())
    assert info["world"] == 2 and info["cursor"] == 12 and info["step"] == 6


def test_ddp_sigterm_checkpoints_on_all_ranks_and_resumes(cu_data, tmp_path):
    args = _index_args(cu_data, tmp_path, "m", "--grad-accum", "2", "--max-steps", "60", "--ckpt-every", "1000")
    log_path = tmp_path / "out.txt"
    with log_path.open("wb") as fh:
        p = _torchrun(args, stdout=fh, stderr=subprocess.STDOUT)
        run_log = tmp_path / "runs" / "m" / "log.jsonl"
        t0 = time.time()
        while time.time() - t0 < 300:
            if run_log.is_file() and len(run_log.read_text().splitlines()) >= 3:
                break
            if p.poll() is not None:
                break
            time.sleep(0.2)
        assert p.poll() is None, log_path.read_text()[-3000:]
        os.killpg(p.pid, signal.SIGTERM)  # like `pkill`: torchrun and both workers get SIGTERM
        p.wait(timeout=120)
    text = log_path.read_text()
    info = json.loads((tmp_path / "runs" / "m" / "ckpt" / "trainer.json").read_text())
    assert 3 <= info["step"] < 60, text[-2000:]
    # 2 ranks x grad_accum 2 = 4 global micro-batches per step, 16 per epoch: no partial step leaked in
    assert info["cursor"] == ((info["step"] - 1) % 4 + 1) * 4 and info["epoch"] == (info["step"] - 1) // 4
    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    assert meta["step"] == info["step"] and meta["final"] is False
    # resume (different world size on purpose: the cursor counts global micro-batches)
    train_main([*args, "--resume"])
    info2 = json.loads((tmp_path / "runs" / "m" / "ckpt" / "trainer.json").read_text())
    assert info2["step"] == 60 and info2["world"] == 1
