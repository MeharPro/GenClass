"""Registry: lazy construction by import path, graceful absence, status, env override."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

from jev_local.engine import registry as reg
from jev_local.engine.heuristic import HeuristicEngine
from jev_local.engine.registry import Registry, default_fast_ckpt


class StubFast:
    name = "jev-local-fast-0.1.0"
    max_tokens = 2048
    built: list[Path] = []

    def __init__(self, ckpt_dir: Path):
        StubFast.built.append(ckpt_dir)
        self.ckpt_dir = ckpt_dir
        self.warmed = 0

    def warmup(self) -> None:
        self.warmed += 1

    def supports(self, req):
        return True

    def count_tokens(self, req):
        return 1

    def evaluate(self, state, questions):
        raise NotImplementedError


class StubGeneral:
    name = "jev-local-general-0.1.0"
    max_tokens = 8192

    def __init__(self, repo: str):
        self.repo = repo
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False


class Boom:
    def __init__(self, *a):
        raise RuntimeError("corrupt checkpoint")


@pytest.fixture
def stub_modules(monkeypatch):
    mod = types.ModuleType("jev_test_stub_engines")
    mod.StubFast, mod.StubGeneral, mod.Boom = StubFast, StubGeneral, Boom
    monkeypatch.setitem(sys.modules, "jev_test_stub_engines", mod)
    monkeypatch.setattr(reg, "FAST_IMPORT", ("jev_test_stub_engines", "StubFast"))
    monkeypatch.setattr(reg, "GENERAL_IMPORT", ("jev_test_stub_engines", "StubGeneral"))
    StubFast.built.clear()
    return mod


@pytest.fixture
def ckpt(tmp_path) -> Path:
    d = tmp_path / "jev-local-fast"
    (d / "backbone").mkdir(parents=True)
    (d / "heads.safetensors").write_bytes(b"")
    return d


def test_missing_checkpoint_and_general_off_leave_only_heuristic(tmp_path, stub_modules):
    r = Registry(fast_ckpt=tmp_path / "nope", load_general="off")
    es = r.engines()
    assert list(es) == ["heuristic"] and isinstance(es["heuristic"], HeuristicEngine)
    assert r.notes["fast"].startswith("missing") and r.notes["general"] == "off"
    assert r.status() == {"fast": "missing", "general": "off", "heuristic": "loaded"}
    assert StubFast.built == []


def test_lazy_construction(ckpt, stub_modules):
    r = Registry(fast_ckpt=ckpt, load_general="off")
    assert not r.ready and StubFast.built == []  # nothing built at construction
    es = r.engines()
    assert r.ready and StubFast.built == [ckpt]
    assert list(es) == ["fast", "heuristic"]
    assert r.engines() is es  # built once


def test_fast_constructor_error_is_not_fatal(ckpt, stub_modules, monkeypatch):
    monkeypatch.setattr(reg, "FAST_IMPORT", ("jev_test_stub_engines", "Boom"))
    r = Registry(fast_ckpt=ckpt, load_general="off")
    assert list(r.engines()) == ["heuristic"]
    assert r.notes["fast"].startswith("error: RuntimeError") and r.status()["fast"] == "error"


def test_load_fast_false(ckpt, stub_modules):
    r = Registry(fast_ckpt=ckpt, load_general="off", load_fast=False)
    assert "fast" not in r.engines() and r.status()["fast"] == "off"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("JEV_LOCAL_FAST_CKPT", str(tmp_path / "x"))
    assert default_fast_ckpt() == tmp_path / "x"
    assert Registry(load_general="off").fast_ckpt == tmp_path / "x"
    monkeypatch.delenv("JEV_LOCAL_FAST_CKPT")
    assert default_fast_ckpt() == reg.PROJECT_ROOT / "models" / "jev-local-fast"


def _needs_mlx() -> None:
    if importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_lm") is None:
        pytest.skip("mlx / mlx-lm not installed: the registry reports the general engine missing before the cache check")


def test_general_absent_when_model_not_cached(stub_modules, monkeypatch, tmp_path):
    _needs_mlx()
    monkeypatch.setattr(reg, "general_model_cached", lambda repo: False)
    r = Registry(fast_ckpt=tmp_path / "nope", general_repo="someone/not-downloaded")
    assert "general" not in r.engines()
    assert "not in the Hugging Face cache" in r.notes["general"]


def test_general_lazy_then_eager(stub_modules, monkeypatch, tmp_path):
    _needs_mlx()
    monkeypatch.setattr(reg, "general_model_cached", lambda repo: True)
    lazy = Registry(fast_ckpt=tmp_path / "nope", general_repo="org/model", load_general="lazy")
    es = lazy.engines()
    assert list(es) == ["general", "heuristic"] and es["general"].repo == "org/model"
    lazy.warm()
    assert lazy.status()["general"] == "lazy"  # lazy: warm() does not load the decoder

    eager = Registry(fast_ckpt=tmp_path / "nope", general_repo="org/model", load_general="eager")
    eager.warm()
    assert eager.status()["general"] == "loaded"
    eager.unload()
    assert not eager.ready


def test_warm_runs_fast_warmup(ckpt, stub_modules):
    r = Registry(fast_ckpt=ckpt, load_general="off")
    r.warm()
    assert r.engines()["fast"].warmed == 1


def test_general_model_cached_local_dir(tmp_path):
    assert reg.general_model_cached(str(tmp_path))
    assert not reg.general_model_cached("jev-local-tests/definitely-not-a-cached-repo")


def test_importing_registry_does_not_import_torch_or_mlx():
    code = "import sys, jev_local.engine.registry, jev_local.server.app; print(sorted(m for m in ('torch','mlx','transformers') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
