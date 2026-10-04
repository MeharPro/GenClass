"""Which engines this process can serve, constructed lazily by import path.

Keys: "fast" (trained encoder, needs a checkpoint dir), "general" (MLX decoder scorer, needs mlx
and the model in the Hugging Face cache) and "heuristic" (always present). A missing checkpoint,
a missing mlx install or an uncached model leaves that engine out; that is not an error, it just
narrows what the auto route can pick. Engine modules are imported only when `engines()` is first
called, so importing the registry never pulls in torch or mlx.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import threading
from pathlib import Path
from typing import Any, Literal

from jev_local.engine.base import Engine
from jev_local.engine.heuristic import HeuristicEngine

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAST_CKPT = PROJECT_ROOT / "models" / "jev-local-fast"
FAST_CKPT_ENV = "JEV_LOCAL_FAST_CKPT"
DEFAULT_GENERAL_REPO = "mlx-community/Qwen3-1.7B-4bit"

FAST_IMPORT = ("jev_local.engine.encoder.engine", "FastEngine")
GENERAL_IMPORT = ("jev_local.engine.decoder.mlx_scorer", "GeneralEngine")

LoadMode = Literal["lazy", "eager", "off"]


def default_fast_ckpt() -> Path:
    env = os.environ.get(FAST_CKPT_ENV, "").strip()
    return Path(env).expanduser() if env else DEFAULT_FAST_CKPT


def _import(path: tuple[str, str]) -> Any:
    module, name = path
    return getattr(importlib.import_module(module), name)


def general_model_cached(repo: str) -> bool:
    """True if `repo` is a local model dir or is already in the HF cache (no network access)."""
    if Path(repo).expanduser().is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    cfg = try_to_load_from_cache(repo, "config.json")
    if not isinstance(cfg, str):
        return False
    return any(Path(cfg).parent.glob("*.safetensors"))


def is_loaded(engine: object) -> bool:
    """Best-effort 'are the weights in memory' for engines that load lazily."""
    for attr in ("loaded", "is_loaded"):
        v = getattr(engine, attr, None)
        if isinstance(v, bool):
            return v
        if callable(v):
            try:
                return bool(v())
            except Exception:
                return False
    for attr in ("model", "_model"):
        if hasattr(engine, attr):
            return getattr(engine, attr) is not None
    return True  # engines without a lazy phase are loaded once constructed


class Registry:
    def __init__(
        self,
        fast_ckpt: Path | None = None,
        general_repo: str | None = None,
        load_general: LoadMode = "lazy",
        *,
        load_fast: bool = True,
        fast_options: dict[str, Any] | None = None,
    ):
        """`fast_ckpt=None` means $JEV_LOCAL_FAST_CKPT or models/jev-local-fast;
        `general_repo=None` means mlx-community/Qwen3-1.7B-4bit. `load_general="off"` and
        `load_fast=False` disable an engine outright.
        `fast_options` are keyword arguments for the fast engine's constructor, chosen by the run config
        (benchmax): `calibration` (path, W9), `max_tokens`, `max_flat_tokens`, `device`, `threads`,
        `drop_header_calibration`. None values are dropped, so the engine keeps its own defaults."""
        self.fast_ckpt = Path(fast_ckpt).expanduser() if fast_ckpt is not None else default_fast_ckpt()
        self.general_repo = general_repo or DEFAULT_GENERAL_REPO
        self.load_general: LoadMode = load_general
        self.load_fast = load_fast
        self.fast_options = {k: v for k, v in (fast_options or {}).items() if v is not None}
        self._engines: dict[str, Engine] | None = None
        self.notes: dict[str, str] = {}  # key -> why it is absent / how it was built
        self._lock = threading.Lock()

    # ------------------------------------------------------------ building

    @property
    def ready(self) -> bool:
        """Engines are built, so `engines()` returns immediately."""
        return self._engines is not None

    def engines(self) -> dict[str, Engine]:
        with self._lock:
            if self._engines is None:
                self._engines = self._build()
            return self._engines

    def _build(self) -> dict[str, Engine]:
        out: dict[str, Engine] = {}
        fast = self._build_fast()
        if fast is not None:
            out["fast"] = fast
        general = self._build_general()
        if general is not None:
            out["general"] = general
        out["heuristic"] = HeuristicEngine()
        self.notes["heuristic"] = "built in"
        return out

    def _build_fast(self) -> Engine | None:
        if not self.load_fast:
            self.notes["fast"] = "off"
            return None
        ckpt = self.fast_ckpt
        if not (ckpt.is_dir() and (ckpt / "heads.safetensors").is_file()):
            self.notes["fast"] = f"missing: no checkpoint at {ckpt}"
            return None
        try:
            engine = _import(FAST_IMPORT)(ckpt, **self.fast_options) if self.fast_options else _import(FAST_IMPORT)(ckpt)
        except Exception as e:  # a broken checkpoint must not take the server down
            log.warning("fast engine unavailable: %s", e, exc_info=True)
            self.notes["fast"] = f"error: {type(e).__name__}: {e}"
            return None
        self.notes["fast"] = f"loaded from {ckpt}" + (f" with {sorted(self.fast_options)}" if self.fast_options else "")
        return engine

    # ------------------------------------------------------------ run.json facts

    def describe(self) -> dict[str, Any]:
        """Checkpoint path, its sha256 manifest, calibration in force and engine limits, for run.json
        (PLAN §6: checkpoint sha256, calibration file). Pure Python; engines are built if needed."""
        engines = self.engines()
        out: dict[str, Any] = {"fast_ckpt": str(self.fast_ckpt), "fast_options": dict(self.fast_options), "notes": dict(self.notes)}
        fast = engines.get("fast")
        if fast is not None:
            out["fast"] = {
                "name": getattr(fast, "name", None),
                "max_tokens": getattr(fast, "max_tokens", None),
                "max_flat_tokens": getattr(fast, "max_flat_tokens", None),
                "device": str(getattr(fast, "device", "")),
                "dtype": str(getattr(fast, "dtype", "")),
                "calibration": getattr(fast, "calibration_info", None),
                "meta": {k: v for k, v in (getattr(fast, "meta", None) or {}).items() if k in ("name", "base", "max_len", "run", "step", "final", "written_at")},
                "checkpoint_sha256": checkpoint_sha256(self.fast_ckpt),
            }
        return out

    def _build_general(self) -> Engine | None:
        if self.load_general == "off":
            self.notes["general"] = "off"
            return None
        if importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_lm") is None:
            self.notes["general"] = "missing: mlx / mlx-lm not installed"
            return None
        if not general_model_cached(self.general_repo):
            # Never download from here: a first request must not silently pull ~1 GB.
            self.notes["general"] = f"missing: {self.general_repo} is not in the Hugging Face cache"
            return None
        try:
            engine = _import(GENERAL_IMPORT)(self.general_repo)
        except Exception as e:
            log.warning("general engine unavailable: %s", e, exc_info=True)
            self.notes["general"] = f"error: {type(e).__name__}: {e}"
            return None
        self.notes["general"] = f"{self.load_general}: {self.general_repo}"
        return engine

    # ------------------------------------------------------------ lifecycle

    def warm(self) -> None:
        """Build engines and do the startup work that should not land on the first request:
        MPS kernel warmup for the fast engine, and loading the decoder when eager."""
        engines = self.engines()
        fast = engines.get("fast")
        if fast is not None and hasattr(fast, "warmup"):
            try:
                fast.warmup()
            except Exception as e:
                log.warning("fast engine warmup failed: %s", e)
        general = engines.get("general")
        if general is not None and self.load_general == "eager" and not is_loaded(general):
            self._load_general(general)

    @staticmethod
    def _load_general(engine: Engine) -> None:
        for attr in ("load", "ensure_loaded", "_ensure_loaded", "warmup"):
            fn = getattr(engine, attr, None)
            if callable(fn):
                fn()
                return
        # No explicit loader: a one-question evaluate triggers the lazy load.
        from jev_local.schema import NoulQuestion

        engine.evaluate("warmup", {"q": NoulQuestion(instructions="Is this a warmup?")})

    def status(self) -> dict[str, str]:
        """key -> "loaded" | "lazy" | "off" | "missing" | "error", for /healthz."""
        engines = self.engines()
        out: dict[str, str] = {}
        for key in ("fast", "general", "heuristic"):
            e = engines.get(key)
            if e is not None:
                out[key] = "loaded" if is_loaded(e) else "lazy"
            else:
                out[key] = self.notes.get(key, "missing").split(":", 1)[0]
        return out

    def unload(self) -> None:
        with self._lock:
            for e in (self._engines or {}).values():
                fn = getattr(e, "unload", None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as err:
                        log.warning("unload failed for %s: %s", getattr(e, "name", e), err)
            self._engines = None


def checkpoint_sha256(ckpt: Path) -> dict[str, str] | None:
    """sha256 of the weight and calibration files of a fast checkpoint (None when it is not one)."""
    import hashlib

    files = ("heads.safetensors", "backbone/model.safetensors", "calibration.json", "meta.json")
    out: dict[str, str] = {}
    for rel in files:
        p = ckpt / rel
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[rel] = h.hexdigest()
    return out or None
