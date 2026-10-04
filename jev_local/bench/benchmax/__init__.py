"""benchmax: run meharsjev through every publisher's own harness (docs/research/benchmax/PLAN.md §6).

A *spec* is one row of `targets.json` `spec_id` (e.g. `deusser_exact@6bbdeb33`) bound to an *adapter* that
reproduces that suite's protocol byte for byte: the harness's pinned commit or dataset revision, its request
shapes, its scorer and its CIs. Adapters never paraphrase a request; they import or rebuild the publisher's
own bytes. `runner.py` holds what every adapter shares: the engine clients (in-process checkpoint, HTTP
server, heuristic), the wire answers (W1–W3), refusal bodies (W4), the overflow switch (W5), the run.json log
(PLAN §6 "every harness run logs ...") and the determinism report (W11).

Plugin registry: `load_specs()` imports `specs_a` (this agent's four suites) and `specs_b` when that module is
present (the other adapter agent's suites). Nothing here imports torch, datasets or transformers at module
level, so the registry and the CLI are importable on the Mac; the heavy clients import lazily.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

SPECS: dict[str, "Spec"] = {}
_LOADED: set[str] = set()
PLUGIN_MODULES = ("jev_local.bench.benchmax.specs_a", "jev_local.bench.benchmax.specs_b")


@dataclass(frozen=True)
class Spec:
    id: str  # exactly the `spec_id` string of bench/public/targets.json
    suite: str  # human name
    adapter: str  # "package.module:ClassName", imported lazily
    harness: dict[str, Any] = field(default_factory=dict)  # repo / commit / hf_id / revision / licence
    verification: str = ""  # PLAN §6 "verification hook"
    counted_rows: str = ""  # PLAN §6 "counted rows"
    description: str = ""
    owner: str = ""  # which adapter agent (a | b)


def register(spec: Spec) -> Spec:
    prev = SPECS.get(spec.id)
    if prev is not None and prev != spec:
        raise ValueError(f"spec {spec.id!r} registered twice with different definitions")
    SPECS[spec.id] = spec
    return spec


def load_specs() -> dict[str, Spec]:
    """Import every plugin module once. A missing `specs_b` is fine; a broken one is not."""
    for mod in PLUGIN_MODULES:
        if mod in _LOADED:
            continue
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name != mod:  # the plugin exists but one of ITS imports is missing: surface it
                raise
            continue
        _LOADED.add(mod)
    return SPECS


def get_spec(spec_id: str) -> Spec:
    specs = load_specs()
    if spec_id not in specs:
        raise KeyError(f"unknown spec {spec_id!r}; known: {', '.join(sorted(specs))}")
    return specs[spec_id]


def make_adapter(spec: Spec, args: Any):
    module, _, cls = spec.adapter.partition(":")
    return getattr(importlib.import_module(module), cls)(spec, args)


def adapter_class(spec: Spec):
    module, _, cls = spec.adapter.partition(":")
    return getattr(importlib.import_module(module), cls)


def runner_adapter_class(spec: Spec):
    """The class `scripts/benchmax.py` drives for `spec`: group-A classes already speak the runner contract
    (`add_arguments` / `verify` / `run`); group-B classes (prepare / items / score) are wrapped by
    `bridge_b.bridge_class` so every spec gets the same clients, counts and run.json (PLAN §6)."""
    cls = adapter_class(spec)
    from jev_local.bench.benchmax.bridge_b import bridge_class, is_group_b_class

    return bridge_class(cls) if is_group_b_class(cls) else cls
