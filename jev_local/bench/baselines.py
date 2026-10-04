"""Open-baseline adapters for jevbench (open-baselines.md §4.0; PLAN §1.5). Interfaces and stubs for phase M0+.

One contract for every system: `decide(state, questions) -> {qid: answer}` with Jev wire shapes and the FULL
probability vector over the declared options; scoring then goes through `jev_local.bench.metrics`, so no
baseline gains or loses from its own confidence convention. All baselines run on the Azure CPU VM, never on
the Mac. Only the HTTP adapter (any `/v1/systemone` server: laya-serve, von serve, opendecider serve,
featherless simple-jev, our own server) is implemented; the in-process adapters are stubs with the exact input
mapping written down, to be filled in when the baseline phase starts.

Shared rules (§4.0):
  * option mapping   choice: options = criteria keys, text = value or the key when null
                     noul:   {"no": criteria.false or "No", "yes": criteria.true or "Yes"}; P(yes) is the answer
                     score:  "level i: <text>" per level
  * state            strings pass through; objects -> json.dumps(state, ensure_ascii=False)
  * truncation       cut the state to the model's budget (keep the beginning), record `truncated`
  * calibration      report raw; plus a refit column: one temperature per (type, K-bucket), fitted on
                     jevbench-dev only (`KBUCKETS`), and a noul threshold intercept
  * noul reporting   accuracy at 0.5 and the JevBench band rule (0.2 / 0.8)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

KBUCKETS = ((2, 2), (3, 5), (6, 10), (11, 30), (31, 100), (101, 255))  # PLAN §3.2 calibration buckets


@runtime_checkable
class Decider(Protocol):
    name: str

    def decide(self, state: Any, questions: Mapping[str, dict]) -> dict[str, dict]: ...


# ---------------------------------------------------------------------------------------------- shared mapping


def render_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def choice_options(q: dict) -> list[tuple[str, str]]:
    return [(k, v if isinstance(v, str) and v else k) for k, v in q["criteria"].items()]


def noul_options(q: dict) -> list[tuple[str, str]]:
    c = q.get("criteria") or {}
    return [("no", c.get("false") or "No"), ("yes", c.get("true") or "Yes")]


def score_options(q: dict) -> list[tuple[str, str]]:
    return [(str(i), f"level {i}: {t}") for i, t in enumerate(q["criteria"])]


def options_for(q: dict) -> list[tuple[str, str]]:
    return {"choice": choice_options, "noul": noul_options, "score": score_options}[q["type"]](q)


def to_answer(q: dict, probs: Mapping[str, float]) -> dict:
    """Wrap a distribution over `options_for(q)` keys as a Jev-shaped answer (unrounded)."""
    z = sum(probs.values()) or 1.0
    p = {k: float(v) / z for k, v in probs.items()}
    if q["type"] == "noul":
        return {"type": "noul", "noul": p.get("yes", 0.0)}
    return {"type": q["type"], "probabilities": p}


def kbucket(kind: str, k: int) -> str:
    if kind == "noul":
        return "noul"
    for lo, hi in KBUCKETS:
        if lo <= k <= hi:
            return f"{kind}:{lo}-{hi}"
    return f"{kind}:>255"


@dataclass
class BaselineSpec:
    key: str
    hf_id: str
    params: str
    adapter: str  # class name below
    contaminated: tuple[str, ...] = ()  # jevbench keys present in its training data: flag those rows
    notes: str = ""


# PLAN §1.5 battery. Contamination flags come from the model cards (open-baselines.md §2).
BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec("jev-local-fast-v1", "models/jev-local-fast", "32M", "SystemOneHTTP", notes="our v1; GEN templates only"),
    BaselineSpec("jev-local-general", "mlx-community/Qwen3-1.7B-4bit", "1.7B", "SystemOneHTTP", notes="MLX; Mac only via overnight gates"),
    BaselineSpec("laya", "convaiinnovations/laya", "421M", "SystemOneHTTP", ("ag_news",), "laya-serve; also head_max_len=512 for >20 options"),
    BaselineSpec("laya-multilingual", "convaiinnovations/laya-multilingual", "322M", "SystemOneHTTP", ("ag_news",)),
    BaselineSpec("gliner2.5-decide", "fastino/GLiNER2.5-Decide", "486M", "GLiNER2Adapter"),
    BaselineSpec("gliner2.5-base", "fastino/gliner2.5-base-v1", "194M", "GLiNER2Adapter"),
    BaselineSpec("deberta-v3-large-zs-c", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c", "435M", "NLIZeroShotAdapter",
                 notes="never the non -c variant (trained on 28 benchmark datasets)"),
    BaselineSpec("deberta-v3-base-zs-c", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c", "184M", "NLIZeroShotAdapter"),
    BaselineSpec("qwen3-reranker-0.6b", "Qwen/Qwen3-Reranker-0.6B", "596M", "QwenRerankerAdapter"),
    BaselineSpec("qwen3-4b-instruct", "Qwen/Qwen3-4B-Instruct-2507", "4B", "SystemOneHTTP", notes="via featherless simple-jev /v1/systemone"),
    BaselineSpec("opendecider-nano", "manjunathshiva/opendecider-nano", "395M", "SystemOneHTTP",
                 ("clinc150", "go_emotions", "paws", "typed_decisions"), "Ettin-400m peer; opendecider serve"),
    BaselineSpec("von-1.3", "wfzyx/von", "395M", "SystemOneHTTP", ("banking77", "dair_emotion", "anli"), "von serve"),
)


# ---------------------------------------------------------------------------------------------- adapters


@dataclass
class SystemOneHTTP:
    """Any server speaking `POST /v1/systemone` (Jev wire format): send the request bytes unchanged."""

    base_url: str
    model: str = "jev"
    name: str = "systemone-http"
    timeout_s: float = 120.0
    headers: dict = field(default_factory=dict)

    def decide(self, state: Any, questions: Mapping[str, dict]) -> dict[str, dict]:
        import httpx

        r = httpx.post(f"{self.base_url.rstrip('/')}/v1/systemone", json={"model": self.model, "state": state,
                       "questions": dict(questions)}, headers=self.headers, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["answers"]


class _Stub:
    name = "stub"
    spec: BaselineSpec | None = None

    def decide(self, state: Any, questions: Mapping[str, dict]) -> dict[str, dict]:
        raise NotImplementedError(f"{type(self).__name__} is a stub until the baseline phase (PLAN §1.5)")


class GLiNER2Adapter(_Stub):
    """gliner2 Classifier: one `ClassificationSchema().single(qid, {label: text}, instruction=instructions)` per
    question (Jev questions are isolated); labels from `options_for`; read `scores.probability(qid, label)` and
    renormalise. The prompt compiler reserves parentheses: rewrite "(...)" as ", ...". Load once on CPU."""


class NLIZeroShotAdapter(_Stub):
    """transformers zero-shot pipeline on a `-c` NLI model. premise = render_state(state).
    choice: hypotheses = option texts (BTZSC hypotheses where the request carries them), else
            'The answer to "{instructions}" is: {text}.'; multi_label=False (softmax over entailment logits).
    noul:   one hypothesis (criteria.true, else 'The answer to "{instructions}" is yes.'); P(yes) = P(entail)
            over {entail, not-entail} (multi_label=True).
    score:  one hypothesis per level, softmax. Context 512 tokens: truncate the premise, record it."""


class QwenRerankerAdapter(_Stub):
    """Qwen3-Reranker yes/no readout with the official template: Instruct = instructions, Query = rendered state,
    Document = '{key}: {text}'. Option score = logit(yes) - logit(no); choice = softmax over options (refit T on
    dev); noul = sigmoid(score)."""


ADAPTERS = {"SystemOneHTTP": SystemOneHTTP, "GLiNER2Adapter": GLiNER2Adapter, "NLIZeroShotAdapter": NLIZeroShotAdapter,
            "QwenRerankerAdapter": QwenRerankerAdapter}


def run_rows(decider: Decider, rows: list[dict]) -> list[dict]:
    """Same record shape as scripts/jevbench.py writes for Jev and ours, so `score` reads it unchanged."""
    out = []
    for r in rows:
        t = time.perf_counter()
        try:
            ans = decider.decide(r["request"]["state"], r["request"]["questions"])
            out.append({"id": r["id"], "ok": True, "model": decider.name, "answers": ans,
                        "ms": (time.perf_counter() - t) * 1e3})
        except NotImplementedError:
            raise
        except Exception as e:
            out.append({"id": r["id"], "ok": False, "permanent": True, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return out
