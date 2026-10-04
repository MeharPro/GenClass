"""Canonical text serialization of state and questions, shared by all engines and the trainer.

State -> ordered segments (one per top-level key of an object state, one per array item, or one
for a string). Question -> a header plus items (choice options, score levels, or noul true/false
criteria). Engines tokenize these texts; they never re-serialize JSON themselves, so the fast
encoder at train time and at run time see exactly the same strings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from jev_local.schema import ChoiceQuestion, Entry, NoulQuestion, Question, ScoreQuestion

MAX_ARRAY_SEGMENTS = 64


def entry_text(e: Entry | None) -> str:
    """Render an entry (str | dict | list | None) as compact, stable text."""
    if e is None:
        return ""
    if isinstance(e, str):
        return e.strip()
    if isinstance(e, list):
        parts = [entry_text(x) for x in e]
        if all(isinstance(x, str) for x in e):
            return "; ".join(p for p in parts if p)
        return json.dumps(e, ensure_ascii=False, separators=(", ", ": "))
    if isinstance(e, dict):
        # "key: value" lines read better for small encoders than raw JSON braces.
        lines = []
        for k, v in e.items():
            nested = isinstance(v, dict) or (isinstance(v, list) and not all(isinstance(x, str) for x in v))
            if nested:
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False, separators=(', ', ': '))}")
            else:
                lines.append(f"{k}: {entry_text(v)}")
        return " | ".join(lines)
    return str(e)


@dataclass(frozen=True)
class Segment:
    key: str  # "" for a bare string state, the object key, or "[i]" for array items
    text: str

    @property
    def digest(self) -> str:
        return hashlib.blake2b(f"{self.key}\x00{self.text}".encode(), digest_size=12).hexdigest()


def _value_text(v: Any) -> str:
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return "\n".join(v)  # element lists etc. read one per line
    return entry_text(v)


def state_segments(state: Entry, max_items: int = MAX_ARRAY_SEGMENTS) -> list[Segment]:
    if isinstance(state, str):
        return [Segment("", state.strip())]
    if isinstance(state, dict):
        return [Segment(str(k), _value_text(v)) for k, v in state.items()]
    if isinstance(state, list):
        segs = [Segment(f"[{i}]", entry_text(x)) for i, x in enumerate(state[:max_items])]
        if len(state) > max_items:
            rest = "\n".join(entry_text(x) for x in state[max_items:])
            segs.append(Segment(f"[{max_items}:]", rest))
        return segs
    return [Segment("", str(state))]


def state_text(state: Entry) -> str:
    """Whole state as one document (used by the decoder engine)."""
    out = []
    for s in state_segments(state):
        out.append(f"{s.key}:\n{s.text}" if s.key else s.text)
    return "\n\n".join(out)


BlockKind = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class QBlock:
    qid: str
    kind: BlockKind
    header: str  # instructions text (question ids are NOT shown to the model, as in Jev)
    items: tuple[str, ...]  # choice: "label: desc" | "label"; score: level texts; noul: (true, false)
    labels: tuple[str, ...]  # choice labels / score "0".."n-1" / noul ("true","false")


DEFAULT_NOUL_INSTR = "Is the statement true of the state?"
DEFAULT_CHOICE_INSTR = "Which option best fits the state?"
DEFAULT_SCORE_INSTR = "Which level best describes the state?"


def _bare_label(label: str, desc: Entry | None) -> bool:
    if desc is None:
        return True
    if isinstance(desc, str):
        d = desc.strip()
        return d == "" or d == label.strip()
    return False


def question_block(qid: str, q: Question) -> QBlock:
    if isinstance(q, NoulQuestion):
        t = entry_text(q.criteria.true) if q.criteria else ""
        f = entry_text(q.criteria.false) if q.criteria else ""
        return QBlock(
            qid,
            "noul",
            entry_text(q.instructions) or DEFAULT_NOUL_INSTR,
            (t or "yes", f or "no"),
            ("true", "false"),
        )
    if isinstance(q, ChoiceQuestion):
        labels = tuple(q.criteria.keys())
        # W8: null, "" and a description equal to the label (DMB's key == value) all mean "read the
        # label by its name alone", so training renders them exactly as the server does.
        items = tuple(
            lab if _bare_label(lab, desc) else f"{lab}: {entry_text(desc)}" for lab, desc in q.criteria.items()
        )
        return QBlock(qid, "choice", entry_text(q.instructions) or DEFAULT_CHOICE_INSTR, items, labels)
    if isinstance(q, ScoreQuestion):
        items = tuple(entry_text(c) for c in q.criteria)
        labels = tuple(str(i) for i in range(len(items)))
        return QBlock(qid, "score", entry_text(q.instructions) or DEFAULT_SCORE_INSTR, items, labels)
    raise TypeError(f"unsupported question: {type(q).__name__}")
