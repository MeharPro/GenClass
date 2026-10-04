"""The fixed question schema the Mac harness sends on every transcript update.

One request carries every question (speculative fan-out): the intent plus the argument questions
for every intent. Code reads only the branch the intent picks. The synthetic-data generator calls
`build_questions` too, so the fast model is trained on exactly this schema.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from rapidfuzz import fuzz, process

from jev_local.harness.catalog import FOLDERS, INTENTS, KEYS, SCROLL_LEVELS
from jev_local.harness.types import Element, Snapshot
from jev_local.schema import ChoiceQuestion, NoulCriteria, NoulQuestion, Question, ScoreQuestion

# Question ids, in request order.
Q_INTENT = "intent"
Q_COMPLETE = "complete"
Q_IS_COMMAND = "is_command"
Q_DESTRUCTIVE = "destructive"
Q_TARGET = "target"
Q_APP = "app"
Q_KEY = "key"
Q_FOLDER = "folder"
Q_TEXT = "text_span"
Q_URL = "url_span"
Q_SCROLL = "scroll_amount"

NONE = "none"

INSTR = {
    Q_INTENT: (
        "Which action does `transcript` ask the computer to do? If several actions are chained, pick the "
        "first one. If the sentence is unfinished, pick the action its words already commit to, or wait "
        "if they do not commit to one yet."
    ),
    Q_COMPLETE: (
        "Does `transcript` already contain a complete first command: the action plus everything it needs, "
        "such as which app, which element, which key, or what text?"
    ),
    Q_IS_COMMAND: "Is `transcript` an instruction for this computer, rather than speech meant for someone else?",
    Q_DESTRUCTIVE: (
        "Would doing what `transcript` asks delete, send, submit, buy, quit, close, or otherwise do "
        "something hard to undo?"
    ),
    Q_TARGET: "Which on-screen element does `transcript` tell the computer to act on?",
    Q_APP: "Which application does `transcript` name?",
    Q_KEY: "Which key or shortcut does `transcript` ask to press?",
    Q_FOLDER: "Which folder does `transcript` ask to open?",
    Q_TEXT: "Which exact words from `transcript` should be typed or searched for?",
    Q_URL: "Which website address does `transcript` mention?",
    Q_SCROLL: "How far does `transcript` ask to scroll?",
}

COMPLETE_CRITERIA = NoulCriteria(
    true="complete, for example: open safari, click send, type hello there, press escape, scroll down",
    false="unfinished, for example: open, click the, type, search for, go to, press",
)
COMMAND_CRITERIA = NoulCriteria(
    true="a request for the computer, for example: open mail, scroll down, click the blue button",
    false="talking to a person or thinking aloud, for example: can you pass the salt, I think it's fine",
)
TARGET_NONE = "no on-screen element is mentioned"
APP_NONE = "no application is mentioned"
TEXT_NONE = "nothing should be typed or searched"
URL_NONE = "no website is mentioned"


def element_line(el: Element) -> str:
    """How an element is shown to the model: `button "Send"`, `text field "To" = "bob" (focused)`."""
    s = f'{el.role} "{el.label}"' if el.label else el.role
    if el.value:
        s += f' = "{el.value}"'
    if el.context:
        s += f" in {el.context}"
    if el.focused:
        s += " (focused)"
    if not el.enabled:
        s += " (disabled)"
    return s


def rank_apps(tail_text: str, names: Iterable[str], running: Sequence[str] = (), max_n: int = 24) -> list[str]:
    """Pick <= max_n app names: fuzzy matches to the transcript first, then running apps, then others."""
    names = list(dict.fromkeys(n for n in names if n))
    out: list[str] = []
    if tail_text.strip() and names:
        hits = process.extract(tail_text, names, scorer=fuzz.partial_ratio, limit=max_n)
        out += [n for n, score, _ in hits if score >= 60]
    for n in list(running) + names:
        if len(out) >= max_n:
            break
        if n not in out:
            out.append(n)
    return out[:max_n]


def build_questions(
    snap: Snapshot | None,
    apps: Sequence[str],
    text_cands: Sequence[str],
    url_cands: Sequence[str],
    max_elements: int = 60,
) -> dict[str, Question]:
    q: dict[str, Question] = {}
    q[Q_INTENT] = ChoiceQuestion(instructions=INSTR[Q_INTENT], criteria=dict(INTENTS))
    q[Q_COMPLETE] = NoulQuestion(instructions=INSTR[Q_COMPLETE], criteria=COMPLETE_CRITERIA)
    q[Q_IS_COMMAND] = NoulQuestion(instructions=INSTR[Q_IS_COMMAND], criteria=COMMAND_CRITERIA)
    q[Q_DESTRUCTIVE] = NoulQuestion(instructions=INSTR[Q_DESTRUCTIVE])
    if snap is not None and snap.elements:
        crit: dict[str, str | None] = {el.eid: element_line(el) for el in snap.elements[:max_elements]}
        crit[NONE] = TARGET_NONE
        q[Q_TARGET] = ChoiceQuestion(instructions=INSTR[Q_TARGET], criteria=crit)
    app_crit: dict[str, str | None] = {a: None for a in apps if a and a != NONE}
    app_crit[NONE] = APP_NONE
    q[Q_APP] = ChoiceQuestion(instructions=INSTR[Q_APP], criteria=app_crit)
    q[Q_KEY] = ChoiceQuestion(instructions=INSTR[Q_KEY], criteria=dict(KEYS))
    q[Q_FOLDER] = ChoiceQuestion(instructions=INSTR[Q_FOLDER], criteria=dict(FOLDERS))
    if text_cands:
        tc: dict[str, str | None] = {c: None for c in text_cands if c.lower() != NONE}
        tc[NONE] = TEXT_NONE
        q[Q_TEXT] = ChoiceQuestion(instructions=INSTR[Q_TEXT], criteria=tc)
    if url_cands:
        uc: dict[str, str | None] = {u: None for u in url_cands if u.lower() != NONE}
        uc[NONE] = URL_NONE
        q[Q_URL] = ChoiceQuestion(instructions=INSTR[Q_URL], criteria=uc)
    q[Q_SCROLL] = ScoreQuestion(instructions=INSTR[Q_SCROLL], criteria=list(SCROLL_LEVELS))
    return q
