"""Deterministic stand-ins for the macOS observer/executor and the decision engine.

Used by the harness tests, by `replay`, and by integration tests (H). Nothing here touches the UI.

- `FakeObserver(snapshot)`: returns a fixed (or settable) `Snapshot`; counts prefetch/invalidate.
- `FakeExecutor()`: records every action it is asked to run and returns ok.
- `ScriptedEngine(lookup)`: an `Engine` whose answers come from `lookup(transcript, questions)`.
- `RuleModel()`: a toy regex "model" usable as a lookup, so replay scripts behave like a
  reasonable System One without weights: first command only, `wait` for unfinished prefixes,
  verbatim span picks, `complete` from whether the command's argument is present yet.
- `SlowClient(inner, latency_ms)`: adds event-loop latency to a client (virtual-time friendly), to
  exercise the stale-answer and coalescing paths.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from jev_local.engine.base import EngineResult, RawDist
from jev_local.harness.catalog import CHAIN_WORDS, FOLDERS, RISK_RE
from jev_local.harness.questions import (
    NONE,
    Q_APP,
    Q_COMPLETE,
    Q_DESTRUCTIVE,
    Q_FOLDER,
    Q_INTENT,
    Q_IS_COMMAND,
    Q_KEY,
    Q_SCROLL,
    Q_TARGET,
    Q_TEXT,
    Q_URL,
)
from jev_local.harness.types import Action, ActionKind, ActionRecord, Element, ExecResult, Snapshot
from jev_local.schema import ChoiceQuestion, Entry, NoulQuestion, Question, ScoreQuestion, SystemOneRequest, SystemOneResponse

# ---------------------------------------------------------------- snapshots


def make_snapshot(
    app_name: str,
    elements: Sequence[Element | tuple[str, str] | tuple[str, str, dict[str, Any]]] = (),
    window_title: str | None = None,
    focused: str | None = None,
    bundle_id: str | None = None,
) -> Snapshot:
    """Build a Snapshot from (role, label[, extra fields]) tuples; eids are e01, e02, ... in order.

    `focused` is a label (or eid) to mark as the focused element.
    """
    els: list[Element] = []
    for i, e in enumerate(elements, 1):
        if isinstance(e, Element):
            els.append(e)
            continue
        role, label, *rest = e
        extra = dict(rest[0]) if rest else {}
        eid = f"e{i:02d}"
        if focused is not None and focused in (label, eid):
            extra["focused"] = True
        els.append(Element(eid=eid, role=role, label=label, **extra))
    focused_eid = next((e.eid for e in els if e.focused), None)
    return Snapshot(
        app_name=app_name,
        bundle_id=bundle_id or f"com.example.{re.sub(r'[^a-z0-9]', '', app_name.lower())}",
        pid=4242,
        window_title=window_title,
        elements=tuple(els),
        taken_at=time.monotonic(),
        focused_eid=focused_eid,
        walk_ms=1.0,
    )


def mail_snapshot() -> Snapshot:
    return make_snapshot(
        "Mail",
        [
            ("button", "Send", {"context": "toolbar"}),
            ("button", "Reply (1 of 3)"),
            ("button", "Reply (2 of 3)"),
            ("button", "Reply (3 of 3)"),
            ("button", "Archive"),
            ("button", "Delete"),
            ("text field", "To"),
            ("text field", "Subject"),
            ("text area", "Message body"),
        ],
        window_title="New Message",
        focused="Message body",
    )


def finder_snapshot() -> Snapshot:
    return make_snapshot(
        "Finder",
        [
            ("button", "Back", {"context": "toolbar"}),
            ("button", "Forward", {"context": "toolbar"}),
            ("text field", "Search"),
            ("row", "Desktop", {"context": "sidebar"}),
            ("row", "Documents", {"context": "sidebar"}),
            ("row", "Downloads", {"context": "sidebar"}),
            ("button", "Move to Trash"),
            ("button", "New Folder"),
        ],
        window_title="Desktop",
        focused="Search",
    )


# ---------------------------------------------------------------- observer / executor


class FakeObserver:
    """Stands in for harness.observe.Observer. `snapshot` may be a Snapshot or a zero-arg callable."""

    def __init__(self, snapshot: Snapshot | Callable[[], Snapshot] | None = None):
        self._snap = snapshot if snapshot is not None else finder_snapshot()
        self.snapshot_calls: list[bool] = []  # allow_cached flag of each call
        self.prefetch_calls = 0
        self.invalidate_calls = 0

    def set(self, snapshot: Snapshot | Callable[[], Snapshot]) -> None:
        self._snap = snapshot

    def snapshot(self, allow_cached: bool = True, max_age_s: float = 1.5) -> Snapshot:
        self.snapshot_calls.append(allow_cached)
        return self._snap() if callable(self._snap) else self._snap

    def prefetch(self) -> None:
        self.prefetch_calls += 1

    def invalidate(self) -> None:
        self.invalidate_calls += 1

    def resolve(self, snap: Snapshot, eid: str) -> Element | None:
        return next((e for e in snap.elements if e.eid == eid), None)

    def focused_element(self) -> Element | None:
        s = self.snapshot()
        return next((e for e in s.elements if e.focused), None)


class FakeExecutor:
    """Records actions instead of performing them. Always ok, always a dry run."""

    def __init__(self, delay_s: float = 0.0, fail_kinds: Sequence[ActionKind] = ()):
        self.delay_s = delay_s
        self.fail_kinds = set(fail_kinds)
        self.actions: list[Action] = []
        self.undone: list[ActionRecord] = []
        self._lock = threading.Lock()
        self.max_concurrent = 0
        self._running = 0

    def run(self, action: Action, snap: Snapshot | None, cancel: threading.Event | None = None) -> ExecResult:
        with self._lock:
            self._running += 1
            self.max_concurrent = max(self.max_concurrent, self._running)
            self.actions.append(action)
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            if cancel is not None and cancel.is_set():
                return ExecResult(ok=False, changed=False, detail="cancelled", elapsed_ms=0.0, dry_run=True)
            ok = action.kind not in self.fail_kinds
            return ExecResult(ok=ok, changed=ok, detail=f"would {action.describe()}", elapsed_ms=0.0,
                              undo_token={"kind": action.kind.value}, dry_run=True)
        finally:
            with self._lock:
                self._running -= 1

    def undo(self, record: ActionRecord) -> ExecResult:
        self.undone.append(record)
        return ExecResult(ok=True, changed=True, detail=f"would undo {record.action.describe()}", elapsed_ms=0.0,
                          dry_run=True)

    @property
    def kinds(self) -> list[str]:
        return [a.kind.value for a in self.actions]


# ---------------------------------------------------------------- scripted engine

Lookup = Callable[[str, Mapping[str, Question]], Mapping[str, Any]]

# Neutral answers for harness questions a lookup leaves out: a command, finished, not destructive.
HARNESS_DEFAULTS: dict[str, Any] = {Q_IS_COMMAND: 0.95, Q_COMPLETE: 0.9, Q_DESTRUCTIVE: 0.02, Q_SCROLL: 1}


def _transcript_of(state: Entry) -> str:
    if isinstance(state, dict):
        t = state.get("transcript")
        return t if isinstance(t, str) else json.dumps(state)
    return state if isinstance(state, str) else json.dumps(state)


def _spread(labels: Sequence[str], given: Mapping[str, float]) -> list[float]:
    """Given probabilities for some labels; the leftover mass is shared by the rest."""
    known = {k: max(0.0, float(v)) for k, v in given.items() if k in labels}
    total = sum(known.values())
    if total > 1.0:
        return [known.get(lab, 0.0) / total for lab in labels]
    rest = [lab for lab in labels if lab not in known]
    share = (1.0 - total) / len(rest) if rest else 0.0
    probs = [known.get(lab, share) for lab in labels]
    s = sum(probs)
    return [p / s for p in probs] if s > 0 else [1.0 / len(labels)] * len(labels)


def spec_to_dist(q: Question, spec: Any, top_p: float = 0.9) -> RawDist:
    """label | {label: p} | float | level int | [p...] | None -> RawDist for question `q`."""
    if isinstance(q, NoulQuestion):
        p = 0.5 if spec is None else float(spec)
        return RawDist("noul", (min(1.0, max(0.0, p)),))
    if isinstance(q, ChoiceQuestion):
        labels = list(q.criteria.keys())
        if spec is None:
            given = {NONE: top_p} if NONE in labels else {}
        elif isinstance(spec, str):
            given = {spec: top_p}
        else:
            given = dict(spec)
        return RawDist("choice", tuple(_spread(labels, given)), tuple(labels))
    if isinstance(q, ScoreQuestion):
        levels = [str(i) for i in range(len(q.criteria))]
        if spec is None:
            spec = 0
        if isinstance(spec, int):
            given = {str(spec): top_p}
        elif isinstance(spec, Mapping):
            given = {str(k): v for k, v in spec.items()}
        else:
            given = {str(i): float(v) for i, v in enumerate(spec)}
        return RawDist("score", tuple(_spread(levels, given)), tuple(levels))
    raise TypeError(type(q).__name__)


class ScriptedEngine:
    """An Engine that answers from `lookup(transcript, questions) -> {qid: spec}` (see `spec_to_dist`)."""

    def __init__(
        self,
        lookup: Lookup,
        name: str = "jev-local-scripted-0.1.0",
        defaults: Mapping[str, Any] | None = None,
        latency_s: float = 0.0,
    ):
        self.lookup = lookup
        self.name = name
        self.max_tokens = 1_000_000
        self.defaults = dict(HARNESS_DEFAULTS if defaults is None else defaults)
        self.latency_s = latency_s
        self.calls: list[str] = []  # transcripts, in call order
        self.max_concurrent = 0
        self._running = 0
        self._lock = threading.Lock()

    def supports(self, req: SystemOneRequest) -> bool:
        return True

    def count_tokens(self, req: SystemOneRequest) -> int:
        return len(req.model_dump_json()) // 4

    def evaluate(self, state: Entry, questions: Mapping[str, Question]) -> EngineResult:
        with self._lock:
            self._running += 1
            self.max_concurrent = max(self.max_concurrent, self._running)
        try:
            t0 = time.perf_counter()
            transcript = _transcript_of(state)
            self.calls.append(transcript)
            if self.latency_s:
                time.sleep(self.latency_s)
            spec = dict(self.defaults)
            spec.update(self.lookup(transcript, questions) or {})
            dists = {qid: spec_to_dist(q, spec.get(qid)) for qid, q in questions.items()}
            return EngineResult(dists=dists, input_tokens=len(json.dumps(state)) // 4, output_tokens=0,
                                engine=self.name, timings_ms={"total": (time.perf_counter() - t0) * 1000})
        finally:
            with self._lock:
                self._running -= 1


def answers_for(questions: Mapping[str, Question], spec: Mapping[str, Any], defaults: Mapping[str, Any] | None = None) -> SystemOneResponse:
    """A SystemOneResponse built exactly as the API would, from a spec (policy/safety unit tests)."""
    from jev_local.api import system_one

    eng = ScriptedEngine(lambda _t, _q: spec, defaults=defaults)
    req = SystemOneRequest(state={"transcript": ""}, model="jev-local", questions=dict(questions))
    return system_one(req, {"fast": eng})


# ---------------------------------------------------------------- toy rule model

_CHAIN_RE = re.compile(r"\s+(?:" + "|".join(re.escape(w) for w in CHAIN_WORDS) + r")\s+")
_VERB_START = (
    r"(?:open|launch|start|switch|quit|close|click|tap|press|hit|select|choose|type|write|enter|dictate|"
    r"search|google|look|go|visit|navigate|scroll|page|undo|new|copy|paste|save|refresh|reload|yes|no|cancel)\b"
)
_KEY_WORDS: dict[str, str] = {
    "enter": "return", "return": "return", "escape": "escape", "esc": "escape", "tab": "tab", "space": "space",
    "spacebar": "space", "delete": "delete", "backspace": "delete", "up": "up", "down": "down", "left": "left",
    "right": "right", "page down": "page_down", "page up": "page_up", "select all": "cmd+a", "copy": "cmd+c",
    "paste": "cmd+v", "cut": "cmd+x", "undo": "cmd+z", "redo": "cmd+shift+z", "save": "cmd+s", "find": "cmd+f",
    "refresh": "cmd+r", "reload": "cmd+r", "command a": "cmd+a", "command c": "cmd+c", "command v": "cmd+v",
    "command s": "cmd+s", "command z": "cmd+z",
}
_ROLE_WORDS = {"button", "link", "tab", "checkbox", "field", "box", "menu", "icon", "row", "item", "option"}
_SIDE_TALK_RE = re.compile(r"^(?:can you|could you|did you|i think|what do you|how was|thanks|thank you|pass the|"
                           r"we should|let's grab|lunch)\b")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[a-z0-9][.!?])\s+", re.I)


def _norm(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[,!?;:\"]", " ", t)
    t = re.sub(r"\.(?=\s|$)", " ", t)  # sentence dots, not the dots in "github.com"
    return re.sub(r"\s+", " ", t).strip()


def _first_clause(t: str) -> str:
    """Cut at the first chain word that is followed by another command verb."""
    for m in _CHAIN_RE.finditer(t):
        if re.match(_VERB_START, t[m.end():]):
            return t[: m.start()]
    return re.sub(r"(?:\s+(?:and|then|also|and then|after that))+$", "", t)


def _label_of(option_text: str | None, label: str) -> str:
    if not option_text:
        return label
    m = re.search(r'"([^"]*)"', option_text)
    return m.group(1) if m else option_text


@dataclass
class RuleModel:
    """Toy System One for replay tests (a callable `Lookup`).

    `complete_p` is what a finished command scores on `complete`: 0.95 acts on the first complete
    partial, 0.75 needs two agreeing partials (the stability rule). `overrides` maps an exact
    normalised transcript to a spec merged over the parse (adversarial and red-team cases).
    """

    complete_p: float = 0.95
    intent_p: float = 0.95
    arg_p: float = 0.9
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __call__(self, transcript: str, questions: Mapping[str, Question]) -> dict[str, Any]:
        t = _norm(transcript)
        # Whisper punctuates pauses: "Open notes. Scroll down." is two commands without a chain word.
        # Like _first_clause, a sentence end only cuts when the next sentence starts with a command
        # verb, so dictation that spans sentences ("type hello. world") stays one payload.
        parts = _SENTENCE_SPLIT_RE.split(transcript.strip())
        head = parts[0]
        for nxt in parts[1:]:
            if re.match(_VERB_START, _norm(nxt)):
                break
            head += " " + nxt
        first = _norm(head)
        out = self.parse(first if first and not _SIDE_TALK_RE.match(t) else t, questions)
        if t in self.overrides:
            out.update(self.overrides[t])
        return out

    # The parse returns a spec dict for ScriptedEngine.
    def parse(self, t: str, questions: Mapping[str, Question]) -> dict[str, Any]:
        if not t or _SIDE_TALK_RE.match(t):
            return {Q_INTENT: {NONE: self.intent_p}, Q_IS_COMMAND: 0.05, Q_COMPLETE: 0.5}
        c = _first_clause(t)
        intent, args, complete = self._command(c, questions)
        spec: dict[str, Any] = {Q_IS_COMMAND: 0.95 if intent != NONE else 0.3}
        spec[Q_INTENT] = {intent: self.intent_p}
        spec[Q_COMPLETE] = self.complete_p if complete else 0.1
        risky = RISK_RE.search(c) or any(isinstance(v, str) and RISK_RE.search(v) for v in args.values())
        spec[Q_DESTRUCTIVE] = 0.9 if risky else 0.03
        for qid in (Q_APP, Q_KEY, Q_FOLDER, Q_TEXT, Q_URL):
            spec.setdefault(qid, {NONE: self.arg_p})
        spec.update({k: v for k, v in args.items() if not k.startswith("_")})
        return spec

    def _command(self, c: str, qs: Mapping[str, Question]) -> tuple[str, dict[str, Any], bool]:
        p = self.arg_p
        if re.match(r"^(?:yes|yeah|yep|sure|confirm|do it|go ahead)\b", c):
            return "confirm", {}, True
        if re.match(r"^(?:no|nope|cancel|stop|never ?mind)\b", c):
            return "cancel", {}, True
        m = re.match(r"^(?:scroll|page)(?: (down|up))?(?: (.*))?$", c)
        if m:
            if not m.group(1):
                return "wait", {}, False
            rest = m.group(2) or ""
            lvl = 0 if re.search(r"little|bit|few", rest) else 2 if re.search(r"all the way|top|bottom|end", rest) else 1
            return f"scroll_{m.group(1)}", {Q_SCROLL: lvl}, True
        if re.match(r"^(?:go back|back)$", c):
            return "go_back", {}, True
        if re.match(r"^(?:open |make )?(?:a )?new tab$", c):
            return "new_tab", {}, True
        if re.match(r"^close (?:the |this )?(?:current )?tab$", c):
            return "close_tab", {}, True
        if c == "undo" or c.startswith("undo "):
            return "undo", {}, True
        m = re.match(r"^(?:press|hit)(?: the)?(?: (.+?))?(?: key)?$", c)
        if m and (m.group(1) or "") in _KEY_WORDS:
            return "press_key", {Q_KEY: {_KEY_WORDS[m.group(1)]: p}}, True
        if m and not m.group(1):
            return "wait", {}, False
        m = re.match(r"^(copy|paste|save|select all|refresh|reload)(?: (?:that|this|it|the page))?$", c)
        if m:
            return "press_key", {Q_KEY: {_KEY_WORDS[m.group(1)]: p}}, True
        m = re.match(r"^(?:search(?: the web| google| online)? for|search|google|look up)(?: (.+))?$", c)
        if m:
            return self._payload("search_web", m.group(1), qs)
        m = re.match(r"^(?:type|write|enter|dictate)(?: (.+))?$", c)
        if m:
            return self._payload("type_text", m.group(1), qs)
        m = re.match(r"^(?:go to|visit|navigate to)(?: (.+))?$", c)
        if m:
            return self._url(m.group(1), qs)
        m = re.match(r"^(?:open|launch|start|switch to)(?: (?:the |my |up )?(.+))?$", c)
        if m:
            x = m.group(1) or ""
            if not x:
                return "wait", {}, False
            fx = re.sub(r"\s+folder$", "", x)
            if fx in FOLDERS and fx != NONE:
                return "open_folder", {Q_FOLDER: {fx: p}}, True
            if re.search(r"\.[a-z]{2,}|\bdot\b", x):
                return self._url(x, qs)
            return self._app("open_app", x, qs)
        m = re.match(r"^(?:quit|close)(?: (.+))?$", c)
        if m:
            return self._app("quit_app", m.group(1) or "", qs) if m.group(1) else ("wait", {}, False)
        m = re.match(r"^(?:click|tap|press|select|choose|hit)(?: on)?(?: the)?(?: (.+))?$", c)
        if m:
            return self._click(m.group(1) or "", qs)
        return NONE, {}, False

    def _payload(self, intent: str, q: str | None, qs: Mapping[str, Question]) -> tuple[str, dict[str, Any], bool]:
        if not q:
            return "wait", {}, False
        opts = [lab for lab in _labels(qs, Q_TEXT) if lab != NONE]
        want = q.strip(" .")
        best = next((o for o in opts if _norm(o) == want), None)
        if best is None:
            pref = [o for o in opts if want.startswith(_norm(o))]
            best = max(pref, key=len) if pref else None
        return intent, {Q_TEXT: {best or NONE: self.arg_p}}, best is not None

    def _url(self, x: str | None, qs: Mapping[str, Question]) -> tuple[str, dict[str, Any], bool]:
        opts = [lab for lab in _labels(qs, Q_URL) if lab != NONE]
        if not x or not opts:
            return ("wait", {}, False) if not x else ("open_url", {Q_URL: {NONE: self.arg_p}}, False)
        return "open_url", {Q_URL: {opts[0]: self.arg_p}}, True

    def _app(self, intent: str, x: str, qs: Mapping[str, Question]) -> tuple[str, dict[str, Any], bool]:
        x = re.sub(r"\s+app$", "", x.strip())
        apps = [a for a in _labels(qs, Q_APP) if a != NONE]
        full = [a for a in apps if x == a.lower() or x.startswith(a.lower() + " ")]
        if full:
            return intent, {Q_APP: {max(full, key=len): self.arg_p}}, True
        if any(a.lower().startswith(x) for a in apps):
            return "wait", {}, False  # "open visual" on the way to "open visual studio code"
        return intent, {Q_APP: {NONE: self.arg_p}}, False

    def _click(self, x: str, qs: Mapping[str, Question]) -> tuple[str, dict[str, Any], bool]:
        words = [w for w in x.split() if w not in ("the", "on", "a")]
        while words and words[-1] in _ROLE_WORDS:
            words.pop()
        want = " ".join(words)
        if not want:
            return "wait", {}, False
        q = qs.get(Q_TARGET)
        opts = {eid: _label_of(desc if isinstance(desc, str) else None, eid).lower()
                for eid, desc in (q.criteria.items() if isinstance(q, ChoiceQuestion) else ()) if eid != NONE}
        exact = [e for e, lab in opts.items() if re.sub(r"\s*\(\d+ of \d+\)$", "", lab) == want]
        hits = exact or [e for e, lab in opts.items() if want in lab]
        if not hits:
            return "click", {Q_TARGET: {NONE: self.arg_p}}, False
        share = self.arg_p / len(hits)  # ambiguous references split the mass -> the target gate clarifies
        return "click", {Q_TARGET: {e: share for e in hits}}, True


def _labels(qs: Mapping[str, Question], qid: str) -> list[str]:
    q = qs.get(qid)
    return list(q.criteria.keys()) if isinstance(q, ChoiceQuestion) else []


# ---------------------------------------------------------------- slow client


class SlowClient:
    """Wraps a DecisionClient and adds `latency_ms` of event-loop time (virtual under replay)."""

    def __init__(self, inner: Any, latency_ms: float):
        self.inner = inner
        self.latency_ms = latency_ms
        self.in_flight = 0
        self.max_in_flight = 0
        self.n_calls = 0

    async def system_one(self, state: Any, questions: Mapping[str, Question]) -> SystemOneResponse:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.n_calls += 1
        try:
            resp = await self.inner.system_one(state, questions)
            await asyncio.sleep(self.latency_ms / 1000.0)
            return resp
        finally:
            self.in_flight -= 1

    async def aclose(self) -> None:
        close = getattr(self.inner, "aclose", None)
        if close is not None:
            await close()


# ---------------------------------------------------------------- one-line controller wiring


def scripted_factory(
    lookup: Lookup | None = None,
    *,
    snapshot: Snapshot | Callable[[], Snapshot] | None = None,
    executor: FakeExecutor | None = None,
    cfg: Any = None,
    log: Any = None,
    latency_ms: float = 0.0,
    hud: Any = None,
    holder: dict[str, Any] | None = None,
    **controller_kw: Any,
) -> Callable[[Callable[[], float]], Any]:
    """A `replay` controller factory: ScriptedEngine(lookup or RuleModel()) in process, fake observer
    and executor, dry-run config, and a disabled audit log unless one is given. `holder` receives
    the controller, engine, client, observer and executor for assertions."""
    from jev_local.harness.client import DecisionClient
    from jev_local.harness.controller import Controller
    from jev_local.harness.log import AuditLog
    from jev_local.harness.types import HarnessConfig

    def factory(clock: Callable[[], float]) -> Any:
        eng = ScriptedEngine(lookup or RuleModel())
        client: Any = DecisionClient("inproc", {"fast": eng})
        if latency_ms:
            client = SlowClient(client, latency_ms)
        obs = FakeObserver(snapshot)
        ex = executor if executor is not None else FakeExecutor()
        ctl = Controller(None, obs, client, ex, cfg or HarnessConfig(), hud=hud, clock=clock,
                         log=log if log is not None else AuditLog(enabled=False), **controller_kw)
        if holder is not None:
            holder.update(controller=ctl, engine=eng, client=client, observer=obs, executor=ex)
        return ctl

    return factory
