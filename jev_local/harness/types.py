"""Dataclasses shared across the harness (speech -> stream -> decide -> policy -> safety -> execute)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal

from jev_local.schema import SystemOneResponse


@dataclass(frozen=True)
class Word:
    text: str
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class TranscriptEvent:
    kind: Literal["speech_start", "partial", "final", "speech_end", "error"]
    seq: int
    uid: str  # physical utterance id from the recognizer
    text: str
    t_mono: float  # time.monotonic() when received by the host
    words: tuple[Word, ...] = ()
    error: str | None = None
    # The recognizer saw no pause between the previous utterance and this one (whisper: it ended on a
    # stall, not on silence), so the two may be one breath. Such an utterance never confirms.
    joined: bool = False


@dataclass(frozen=True)
class Element:
    eid: str  # "e01".. assigned per snapshot, in screen order
    role: str  # humanized role, e.g. "button", "text field", "link"
    label: str  # title/description/placeholder, <= 60 chars, may carry " (2 of 3)"
    value: str | None = None  # current value (text fields, checkboxes), <= 60 chars
    frame: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x, y, w, h (screen points)
    actions: tuple[str, ...] = ()  # AX action names, e.g. ("AXPress",)
    path: tuple[int, ...] = ()  # child-index path from the window, for re-resolving stale refs
    secure: bool = False
    focused: bool = False
    enabled: bool = True
    context: str | None = None  # nearby container, e.g. "toolbar", "sidebar"


@dataclass(frozen=True)
class Snapshot:
    app_name: str
    bundle_id: str
    pid: int
    window_title: str | None
    elements: tuple[Element, ...]
    taken_at: float  # time.monotonic()
    partial: bool = False  # walk hit its deadline
    focused_eid: str | None = None
    walk_ms: float = 0.0


@dataclass(frozen=True)
class Tail:
    """The not-yet-consumed part of the current utterance, as seen by the decision loop."""

    vid: str  # virtual utterance id: "<uid>+<gen>"
    text: str  # words[cursor:] joined
    words: tuple[str, ...]  # tail words only
    cursor: int  # number of words of the physical utterance already consumed
    is_final: bool
    silent_ms: int
    uid: str = ""
    joined: bool = False  # see TranscriptEvent.joined


class ActionKind(StrEnum):
    # Values equal the intent labels in catalog.INTENTS.
    OPEN_APP = "open_app"
    QUIT_APP = "quit_app"
    CLICK = "click"
    TYPE_TEXT = "type_text"
    SEARCH_WEB = "search_web"
    OPEN_URL = "open_url"
    PRESS_KEY = "press_key"
    SCROLL_DOWN = "scroll_down"
    SCROLL_UP = "scroll_up"
    OPEN_FOLDER = "open_folder"
    GO_BACK = "go_back"
    NEW_TAB = "new_tab"
    CLOSE_TAB = "close_tab"
    UNDO = "undo"
    CONFIRM = "confirm"
    CANCEL = "cancel"


class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2  # needs spoken/keyboard confirmation
    DENY = 3  # never executed


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    source_vid: str
    confidence: float  # min over the judgements used
    target_eid: str | None = None
    target_label: str | None = None  # "button \"Send\"" (for logs, HUD, risk lexicon)
    app: str | None = None
    text: str | None = None  # verbatim span of the transcript (type_text / search_web)
    key: str | None = None  # catalog.KEYS label, e.g. "cmd+c"
    url: str | None = None
    folder: str | None = None  # catalog.FOLDERS label
    amount: int | None = None  # scroll level 0/1/2
    consumed_words: int = 0  # how many tail words this action accounts for

    def describe(self) -> str:
        k = self.kind.value
        arg = self.target_label or self.app or self.text or self.key or self.url or self.folder
        return f"{k}({arg!r})" if arg else k


Verdict = Literal["act", "wait", "ignore", "confirm", "clarify", "deny"]


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    action: Action | None
    reason: str
    seq: int
    retry_in_ms: int | None = None
    answers: SystemOneResponse | None = None
    latency_ms: float = 0.0
    risk: RiskLevel = RiskLevel.LOW


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    changed: bool
    detail: str
    elapsed_ms: float
    undo_token: dict[str, Any] | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class ActionRecord:
    said: str
    action: Action
    outcome: str  # "ok" | "failed: ..." | "dry-run"
    t: float
    undo_token: dict[str, Any] | None = None


@dataclass(frozen=True)
class Thresholds:
    is_command: float = 0.5
    intent_conf: float = 0.45  # choice confidence (not raw p) for the intent question
    intent_top_p: float = 0.55  # raw top probability for the intent question
    complete: float = 0.65
    stable_complete: float = 0.85  # act on a single partial only above this; else need 2 agreeing evals
    target_conf: float = 0.45
    target_top_p: float = 0.35
    span_top_p: float = 0.35
    app_top_p: float = 0.50
    key_top_p: float = 0.60
    folder_top_p: float = 0.50
    destructive: float = 0.5
    high_risk_intent_p: float = 0.85
    high_risk_target_p: float = 0.75
    confirm_p: float = 0.8
    silence_complete_ms: int = 900
    payload_silence_ms: int = 600
    debounce_ms: int = 120
    confirm_timeout_ms: int = 8000


@dataclass
class HarnessConfig:
    dry_run: bool = True
    backend: Literal["inproc", "http", "hosted"] = "inproc"
    base_url: str = "http://127.0.0.1:8765"
    model: str = "jev-local"
    search_url: str = "https://www.google.com/search?q={q}"
    allow_apps: tuple[str, ...] = ()  # apps exempt from the hard-deny list
    max_elements: int = 60
    max_apps: int = 24
    log_redact: bool = True
    thresholds: Thresholds = field(default_factory=Thresholds)
