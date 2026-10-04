"""Safety gating between the policy and the executor (SPEC §4.7 layers 1–5).

Deterministic layers come first, because a confident mis-hear passes every model gate:

1. **Hard deny.** Acting inside or opening a deny-listed app (System Settings, Keychain, terminals,
   apps that run what is typed: Claude, Script Editor, Automator, code editors; password managers,
   banking/trading). Typing or pasting into a secure/password-like field. Return in a terminal.
   Non-http(s) URLs. Unknown folders.
2. **Risk lexicon.** quit_app / close_tab, or a risk word ("send", "delete", "buy", "allow", ...) in
   the target label, the focused control that Return/Space would press, the payload, the key, or
   the words the user said. Return in a chat app or message composer (it sends), or with an alert
   whose default button is destructive ("Empty Trash"). Any of these makes the action HIGH, which
   needs a spoken confirmation.
3. **Model gate.** `destructive >= T.destructive` -> confirm. HIGH risk also needs intent
   p >= .85 and (for clicks) target p >= .75, else clarify: never ask "confirm?" for a guess.
4. **Confirmation.** It never comes from the virtual utterance that proposed the action, nor from
   words already heard when the prompt appeared, nor from an utterance the recognizer could not
   separate from the previous one by a pause (`confirmation_allowed`), and it needs an explicit
   yes-word (`AFFIRM_RE`: not "yeah"/"sure"/"okay"). Any negation cancels.
5. **Rate limit.** At most 3 executed actions per second (`RateLimiter`).
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from jev_local.harness.catalog import DENY_APP_NAMES, DENY_APP_PATTERNS, FOLDER_PATHS, RISK_RE, SECURE_FIELD_RE
from jev_local.harness.questions import Q_DESTRUCTIVE, Q_INTENT, Q_TARGET
from jev_local.harness.types import Action, ActionKind, Decision, Element, HarnessConfig, RiskLevel, Snapshot, Tail
from jev_local.schema import ChoiceAnswer, NoulAnswer, SystemOneResponse

# Risky phrasings missing from catalog.RISK_WORDS (reported to the lead). "Don't Save" discards
# work in every macOS save dialog; "clean up my desktop" is the documented bulk-move failure
# (SPEC §2.7); the others end sessions or wipe state.
EXTRA_RISK_RE = re.compile(
    r"\b(don'?t save|do not save|revert|shut ?down|restart|force quit|deactivate|move to (?:the )?bin|"
    r"delete all|clear (?:all|history|everything)|leave (?:the )?(?:meeting|call|group|channel)|"
    r"clean(?:ing)? (?:up|out)|tidy(?: up)?|get rid of|throw (?:it |them |that |this )?away)\b",
    re.IGNORECASE,
)
TERMINAL_RE = re.compile(r"terminal|iterm|warp|ghostty|kitty|alacritty|wezterm|\bhyper\b|tabby|termius", re.IGNORECASE)
# Explicit yes-words only. "yeah", "sure", "okay", "yep" are left out on purpose: they are what people
# say to each other in side talk, and "Yeah." / "Okay." are classic whisper hallucinations on
# silence, so they must never confirm a risky action. The prompt asks for "confirm".
AFFIRM_RE = re.compile(r"\b(yes|confirm(?:ed)?|do it|go ahead|proceed|affirmative)\b", re.IGNORECASE)
# Chat apps, where Return in the message box sends it to other people. (Not mail clients: there
# Return in the body is a new line; sending is cmd+shift+d or the Send button, both risky already.)
MESSAGING_APP_RE = re.compile(
    r"\b(messages|slack|discord|whatsapp|telegram|signal|microsoft teams|teams|wechat|line|skype|zoom|"
    r"messenger|webex|mattermost|element|beeper)\b",
    re.IGNORECASE,
)
# Focused controls where Return sends or posts what was typed.
# ("Message body" is a mail body, where Return is a new line.)
COMPOSER_RE = re.compile(r"\b(i?message(?!\s+body)|reply|chat|comment|compose|tweet|post|send|write a|say something)\b",
                         re.IGNORECASE)
_DIALOG_RE = re.compile(r"dialog|sheet|alert", re.IGNORECASE)
NEGATE_RE = re.compile(
    r"\b(no|nope|nah|don'?t|do not|not|cancel|stop|never ?mind|abort|wait|hold on)\b", re.IGNORECASE
)

_HOST_RE = re.compile(r"^(?:localhost|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}|\d{1,3}(?:\.\d{1,3}){3})$",
                      re.IGNORECASE)

# Minimum time from the confirmation prompt to a "yes" in the same physical utterance: people
# need ~0.5 s to read a prompt and start speaking, and the recognizer adds ~0.3 s.
CONFIRM_REACTION_S = 0.8

# Actions that operate inside the frontmost app (so the frontmost app's deny status applies).
IN_APP_KINDS = frozenset({
    ActionKind.CLICK, ActionKind.TYPE_TEXT, ActionKind.PRESS_KEY, ActionKind.SCROLL_DOWN, ActionKind.SCROLL_UP,
    ActionKind.GO_BACK, ActionKind.NEW_TAB, ActionKind.CLOSE_TAB, ActionKind.UNDO,
})
# Actions whose safety checks need to see the screen (target / focused field).
SCREEN_KINDS = frozenset({ActionKind.CLICK, ActionKind.TYPE_TEXT, ActionKind.PRESS_KEY})
ALWAYS_HIGH = frozenset({ActionKind.QUIT_APP, ActionKind.CLOSE_TAB})
MEDIUM_KINDS = frozenset({ActionKind.TYPE_TEXT, ActionKind.SEARCH_WEB, ActionKind.OPEN_URL, ActionKind.PRESS_KEY})
PASSTHROUGH = frozenset({ActionKind.CONFIRM, ActionKind.CANCEL})


# ---------------------------------------------------------------- helpers


def is_risky_text(s: str | None) -> bool:
    return bool(s) and bool(RISK_RE.search(s) or EXTRA_RISK_RE.search(s))  # type: ignore[arg-type]


def is_denied_app(name: str | None, cfg: HarnessConfig) -> bool:
    if not name:
        return False
    n = name.lower().strip()
    if any(n == a.lower() for a in cfg.allow_apps):
        return False
    return n in DENY_APP_NAMES or any(p in n for p in DENY_APP_PATTERNS)


def _element(snap: Snapshot | None, eid: str | None) -> Element | None:
    if snap is None or eid is None:
        return None
    return next((e for e in snap.elements if e.eid == eid), None)


def focused_element(snap: Snapshot | None) -> Element | None:
    if snap is None:
        return None
    return _element(snap, snap.focused_eid) or next((e for e in snap.elements if e.focused), None)


def _is_secure(el: Element | None) -> bool:
    if el is None:
        return False
    return el.secure or bool(SECURE_FIELD_RE.search(el.label or "")) or bool(SECURE_FIELD_RE.search(el.role or ""))


def _url_ok(url: str | None) -> bool:
    if not url:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or "@" in parts.netloc:
        return False
    try:
        host = parts.hostname or ""
    except ValueError:
        return False
    return bool(_HOST_RE.match(host))


# ---------------------------------------------------------------- classification


def deny_reason(action: Action, snap: Snapshot | None, cfg: HarnessConfig) -> str | None:
    """Why this action is hard-denied, or None."""
    k = action.kind
    if k in (ActionKind.OPEN_APP, ActionKind.QUIT_APP) and is_denied_app(action.app, cfg):
        return f"{action.app} is on the deny list (add it to allow_apps to permit)"
    if k in IN_APP_KINDS and snap is not None and is_denied_app(snap.app_name, cfg):
        return f"acting inside {snap.app_name} is denied (add it to allow_apps to permit)"
    focused = focused_element(snap)
    if k == ActionKind.TYPE_TEXT and _is_secure(focused):
        return "never types into a password or secure field"
    if k == ActionKind.PRESS_KEY and action.key == "cmd+v" and _is_secure(focused):
        return "never pastes into a password or secure field"
    if k == ActionKind.PRESS_KEY and action.key == "return" and snap is not None and TERMINAL_RE.search(snap.app_name):
        return "never presses Return in a terminal"
    if k == ActionKind.OPEN_URL and not _url_ok(action.url):
        return f"only plain http(s) URLs are opened, not {action.url!r}"
    if k == ActionKind.OPEN_FOLDER and action.folder not in FOLDER_PATHS:
        return f"unknown folder {action.folder!r}"
    return None


def classify_risk(action: Action, snap: Snapshot | None, cfg: HarnessConfig, said: str | None = None) -> RiskLevel:
    """Deterministic risk class. `said` is the transcript words the action was decided from."""
    if action.kind in PASSTHROUGH:
        return RiskLevel.LOW
    if deny_reason(action, snap, cfg) is not None:
        return RiskLevel.DENY
    if action.kind in ALWAYS_HIGH:
        return RiskLevel.HIGH
    el = _element(snap, action.target_eid)
    texts = [action.target_label, el.label if el else None, action.text, action.key, said]
    if any(is_risky_text(t) for t in texts):
        return RiskLevel.HIGH
    if action.kind == ActionKind.PRESS_KEY and action.key in ("return", "space"):
        # Return/Space activate the focused control: pressing Return on a focused "Send" button is a send.
        f = focused_element(snap)
        if f is not None and is_risky_text(f.label):
            return RiskLevel.HIGH
    if action.kind == ActionKind.PRESS_KEY and action.key == "return" and snap is not None and return_is_risky(snap):
        return RiskLevel.HIGH
    if action.kind in MEDIUM_KINDS:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def return_is_risky(snap: Snapshot) -> bool:
    """Does Return do something irreversible here, whatever the focused control's label says?

    - In a chat app, Return in the message box sends it to other people ("iMessage",
      "Message #general" contain no risk word), and a composer-like focused field anywhere too.
    - In an alert or sheet, Return presses the *default* button, not the focused control: with
      Finder's "permanently erase the items in the Trash?" up, Return empties the Trash.
    """
    f = focused_element(snap)
    if MESSAGING_APP_RE.search(snap.app_name or ""):
        return True
    if f is not None and COMPOSER_RE.search(f"{f.label or ''} {f.context or ''}"):
        return True
    for e in snap.elements:
        if e.role == "button" and _DIALOG_RE.search(e.context or "") and is_risky_text(e.label):
            return True
    return bool(snap.window_title) and "?" in (snap.window_title or "") and is_risky_text(snap.window_title)


def _top_p(resp: SystemOneResponse | None, qid: str, label: str | None = None) -> float:
    a = resp.answers.get(qid) if resp is not None else None
    if not isinstance(a, ChoiceAnswer):
        return 0.0
    return float(a.probabilities.get(label if label is not None else a.choice, 0.0))


def _noul(resp: SystemOneResponse | None, qid: str) -> float | None:
    a = resp.answers.get(qid) if resp is not None else None
    return float(a.noul) if isinstance(a, NoulAnswer) else None


def gate(decision: Decision, snap: Snapshot | None, cfg: HarnessConfig, said: str | None = None) -> Decision:
    """Upgrade an `act` to confirm / clarify / deny when it is risky. Other verdicts pass through."""
    a = decision.action
    if decision.verdict != "act" or a is None or a.kind in PASSTHROUGH:
        return decision
    T = cfg.thresholds
    if snap is None and a.kind in SCREEN_KINDS:
        return replace(decision, verdict="clarify", reason=f"{a.describe()}: the screen could not be read")
    risk = classify_risk(a, snap, cfg, said)
    if risk == RiskLevel.DENY:
        return replace(decision, verdict="deny", risk=risk, reason=f"denied: {deny_reason(a, snap, cfg)}")
    resp = decision.answers
    if risk == RiskLevel.HIGH:
        ip = _top_p(resp, Q_INTENT)
        tp = _top_p(resp, Q_TARGET, a.target_eid) if a.kind == ActionKind.CLICK else 1.0
        if ip < T.high_risk_intent_p or tp < T.high_risk_target_p:
            return replace(decision, verdict="clarify", risk=risk,
                           reason=f"risky {a.describe()} but not sure enough (intent p={ip:.2f}, target p={tp:.2f})")
        return replace(decision, verdict="confirm", risk=risk, reason=f"risky: confirm {a.describe()}?")
    destructive = _noul(resp, Q_DESTRUCTIVE)
    if destructive is not None and destructive >= T.destructive:
        return replace(decision, verdict="confirm", risk=max(risk, RiskLevel.MEDIUM),
                       reason=f"looks destructive ({destructive:.2f}): confirm {a.describe()}?")
    return replace(decision, risk=risk)


def confirmation_allowed(
    pending: Action, tail: Tail, mark: tuple[Any, ...] | None, heard_at: float | None = None
) -> bool:
    """A "yes" counts only from a different virtual utterance, spoken in reaction to the prompt.

    `mark` is (uid, words heard, time) when the action was proposed. Within that physical
    utterance, words up to the mark were spoken before the prompt existed, and a word heard less
    than CONFIRM_REACTION_S after it was planned before the user could have read the prompt
    ("quit safari yes do it" in one breath). `heard_at` is when the tail's first word was heard.
    """
    if tail.vid == pending.source_vid:
        return False
    if tail.joined:
        # The recognizer never saw a pause before these words (whisper stalled and the tracker ended
        # the previous utterance on a timer): "quit textedit yes do it" may be a single breath.
        return False
    if mark is not None and tail.uid == mark[0]:
        if tail.cursor < mark[1]:
            return False
        if len(mark) > 2 and (heard_at is None or heard_at - mark[2] < CONFIRM_REACTION_S):
            return False
    return True


class RateLimiter:
    """At most `max_actions` executions in any `window_s` window (SPEC §4.7 layer 5)."""

    def __init__(self, max_actions: int = 3, window_s: float = 1.0):
        self.max_actions = max_actions
        self.window_s = window_s
        self._times: deque[float] = deque()

    def wait_s(self, now: float | None = None) -> float:
        """0 if an action may run now, else seconds until one may."""
        now = time.monotonic() if now is None else now
        while self._times and now - self._times[0] >= self.window_s:
            self._times.popleft()
        if len(self._times) < self.max_actions:
            return 0.0
        return self.window_s - (now - self._times[0])

    def record(self, now: float | None = None) -> None:
        self._times.append(time.monotonic() if now is None else now)
