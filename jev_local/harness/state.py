"""Build the `state` object for a harness decision.

Key order is stable context first and the transcript last. Screen elements are NOT in the state:
they are the options of the `target` question, so each element is scored against the transcript
directly (cheaper and easier for a small encoder than pointer-matching ids back into the state).
"""

from __future__ import annotations

from typing import Any, Sequence

from jev_local.harness.questions import element_line
from jev_local.harness.types import Action, ActionRecord, Snapshot


def describe_record(r: ActionRecord) -> str:
    a = r.action
    arg = a.target_label or a.app or (f'"{a.text}"' if a.text else None) or a.key or a.url or a.folder
    s = a.kind.value.replace("_", " ")
    return f"{s} {arg}" if arg else s


def build_state(
    transcript: str,
    snap: Snapshot | None,
    history: Sequence[ActionRecord] = (),
    pending: Action | None = None,
) -> dict[str, Any]:
    if snap is not None:
        screen = snap.app_name + (f', window "{snap.window_title}"' if snap.window_title else "")
        focused = next((element_line(e) for e in snap.elements if e.focused), "nothing")
    else:
        screen, focused = "unknown", "nothing"
    return {
        "screen": screen,
        "focused": focused,
        "recent_actions": [describe_record(r) for r in history[-3:]] or "none",
        "pending": f"{pending.describe()} is waiting for the user to confirm" if pending else "none",
        "transcript": transcript,
    }
