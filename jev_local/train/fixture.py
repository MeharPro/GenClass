"""Tiny procedural harness examples in the CONTRACT "D" training format.

This is NOT the data generator (agent D owns jev_local/data). It exists so the encoder tests and
benchmarks can run before real data exists, and so their inputs go through exactly the runtime
path: `build_state`, `build_questions`, `rank_apps`, `extract_text_candidates`,
`extract_url_candidates`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from jev_local.harness.catalog import RISK_RE
from jev_local.harness.questions import build_questions, rank_apps
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates
from jev_local.harness.state import build_state
from jev_local.harness.types import Element, Snapshot
from jev_local.schema import question_to_json

APPS = [
    "Safari", "Notes", "Mail", "Finder", "Slack", "Spotify", "Calendar", "Messages", "Music", "Photos",
    "Preview", "Xcode", "Visual Studio Code", "Google Chrome", "Reminders", "Pages", "Numbers", "Keynote",
    "Zoom", "Discord", "Figma", "Notion", "Maps", "Weather", "Books", "TextEdit", "Calculator", "Podcasts",
]
LABELS = [
    "Send", "Reply", "Reply All", "Forward", "Archive", "Delete", "Cancel", "OK", "Save", "Don't Save",
    "New Message", "Search", "Compose", "Settings", "Share", "Back", "Next", "Play", "Pause", "Subscribe",
    "Sign In", "Download", "Upload", "Attach", "Bold", "Italic", "Inbox", "Drafts", "Sent", "Trash",
    "Close", "Minimize", "Zoom", "Refresh", "Bookmarks", "History", "Downloads", "Home", "Profile", "Help",
]
ROLES = ["button", "button", "button", "link", "text field", "checkbox", "menu item", "tab", "row"]
CONTEXTS = [None, None, "toolbar", "sidebar", "dialog"]
PAYLOADS = ["hello world", "see you at noon", "the quarterly report", "thanks so much", "meeting notes",
            "cheap flights to tokyo", "pasta recipes", "weather tomorrow"]
SIDE_TALK = ["can you pass the salt", "I think it's fine honestly", "what did you have for lunch",
             "yeah that sounds good to me", "hold on I'm on the phone"]
KEYS = {"escape": "escape", "enter": "return", "tab": "tab", "copy": "cmd+c", "paste": "cmd+v", "save": "cmd+s"}
FOLDERS = ["downloads", "documents", "desktop", "pictures"]


def make_snapshot(rng: random.Random, n_elements: int) -> Snapshot:
    labels = rng.sample(LABELS, k=min(n_elements, len(LABELS)))
    while len(labels) < n_elements:
        labels.append(f"{rng.choice(LABELS)} {len(labels)}")
    focus = rng.randrange(n_elements)
    els = []
    for i, lab in enumerate(labels):
        role = rng.choice(ROLES)
        els.append(Element(
            eid=f"e{i + 1:02d}", role=role, label=lab,
            value=rng.choice(PAYLOADS) if role == "text field" and rng.random() < 0.3 else None,
            context=rng.choice(CONTEXTS), focused=(i == focus), enabled=rng.random() > 0.05,
            actions=("AXPress",),
        ))
    app = rng.choice(APPS)
    return Snapshot(app_name=app, bundle_id=f"com.example.{app.lower().replace(' ', '')}", pid=100,
                    window_title=rng.choice([None, "Inbox", "Untitled", "Home"]), elements=tuple(els),
                    taken_at=0.0, focused_eid=els[focus].eid)


def _command(rng: random.Random, snap: Snapshot) -> tuple[str, dict[str, Any], bool]:
    """-> (transcript, labels, keep). Labels follow CONTRACT "D" semantics for this small grammar."""
    el = rng.choice(snap.elements)
    app = rng.choice(APPS)
    kind = rng.choice(["open_app", "click", "type_text", "scroll_down", "press_key", "open_folder", "none", "wait"])
    lab: dict[str, Any] = {"intent": kind}
    if kind == "open_app":
        text = rng.choice(["open {a}", "launch {a}", "switch to {a}", "please open {a}"]).format(a=app)
        lab.update(app=app)
    elif kind == "click":
        text = rng.choice(["click {l}", "press the {l} button", "click on {l}", "hit {l}"]).format(l=el.label.lower())
        lab.update(target=el.eid)
    elif kind == "type_text":
        pay = rng.choice(PAYLOADS)
        text = rng.choice(["type {p}", "write {p}", "enter {p}"]).format(p=pay)
        lab.update(text_span=pay)
    elif kind == "scroll_down":
        lvl = rng.randrange(3)
        text = ["scroll down a little", "scroll down a page", "scroll all the way down"][lvl]
        lab.update(scroll_amount=lvl)
    elif kind == "press_key":
        spoken = rng.choice(list(KEYS))
        text = f"press {spoken}"
        lab.update(key=KEYS[spoken])
    elif kind == "open_folder":
        f = rng.choice(FOLDERS)
        text = f"open my {f} folder"
        lab.update(folder=f)
    elif kind == "wait":
        text = rng.choice(["open", "click the", "type", "press"])
    else:
        text = rng.choice(SIDE_TALK)
    return text, lab, True


def make_example(rng: random.Random, idx: int, n_elements: int | None = None) -> dict[str, Any] | None:
    snap = make_snapshot(rng, n_elements or rng.randint(5, 30))
    text, lab, _ = _command(rng, snap)
    kind = lab["intent"]
    apps = rank_apps(text, APPS, running=[snap.app_name], max_n=24)
    tc = extract_text_candidates(text)
    uc = extract_url_candidates(text)
    state = build_state(text, snap)
    qs = build_questions(snap, apps, tc, uc)
    labels: dict[str, Any] = {
        "intent": {"type": "choice", "label": kind},
        "complete": {"type": "noul", "p": 0.0 if kind == "wait" else 1.0},
        "is_command": {"type": "noul", "p": 0.0 if kind == "none" else 1.0},
        "target": {"type": "choice", "label": lab.get("target", "none")},
        "app": {"type": "choice", "label": lab.get("app", "none")},
        "key": {"type": "choice", "label": lab.get("key", "none")},
        "folder": {"type": "choice", "label": lab.get("folder", "none")},
    }
    target_label = next((e.label for e in snap.elements if e.eid == lab.get("target")), "")
    risky = kind == "click" and bool(RISK_RE.search(target_label))
    labels["destructive"] = {"type": "noul", "p": 1.0 if risky else 0.0}
    if "text_span" in qs:
        gold = lab.get("text_span", "none")
        if gold != "none" and gold not in qs["text_span"].criteria:
            return None  # contract: drop when the payload is not among the candidates
        labels["text_span"] = {"type": "choice", "label": gold}
    if "url_span" in qs:
        labels["url_span"] = {"type": "choice", "label": "none"}
    if kind == "scroll_down":
        labels["scroll_amount"] = {"type": "score", "level": lab["scroll_amount"]}
    return {
        "id": f"fx-{idx:06d}",
        "split": "train",
        "family": f"fixture/{kind}",
        "state": state,
        "questions": {qid: question_to_json(q) for qid, q in qs.items()},
        "labels": labels,
        "meta": {"full_text": text, "prefix": kind == "wait", "n_words": len(text.split()),
                 "gold": {"kind": kind, "target": lab.get("target")}},
    }


def make_examples(n: int, seed: int = 0, n_elements: int | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < n:
        ex = make_example(rng, i, n_elements)
        i += 1
        if ex is not None:
            out.append(ex)
    return out


def write_jsonl(path: str | Path, examples: list[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return p


def write_splits(out_dir: str | Path, n_train: int, n_dev: int, seed: int = 0, n_elements: int | None = None) -> Path:
    d = Path(out_dir)
    write_jsonl(d / "train.jsonl", make_examples(n_train, seed, n_elements))
    write_jsonl(d / "dev.jsonl", make_examples(n_dev, seed + 1000, n_elements))
    return d
