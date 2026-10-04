"""Generate Python-harness fixtures for GenClass's JS parity tests (torch-free; runs on the Mac).

    .venv/bin/python extension/genclass/scripts/make_py_fixtures.py

Writes extension/genclass/test/fixtures/{spans,questions,policy,stream}_py.json. Choice criteria and probabilities
are written as [label, value] pairs so integer-like labels keep their order in JS.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from jev_local.harness import spans
from jev_local.harness.catalog import INTENTS, KEYS, SCROLL_LEVELS
from jev_local.harness.policy import PolicyContext, evaluate_policy, pick_of
from jev_local.harness.questions import build_questions, element_line, rank_apps
from jev_local.harness.safety import gate
from jev_local.harness.state import build_state
from jev_local.harness.stream import Stream
from jev_local.harness.types import Action, ActionKind, ActionRecord, Element, HarnessConfig, Snapshot, Tail, Thresholds, TranscriptEvent
from jev_local.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "extension" / "genclass" / "test" / "fixtures"
rng = random.Random(1234)

# ---------------------------------------------------------------- transcripts
cases = json.loads((ROOT / "tests" / "fixtures" / "harness_cases.json").read_text())
reqs = json.loads((OUT / "requests50.json").read_text())
texts = [c["transcript"] for c in cases["cases"]] + [r["state"]["transcript"] for r in reqs]
texts += [
    "search for alan turing", "search for", "type hello world and press enter", 'type "salt and pepper" in the search box',
    "um please search google for best pizza near me thanks", "go to wikipedia dot org and scroll down", "open youtube website",
    "visit github.com/anthropics please", "navigate to news dot ycombinator dot com", "w w w dot example dot com",
    "the second one", "number 3", "option two", "3", "last one", "pick for", "click the blue button",
    "type running late. see you soon", "type hello world. Press enter.", "open notes. scroll down.", "undo. scroll down",
    "Open. Notes.", "find me cheap flights to tokyo in the search field and then press enter",
    "write down buy milk in the note", "put 42 in the quantity field", "enter my email in the email box",
    "look up the weather for tomorrow", "search amazon for usb c cables", "search the web for jev typesafe",
    "go to http://localhost:8080/test and click login", "scroll down a little bit please", "scroll all the way to the bottom",
    "press command shift z", "close this tab", "new tab and go to gmail.com", "go back", "go forward to the next page",
    "hey can you pass the salt", "I think it's fine", "quit safari", "delete all the emails", "yes do it", "no don't",
    "confirm", "cancel that", "café au lait", "type 'it's fine' now", "dictate dear bob, thanks for the note.",
    "search for “quoted thing” please", "type    spaced   out", "open the downloads folder and scroll down",
    "click on sign in and then type my name", "and then scroll up", "um uh so open safari",
]
texts = list(dict.fromkeys(t for t in texts if t))

span_fx = []
for t in texts:
    words = t.split()
    tc = spans.extract_text_candidates(t)
    cf = {}
    for intent in ("click", "open_app", "type_text", "search_web", "undo", "scroll_down", "none", "cancel", "confirm"):
        cf[intent] = spans.consumed_for(words, intent)
    cf_span = {c: spans.consumed_for(words, "type_text", c) for c in tc[:4]}
    span_fx.append({
        "text": t,
        "text_cands": tc,
        "url_cands": spans.extract_url_candidates(t),
        "spoken_url": spans.normalize_spoken_url(t),
        "pick": [spans.parse_candidate_pick(t, n) for n in (1, 3, 5, 12)],
        "consumed": cf,
        "consumed_span": [[c, n] for c, n in cf_span.items()],
        "strip_chain": spans.strip_leading_chain(words),
        "ends": [spans.ends_sentence(w) for w in words],
    })
(OUT / "spans_py.json").write_text(json.dumps(span_fx, indent=0, ensure_ascii=False))

# ---------------------------------------------------------------- questions + state


def snap_from(screen: dict) -> Snapshot:
    els = tuple(Element(eid=e["eid"], role=e["role"], label=e.get("label", ""), value=e.get("value"),
                        context=e.get("context"), focused=e.get("focused", False), enabled=e.get("enabled", True),
                        secure=e.get("secure", False)) for e in screen["elements"])
    focused = next((e.eid for e in els if e.focused), None)
    return Snapshot(app_name=screen["app_name"], bundle_id=screen.get("bundle_id", ""), pid=1,
                    window_title=screen.get("window_title"), elements=els, taken_at=0.0, focused_eid=focused)


def qjson(qs):
    out = {}
    for k, q in qs.items():
        d = q.model_dump(mode="json", exclude_none=False)
        if d["type"] == "choice":
            d["criteria"] = [[lab, desc] for lab, desc in q.criteria.items()]
        out[k] = d
    return out


q_fx = []
hist = [ActionRecord(said="open safari", action=Action(ActionKind.OPEN_APP, "u1+0", 0.9, app="Safari"), outcome="ok", t=0.0),
        ActionRecord(said="type hi", action=Action(ActionKind.TYPE_TEXT, "u1+1", 0.9, text="hi there"), outcome="ok", t=0.0),
        ActionRecord(said="click send", action=Action(ActionKind.CLICK, "u2+0", 0.9, target_eid="e01", target_label='button "Send"'), outcome="ok", t=0.0),
        ActionRecord(said="scroll", action=Action(ActionKind.SCROLL_DOWN, "u3+0", 0.9, amount=1), outcome="ok", t=0.0)]
pend = Action(ActionKind.QUIT_APP, "u4+0", 0.9, app="Safari")
pend2 = Action(ActionKind.CLICK, "u4+0", 0.9, target_eid="e02", target_label="button \"Don't Save\"")
for c in cases["cases"]:
    scr = cases["screens"][c["screen"]]
    snap = snap_from(scr)
    t = c["transcript"]
    apps = rank_apps(t, cases["apps"], running=scr.get("running", []), max_n=24)
    qs = build_questions(snap, apps, spans.extract_text_candidates(t), spans.extract_url_candidates(t))
    h = hist[: rng.randint(0, 4)]
    p = rng.choice([None, pend, pend2])
    q_fx.append({
        "id": c["id"], "screen": scr, "transcript": t, "apps": apps, "questions": qjson(qs),
        "element_lines": [element_line(e) for e in snap.elements],
        "history": [{"kind": r.action.kind.value, "target_label": r.action.target_label, "app": r.action.app,
                     "text": r.action.text, "key": r.action.key, "amount": r.action.amount} for r in h],
        "pending": None if p is None else {"kind": p.kind.value, "app": p.app, "target_label": p.target_label, "target_eid": p.target_eid},
        "state": build_state(t, snap, h, p),
    })
rank_fx = [{"text": t, "running": ["Safari", "Notes"], "out": rank_apps(t, cases["apps"], running=["Safari", "Notes"], max_n=24)}
           for t in texts[:60]]
(OUT / "questions_py.json").write_text(json.dumps({"cases": q_fx, "apps": cases["apps"], "rank": rank_fx}, indent=0, ensure_ascii=False))

# ---------------------------------------------------------------- policy + safety (synthetic answers)


def rand_dist(labels, top=None, sharp=None):
    sharp = sharp if sharp is not None else rng.choice([0.3, 0.6, 0.8, 0.95, 0.99])
    top = top if top is not None else rng.choice(labels)
    rest = [lab for lab in labels if lab != top]
    p = {top: sharp}
    w = [rng.random() for _ in rest]
    s = sum(w) or 1.0
    for lab, x in zip(rest, w):
        p[lab] = (1 - sharp) * x / s
    return {lab: round(p[lab], 2) for lab in labels}


def choice_ans(labels, top=None, sharp=None):
    pr = rand_dist(labels, top, sharp)
    best = max(labels, key=lambda lab: (pr[lab], -labels.index(lab)))
    k = len(labels)
    conf = round(max(0.0, min(1.0, (k * pr[best] - 1) / (k - 1))), 2) if k > 1 else 1.0
    return ChoiceAnswer(choice=best, confidence=conf, probabilities=pr)


def score_ans():
    pr = rand_dist(["0", "1", "2"])
    return ScoreAnswer(score=round(sum(int(k) * v for k, v in pr.items()), 2), confidence=0.5,
                       legend={str(i): s for i, s in enumerate(SCROLL_LEVELS)}, probabilities=pr)


pol_fx = []
cfg = HarnessConfig()
T = Thresholds()
screens = list(cases["screens"].values())
web_screen = {"app_name": "Google Chrome", "window_title": "Checkout - Shop", "elements": [
    {"eid": "e01", "role": "button", "label": "Place order"}, {"eid": "e02", "role": "link", "label": "Home"},
    {"eid": "e03", "role": "text field", "label": "Password", "secure": True, "focused": True},
    {"eid": "e04", "role": "button", "label": "Search"}, {"eid": "e05", "role": "button", "label": "Delete account", "enabled": False}]}
screens.append(web_screen)
intent_labels = list(INTENTS)
for i in range(700):
    scr = rng.choice(screens + [None])
    snap = snap_from(scr) if scr else None
    t = rng.choice(texts)
    words = tuple(t.split())
    uid = rng.choice(["u1", "u2"])
    tail = Tail(vid=f"{uid}+{rng.randint(0, 2)}", text=t, words=words, cursor=rng.randint(0, 3), is_final=rng.random() < 0.3,
                silent_ms=rng.choice([0, 100, 650, 950, 2000]), uid=uid, joined=rng.random() < 0.1)
    tc = spans.extract_text_candidates(t)
    uc = spans.extract_url_candidates(t)
    apps = rank_apps(t, cases["apps"], max_n=24)
    qs = build_questions(snap, apps, tc, uc)
    ans = {}
    for qid, q in qs.items():
        if q.type == "noul":
            ans[qid] = NoulAnswer(noul=round(rng.choice([0.02, 0.3, 0.55, 0.7, 0.9, 0.99, rng.random()]), 2))
        elif q.type == "score":
            ans[qid] = score_ans()
        else:
            labels = list(q.criteria)
            top = None
            if qid == "intent":
                top = rng.choice(intent_labels)
            elif qid == "text_span" and rng.random() < 0.8:
                top = labels[0]
            ans[qid] = choice_ans(labels, top)
    resp = SystemOneResponse(model="fixture", answers=ans, usage=Usage(input_tokens=0, output_tokens=0))
    pending = rng.choice([None, None, None, pend, pend2, Action(ActionKind.CLICK, tail.vid, 0.9, target_eid="e01", target_label='button "Send"')])
    mark = None
    if pending is not None:
        mark = rng.choice([None, (uid, rng.randint(0, 5), 10.0), ("u9", 1, 10.0)])
    prev = rng.choice([None, (tail.vid, ans["intent"].choice, None)])
    ctx = PolicyContext(prev_pick=prev, pending=pending, pending_mark=mark,
                        tail_heard_at=rng.choice([None, 10.2, 11.5]), stale=rng.random() < 0.15)
    d = evaluate_policy(resp, tail, snap, ctx, T)
    a = d.action
    said = " ".join(tail.words[: a.consumed_words]) if a is not None and a.consumed_words else tail.text
    g = gate(d, snap, cfg, said=said)

    def enc_action(x):
        if x is None:
            return None
        dd = asdict(x)
        dd["kind"] = x.kind.value
        return dd

    def enc_ans(a_):
        if isinstance(a_, ChoiceAnswer):
            return {"type": "choice", "choice": a_.choice, "confidence": a_.confidence, "probabilities": [[k, v] for k, v in a_.probabilities.items()]}
        if isinstance(a_, NoulAnswer):
            return {"type": "noul", "noul": a_.noul}
        return {"type": "score", "score": a_.score, "confidence": a_.confidence, "probabilities": [[k, v] for k, v in a_.probabilities.items()]}

    pol_fx.append({
        "screen": scr, "tail": asdict(tail), "answers": {k: enc_ans(v) for k, v in ans.items()},
        "ctx": {"prev_pick": list(prev) if prev else None, "pending": enc_action(pending), "pending_mark": list(mark) if mark else None,
                "tail_heard_at": ctx.tail_heard_at, "stale": ctx.stale},
        "policy": {"verdict": d.verdict, "reason": d.reason, "retry_in_ms": d.retry_in_ms, "action": enc_action(a),
                   "pick": list(pick_of(a)) if pick_of(a) else None},
        "gate": {"verdict": g.verdict, "reason": g.reason, "risk": int(g.risk)},
    })
(OUT / "policy_py.json").write_text(json.dumps(pol_fx, ensure_ascii=False))

# ---------------------------------------------------------------- stream
st_fx = []
scripts = [
    [("partial", "u1", "open", 0.0), ("partial", "u1", "open notes", 0.2), ("consume", 2, 0.25), ("partial", "u1", "open notes and", 0.4),
     ("partial", "u1", "open notes and type hello", 0.6), ("final", "u1", "Open Notes and type hello.", 1.0)],
    [("partial", "u1", "scroll down", 0.0), ("consume", 2, 0.1), ("partial", "u1", "scroll town", 0.3), ("partial", "u1", "scroll down please", 0.5)],
    [("partial", "u1", "go back", 0.0), ("consume", 2, 0.1), ("final", "u1", "go back", 0.4), ("partial", "u2", "go back", 0.6),
     ("partial", "u2", "go back and scroll down", 0.9)],
    [("partial", "a", "um so open safari", 0.0), ("consume", 2, 0.1), ("partial", "a", "um so open safari then", 0.2),
     ("partial", "a", "um so open safari then new tab", 0.4), ("drop",), ("partial", "a", "um so open safari then new tab and close", 0.6),
     ("partial", "b", "close it", 3.0)],
    [("partial", "x", "click the", 0.0), ("partial", "x", "click", 0.1), ("partial", "x", "click the send button", 0.3), ("consume", 4, 0.35),
     ("partial", "x", "click the send button after that", 0.5), ("partial", "x", "click the send button after that undo", 0.7), ("skip",),
     ("partial", "x", "click the send button after that undo now", 0.9), ("partial", "y", "click the send button", 1.0)],
]
for sc in scripts:
    s = Stream()
    rec = []
    for stp in sc:
        if stp[0] in ("partial", "final"):
            kind, uid, text, t = stp
            tl = s.update(TranscriptEvent(kind=kind, seq=0, uid=uid, text=text, t_mono=t), t)
            rec.append({"op": kind, "uid": uid, "text": text, "t": t, "tail": asdict(tl) if tl else None,
                        "frozen": s.frozen, "dropped": s.dropped, "vid": s.vid, "consumed": s.consumed_text})
        elif stp[0] == "consume":
            s.consume(stp[1])
            rec.append({"op": "consume", "n": stp[1], "t": stp[2], "vid": s.vid, "consumed": s.consumed_text, "cursor": s.cursor})
        elif stp[0] == "drop":
            s.drop_utterance()
            rec.append({"op": "drop", "vid": s.vid})
        elif stp[0] == "skip":
            s.skip_to_end()
            rec.append({"op": "skip", "vid": s.vid})
    st_fx.append(rec)
(OUT / "stream_py.json").write_text(json.dumps(st_fx, ensure_ascii=False))
print("wrote", len(span_fx), "span,", len(q_fx), "question,", len(pol_fx), "policy,", len(st_fx), "stream fixtures")
