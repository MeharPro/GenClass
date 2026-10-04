"""AuditLog: date-stamped JSONL, JSON conversion of harness types, redaction, never raising."""

from __future__ import annotations

import datetime as dt
import json

from jev_local.harness.fakes import answers_for, finder_snapshot
from jev_local.harness.log import AuditLog, redact_transcript, to_jsonable
from jev_local.harness.questions import build_questions
from jev_local.harness.types import Action, ActionKind, Decision, RiskLevel, Thresholds


def test_date_stamped_file_and_record_shape(tmp_path):
    t = dt.datetime(2026, 9, 24, 23, 59, 59).timestamp()
    log = AuditLog(tmp_path, redact=False, wallclock=lambda: t)
    log.write("decision", verdict="act", n=3, th=Thresholds())
    assert log.flush()
    p = tmp_path / "2026-09-24.jsonl"
    assert p.exists() and log.path_for() == p
    rec = json.loads(p.read_text().strip())
    assert rec["kind"] == "decision" and rec["verdict"] == "act" and rec["n"] == 3
    assert rec["ts"].startswith("2026-09-24T23:59:59")
    assert rec["th"]["debounce_ms"] == 120
    log.wallclock = lambda: t + 2  # midnight rolls over mid-session
    log.write("exec")
    log.close()
    assert (tmp_path / "2026-09-25.jsonl").exists()
    assert log.n_written == 2


def test_to_jsonable_handles_harness_types():
    snap = finder_snapshot()
    qs = build_questions(snap, ["Notes", "Safari"], [], [])
    resp = answers_for(qs, {"intent": "scroll_down"})
    d = Decision("act", Action(ActionKind.SCROLL_DOWN, "u1+0", 0.9, amount=1), "why", 7, answers=resp,
                 risk=RiskLevel.MEDIUM)
    j = to_jsonable(d)
    json.dumps(j)
    assert j["action"]["kind"] == "scroll_down" and j["risk"] == 1
    assert j["answers"]["answers"]["intent"]["choice"] == "scroll_down"
    assert len(j["answers"]["answers"]["target"]["top"]) == 3  # compact: top-3 only
    assert to_jsonable((1, 2.123456789, float("nan"))) == [1, 2.1235, None]


def test_redaction_of_payloads_everywhere(tmp_path):
    log = AuditLog(tmp_path, redact=True)
    a = Action(ActionKind.TYPE_TEXT, "u1+0", 0.9, text="my pin is 4321")
    log.write("decision", text="type my pin is 4321", reason=f"{a.describe()} (conf=0.9)", action=a,
              answers={"text_span": {"choice": "my pin is 4321", "confidence": 0.8, "top": {"my pin is 4321": 0.9}}})
    log.write("transcript", text="open notes and type my pin is 4321")
    log.flush()
    raw = "".join(p.read_text() for p in tmp_path.glob("*.jsonl"))
    assert "4321" not in raw
    recs = log.read()
    assert recs[0]["action"]["text"] == "<14 chars>"
    assert recs[0]["answers"]["text_span"]["choice"] == "<14 chars>"
    assert recs[1]["text"] == "open notes and type <14 chars>"


def test_redact_transcript():
    assert redact_transcript("open safari") == "open safari"
    assert redact_transcript("search for cheap flights") == "search for <13 chars>"
    assert redact_transcript("type") == "type"


def test_no_redaction_when_disabled(tmp_path):
    log = AuditLog(tmp_path, redact=False)
    log.write("exec", action=Action(ActionKind.TYPE_TEXT, "u1+0", 0.9, text="hello"))
    assert log.read()[0]["action"]["text"] == "hello"


def test_write_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log = AuditLog(blocker / "sub")  # a directory under a regular file cannot exist
    log.write("decision", a=1)
    assert log.flush()  # the failed record counts as done; flush does not hang
    assert not log.enabled and log.error
    log.write("decision", a=2)  # disabled, still no exception
    off = AuditLog(tmp_path / "off", enabled=False)
    off.write("x")
    assert not (tmp_path / "off").exists()


def test_background_writer_keeps_order_and_synchronous_mode_works(tmp_path):
    log = AuditLog(tmp_path / "bg", redact=False)
    for i in range(500):
        log.write("n", i=i)
    assert [r["i"] for r in log.read()] == list(range(500))
    sync = AuditLog(tmp_path / "sync", redact=False, background=False)
    sync.write("n", i=1)
    assert (tmp_path / "sync").exists() and sync.n_written == 1


def test_unserializable_record_is_dropped_not_raised(tmp_path):
    class Weird:
        def __repr__(self):
            raise RuntimeError("no repr")

    log = AuditLog(tmp_path)
    log.write("x", thing=Weird())
    assert log.error and "serialize" in log.error
