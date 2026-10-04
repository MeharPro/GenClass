"""Stream: consumed-prefix cursor, virtual utterances, revision handling, fired set."""

from __future__ import annotations

from jev_local.harness.stream import Stream, core_words, match_prefix
from jev_local.harness.types import TranscriptEvent

_seq = 0


def ev(text: str, uid: str = "u1", kind: str = "partial") -> TranscriptEvent:
    global _seq
    _seq += 1
    return TranscriptEvent(kind=kind, seq=_seq, uid=uid, text=text, t_mono=0.0)  # type: ignore[arg-type]


def test_tail_vid_and_empty():
    s = Stream()
    t = s.update(ev("open"), 0.0)
    assert t is not None and t.text == "open" and t.vid == "u1+0" and t.cursor == 0 and not t.is_final
    assert t.uid == "u1"
    assert s.update(ev("", kind="partial"), 0.1) is None  # no words -> nothing to decide
    assert s.update(TranscriptEvent("speech_start", 0, "", "", 0.0), 0.1) is None


def test_consume_starts_virtual_utterance_and_strips_chain_words():
    s = Stream()
    t = s.update(ev("open notes"), 0.0)
    assert t is not None
    s.consume(2)
    assert s.vid == "u1+1"
    assert s.update(ev("open notes and"), 0.3) is None  # only a chain word is new
    t = s.update(ev("open notes and then um type hello"), 0.6)
    assert t is not None and t.text == "type hello" and t.vid == "u1+1" and t.cursor == 5


def test_prefix_match_ignores_case_punctuation_and_fillers():
    s = Stream()
    s.update(ev("open notes"), 0.0)
    s.consume(2)
    assert s.update(ev("Open Notes."), 0.1) is None
    assert s.update(ev("um, Open, notes"), 0.2) is None
    t = s.update(ev("Um open Notes, and type hi"), 0.3)
    assert t is not None and t.text == "type hi"
    assert s.revisions == []


def test_dropped_chain_word_in_revision_is_not_a_rewrite():
    s = Stream()
    s.update(ev("open notes and"), 0.0)
    s.consume(3)  # consumed_for consumes the trailing "and"
    t = s.update(ev("open notes type hello"), 0.3)  # the recognizer dropped "and"
    assert t is not None and t.text == "type hello"
    assert s.revisions == []


def test_revision_of_consumed_words_freezes_and_logs():
    logs = []
    s = Stream(log=lambda kind, **f: logs.append((kind, f)))
    s.update(ev("open notes"), 0.0)
    s.consume(2)
    assert s.update(ev("open nodes and type hello"), 0.3) is None
    assert s.current(0.4) is None  # frozen: the rest of a rewritten utterance is not trusted
    assert len(s.revisions) == 1 and s.revisions[0]["consumed"] == "open notes"
    assert any(k == "revised_after_act" for k, _ in logs)
    assert s.update(ev("open nodes and type hello world"), 0.5) is None
    assert len(s.revisions) == 1  # logged once per freeze
    # the recognizer flips back into agreement: the stream resumes after the consumed prefix
    t = s.update(ev("open notes and type hello world"), 0.7)
    assert t is not None and t.text == "type hello world"


def test_transient_retraction_is_not_a_revision():
    s = Stream()
    s.update(ev("open notes"), 0.0)
    s.consume(2)
    assert s.update(ev("open"), 0.1) is None
    assert s.revisions == [] and not s.frozen
    t = s.update(ev("open notes then scroll down"), 0.2)
    assert t is not None and t.text == "scroll down"


def test_final_sets_is_final_and_same_text_final_has_no_new_words():
    s = Stream()
    s.update(ev("scroll down"), 0.0)
    t = s.update(ev("scroll down", kind="final"), 0.5)
    assert t is not None and t.is_final and t.silent_ms == 500  # the words did not change
    s.consume(2)
    assert s.update(ev("Scroll down.", kind="final"), 0.6) is None


def test_silent_ms_counts_from_last_word_change():
    s = Stream()
    s.update(ev("type hello"), 1.0)
    s.update(ev("type hello"), 1.3)  # identical partial: not a change
    t = s.current(1.9)
    assert t is not None and t.silent_ms == 900
    assert not s.changed


def test_new_uid_resets_and_late_events_are_ignored():
    logs = []
    s = Stream(log=lambda kind, **f: logs.append(kind))
    s.update(ev("scroll down", uid="u1", kind="final"), 0.0)
    s.consume(2)
    t = s.update(ev("go back", uid="u2"), 3.0)
    assert t is not None and t.vid == "u2+0" and t.text == "go back"
    assert s.update(ev("scroll down please", uid="u1", kind="final"), 3.1) is None
    assert "late_event" in logs


def test_uid_echo_without_final_inherits_consumed_prefix():
    logs = []
    s = Stream(log=lambda kind, **f: logs.append(kind))
    s.update(ev("open notes", uid="u1"), 0.0)
    s.consume(2)
    # the recognizer restarts under u2 and re-emits the words it already sent (no final for u1)
    assert s.update(ev("open", uid="u2"), 0.5) is None  # too short to tell yet
    assert s.update(ev("open notes", uid="u2"), 0.8) is None
    t = s.update(ev("open notes and type hi", uid="u2"), 1.1)
    assert t is not None and t.text == "type hi"
    assert "uid_echo" in logs


def test_new_uid_that_only_starts_like_the_echo_is_fresh_speech():
    logs = []
    s = Stream(log=lambda kind, **f: logs.append(kind))
    s.update(ev("open notes", uid="u1"), 0.0)
    s.consume(2)
    assert s.update(ev("open", uid="u2"), 0.5) is None
    t = s.update(ev("open safari", uid="u2"), 0.8)
    assert t is not None and t.text == "open safari" and t.cursor == 0
    assert "echo_rejected" in logs


def test_duplicate_final_under_fresh_uid_is_an_echo():
    s = Stream()
    s.update(ev("open notes", uid="u1", kind="final"), 0.0)
    s.consume(2)
    assert s.update(ev("Open notes.", uid="u2", kind="final"), 0.1) is None  # within dup window
    # a genuinely new "open notes" long after the final is a new command
    s2 = Stream()
    s2.update(ev("open notes", uid="u1", kind="final"), 0.0)
    s2.consume(2)
    t = s2.update(ev("open notes", uid="u2"), 5.0)
    assert t is not None and t.text == "open notes"


def test_matches_checks_position_and_words():
    s = Stream()
    t1 = s.update(ev("open notes"), 0.0)
    assert t1 is not None
    s.update(ev("open notes and type"), 0.3)
    assert s.matches(t1, 2)  # the stale tail's words are still at the same place
    s.update(ev("open nodes and type"), 0.4)  # (nothing consumed yet, so this is allowed)
    assert not s.matches(t1, 2)


def test_fired_set_skip_to_end_drop_and_reset():
    s = Stream()
    t = s.update(ev("scroll down and go back"), 0.0)
    assert t is not None
    key = (t.vid, t.cursor, 3)
    assert not s.already_fired(key)
    s.mark_fired(key)
    assert s.already_fired(key)
    s.skip_to_end()
    assert s.current(0.1) is None
    t2 = s.update(ev("scroll down and go back and undo"), 0.2)
    assert t2 is not None and t2.text == "undo"
    s.drop_utterance()
    assert s.update(ev("scroll down and go back and undo and new tab"), 0.3) is None
    t3 = s.update(ev("new tab", uid="u2"), 5.0)
    assert t3 is not None and t3.text == "new tab"
    s.reset()
    assert s.uid == "" and s.current(6.0) is None and not s.already_fired(key)


def test_helpers():
    assert core_words(["Um,", "Open", "Notes!", "--"]) == ["open", "notes"]
    assert match_prefix(["open", "notes", "and"], ["open", "notes"]) == 2
    assert match_prefix(["open"], ["open", "notes"]) == -1
    assert match_prefix(["close", "notes"], ["open", "notes"]) is None
    assert match_prefix(["anything"], []) == 0


def test_echo_of_a_dropped_utterance_stays_dropped():
    s = Stream()
    s.update(ev("open terminal and", uid="u1"), 0.0)
    s.drop_utterance()  # denied
    assert s.update(ev("open terminal and type ls", uid="u1"), 0.3) is None
    # the recognizer restarts under a new uid, re-sending the denied words and more
    assert s.update(ev("open terminal and type ls", uid="u2"), 0.8) is None
    assert s.update(ev("open terminal and type ls now", uid="u2"), 1.0) is None
    t = s.update(ev("scroll down", uid="u3"), 5.0)
    assert t is not None and t.text == "scroll down"
