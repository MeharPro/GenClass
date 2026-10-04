"""replay: event timing, the mid-sentence report, expectations, script loading, virtual time."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from jev_local.harness.fakes import scripted_factory
from jev_local.harness.replay import (
    ReplayUtterance,
    VirtualTimeLoop,
    load_script,
    replay,
    run_replay,
    run_virtual,
)


class Recorder:
    """A controller stand-in that records event timing."""

    def __init__(self, clock):
        from jev_local.harness.types import Thresholds

        self.clock, self.T, self.events = clock, Thresholds(), []
        self.history, self.decisions, self.stats = [], [], {}
        self.stream = type("S", (), {"revisions": []})()

    async def on_event(self, ev):
        self.events.append((round(self.clock(), 3), ev.kind, ev.uid, ev.text))

    async def drain(self):
        pass

    def idle(self):
        return True

    async def aclose(self):
        pass


def test_event_timing():
    rec = {}

    def factory(clock):
        rec["c"] = Recorder(clock)
        return rec["c"]

    run_replay([ReplayUtterance("open notes now"), ReplayUtterance("go back", final="")], factory,
               word_ms=100, final_delay_ms=300, pause_ms=1000)
    ev = rec["c"].events
    assert ev[:6] == [
        (0.0, "speech_start", "u1", ""),
        (0.1, "partial", "u1", "open"),
        (0.2, "partial", "u1", "open notes"),
        (0.3, "partial", "u1", "open notes now"),
        (0.6, "final", "u1", "open notes now"),
        (0.6, "speech_end", "u1", ""),
    ]
    # next utterance starts pause_ms after the final; it has no final at all
    assert ev[6] == (1.6, "speech_start", "u2", "")
    assert [e[1] for e in ev[7:]] == ["partial", "partial", "speech_end"]


def test_report_mid_sentence_metrics_and_summary():
    rep = run_replay([ReplayUtterance("scroll down and go back")], scripted_factory())
    u = rep.utterances[0]
    assert [a.kind for a in u.actions] == ["scroll_down", "go_back"]
    first = u.actions[0]
    assert first.before_final and first.at_word == 2 and first.t_ms < 0
    s = rep.summary()
    assert s["actions"] == 2 and s["closed_set_before_final_rate"] == 1.0
    assert s["last_word_to_action_ms_p50"] is not None
    table = rep.table()
    assert "scroll_down" in table and "yes" in table


def test_expectations_report_failures():
    rep = run_replay([ReplayUtterance("scroll down", expect=[{"kind": "scroll_up"}]),
                      ReplayUtterance("can you pass the salt", expect=[]),
                      ReplayUtterance("go back", expect=[{"kind": "go_back", "max_t_ms": -1000}])],
                     scripted_factory())
    assert not rep.ok
    fails = rep.failures
    assert any("kind='scroll_down', expected 'scroll_up'" in f for f in fails)
    assert any("t_ms" in f for f in fails)
    assert rep.utterances[1].ok


def test_load_script(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join([
        "# comment",
        json.dumps({"text": "open notes and type hi", "expect": [{"kind": "open_app"}, {"kind": "type_text"}]}),
        json.dumps("scroll down"),
        "",
    ]))
    script = load_script(p)
    assert [u.text for u in script] == ["open notes and type hi", "scroll down"]
    rep = run_replay(script, scripted_factory())
    assert rep.ok, rep.failures


def test_virtual_time_is_fast_and_deterministic():
    t0 = time.perf_counter()
    reps = [run_replay([ReplayUtterance("open notes and type hello world")] * 3, scripted_factory())
            for _ in range(2)]
    assert time.perf_counter() - t0 < 5.0  # ~13 s of speech, replayed twice
    assert [a.t_ms for a in reps[0].actions] == [a.t_ms for a in reps[1].actions]


def test_virtual_loop_freezes_time_during_threads():
    async def main():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await asyncio.to_thread(time.sleep, 0.05)  # real work: zero virtual time
        t1 = loop.time()
        await asyncio.sleep(10)  # virtual: instant
        return t1 - t0, loop.time() - t1

    assert isinstance(VirtualTimeLoop(), asyncio.AbstractEventLoop)
    dt_thread, dt_sleep = run_virtual(main())
    assert dt_thread == 0.0 and dt_sleep == pytest.approx(10.0)


@pytest.mark.slow
def test_real_time_replay_matches_virtual():
    script = [ReplayUtterance("open notes and type hello world")]
    real = run_replay(script, scripted_factory(), virtual=False)
    virt = run_replay(script, scripted_factory())
    assert [a.kind for a in real.actions] == [a.kind for a in virt.actions]
    for r, v in zip(real.actions, virt.actions):
        assert abs(r.t_ms - v.t_ms) < 150  # engine + scheduling overhead on a busy machine
