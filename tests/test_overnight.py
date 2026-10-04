"""Gate logic of scripts/overnight.py, checked against the 2026-09-27 05:04 crash readings."""

import datetime as dt
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("overnight", Path(__file__).parents[1] / "scripts" / "overnight.py")
o = importlib.util.module_from_spec(spec)
sys.modules["overnight"] = o  # dataclasses resolve annotations via sys.modules
spec.loader.exec_module(o)

TRAIN = next(j for j in o.JOBS if j.name == "train")

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     4078.
Pages active:                                  77440.
Pages inactive:                                91540.
Pages speculative:                              8538.
Pages purgeable:                                   2.
Swapins:                                       44429.
Swapouts:                                     139184.
"""


def health(**kw):
    base = dict(t=0.0, ram_free_gb=6.0, ram_free_pct=75.0, swap_used_gb=0.5, swapins_per_s=0.0,
                disk_free_gb=9.0, load1=1.0, window=True, paused=False)
    return o.Health(**{**base, **kw})


def test_parse_vm_stat():
    vs = o.parse_vm_stat(VM_STAT)
    assert vs.page_size == 16384 and vs.swapins == 44429
    assert vs.ram_free_bytes == (4078 + 91540 + 8538 + 2) * 16384


def test_parse_swap():
    assert o.parse_swap_used_gb("vm.swapusage: total = 8192.00M  used = 7080.75M  free = 1111.25M") == 7080.75 / 1024
    assert o.parse_swap_used_gb("total = 2.00G  used = 1.50G  free = 0.50G") == 1.5


def test_crash_conditions_never_start():
    # 05:01:47: 26% free (~2.1 GB), swap 7.9 of 8 GB, 679 swap-ins/s
    b = o.start_blockers(health(ram_free_gb=2.1, ram_free_pct=26, swap_used_gb=7.9, swapins_per_s=679), TRAIN)
    assert any("swap used" in x for x in b) and any("RAM free" in x for x in b) and any("swap-ins" in x for x in b)
    # a momentary 55% free with full swap (what the old gate accepted at 05:03:15) is still refused
    assert o.start_blockers(health(ram_free_gb=4.4, ram_free_pct=55, swap_used_gb=7.9), TRAIN)


def test_start_needs_peak_plus_2gb():
    need = TRAIN.peak_gb + 2
    assert o.start_blockers(health(ram_free_gb=need - 0.1), TRAIN)
    assert o.start_blockers(health(ram_free_gb=need + 0.1), TRAIN) == []


def test_stop_rules_fire_on_crash_readings():
    # 05:02:47: 8.9% free, 1,784 swap-ins/s -> already past the preempt line
    why, _ = o.stop_verdict(health(ram_free_pct=8.9, swapins_per_s=1784), prev_swapins=679)
    assert why
    # 05:03:48: 9% free, 5,314 swap-ins/s -> severe, kill without waiting for a checkpoint
    why, kill = o.stop_verdict(health(ram_free_pct=9.0, swapins_per_s=5314), prev_swapins=1784)
    assert why and kill
    assert o.stop_verdict(health(ram_free_pct=11.9), 0)[0]
    assert o.stop_verdict(health(swapins_per_s=1200), prev_swapins=1100)[0]
    assert o.stop_verdict(health(swapins_per_s=1200), prev_swapins=200)[0] is None  # needs 2 samples
    assert o.stop_verdict(health(), 0) == (None, False)


def test_window_and_next_night():
    assert o.in_window(dt.datetime(2026, 9, 27, 23, 0)) and o.in_window(dt.datetime(2026, 9, 28, 3, 0))
    assert not o.in_window(dt.datetime(2026, 9, 27, 9, 30)) and not o.in_window(dt.datetime(2026, 9, 27, 14, 0))
    # preempted at 05:04 -> tonight 23:00; preempted at 23:30 -> tomorrow 23:00
    assert o.next_window_start(dt.datetime(2026, 9, 27, 5, 4)) == dt.datetime(2026, 9, 27, 23, 0)
    assert o.next_window_start(dt.datetime(2026, 9, 27, 23, 30)) == dt.datetime(2026, 9, 28, 23, 0)


def test_hold_blocks_start_but_not_running_job():
    assert any("HOLD" in x for x in o.start_blockers(health(held=True), TRAIN))
    assert o.stop_verdict(health(held=True), 0) == (None, False)
