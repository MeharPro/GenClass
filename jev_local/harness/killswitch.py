"""Global kill switch: ⌃⌥⎋ (control-option-escape) stops the voice harness from any app.

Two read-only ways to see the keystroke, tried in order:

1. A listen-only `CGEventTap` on key-down events (needs the Input Monitoring permission for the
   app running Python, which `CGPreflightListenEventAccess()` reports without prompting). The tap
   never modifies or swallows events.
2. Polling `CGEventSourceKeyState` / `CGEventSourceFlagsState` every 30 ms. No permission
   prompt; may see nothing if the OS withholds key state.

Ctrl-C in the terminal always works too (the demo installs a SIGINT handler). The switch runs
on a daemon thread and calls `on_trigger` at most once per press, from that thread: callers hop
to their event loop with `loop.call_soon_threadsafe`.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

ESC_KEYCODE = 53


def _is_combo(flags: int, Q) -> bool:  # noqa: ANN001
    return bool(flags & Q.kCGEventFlagMaskControl) and bool(flags & Q.kCGEventFlagMaskAlternate)


class KillSwitch:
    def __init__(self, on_trigger: Callable[[], None], *, poll_s: float = 0.03, prefer_tap: bool = True):
        self.on_trigger = on_trigger
        self.poll_s = poll_s
        self.prefer_tap = prefer_tap
        self.mode: str = "off"  # "tap" | "poll" | "off"
        self.triggered = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._runloop = None
        self._ready = threading.Event()

    def start(self) -> str:
        """Start listening; returns the mode in use ("tap", "poll" or "off" when Quartz is missing)."""
        try:
            import Quartz as Q  # noqa: N812
        except Exception:  # noqa: BLE001 - not on macOS / no pyobjc: Ctrl-C only
            self.mode = "off"
            return self.mode
        use_tap = self.prefer_tap and bool(getattr(Q, "CGPreflightListenEventAccess", lambda: False)())
        target = self._run_tap if use_tap else self._run_poll
        self._thread = threading.Thread(target=target, args=(Q,), name="jev-killswitch", daemon=True)
        self._thread.start()
        self._ready.wait(2.0)
        return self.mode

    def stop(self) -> None:
        self._stop.set()
        rl = self._runloop
        if rl is not None:
            try:
                import Quartz as Q  # noqa: N812

                Q.CFRunLoopStop(rl)
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(1.0)

    def _fire(self) -> None:
        self.triggered += 1
        try:
            self.on_trigger()
        except Exception:  # noqa: BLE001 - the switch must keep working
            pass

    # ------------------------------------------------------------ event tap

    def _run_tap(self, Q) -> None:  # noqa: ANN001, N803
        holder: dict = {}

        def cb(proxy, etype, event, refcon):  # noqa: ANN001
            if etype in (Q.kCGEventTapDisabledByTimeout, Q.kCGEventTapDisabledByUserInput):
                if holder.get("tap") is not None:
                    Q.CGEventTapEnable(holder["tap"], True)
                return event
            if etype == Q.kCGEventKeyDown:
                code = Q.CGEventGetIntegerValueField(event, Q.kCGKeyboardEventKeycode)
                if code == ESC_KEYCODE and _is_combo(Q.CGEventGetFlags(event), Q):
                    self._fire()
            return event

        tap = Q.CGEventTapCreate(Q.kCGSessionEventTap, Q.kCGHeadInsertEventTap, Q.kCGEventTapOptionListenOnly,
                                 Q.CGEventMaskBit(Q.kCGEventKeyDown), cb, None)
        if tap is None:
            self._run_poll(Q)
            return
        holder["tap"] = tap
        src = Q.CFMachPortCreateRunLoopSource(None, tap, 0)
        rl = Q.CFRunLoopGetCurrent()
        Q.CFRunLoopAddSource(rl, src, Q.kCFRunLoopCommonModes)
        Q.CGEventTapEnable(tap, True)
        self._runloop = rl
        self.mode = "tap"
        self._ready.set()
        while not self._stop.is_set():
            Q.CFRunLoopRunInMode(Q.kCFRunLoopDefaultMode, 0.25, False)
        Q.CGEventTapEnable(tap, False)

    # ------------------------------------------------------------ polling

    def _run_poll(self, Q) -> None:  # noqa: ANN001, N803
        self.mode = "poll"
        self._ready.set()
        state = Q.kCGEventSourceStateHIDSystemState
        down = False
        while not self._stop.is_set():
            try:
                now = bool(Q.CGEventSourceKeyState(state, ESC_KEYCODE)) and _is_combo(Q.CGEventSourceFlagsState(state), Q)
            except Exception:  # noqa: BLE001
                now = False
            if now and not down:
                self._fire()
            down = now
            time.sleep(self.poll_s)
