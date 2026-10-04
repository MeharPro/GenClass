"""whisper.cpp streaming speech-to-text as a Controller speech source.

`WhisperSource` wraps `whisper-stream` (whisper.cpp's examples/stream, Homebrew build) as a
subprocess and turns its terminal output into `TranscriptEvent`s:
speech_start / partial / final / speech_end. `FileWhisperSource` produces the *same* terminal
output from an audio file with `whisper-cli`, so tests and offline demos exercise exactly the
same parser and utterance tracker as the live microphone.

What whisper-stream prints (examples/stream/stream.cpp, v1.8.x, sliding-window mode `--step > 0`)
-----------------------------------------------------------------------------------------------
- `[Start speaking]\\n` once, then every `step_ms` of audio one inference over a window made of
  the previous window plus the new step (capped at `length_ms + keep_ms`).
- Each result is printed as `ESC[2K\\r` + 100 spaces + `ESC[2K\\r` + text, with no newline:
  the line is *rewritten* every step and earlier words may change.
- Every `n_new_line = max(1, length_ms/step_ms - 1)` steps it prints `\\n` and the window
  restarts from only the last `keep_ms` of audio. The line above is then final, and the next
  line starts with the next words (the ~200 ms overlap can repeat a word).
- Silence is still transcribed: `[BLANK_AUDIO]`, `(silence)`, `[Music]`, `♪`, `Thank you.` ...
- There is no VAD in this mode and no word timestamps.

From that, this module builds one growing utterance at a time:

    WhisperStreamParser   bytes -> ("window", text) / ("commit",)   (ANSI clears, \\r, \\n)
    clean_text            drops bracketed/parenthesised annotations and music marks
    UtteranceTracker      windows -> utterance events:
      - utterance text = words from earlier lines of this utterance + the new part of the window;
      - words of utterances already finalised in the same line are skipped by aligning them
        (difflib) against each new window, so re-punctuation or small rewrites don't leak them;
      - a partial is emitted only when the normalised words change (case/punctuation flips do
        not reset the controller's silence timers);
      - a final + speech_end after the words stop changing for `final_windows` windows and
        `final_after_ms` (silence: whisper keeps re-transcribing the same words);
      - a new utterance whose words are a known hallucination ("thank you.", "you") is ignored.

Revisions of words already acted on are not handled here: `Stream` keeps a consumed-prefix
cursor (case/punctuation insensitive) and logs `revised_after_act`, so nothing runs twice.

Process hygiene: whisper-stream rewrites its line every step, so if this process dies the next
write gets SIGPIPE and it exits within one step. `stop()` sends SIGTERM (whisper-stream's SDL
loop exits on it) and SIGKILL after a grace period; an atexit hook and a pidfile
(`~/.jev-local/whisper-stream.pid`, checked by `kill_stale()`) cover the rest.
"""

from __future__ import annotations

import asyncio
import atexit
import difflib
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable, Sequence

from jev_local.harness.types import TranscriptEvent

STREAM_BIN = shutil.which("whisper-stream") or "/opt/homebrew/bin/whisper-stream"
CLI_BIN = shutil.which("whisper-cli") or "/opt/homebrew/bin/whisper-cli"
MODEL_DIR = Path.home() / ".jev-local" / "models" / "whisper"
PIDFILE = Path.home() / ".jev-local" / "whisper-stream.pid"
SAMPLE_RATE = 16000
CLEAR = "\x1b[2K"

# Low-latency defaults: a 500 ms step keeps base.en (~150-300 ms per window on an M1 with Metal)
# inside real time; an 8 s line means a window reset (where a word can be cut) every 7.5 s.
DEFAULT_STEP_MS = 500
DEFAULT_LENGTH_MS = 8000
DEFAULT_KEEP_MS = 200


def model_path(name: str = "base.en", model_dir: Path = MODEL_DIR) -> Path:
    """'base.en' -> ~/.jev-local/models/whisper/ggml-base.en.bin (a path is returned unchanged)."""
    p = Path(name).expanduser()
    if p.suffix == ".bin" or p.exists():
        return p
    return model_dir / f"ggml-{name}.bin"


# ---------------------------------------------------------------- text cleaning

# Non-speech annotations whisper emits for silence, noise or music.
_ANNOT_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪[^♪]*♪|[♪♫]")
_UNCLOSED_RE = re.compile(r"[\[(*][^\])*]*$")  # "(music" cut off at the end of a window
_SPACE_RE = re.compile(r"\s+")
_NORM_RE = re.compile(r"[\W_]+", re.UNICODE)
# Whole-utterance texts whisper hallucinates on silence or breath noise. They are only dropped
# when they are *all* that a new utterance contains (or a trailing sentence after real words).
HALLUCINATIONS = frozenset({
    "you", "thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching",
    "thanks for watching and see you next time", "bye", "bye bye", "okay", "so", "the", "uh", "um", "hmm",
    "please subscribe", "subtitles by the amaraorg community", "i", "oh", "ah", "huh", "mm", "mhm",
})


def clean_text(text: str) -> str:
    """Drop annotations ('[BLANK_AUDIO]', '(music)', '♪ ... ♪', '*sigh*') and speaker turns.

    whisper marks a change of speaker with '>>'. A leading mark is dropped; everything after a
    mark that follows words is another person talking ("Quit TextEdit. >> Yes, do it."), so it is
    cut off instead of being merged into the user's words (it could confirm a pending action)."""
    t = _ANNOT_RE.sub(" ", text)
    t = _UNCLOSED_RE.sub(" ", t)
    t = t.strip(" -")
    while t.startswith(">>"):
        t = t[2:].strip(" -")
    cut = t.find(">>")
    if cut >= 0:
        t = t[:cut]
    return _SPACE_RE.sub(" ", t).strip(" -")


def norm(w: str) -> str:
    return _NORM_RE.sub("", w.lower())


def norm_words(words: Iterable[str]) -> list[str]:
    return [n for n in (norm(w) for w in words) if n]


def is_hallucination(words: Sequence[str]) -> bool:
    n = norm_words(words)
    return not n or " ".join(n) in HALLUCINATIONS or "".join(n) in HALLUCINATIONS


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def strip_trailing_hallucination(text: str) -> str:
    """'Open notes. Thank you.' -> 'Open notes.' (only a separate trailing sentence is dropped)."""
    parts = _SENTENCE_RE.split(text.strip())
    while len(parts) > 1 and is_hallucination(parts[-1].split()):
        parts.pop()
    return " ".join(parts)


# ---------------------------------------------------------------- terminal parser


class WhisperStreamParser:
    """Turns whisper-stream stdout into ('window', text) and ('commit',) items.

    A window's text is complete when the next clear (`ESC[2K`) or newline arrives; `flush()`
    (called after a short read idle) emits a window whose terminator has not arrived yet, so a
    result is never held back for a whole step. Whitespace-only segments are the 100-space line
    wipe, not a result.
    """

    def __init__(self) -> None:
        self._buf = ""  # text since the last clear / \r / \n
        self._emitted: str | None = None  # the pending text already emitted by flush()
        self._esc = ""  # an escape sequence split across reads
        self.started = False  # saw "[Start speaking]"

    def feed(self, data: str) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = []
        s = self._esc + data
        self._esc = ""
        i = 0
        while i < len(s):
            c = s[i]
            if c == "\x1b":
                j = s.find("K", i)  # ESC [ 2 K (the only sequence whisper-stream prints)
                if j < 0 or j - i > 4:
                    if j < 0 and len(s) - i <= 4:
                        self._esc = s[i:]  # incomplete sequence: wait for the rest
                        break
                    i += 1
                    continue
                self._end_segment(out)
                i = j + 1
                continue
            if c == "\r":
                self._end_segment(out)
            elif c == "\n":
                self._end_segment(out)
                out.append(("commit",))
            else:
                self._buf += c
            i += 1
        return out

    def flush(self) -> list[tuple[str, ...]]:
        """Emit the pending (unterminated) window text if it is new."""
        t = self._buf.strip()
        if not t or t == self._emitted:
            return []
        self._emitted = t
        return self._window(t)

    def _end_segment(self, out: list[tuple[str, ...]]) -> None:
        t = self._buf.strip()
        self._buf = ""
        already = t == self._emitted
        self._emitted = None
        if not t or already:
            return
        out.extend(self._window(t))

    def _window(self, t: str) -> list[tuple[str, ...]]:
        if t == "[Start speaking]":
            self.started = True
            return [("start",)]
        return [("window", t)]


# ---------------------------------------------------------------- utterance tracker


def align_skip(done: Sequence[str], window: Sequence[str]) -> int:
    """How many leading words of `window` belong to `done` (normalised words of finished speech).

    Uses difflib so a re-punctuated or slightly rewritten prefix still lines up; unmatched
    trailing done-words are assumed to map one-to-one onto the following window words.
    """
    if not done or not window:
        return 0
    wn = [norm(w) for w in window]
    sm = difflib.SequenceMatcher(None, list(done), wn, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    matched = sum(b.size for b in blocks)
    if not blocks:
        # Nothing lines up: the finished words are not in this window (a dropped hallucination such
        # as "You" before "Undo that.", or whisper leaving a finished sentence out). Skipping by
        # count would eat the first real words of a new command.
        return 0
    if matched < max(1, len(done) // 2):
        return min(len(done), len(window))
    last = blocks[-1]
    a_end, b_end = last.a + last.size, last.b + last.size
    return min(len(window), b_end + (len(done) - a_end))


_TRAIL_PUNCT_RE = re.compile(r"[^\w]*$", re.UNICODE)


def stabilize(old: Sequence[str], new: Sequence[str]) -> list[str]:
    """Keep the already-emitted tokens where a new window only re-spaces, re-cases or re-punctuates
    them ('Open TextEdit' -> 'Open text edit,' stays 'Open TextEdit'), so the stream's consumed
    prefix keeps matching. The longest old prefix whose letters equal a prefix of the new window's
    letters is kept, and the rest of the new window is appended."""
    if not old:
        return list(new)
    cum_new: dict[str, int] = {"": 0}
    acc = ""
    for k, w in enumerate(new, 1):
        acc += norm(w)
        cum_new.setdefault(acc, k)
    best_i, best_k = 0, 0
    acc = ""
    for i, w in enumerate(old, 1):
        acc += norm(w)
        k = cum_new.get(acc)
        if k is not None:
            best_i, best_k = i, k
    if best_i == best_k and all(norm(o) == norm(n) for o, n in zip(old[:best_i], new[:best_k])):
        kept = list(new[:best_k])  # same split: take the new casing and punctuation
    else:
        kept = list(old[:best_i])
        if kept and best_k:  # re-spaced: old tokens, but the punctuation the window now ends them with
            kept[-1] = _TRAIL_PUNCT_RE.sub("", kept[-1]) + _TRAIL_PUNCT_RE.search(new[best_k - 1]).group(0)
    return kept + list(new[best_k:])


@dataclass
class TrackerStats:
    windows: int = 0
    commits: int = 0
    blank_windows: int = 0
    hallucinations_dropped: int = 0
    repeats_dropped: int = 0  # whisper re-printing the last finished utterance in the same line
    utterances: int = 0
    partials: int = 0
    finals: int = 0
    stall_finals: int = 0
    overlap_deduped: int = 0
    first_window_t: float | None = None
    last_window_t: float | None = None
    last_speech_t: float | None = None
    window_period_ms: float | None = None  # moving average of the time between windows
    max_window_gap_ms: float = 0.0


class UtteranceTracker:
    """Sliding-window whisper results -> one growing utterance at a time (see module docstring).

    Rules added for whisper-stream's quirks (each one is pinned by a test):
    - a finished utterance re-printed later in the same line ("Press enter. Press enter." over
      trailing silence: greedy decoding repeats itself) is dropped, not run again; a new utterance
      whose words are still a prefix of the last finished one waits for the next window;
    - a word cut by the line reset ("wor-" at the end of the old line, "world." at the start of the
      new one) replaces the cut word instead of being appended to it;
    - after a line reset outside an utterance, the kept ~200 ms of audio ("down." after
      "Scroll down.") is skipped instead of starting a phantom utterance;
    - a window that flips back to one of the last few same-length variants ("world."/"word.") counts
      as unchanged, so the final is not postponed until the next line reset;
    - an utterance that starts right after a *stall* final (no windows for `stall_final_ms`, so no
      pause was actually observed) is marked `joined`: it may be the same breath as the previous
      one, and it can never confirm a pending action.
    """

    def __init__(
        self,
        *,
        final_windows: int = 2,
        final_after_ms: int = 900,
        stall_final_ms: int = 2500,
        recent_variants: int = 3,
        uid_prefix: str = "w",
        clock: Callable[[], float] = time.monotonic,
    ):
        self.final_windows = final_windows
        self.final_after_ms = final_after_ms
        self.stall_final_ms = stall_final_ms  # no window at all for this long also ends speech
        self.uid_prefix = uid_prefix
        self.clock = clock
        self.stats = TrackerStats()
        self._seq = 0
        self._n_utt = 0
        self.uid: str | None = None  # current utterance (None between utterances)
        self._base: list[str] = []  # utterance words from earlier lines
        self._line: list[str] = []  # current window words
        self._done: list[str] = []  # normalised words of finished utterances in this line
        self._line_in_utt = False  # this line started inside the current utterance (overlap dedupe)
        self._words: list[str] = []  # last emitted utterance words
        self._recent: deque[tuple[str, ...]] = deque(maxlen=max(1, recent_variants))
        self._last_change = 0.0
        self._same = 0  # consecutive windows without a normalised change
        self._carry: str | None = None  # last word of the previous line, when it ended outside an utterance
        self._prev_final: list[str] = []  # normalised words of the last finished utterance
        self._prev_final_line = -1  # stats.commits when it finished (equal => same whisper line)
        self._after_stall = False  # the last utterance ended by a stall and no quiet window came since
        self._joined = False  # the current utterance started right after a stall final

    # ------------------------------------------------------------ inputs

    def handle(self, item: tuple[str, ...], now: float | None = None) -> list[TranscriptEvent]:
        now = self.clock() if now is None else now
        if item[0] == "window":
            return self.window(item[1], now)
        if item[0] == "commit":
            return self.commit(now)
        return []

    def window(self, raw: str, now: float | None = None) -> list[TranscriptEvent]:
        now = self.clock() if now is None else now
        st = self.stats
        st.windows += 1
        if st.last_window_t is not None:
            gap = max(0.0, (now - st.last_window_t) * 1000.0)
            st.window_period_ms = gap if st.window_period_ms is None else 0.7 * st.window_period_ms + 0.3 * gap
            st.max_window_gap_ms = max(st.max_window_gap_ms, gap)
        st.first_window_t = st.first_window_t if st.first_window_t is not None else now
        st.last_window_t = now
        text = strip_trailing_hallucination(clean_text(raw))
        words = text.split()
        if not norm_words(words):
            st.blank_windows += 1
        self._line = words
        skip = align_skip(self._done, words)
        if skip == 0 and self._carry and words and norm(words[0]) == self._carry:
            skip = 1  # the ~200 ms of audio kept across the line reset re-transcribed
        new = words[skip:]
        if self.uid is None:
            if not norm_words(new):
                self._after_stall = False  # a window with no new words: the pause was observed
                return []
            if is_hallucination(new):
                # Treat it like finished speech, so the real words that follow in this line align past it.
                st.hallucinations_dropped += 1
                self._done = norm_words(words)
                return []
            if self._held_as_repeat(new):
                return []
        base = self._base
        if base and self._line_in_utt and new:
            if is_hallucination(new):
                new = []  # a new line holding only "you" / "thank you" adds nothing to the utterance
            else:
                old, first = norm(base[-1]), norm(new[0])
                if first == old:
                    # The window restarted with keep_ms of old audio: the boundary word can repeat.
                    new = new[1:]
                    st.overlap_deduped += 1
                elif old and first.startswith(old) and (len(old) >= 3 or base[-1].endswith("-")):
                    base = base[:-1]  # "wor-" | "world.": the reset cut the word; keep the whole one
                    st.overlap_deduped += 1
                elif first and old.startswith(first) and len(first) >= 3:
                    new = new[1:]  # the new line starts with the tail of a word already heard whole
                    st.overlap_deduped += 1
        utt = stabilize(self._words, base + new)
        if not norm_words(utt):
            return self._tick_same(now)
        out: list[TranscriptEvent] = []
        if self.uid is None:
            self._start(now, out)
        nu = tuple(norm_words(utt))
        cur = tuple(norm_words(self._words))
        if nu == cur or (nu in self._recent and len(nu) == len(cur)):
            # Unchanged, or flipping back to a variant just seen ("world." / "word." over silence).
            self._same += 1
            return out + self._maybe_final(now)
        self._same = 0
        self._words = utt
        self._recent.append(nu)
        self._last_change = now
        st.last_speech_t = now
        st.partials += 1
        out.append(self._ev("partial", " ".join(utt), now))
        return out

    def commit(self, now: float | None = None) -> list[TranscriptEvent]:
        """A newline: the line is final and the next window starts from fresh audio."""
        now = self.clock() if now is None else now
        self.stats.commits += 1
        if self.uid is not None:
            # The emitted words are the utterance so far (earlier lines + this line's new part);
            # the next line only adds to them.
            self._base = list(self._words)
            self._line_in_utt = True
            self._carry = None
        else:
            # The next line starts with keep_ms of this line's audio: its last word may reappear.
            last = norm_words(self._line[-1:])
            self._carry = last[0] if last else None
        self._line, self._done = [], []
        return []

    def tick(self, now: float | None = None) -> list[TranscriptEvent]:
        """Called periodically: ends the utterance when whisper itself stalls (no windows at all)."""
        now = self.clock() if now is None else now
        if self.uid is None:
            return []
        last = self.stats.last_window_t or self._last_change
        if (now - last) * 1000.0 >= self.stall_final_ms and (now - self._last_change) * 1000.0 >= self.final_after_ms:
            self.stats.stall_finals += 1
            return self._finish(now, stall=True)
        return []

    def close(self, now: float | None = None) -> list[TranscriptEvent]:
        now = self.clock() if now is None else now
        return self._finish(now) if self.uid is not None else []

    # ------------------------------------------------------------ helpers

    def _held_as_repeat(self, new: Sequence[str]) -> bool:
        """True when `new` (the start of a would-be utterance) is whisper re-printing the utterance
        that just finished in this same line. An exact repeat is dropped for good (its words join
        the finished ones); a prefix of it waits for the next window to tell."""
        pf = self._prev_final
        if not pf or self._prev_final_line != self.stats.commits:
            return False
        nn = norm_words(new)
        if nn != pf[: len(nn)]:
            return False
        if len(nn) == len(pf):
            self.stats.repeats_dropped += 1
            self._done = norm_words(self._line)
        return True

    def _tick_same(self, now: float) -> list[TranscriptEvent]:
        if self.uid is None:
            return []
        self._same += 1
        return self._maybe_final(now)

    def _maybe_final(self, now: float) -> list[TranscriptEvent]:
        if self.uid is None:
            return []
        if self._same >= self.final_windows and (now - self._last_change) * 1000.0 >= self.final_after_ms:
            return self._finish(now)
        return []

    def _start(self, now: float, out: list[TranscriptEvent]) -> None:
        self._n_utt += 1
        self.uid = f"{self.uid_prefix}{self._n_utt}"
        self._words, self._same = [], 0
        self._recent.clear()
        self._line_in_utt = False
        self._joined, self._after_stall = self._after_stall, False
        self.stats.utterances += 1
        out.append(self._ev("speech_start", "", now))

    def _finish(self, now: float, *, stall: bool = False) -> list[TranscriptEvent]:
        text = " ".join(self._words)
        out = [self._ev("final", text, now), self._ev("speech_end", "", now)]
        self.stats.finals += 1
        # Everything in the current window now belongs to a finished utterance.
        self._done = norm_words(self._line) if self._line else self._done
        self._prev_final = norm_words(self._words)
        self._prev_final_line = self.stats.commits
        self._after_stall = stall
        self._base, self._words, self.uid, self._same = [], [], None, 0
        self._line_in_utt = False
        self._joined = False
        return out

    def _ev(self, kind: str, text: str, now: float) -> TranscriptEvent:
        self._seq += 1
        return TranscriptEvent(kind=kind, seq=self._seq, uid=self.uid or "", text=text, t_mono=now,  # type: ignore[arg-type]
                               joined=self._joined and self.uid is not None)


# ---------------------------------------------------------------- live source (microphone)

_FATAL_STDERR = (
    ("audio.init() failed", "mic_unavailable"),
    ("failed to initialize whisper context", "model_failed"),
    ("failed to open", "model_failed"),
    ("error: unknown argument", "bad_args"),
    ("failed to process audio", "inference_failed"),
)
_live: set[int] = set()  # pids of whisper-stream processes started by this interpreter


def _kill_all_live() -> None:
    for pid in list(_live):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        _live.discard(pid)
    try:
        PIDFILE.unlink()
    except OSError:
        pass


atexit.register(_kill_all_live)


def kill_stale(pidfile: Path = PIDFILE) -> int | None:
    """Kill a whisper-stream left behind by a crashed earlier run (pid recorded in `pidfile`)."""
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        cmd = subprocess.run(["ps", "-o", "comm=", "-p", str(pid)], capture_output=True, text=True, timeout=2).stdout
    except Exception:  # noqa: BLE001
        cmd = ""
    if "whisper" not in cmd:
        pidfile.unlink(missing_ok=True)
        return None
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    pidfile.unlink(missing_ok=True)
    return pid


@dataclass
class SourceInfo:
    """What a source reports for the HUD / report (filled while it runs)."""

    cmd: list[str] = field(default_factory=list)
    pid: int | None = None
    started_at: float | None = None
    ready_at: float | None = None  # "[Start speaking]" seen
    # Windows that arrived more than ~1.3 steps after the previous one. In sliding-window mode
    # whisper-stream keeps only the last step of audio after a slow inference and drops the rest,
    # silently ("cannot process audio fast enough" is unreachable there), so this is measured.
    dropped_audio: int = 0
    stderr_tail: list[str] = field(default_factory=list)
    exit_code: int | None = None


class WhisperSource:
    """Speech source protocol (`start` / `stop` / `events`) over a `whisper-stream` subprocess."""

    def __init__(
        self,
        model: str | Path = "base.en",
        *,
        binary: str | Sequence[str] = STREAM_BIN,
        step_ms: int = DEFAULT_STEP_MS,
        length_ms: int = DEFAULT_LENGTH_MS,
        keep_ms: int = DEFAULT_KEEP_MS,
        threads: int = 4,
        capture_id: int = -1,
        max_tokens: int = 32,
        audio_ctx: int = 0,
        use_gpu: bool = True,
        extra_args: Sequence[str] = (),
        final_windows: int = 2,
        final_after_ms: int = 900,
        silence_hint_s: float = 10.0,
        idle_flush_ms: int = 25,
        clock: Callable[[], float] = time.monotonic,
        pidfile: Path | None = PIDFILE,
    ):
        self.model = model_path(str(model))
        self.binary = [binary] if isinstance(binary, str) else list(binary)
        self.step_ms, self.length_ms, self.keep_ms = step_ms, length_ms, keep_ms
        self.args = [
            "-m", str(self.model), "-t", str(threads), "--step", str(step_ms), "--length", str(length_ms),
            "--keep", str(keep_ms), "-c", str(capture_id), "-mt", str(max_tokens), "-ac", str(audio_ctx),
            *([] if use_gpu else ["-ng"]), *extra_args,
        ]
        self.clock = clock
        self.parser = WhisperStreamParser()
        self.tracker = UtteranceTracker(final_windows=final_windows, final_after_ms=final_after_ms, clock=clock)
        self.info = SourceInfo()
        self.silence_hint_s = silence_hint_s
        self.idle_flush_s = idle_flush_ms / 1000.0
        self.pidfile = pidfile
        self._proc: asyncio.subprocess.Process | None = None
        self._q: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._hinted = False
        self._slow_hinted = False
        self._last_window_at: float | None = None
        self._stopping = False

    @property
    def stats(self) -> TrackerStats:
        return self.tracker.stats

    @property
    def cadence_ms(self) -> float:
        """How often the transcript can change: the step, or the measured window period when slower.
        The Controller derives its silence gates from it (one window without a new word is not silence)."""
        return max(float(self.step_ms), self.stats.window_period_ms or 0.0)

    async def start(self) -> None:
        if self._proc is not None:
            return
        if self.binary == [STREAM_BIN] and not Path(STREAM_BIN).exists():
            raise FileNotFoundError("whisper-stream not found: brew install whisper-cpp")
        if not self.model.exists() and self.binary == [STREAM_BIN]:
            raise FileNotFoundError(f"whisper model missing: {self.model}")
        cmd = [*self.binary, *self.args]
        self.info.cmd = cmd
        self.info.started_at = self.clock()
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        self.info.pid = self._proc.pid
        _live.add(self._proc.pid)
        if self.pidfile is not None:
            try:
                self.pidfile.parent.mkdir(parents=True, exist_ok=True)
                self.pidfile.write_text(str(self._proc.pid))
            except OSError:
                pass
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._read_stdout()), loop.create_task(self._read_stderr()),
                       loop.create_task(self._ticker())]

    async def stop(self) -> None:
        self._stopping = True
        p = self._proc
        if p is not None and p.returncode is None:
            try:
                p.terminate()
                await asyncio.wait_for(p.wait(), 1.5)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    p.kill()
                    await p.wait()
                except ProcessLookupError:
                    pass
        if p is not None:
            _live.discard(p.pid)
            self.info.exit_code = p.returncode
            if self.pidfile is not None:
                try:
                    if self.pidfile.read_text().strip() == str(p.pid):
                        self.pidfile.unlink()
                except OSError:
                    pass
        for t in self._tasks:
            t.cancel()
        for ev in self.tracker.close(self.clock()):
            self._q.put_nowait(ev)
        self._q.put_nowait(None)

    def kill_now(self) -> None:
        """Synchronous hard stop (signal handlers, kill switch): never blocks."""
        p = self._proc
        if p is not None and p.returncode is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            ev = await self._q.get()
            if ev is None:
                return
            yield ev

    # ------------------------------------------------------------ readers

    def _emit(self, items: list[tuple[str, ...]]) -> None:
        now = self.clock()
        for it in items:
            if it[0] == "start":
                self.info.ready_at = now
                continue
            if it[0] == "window":
                self._check_period(now)
            for ev in self.tracker.handle(it, now):
                self._q.put_nowait(ev)

    def _check_period(self, now: float) -> None:
        """Count windows that came late: whisper-stream then dropped the audio in between."""
        last, self._last_window_at = self._last_window_at, now
        if last is None:
            return
        gap_ms = (now - last) * 1000.0
        if gap_ms <= max(1.3 * self.step_ms, self.step_ms + 100.0):
            return
        self.info.dropped_audio += 1
        if not self._slow_hinted:
            self._slow_hinted = True
            self._q.put_nowait(TranscriptEvent(
                kind="error", seq=0, uid="", t_mono=now, error="whisper_slow",
                text=f"whisper is slower than real time (a window took {gap_ms:.0f} ms for a {self.step_ms} ms step): "
                     "audio is being dropped and words can be lost - raise --step-ms, use a smaller whisper "
                     "model, or free memory"))

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        out = self._proc.stdout
        try:
            while True:
                try:
                    data = await asyncio.wait_for(out.read(4096), self.idle_flush_s)
                except asyncio.TimeoutError:
                    self._emit(self.parser.flush())
                    continue
                if not data:
                    break
                self._emit(self.parser.feed(data.decode("utf-8", "replace")))
        finally:
            code = await self._proc.wait()
            self.info.exit_code = code
            if not self._stopping:
                for ev in self.tracker.close(self.clock()):
                    self._q.put_nowait(ev)
                if code not in (0, None, -signal.SIGTERM, -signal.SIGKILL):
                    msg = " | ".join(self.info.stderr_tail[-3:]) or f"exit code {code}"
                    self._q.put_nowait(TranscriptEvent(kind="error", seq=0, uid="", text=msg,
                                                       t_mono=self.clock(), error="whisper_exited"))
                self._q.put_nowait(None)

    async def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        async for raw in self._proc.stderr:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            self.info.stderr_tail = (self.info.stderr_tail + [line])[-20:]
            if "cannot process audio fast enough" in line:
                self.info.dropped_audio += 1
                continue
            for pat, code in _FATAL_STDERR:
                if pat in line:
                    self._q.put_nowait(TranscriptEvent(kind="error", seq=0, uid="", text=line,
                                                       t_mono=self.clock(), error=code))

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            now = self.clock()
            for ev in self.tracker.tick(now):
                self._q.put_nowait(ev)
            st = self.stats
            if (not self._hinted and st.first_window_t is not None and st.utterances == 0
                    and now - st.first_window_t >= self.silence_hint_s):
                self._hinted = True
                self._q.put_nowait(TranscriptEvent(
                    kind="error", seq=0, uid="", t_mono=now, error="no_speech_heard",
                    text=f"only silence for {self.silence_hint_s:.0f}s - is the Microphone allowed for this "
                         "terminal app (System Settings > Privacy & Security > Microphone)?"))


# ---------------------------------------------------------------- file replay (tests, offline demo)


def to_pcm16k(audio: str | Path, out_wav: str | Path) -> None:
    """Any audio file macOS can read (aiff from `say -o`, wav, m4a) -> 16 kHz mono 16-bit WAV."""
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(audio), str(out_wav)],
                   check=True, capture_output=True, timeout=60)


def read_wav(path: str | Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16 kHz mono 16-bit")
        return w.readframes(w.getnframes())


def write_wav(path: str | Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def speech_spans(pcm: bytes, frame_ms: int = 20, thresh: int = 500, min_gap_ms: int = 400) -> list[tuple[float, float]]:
    """Rough (start_s, end_s) speech spans by frame peak amplitude (for end-of-speech latency)."""
    import array

    a = array.array("h")
    a.frombytes(pcm)
    n = SAMPLE_RATE * frame_ms // 1000
    spans: list[list[float]] = []
    for i in range(0, len(a), n):
        fr = a[i : i + n]
        if fr and max(abs(min(fr)), abs(max(fr))) >= thresh:
            t0, t1 = i / SAMPLE_RATE, (i + len(fr)) / SAMPLE_RATE
            if spans and t0 - spans[-1][1] <= min_gap_ms / 1000.0:
                spans[-1][1] = t1
            else:
                spans.append([t0, t1])
    return [(round(s, 3), round(e, 3)) for s, e in spans]


Segments = tuple[tuple[int, int], ...]


def _tail(segs: Segments, n: int) -> list[tuple[int, int]]:
    """The last `n` samples of a list of (start, end) segments."""
    out: list[tuple[int, int]] = []
    for s, e in reversed(segs):
        if n <= 0:
            break
        take = min(n, e - s)
        out.insert(0, (e - take, e))
        n -= take
    return out


def stream_schedule(n_samples: int, step_ms: int, length_ms: int, keep_ms: int,
                    latency_ms: float = 0.0) -> list[tuple[Segments, bool, int, float]]:
    """Every window whisper-stream transcribes: (audio segments, newline_after, capture sample, emit s).

    Mirrors examples/stream/stream.cpp (sliding-window mode): a window is the end of the previous
    window (at most `keep + len - step` samples) plus `step` new samples; every n_new_line steps the
    old audio is cut to the last `keep` samples and a newline is printed. The new audio is
    `audio.get(step_ms)` taken once the previous inference has finished: when inference (`latency_ms`)
    is slower than the step, that is the *last* step before the capture, and the audio in between
    is dropped (the window is then not contiguous). The result is printed `latency_ms` later.
    """
    keep_ms = min(keep_ms, step_ms)
    length_ms = max(length_ms, step_ms)
    step = step_ms * SAMPLE_RATE // 1000
    ln = length_ms * SAMPLE_RATE // 1000
    keep = keep_ms * SAMPLE_RATE // 1000
    n_new_line = max(1, length_ms // step_ms - 1)
    lat_s = max(0.0, latency_ms) / 1000.0
    out: list[tuple[Segments, bool, int, float]] = []
    old: Segments = ()
    cap, it = step, 0
    while cap <= n_samples:
        take = min(sum(e - s for s, e in old), max(0, keep + ln - step))
        segs: list[tuple[int, int]] = _tail(old, take)
        if segs and segs[-1][1] == cap - step:
            segs[-1] = (segs[-1][0], cap)  # contiguous: one segment
        else:
            segs.append((cap - step, cap))
        it += 1
        newline = it % n_new_line == 0
        emit = cap / SAMPLE_RATE + lat_s
        out.append((tuple(segs), newline, cap, emit))
        old = tuple(_tail(tuple(segs), keep)) if newline else tuple(segs)
        cap = max(cap + step, int(round(emit * SAMPLE_RATE)))  # the next capture waits for the inference
    return out


def stream_windows(n_samples: int, step_ms: int, length_ms: int, keep_ms: int) -> list[tuple[int, int, bool]]:
    """The audio windows whisper-stream transcribes in real time, as (start, end, newline_after)
    sample offsets (inference faster than the step: every window is contiguous)."""
    return [(segs[0][0], segs[-1][1], nl) for segs, nl, _cap, _t in stream_schedule(n_samples, step_ms, length_ms, keep_ms)]


@dataclass
class FileInfo:
    audio: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    spans: list[tuple[float, float]] = field(default_factory=list)  # speech spans in audio time
    file_spans: list[tuple[float, float]] = field(default_factory=list)  # each input file's speech, merged
    n_windows: int = 0
    whisper_s: float = 0.0  # wall time of the whisper-cli batch
    load_ms: float | None = None
    per_window_ms: float = 0.0  # simulated inference latency per window
    dropped_s: float = 0.0  # audio whisper-stream would have dropped (inference slower than the step)
    t0: float | None = None  # clock() at audio time 0
    windows: list[tuple[float, float, str]] = field(default_factory=list)  # (audio_end_s, emit_s, text)


class FileWhisperSource:
    """Replays audio files through whisper.cpp as if they were spoken into whisper-stream.

    `prepare()` (run automatically by `start`) converts the files to 16 kHz, joins them with
    `gap_ms` of silence, cuts the exact windows whisper-stream would transcribe, runs whisper-cli
    once over all windows (one model load), and records each window's text. `events()` then
    plays the equivalent whisper-stream terminal output in real time: window i appears at
    audio_end_i + per-window inference latency (measured from the batch, serialized like the
    real loop), and goes through the same parser and tracker as the live source.
    """

    def __init__(
        self,
        audio: str | Path | Sequence[str | Path],
        model: str | Path = "base.en",
        *,
        cli: str = CLI_BIN,
        step_ms: int = DEFAULT_STEP_MS,
        length_ms: int = DEFAULT_LENGTH_MS,
        keep_ms: int = DEFAULT_KEEP_MS,
        threads: int = 4,
        use_gpu: bool = True,
        lead_ms: int = 300,
        gap_ms: int = 2500,
        trail_ms: int = 2500,
        latency_ms: float | None = None,
        realtime: bool = True,
        final_windows: int = 2,
        final_after_ms: int = 900,
        clock: Callable[[], float] = time.monotonic,
        workdir: str | Path | None = None,
        texts: Sequence[str] | None = None,  # precomputed window texts (tests: no whisper run)
    ):
        files = [audio] if isinstance(audio, (str, Path)) else list(audio)
        self.files = [Path(f) for f in files]
        self.model = model_path(str(model))
        self.cli = cli
        self.step_ms, self.length_ms, self.keep_ms = step_ms, length_ms, keep_ms
        self.threads, self.use_gpu = threads, use_gpu
        self.lead_ms, self.gap_ms, self.trail_ms = lead_ms, gap_ms, trail_ms
        self.latency_ms = latency_ms
        self.realtime = realtime
        self.clock = clock
        self.workdir = Path(workdir) if workdir else None
        self.parser = WhisperStreamParser()
        self.tracker = UtteranceTracker(final_windows=final_windows, final_after_ms=final_after_ms, clock=clock)
        self.info = FileInfo(audio=[str(f) for f in self.files])
        self._texts = list(texts) if texts is not None else None
        self._plan: list[tuple[Segments, bool, int, float]] = []
        self._prepared = False
        self._stopped = False
        self._cli: subprocess.Popen | None = None  # the running whisper-cli (killed by kill_now)

    @property
    def stats(self) -> TrackerStats:
        return self.tracker.stats

    @property
    def cadence_ms(self) -> float:
        """See WhisperSource.cadence_ms."""
        return max(float(self.step_ms), self.stats.window_period_ms or 0.0)

    def _schedule(self, n_samples: int, latency_ms: float) -> None:
        self._plan = stream_schedule(n_samples, self.step_ms, self.length_ms, self.keep_ms, latency_ms)
        self.info.n_windows = len(self._plan)
        covered = sum(self.step_ms for _ in self._plan) / 1000.0
        self.info.dropped_s = round(max(0.0, (self._plan[-1][2] / SAMPLE_RATE if self._plan else 0.0) - covered), 3)

    def prepare(self) -> None:
        if self._prepared:
            return
        tmp = Path(tempfile.mkdtemp(prefix="jev-whisper-", dir=self.workdir))
        try:
            silence = lambda ms: b"\x00\x00" * (SAMPLE_RATE * ms // 1000)  # noqa: E731
            pcm = silence(self.lead_ms)
            file_spans: list[tuple[float, float]] = []
            for i, f in enumerate(self.files):
                if f.suffix.lower() == ".wav":
                    try:
                        data = read_wav(f)
                    except ValueError:
                        to_pcm16k(f, tmp / f"in{i}.wav")
                        data = read_wav(tmp / f"in{i}.wav")
                else:
                    to_pcm16k(f, tmp / f"in{i}.wav")
                    data = read_wav(tmp / f"in{i}.wav")
                off = len(pcm) / 2 / SAMPLE_RATE
                sp = speech_spans(data)
                if sp:
                    file_spans.append((round(off + sp[0][0], 3), round(off + sp[-1][1], 3)))
                pcm += data + silence(self.gap_ms if i < len(self.files) - 1 else self.trail_ms)
            self.info.duration_s = len(pcm) / 2 / SAMPLE_RATE
            self.info.spans = speech_spans(pcm)
            self.info.file_spans = file_spans
            n = len(pcm) // 2
            self._schedule(n, self.latency_ms or 0.0)
            if self._texts is None:
                self._texts = self._transcribe(pcm, tmp)
                if self.latency_ms is None and self.info.per_window_ms > self.step_ms:
                    # Slower than real time: whisper-stream would drop audio between windows. Cut the
                    # windows it would really see (fewer, with gaps) and transcribe those instead.
                    measured = self.info.per_window_ms
                    self._schedule(n, measured)
                    self._texts = self._transcribe(pcm, tmp)
                    self.info.per_window_ms = measured
            else:  # precomputed texts (tests): one per window, the last one repeated
                pad = self._texts[-1] if self._texts else ""
                self._texts = (self._texts + [pad] * len(self._plan))[: len(self._plan)]
            if self.latency_ms is None:
                self.latency_ms = self.info.per_window_ms
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self._prepared = True

    def _transcribe(self, pcm: bytes, tmp: Path) -> list[str]:
        paths = []
        for k, (segs, _nl, _cap, _t) in enumerate(self._plan):
            p = tmp / f"w{k:04d}.wav"
            write_wav(p, b"".join(pcm[2 * s : 2 * e] for s, e in segs))
            paths.append(p)
        # Same decoding settings as whisper-stream: greedy, no text context, no timestamps.
        cmd = [self.cli, "-m", str(self.model), "-t", str(self.threads), "-bs", "1", "-bo", "1", "-mc", "0",
               "-nt", "-otxt", *([] if self.use_gpu else ["-ng"])]
        texts: list[str] = []
        t0 = time.perf_counter()
        stderr = ""
        for i in range(0, len(paths), 200):  # bounded argv
            chunk = paths[i : i + 200]
            if self._stopped:
                raise RuntimeError("stopped")
            # Popen (not run): kill_now() and the atexit hook can then stop it, so a Ctrl-C / SIGTERM
            # while the files are being transcribed never leaves whisper-cli running.
            proc = subprocess.Popen([*cmd, *map(str, chunk)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            self._cli = proc
            _live.add(proc.pid)
            try:
                _out, err = proc.communicate(timeout=900)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise RuntimeError("whisper-cli timed out") from None
            finally:
                _live.discard(proc.pid)
                self._cli = None
            stderr += err
            if proc.returncode != 0:
                raise RuntimeError(f"whisper-cli failed ({proc.returncode}): {err[-500:]}")
            for p in chunk:
                txt = Path(str(p) + ".txt")
                texts.append(" ".join(txt.read_text(encoding="utf-8").split()) if txt.exists() else "")
        self.info.whisper_s = time.perf_counter() - t0
        m = re.findall(r"load time\s*=\s*([\d.]+)\s*ms", stderr)
        self.info.load_ms = sum(float(x) for x in m) if m else None
        work_ms = self.info.whisper_s * 1000.0 - (self.info.load_ms or 0.0)
        self.info.per_window_ms = round(max(work_ms, 0.0) / max(1, len(paths)), 1)
        return texts

    def terminal_output(self) -> list[tuple[float, str]]:
        """(emit time in audio seconds, whisper-stream stdout chunk) for every step."""
        self.prepare()
        assert self._texts is not None
        lat = self.latency_ms or 0.0
        if self._plan and abs((self._plan[0][3] - self._plan[0][2] / SAMPLE_RATE) * 1000.0 - lat) > 0.5:
            # The latency became known after the windows were cut (measured): re-time them. When it is
            # slower than the step, prepare() already re-cut the windows with it.
            self._plan = [(segs, nl, cap, cap / SAMPLE_RATE + lat / 1000.0) for segs, nl, cap, _t in self._plan]
        out: list[tuple[float, str]] = [(0.0, "[Start speaking]\n")]
        self.info.windows = []
        for (_segs, newline, cap, t_emit), text in zip(self._plan, self._texts):
            t_audio = cap / SAMPLE_RATE
            chunk = f"{CLEAR}\r{' ' * 100}{CLEAR}\r {text}"
            if newline:
                chunk += "\n"
            out.append((t_emit, chunk))
            self.info.windows.append((round(t_audio, 3), round(t_emit, 3), text))
        return out

    async def start(self) -> None:
        if not self._prepared:
            await asyncio.to_thread(self.prepare)

    async def stop(self) -> None:
        self._stopped = True

    def kill_now(self) -> None:
        self._stopped = True
        p = self._cli
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass

    def audio_time_to_clock(self, t_audio: float) -> float | None:
        return None if self.info.t0 is None else self.info.t0 + t_audio

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        chunks = self.terminal_output()
        loop = asyncio.get_running_loop()
        t0 = self.clock()
        self.info.t0 = t0
        tick = 0.1
        next_tick = tick
        for t_emit, chunk in chunks:
            while self.realtime and next_tick < t_emit:
                await self._sleep_until(t0 + next_tick)
                for ev in self.tracker.tick(self.clock()):
                    yield ev
                next_tick += tick
            if self._stopped:
                break
            if self.realtime:
                await self._sleep_until(t0 + t_emit)
            for item in self.parser.feed(chunk) + self.parser.flush():
                if item[0] == "start":
                    continue
                for ev in self.tracker.handle(item, self.clock()):
                    yield ev
        for ev in self.tracker.close(self.clock()):
            yield ev
        del loop

    async def _sleep_until(self, t: float) -> None:
        d = t - self.clock()
        if d > 0:
            await asyncio.sleep(d)
