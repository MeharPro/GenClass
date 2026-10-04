"""Memory-safe overnight job runner for this 8 GB Mac.

Runs heavy jobs (training, calibration, eval) one at a time, only when the machine has real
headroom, and stops them as soon as it starts thrashing. It is stdlib-only and tiny, and it runs
detached from any Claude session.

History: the first version gated on kern.memorystatus_level and pageouts. On 2026-09-27 at 05:04
it started training while swap was 7.9/8 GB full and Chrome held 8.8 GB. The Mac thrashed
(5,314 swap-ins/s, 97 °C) and crashed within ~90 s, before any preemption rule fired. The gates
below follow the rules the "Mac crash prevention" session asked for; they are deliberately
conservative:

  start  only if ALL hold for START_STABLE_S straight:
           swap used < 3 GB, RAM free >= job peak + 2 GB, swap-ins/s < 100, disk >= 4 GB,
           inside the 23:00-09:30 window, no PAUSE file, not already preempted tonight
  stop   immediately if RAM free < 12%, or swap-ins/s >= 1000 on 2 consecutive samples,
           or disk < 2.5 GB, the window ends, PAUSE/STOP appears
           (SIGKILL straight away if it is already severe: RAM free < 8% or swap-ins/s >= 3000)
  after  any preemption (ours or an external kill): no more jobs until the next night's window

  RAM free = (free + inactive + speculative + purgeable pages) x page size, from vm_stat.

Other sessions/people can steer it without killing anything:
  touch ~/.jev-local/overnight/PAUSE    # preempt now; hold until the file is removed
  touch ~/.jev-local/overnight/STOP     # preempt and exit the runner
  touch ~/.jev-local/overnight/HOLD     # don't START new jobs while present (expires after 2 h);
                                        # used by build agents so they never overlap training
  cat   ~/.jev-local/overnight/status.json

Usage:
  .venv/bin/python scripts/overnight.py --daemon      # start detached
  .venv/bin/python scripts/overnight.py --check       # one health sample + start verdict for `train`
  .venv/bin/python scripts/overnight.py --status      # print status.json + whether the runner is alive
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path.home() / ".jev-local" / "overnight"
PY = str(ROOT / ".venv" / "bin" / "python")
LOGS = ROOT / "runs" / "overnight"
GB = 1024**3

# ---------------------------------------------------------------- policy

START_SWAP_MAX_GB = 3.0
START_HEADROOM_GB = 2.0  # RAM free must cover the job's peak plus this
START_SWAPINS_MAX = 100  # per second
START_STABLE_S = 60  # every start condition must hold this long without a break
START_DISK_GB = 4.0
STOP_FREE_PCT = 12.0
STOP_SWAPINS = 1000  # per second, on 2 consecutive samples
KILL_FREE_PCT = 8.0  # already severe: SIGKILL, don't wait for a graceful checkpoint
KILL_SWAPINS = 3000
STOP_DISK_GB = 2.5
WINDOW = ((23, 0), (9, 30))  # local time; heavy jobs only inside [start, end)
SAMPLE_S = 5
TERM_GRACE_S = 30  # SIGTERM -> trainer finishes its step and checkpoints; then SIGKILL
MAX_FAILS = 2
HOLD_TTL_S = 2 * 3600  # a HOLD left behind by a crashed session stops mattering after this
# Exit codes that mean "interrupted": the trainer's SystemExit(130) after SIGINT/SIGTERM, and raw
# signal deaths (SIGTERM, SIGKILL, jetsam memory kills). Treated as a preemption, never as done.
PREEMPTED_RCS = frozenset({130, 143, 137, -signal.SIGTERM, -signal.SIGKILL, -signal.SIGINT})

JOB_ENV = {
    # Hard cap on MPS allocations as a fraction of the GPU's recommended working set (~5.3 GB on
    # this M1), about 1.9 GB: a job that needs more fails cleanly instead of swapping the Mac.
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.35",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.25",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "OMP_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
}


@dataclass
class Job:
    name: str
    cmd: list[str]
    peak_gb: float  # expected peak footprint (MPS + RSS); smoke at batch 4 measured 2.29 + 0.44 GB
    after: str | None = None  # job that must be done first
    optional_file: str | None = None  # skip the job if this file doesn't exist


TRAIN = [PY, "-m", "jev_local.train.train", "--data", "data/cu", "--extra", "data/gen",
         "--base", "models/base/ettin-encoder-32m"]  # in-project copy: the HF cache gets wiped here
JOBS = [
    Job("smoke", TRAIN + ["--out", "models/smoke", "--run-name", "smoke", "--limit", "48", "--batch", "2",
                          "--grad-accum", "8", "--max-steps", "12", "--log-every", "2", "--ckpt-every", "1000"],
        peak_gb=2.5),
    # batch 2 x accum 8 keeps the effective batch at 16 while roughly halving activation memory
    Job("train", TRAIN + ["--out", "models/jev-local-fast", "--epochs", "1", "--batch", "2", "--grad-accum", "8",
                          "--ckpt-every", "150", "--resume"], peak_gb=2.5, after="smoke"),
    Job("calibrate", [PY, "-m", "jev_local.engine.encoder.calibrate", "--ckpt", "models/jev-local-fast",
                      "--data", "data/cu", "--extra", "data/gen", "--split", "dev", "--batch", "4"],
        peak_gb=1.5, after="train"),
    Job("eval", [PY, "-m", "jev_local.train.eval", "--ckpt", "models/jev-local-fast", "--data", "data/cu",
                 "--extra", "data/gen", "--split", "test", "--limit", "3000", "--batch", "4"],
        peak_gb=1.5, after="calibrate"),
    Job("bench", [PY, "scripts/bench_encoder.py", "--ckpt", "models/jev-local-fast"], peak_gb=1.5,
        after="eval", optional_file="scripts/bench_encoder.py"),
]

# ---------------------------------------------------------------- health


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


@dataclass(frozen=True)
class VmStat:
    page_size: int
    free: int
    inactive: int
    speculative: int
    purgeable: int
    swapins: int

    @property
    def ram_free_bytes(self) -> int:
        return (self.free + self.inactive + self.speculative + self.purgeable) * self.page_size


def parse_vm_stat(text: str) -> VmStat:
    ps = re.search(r"page size of (\d+) bytes", text)

    def num(label: str) -> int:
        m = re.search(rf"{label}:\s+(\d+)", text)
        return int(m.group(1)) if m else 0

    return VmStat(int(ps.group(1)) if ps else 16384, num("Pages free"), num("Pages inactive"),
                  num("Pages speculative"), num("Pages purgeable"), num("Swapins"))


def parse_swap_used_gb(text: str) -> float:
    m = re.search(r"used = ([\d.]+)([MG])", text)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v / 1024 if m.group(2) == "M" else v


def mem_total_bytes() -> int:
    try:
        return int(_run(["sysctl", "-n", "hw.memsize"]).strip())
    except ValueError:
        return 8 * GB


def in_window(now: dt.datetime) -> bool:
    (sh, sm), (eh, em) = WINDOW
    t = now.hour * 60 + now.minute
    start, end = sh * 60 + sm, eh * 60 + em
    return (t >= start or t < end) if start > end else (start <= t < end)


def next_window_start(now: dt.datetime) -> dt.datetime:
    """Start of the next overnight window that begins after the current one (if any) ends."""
    (sh, sm), _ = WINDOW
    cand = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    if cand <= now:  # already past today's start (e.g. 23:30): the next night is tomorrow
        cand += dt.timedelta(days=1)
    return cand


@dataclass
class Health:
    t: float
    ram_free_gb: float
    ram_free_pct: float
    swap_used_gb: float
    swapins_per_s: float
    disk_free_gb: float
    load1: float
    window: bool
    paused: bool
    held: bool = False


def hold_active() -> bool:
    p = STATE / "HOLD"
    try:
        return time.time() - p.stat().st_mtime < HOLD_TTL_S
    except FileNotFoundError:
        return False


class Monitor:
    def __init__(self) -> None:
        self.total = mem_total_bytes()
        self.prev: tuple[float, int] | None = None

    def sample(self, now: dt.datetime | None = None) -> Health:
        t = time.time()
        vs = parse_vm_stat(_run(["vm_stat"]))
        rate = 0.0
        if self.prev is not None and t > self.prev[0]:
            rate = max(0.0, (vs.swapins - self.prev[1]) / (t - self.prev[0]))
        self.prev = (t, vs.swapins)
        return Health(
            t=t,
            ram_free_gb=round(vs.ram_free_bytes / GB, 2),
            ram_free_pct=round(100 * vs.ram_free_bytes / self.total, 1),
            swap_used_gb=round(parse_swap_used_gb(_run(["sysctl", "vm.swapusage"])), 2),
            swapins_per_s=round(rate, 1),
            disk_free_gb=round(shutil.disk_usage("/").free / 1e9, 2),
            load1=round(os.getloadavg()[0], 2),
            window=in_window(now or dt.datetime.now()),
            paused=(STATE / "PAUSE").exists(),
            held=hold_active(),
        )


def start_blockers(h: Health, job: Job, ignore_window: bool = False) -> list[str]:
    """Every reason `job` may not start right now (empty list = OK for this sample)."""
    out = []
    if h.paused:
        out.append("PAUSE file present")
    if h.held:
        out.append("HOLD file present (another heavy task is running)")
    if not (h.window or ignore_window):
        out.append("outside the 23:00-09:30 window")
    if h.swap_used_gb >= START_SWAP_MAX_GB:
        out.append(f"swap used {h.swap_used_gb:.1f} GB >= {START_SWAP_MAX_GB:.0f} GB")
    need = job.peak_gb + START_HEADROOM_GB
    if h.ram_free_gb < need:
        out.append(f"RAM free {h.ram_free_gb:.1f} GB < {need:.1f} GB ({job.name} peak {job.peak_gb} + 2)")
    if h.swapins_per_s >= START_SWAPINS_MAX:
        out.append(f"swap-ins {h.swapins_per_s:.0f}/s >= {START_SWAPINS_MAX}/s")
    if h.disk_free_gb < START_DISK_GB:
        out.append(f"disk free {h.disk_free_gb:.1f} GB < {START_DISK_GB:.0f} GB")
    return out


def stop_verdict(h: Health, prev_swapins: float, ignore_window: bool = False) -> tuple[str | None, bool]:
    """(reason to preempt or None, kill_now) for a running job."""
    if h.ram_free_pct < KILL_FREE_PCT or h.swapins_per_s >= KILL_SWAPINS:
        return f"severe: RAM free {h.ram_free_pct:.0f}%, swap-ins {h.swapins_per_s:.0f}/s", True
    if h.ram_free_pct < STOP_FREE_PCT:
        return f"RAM free {h.ram_free_pct:.0f}% < {STOP_FREE_PCT:.0f}%", False
    if h.swapins_per_s >= STOP_SWAPINS and prev_swapins >= STOP_SWAPINS:
        return f"swap-ins {h.swapins_per_s:.0f}/s on 2 samples", False
    if h.disk_free_gb < STOP_DISK_GB:
        return f"disk free {h.disk_free_gb:.1f} GB", False
    if h.paused:
        return "PAUSE requested", False
    if not (h.window or ignore_window):
        return "overnight window ended", False
    return None, False


# ---------------------------------------------------------------- runner


@dataclass
class Status:
    pid: int = 0
    started: str = ""
    updated: str = ""
    state: str = "idle"  # waiting | running | done-for-tonight | done | blocked | stopped
    reason: str = ""
    job: str | None = None
    job_pid: int | None = None
    done: list[str] = field(default_factory=list)
    failed: dict[str, int] = field(default_factory=dict)
    preemptions: list[dict] = field(default_factory=list)
    blocked_until: str | None = None
    stable_since: str | None = None
    health: dict = field(default_factory=dict)
    progress: dict = field(default_factory=dict)


def train_progress() -> dict:
    for name in ("jev-local-fast", "smoke"):
        p = ROOT / "runs" / name / "log.jsonl"
        if p.is_file():
            try:
                with p.open("rb") as f:
                    f.seek(max(0, p.stat().st_size - 4000))
                    rec = json.loads(f.read().decode(errors="ignore").strip().splitlines()[-1])
                keep = ("step", "epoch", "loss", "examples_per_s", "tokens_per_s", "mps_mb", "rss_peak_mb", "elapsed_s")
                return {"run": name, **{k: rec[k] for k in keep if k in rec}}
            except (OSError, ValueError, IndexError):
                pass
    return {}


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class Runner:
    def __init__(self, ignore_window: bool) -> None:
        self.ignore_window = ignore_window
        self.mon = Monitor()
        self.st = Status(pid=os.getpid(), started=_now())
        prev = STATE / "status.json"
        if prev.is_file():  # keep queue position and tonight's preemption block across restarts
            try:
                old = json.loads(prev.read_text())
                self.st.done = old.get("done", [])
                self.st.failed = old.get("failed", {})
                pre = old.get("preemptions", [])
                self.st.preemptions = pre if isinstance(pre, list) else []
                self.st.blocked_until = old.get("blocked_until")
            except ValueError:
                pass
        self.proc: subprocess.Popen | None = None
        self.stable_since: float | None = None
        self.prev_swapins = 0.0
        self.stop_cause: str | None = None

    def log(self, msg: str) -> None:
        print(f"{_now()} {msg}", flush=True)

    def save(self, h: Health | None = None) -> None:
        self.st.updated = _now()
        if h:
            self.st.health = asdict(h)
        self.st.progress = train_progress()
        self.st.stable_since = (dt.datetime.fromtimestamp(self.stable_since).isoformat(timespec="seconds")
                                if self.stable_since else None)
        tmp = STATE / "status.json.tmp"
        tmp.write_text(json.dumps(asdict(self.st), indent=1))
        tmp.replace(STATE / "status.json")

    def next_job(self) -> Job | None:
        for j in JOBS:
            if j.name in self.st.done or self.st.failed.get(j.name, 0) >= MAX_FAILS:
                continue
            if j.after and j.after not in self.st.done:
                return None
            if j.optional_file and not (ROOT / j.optional_file).exists():
                self.st.done.append(j.name)
                self.log(f"skip {j.name}: {j.optional_file} missing")
                continue
            return j
        return None

    def block_for_tonight(self, why: str, h: Health) -> None:
        until = next_window_start(dt.datetime.now())
        self.st.blocked_until = until.isoformat(timespec="minutes")
        self.st.preemptions.append({"t": _now(), "job": self.st.job, "why": why,
                                    "ram_free_pct": h.ram_free_pct, "swapins_per_s": h.swapins_per_s})
        self.st.state, self.st.reason = "done-for-tonight", f"preempted ({why}); next try {self.st.blocked_until}"
        self.log(self.st.reason)

    def start(self, j: Job) -> None:
        LOGS.mkdir(parents=True, exist_ok=True)
        logf = (LOGS / f"{j.name}.log").open("a")
        logf.write(f"\n===== {_now()} start {' '.join(j.cmd[1:])}\n")
        logf.flush()
        self.proc = subprocess.Popen(["nice", "-n", "10", *j.cmd], cwd=ROOT, env={**os.environ, **JOB_ENV},
                                     stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        self.st.job, self.st.job_pid, self.st.state, self.st.reason = j.name, self.proc.pid, "running", ""
        self.log(f"start {j.name} pid={self.proc.pid} (peak ~{j.peak_gb} GB)")

    def preempt(self, why: str, kill_now: bool, h: Health) -> None:
        assert self.proc is not None
        self.log(f"preempt {self.st.job}: {why}{' (SIGKILL)' if kill_now else ''}")
        if kill_now:
            self.proc.kill()
            self.proc.wait()
        else:
            self.proc.send_signal(signal.SIGTERM)  # trainer: finish the step, checkpoint, exit 130
            try:
                self.proc.wait(timeout=TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                self.log(f"no exit {TERM_GRACE_S}s after SIGTERM; SIGKILL")
                self.proc.kill()
                self.proc.wait()
        self.block_for_tonight(why, h)
        self.proc, self.st.job, self.st.job_pid = None, None, None

    def run(self) -> None:
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "runner.pid").write_text(str(os.getpid()))
        # Say WHY we stop: on 2026-09-28 13:42 the runner exited two minutes after a wake from sleep
        # and nobody had written STOP, so an unlogged SIGTERM is the prime suspect.
        def on_term(signum, _frame):
            self.stop_cause = f"signal {signal.Signals(signum).name} (parent pid {os.getppid()})"
            self.log(f"received {self.stop_cause}; stopping after this sample")
            (STATE / "STOP").touch()

        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGINT, on_term)
        signal.signal(signal.SIGHUP, lambda *_: self.log("ignored SIGHUP"))  # a hangup is not a stop request
        self.log(f"runner up pid={os.getpid()} done={self.st.done} blocked_until={self.st.blocked_until}")
        self.mon.sample()  # prime the swap-in rate
        time.sleep(SAMPLE_S)
        while True:
            h = self.mon.sample()
            stop_req = (STATE / "STOP").exists()

            if self.proc is not None:
                rc = self.proc.poll()
                if rc is None:
                    why, kill_now = stop_verdict(h, self.prev_swapins, self.ignore_window)
                    if stop_req and not why:
                        why = "STOP requested"
                    if why:
                        self.preempt(why, kill_now, h)
                else:
                    name = self.st.job or "?"
                    if rc == 0:
                        self.st.done.append(name)
                        self.st.state = "idle"
                        self.log(f"done {name}")
                    elif rc in PREEMPTED_RCS:
                        self.block_for_tonight(f"{name} stopped externally (rc={rc})", h)
                    else:
                        self.st.failed[name] = self.st.failed.get(name, 0) + 1
                        self.st.state = "idle"
                        self.log(f"FAILED {name} rc={rc} (attempt {self.st.failed[name]}); see runs/overnight/{name}.log")
                    self.proc, self.st.job, self.st.job_pid = None, None, None
                    self.stable_since = None
            self.prev_swapins = h.swapins_per_s

            if stop_req and self.proc is None:
                self.st.state = "stopped"
                self.st.reason = f"stopped by {self.stop_cause or 'STOP file'}"
                self.save(h)
                (STATE / "STOP").unlink(missing_ok=True)
                self.log(f"runner stopped: {self.stop_cause or 'STOP file'}")
                return

            if self.proc is None:
                job = self.next_job()
                if job is None:
                    pending = [j.name for j in JOBS if j.name not in self.st.done]
                    self.st.state = "done" if not pending else "blocked"
                    self.st.reason = "queue finished" if not pending else f"blocked: {pending} (a dependency failed)"
                    self.save(h)
                    self.log(self.st.reason)
                    return
                if self.st.blocked_until and dt.datetime.now() < dt.datetime.fromisoformat(self.st.blocked_until):
                    self.st.state = "done-for-tonight"
                    self.stable_since = None
                else:
                    self.st.blocked_until = None
                    blockers = start_blockers(h, job, self.ignore_window)
                    if blockers:
                        self.stable_since = None
                        self.st.state, self.st.reason = ("paused" if h.paused else "waiting"), "; ".join(blockers)
                    else:
                        self.stable_since = self.stable_since or h.t
                        held = h.t - self.stable_since
                        if held >= START_STABLE_S:
                            self.start(job)
                            self.stable_since = None
                        else:
                            self.st.state, self.st.reason = "waiting", f"conditions OK for {held:.0f}/{START_STABLE_S}s"
            self.save(h)
            time.sleep(SAMPLE_S)


def daemonize(log_path: Path) -> None:
    """Double fork so the runner outlives the shell and any Claude session that started it."""
    if os.fork():
        os._exit(0)
    os.setsid()
    if os.fork():
        os._exit(0)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)


def running_pid() -> int | None:
    p = STATE / "runner.pid"
    if not p.is_file():
        return None
    try:
        pid = int(p.read_text())
    except ValueError:
        return None
    cmd = _run(["ps", "-p", str(pid), "-o", "command="])  # pid may have been reused after a reboot
    return pid if "overnight.py" in cmd else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--ignore-window", action="store_true", help="allow heavy jobs outside the overnight window")
    args = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)

    if args.status:
        p = STATE / "status.json"
        print(p.read_text() if p.is_file() else "{}")
        print(f"runner alive: {running_pid()}")
        return
    if args.check:
        mon = Monitor()
        mon.sample()
        time.sleep(SAMPLE_S)
        h = mon.sample()
        job = next(j for j in JOBS if j.name == "train")
        b = start_blockers(h, job, args.ignore_window)
        print(json.dumps({"health": asdict(h), "train_can_start": not b, "blockers": b}, indent=1))
        return
    pid = running_pid()
    if pid:
        raise SystemExit(f"runner already running (pid {pid})")
    (STATE / "STOP").unlink(missing_ok=True)
    if args.daemon:
        daemonize(LOGS / "runner.log")
    Runner(args.ignore_window).run()


if __name__ == "__main__":
    main()
