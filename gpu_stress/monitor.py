"""Background GPU telemetry thread + per-step windows.

The monitor samples NVML at a fixed interval for the whole run.  Each pipeline
step opens a window; when it closes, the samples taken inside it are reduced
to a summary (max temp, mean power, min clock, throttle reasons seen, ...) that
the evaluator scores.
"""
from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field

from .nvml import BAD_THROTTLE, SOFT_THROTTLE, Nvml, Sample


@dataclass
class WindowSummary:
    samples: int = 0
    duration_s: float = 0.0
    temp_max: int | None = None
    temp_start: int | None = None
    temp_end: int | None = None
    power_mean_w: float | None = None
    power_max_w: float | None = None
    util_mean: float | None = None
    mem_pct_max: float | None = None
    sm_clock_min: int | None = None
    sm_clock_mean: float | None = None
    sm_clock_max: int | None = None
    mem_clock_min: int | None = None
    throttle_reasons: list[str] = field(default_factory=list)
    throttle_pct: float = 0.0        # % of samples with a bad/soft throttle reason
    hard_throttle_pct: float = 0.0   # % of samples with thermal/HW slowdown

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def summarize(samples: list[Sample]) -> WindowSummary:
    w = WindowSummary(samples=len(samples))
    if not samples:
        return w
    w.duration_s = samples[-1].t - samples[0].t

    def col(name):
        return [getattr(s, name) for s in samples if getattr(s, name) is not None]

    temps = col("temp")
    if temps:
        w.temp_max, w.temp_start, w.temp_end = max(temps), temps[0], temps[-1]
    power = col("power_w")
    if power:
        w.power_mean_w, w.power_max_w = statistics.fmean(power), max(power)
    util = col("util")
    if util:
        w.util_mean = statistics.fmean(util)
    mem = [s.mem_pct for s in samples if s.mem_pct is not None]
    if mem:
        w.mem_pct_max = max(mem)
    sm = col("sm_clock")
    if sm:
        w.sm_clock_min, w.sm_clock_mean, w.sm_clock_max = min(sm), statistics.fmean(sm), max(sm)
    mc = col("mem_clock")
    if mc:
        w.mem_clock_min = min(mc)
    seen: set[str] = set()
    bad = hard = 0
    loaded = 0
    for s in samples:
        # Parked/idle GPUs (laptops especially) report thermal/power reasons
        # that mean nothing; only judge samples that are actually under load.
        if "gpu_idle" in s.throttle or (s.util is not None and s.util < 30):
            continue
        loaded += 1
        r = set(s.throttle)
        seen |= r
        if r & (BAD_THROTTLE | SOFT_THROTTLE):
            bad += 1
        if r & BAD_THROTTLE:
            hard += 1
    w.throttle_reasons = sorted(seen)
    w.throttle_pct = 100.0 * bad / loaded if loaded else 0.0
    w.hard_throttle_pct = 100.0 * hard / loaded if loaded else 0.0
    return w


class Monitor:
    def __init__(self, device: int = 0, interval: float = 0.25) -> None:
        self.nvml = Nvml(device)
        self.interval = interval
        self.samples: list[Sample] = []
        self.marks: list[tuple[float, str]] = []      # (t, "step:start"/"step:end")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.t0 = time.time()

    @property
    def available(self) -> bool:
        return self.nvml.ok

    def start(self) -> Monitor:
        if not self.nvml.ok:
            return self
        self.t0 = time.time()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            s = self.nvml.sample(time.time() - self.t0)
            with self._lock:
                self.samples.append(s)
            self._stop.wait(self.interval)

    def now(self) -> float:
        return time.time() - self.t0

    def latest(self) -> Sample | None:
        with self._lock:
            return self.samples[-1] if self.samples else None

    def snapshot(self) -> list[Sample]:
        with self._lock:
            return list(self.samples)

    def window(self, name: str) -> Window:
        return Window(self, name)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.nvml.close()

    def to_rows(self) -> list[dict]:
        return [
            {"t": round(s.t, 3), "util": s.util, "mem_pct": None if s.mem_pct is None else round(s.mem_pct, 2),
             "temp": s.temp, "power_w": s.power_w, "sm_clock": s.sm_clock, "mem_clock": s.mem_clock,
             "fan": s.fan, "throttle": "|".join(s.throttle)}
            for s in self.snapshot()
        ]


class Window:
    def __init__(self, mon: Monitor, name: str) -> None:
        self.mon = mon
        self.name = name
        self.t_start = 0.0
        self.t_end = 0.0

    def __enter__(self) -> Window:
        self.t_start = self.mon.now()
        self.mon.marks.append((self.t_start, f"{self.name}:start"))
        return self

    def __exit__(self, *exc) -> None:
        self.t_end = self.mon.now()
        self.mon.marks.append((self.t_end, f"{self.name}:end"))

    def summary(self) -> WindowSummary:
        samples = [s for s in self.mon.snapshot() if self.t_start <= s.t <= self.t_end]
        return summarize(samples)
