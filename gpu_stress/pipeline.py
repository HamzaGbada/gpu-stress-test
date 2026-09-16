"""Generic multi-step runner shared by the torch and lite pipelines.

A Step is a callable ``fn(ctx) -> dict`` of metrics plus an optional
``evaluate(metrics, telemetry, ctx) -> list[Finding]``.  The runner opens a
telemetry window around each step, catches OOM / errors so one failing step
never kills the run, applies the generic telemetry rules, and produces an
ordered list of StepResult objects that the reporter turns into files.
"""
from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .evaluate import Finding, telemetry_findings, worst
from .monitor import Monitor, WindowSummary


class StepSkipped(Exception):
    """Raise inside a step to mark it skipped (e.g. unsupported hardware)."""


@dataclass
class Step:
    name: str
    fn: Callable[[Context], dict]
    description: str = ""
    heavy: bool = False          # apply thermal / throttle / clock rules
    evaluate: Callable[[dict, WindowSummary, Context], list[Finding]] | None = None


@dataclass
class StepResult:
    name: str
    status: str                  # pass | warn | fail | skipped | error
    duration_s: float
    metrics: dict = field(default_factory=dict)
    telemetry: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status, "duration_s": round(self.duration_s, 3),
            "metrics": self.metrics, "telemetry": self.telemetry,
            "findings": [f.as_dict() for f in self.findings], "error": self.error,
        }


@dataclass
class Context:
    """Everything a step may need. ``config`` is the parsed CLI namespace."""
    config: Any
    monitor: Monitor
    device: dict = field(default_factory=dict)      # static device facts (for theoretical peaks)
    results: list[StepResult] = field(default_factory=list)
    oom_types: tuple[type, ...] = ()                # exceptions treated as out-of-memory
    extra: dict = field(default_factory=dict)       # backend-specific handles (cuda ctx, modules...)
    log_lines: list[str] = field(default_factory=list)

    def log(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.log_lines.append(msg)

    def result(self, name: str) -> StepResult | None:
        return next((r for r in self.results if r.name == name), None)


class Pipeline:
    def __init__(self, steps: list[Step], ctx: Context) -> None:
        self.steps = steps
        self.ctx = ctx

    def run(self, only: set[str] | None = None, skip: set[str] | None = None) -> list[StepResult]:
        ctx = self.ctx
        selected = [s for s in self.steps if (not only or s.name in only) and (not skip or s.name not in skip)]
        ctx.log(f"\nPipeline: {' -> '.join(s.name for s in selected)}\n")
        for i, step in enumerate(selected, 1):
            ctx.log("=" * 72)
            ctx.log(f"[{i}/{len(selected)}] {step.name}  {('- ' + step.description) if step.description else ''}")
            ctx.log("=" * 72)
            res = self._run_step(step)
            ctx.results.append(res)
            self._print_result(res)
        return ctx.results

    def _run_step(self, step: Step) -> StepResult:
        ctx = self.ctx
        t0 = time.time()
        metrics: dict = {}
        findings: list[Finding] = []
        error = None
        status = "pass"
        with ctx.monitor.window(step.name) as win:
            try:
                metrics = step.fn(ctx) or {}
            except StepSkipped as e:
                status, error = "skipped", str(e)
            except ctx.oom_types as e:  # type: ignore[misc]
                status, error = "error", f"out of memory: {e}"
                findings.append(Finding("fail", f"step ran out of GPU memory: {str(e).splitlines()[0][:200]}"))
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 - one step failing must not kill the run
                status, error = "error", f"{type(e).__name__}: {e}"
                ctx.log(traceback.format_exc())
                findings.append(Finding("fail", f"step crashed: {type(e).__name__}: {str(e)[:200]}"))
        duration = time.time() - t0
        tele = win.summary()
        if status not in ("skipped",):
            if step.heavy:
                findings += telemetry_findings(tele, ctx.device)
            if step.evaluate and status != "error":
                try:
                    findings += step.evaluate(metrics, tele, ctx)
                except Exception as e:  # noqa: BLE001
                    findings.append(Finding("warn", f"evaluation failed: {type(e).__name__}: {e}"))
            if status == "pass":
                status = worst(findings)
        return StepResult(step.name, status, duration, metrics, tele.as_dict(), findings, error)

    def _print_result(self, r: StepResult) -> None:
        ctx = self.ctx
        tag = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skipped": "SKIP", "error": "ERR "}[r.status]
        line = f"  -> [{tag}] {r.name} in {r.duration_s:.1f}s"
        t = r.telemetry
        if t.get("temp_max") is not None:
            line += f" | max {t['temp_max']}°C"
        if t.get("power_mean_w") is not None:
            line += f" | avg {t['power_mean_w']:.0f} W"
        if t.get("sm_clock_min") is not None:
            line += f" | SM {t['sm_clock_min']}-{t['sm_clock_max']} MHz"
        ctx.log(line)
        if r.error:
            ctx.log(f"     {r.error}")
        for f in r.findings:
            ctx.log(f"     {f.level.upper():5s} {f.message}")
        ctx.log("")


def overall_status(results: list[StepResult]) -> str:
    order = {"pass": 0, "skipped": 0, "warn": 1, "fail": 2, "error": 2}
    if not results:
        return "skipped"
    return max(results, key=lambda r: order[r.status]).status if any(
        r.status not in ("pass", "skipped") for r in results) else "pass"
