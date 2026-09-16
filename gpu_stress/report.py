"""Console / JSON / CSV / Markdown reporting for a pipeline run."""
from __future__ import annotations

import csv
import json
import os
import platform
from datetime import datetime

from . import __version__
from .pipeline import Context, StepResult, overall_status


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:,.1f}"
    return str(v)


def key_metrics(r: StepResult) -> str:
    """Pick the human-interesting numbers of a step for the summary table."""
    m = r.metrics
    picks = []
    for k in ("tflops", "tflops_mean", "gbs", "gbs_copy_kernel", "errors", "imgs_per_sec", "steps_per_sec",
              "tested_gb", "batch_size", "dataset_images", "speedup", "gflops"):
        if k in m and m[k] is not None:
            picks.append(f"{k}={_fmt(m[k])}")
    return ", ".join(picks)


def print_summary(ctx: Context, backend: str) -> str:
    results = ctx.results
    status = overall_status(results)
    ctx.log("")
    ctx.log("#" * 72)
    ctx.log(f"#  FINAL REPORT ({backend})  -  overall: {status.upper()}")
    ctx.log("#" * 72)
    ctx.log(f"{'step':22s} {'status':8s} {'time':>7s}  {'maxT':>5s}  {'avgW':>6s}  key metrics")
    for r in results:
        t = r.telemetry
        temp = f"{t['temp_max']}°" if t.get("temp_max") is not None else "-"
        pw = f"{t['power_mean_w']:.0f}" if t.get("power_mean_w") is not None else "-"
        ctx.log(f"{r.name:22s} {r.status.upper():8s} {r.duration_s:6.1f}s  {temp:>5s}  {pw:>6s}  {key_metrics(r)}")
    fails = [(r.name, f) for r in results for f in r.findings if f.level == "fail"]
    warns = [(r.name, f) for r in results for f in r.findings if f.level == "warn"]
    if fails or warns:
        ctx.log("")
        for name, f in fails:
            ctx.log(f"  FAIL  [{name}] {f.message}")
        for name, f in warns:
            ctx.log(f"  WARN  [{name}] {f.message}")
    ctx.log("")
    return status


def write_reports(ctx: Context, backend: str, out_dir: str = "results", tag: str | None = None) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    status = overall_status(ctx.results)
    base = os.path.join(out_dir, f"{backend}_{ts}")
    paths = {}

    doc = {
        "version": __version__,
        "backend": backend,
        "timestamp": ts,
        "host": platform.node(),
        "python": platform.python_version(),
        "overall": status,
        "device": ctx.device,
        "config": {k: v for k, v in vars(ctx.config).items()} if hasattr(ctx.config, "__dict__") else {},
        "steps": [r.as_dict() for r in ctx.results],
        "telemetry_marks": ctx.monitor.marks,
    }
    paths["json"] = base + ".json"
    with open(paths["json"], "w") as f:
        json.dump(doc, f, indent=2, default=str)

    rows = ctx.monitor.to_rows()
    if rows:
        paths["telemetry_csv"] = base + "_telemetry.csv"
        with open(paths["telemetry_csv"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    paths["metrics_csv"] = base + "_metrics.csv"
    with open(paths["metrics_csv"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "status", "metric", "value"])
        for r in ctx.results:
            for k, v in r.metrics.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    w.writerow([r.name, r.status, k, v])

    paths["markdown"] = base + ".md"
    with open(paths["markdown"], "w") as f:
        d = ctx.device
        f.write(f"# GPU stress report - {d.get('name', '?')}\n\n")
        f.write(f"- backend: `{backend}`  - date: {ts}  - host: {platform.node()}\n")
        f.write(f"- driver: {d.get('driver_version', '?')}  - CC {d.get('compute_capability', '?')}  "
                f"- {d.get('sm_count', '?')} SMs  - {(d.get('total_mem_bytes') or 0)/2**30:.1f} GiB\n")
        f.write(f"- **overall: {status.upper()}**\n\n")
        f.write("| step | status | time | max °C | avg W | min SM MHz | key metrics |\n|---|---|---|---|---|---|---|\n")
        for r in ctx.results:
            t = r.telemetry
            f.write(f"| {r.name} | {r.status} | {r.duration_s:.1f}s | {t.get('temp_max', '-')} | "
                    f"{_fmt(t['power_mean_w']) if t.get('power_mean_w') is not None else '-'} | "
                    f"{t.get('sm_clock_min', '-')} | {key_metrics(r)} |\n")
        f.write("\n## Findings\n\n")
        for r in ctx.results:
            for fd in r.findings:
                f.write(f"- **{fd.level}** [{r.name}] {fd.message}\n")
            if r.error:
                f.write(f"- **error** [{r.name}] {r.error}\n")
        f.write("\n## Metrics\n\n")
        for r in ctx.results:
            f.write(f"### {r.name}\n\n```json\n{json.dumps(r.metrics, indent=2, default=str)}\n```\n\n")

    paths["log"] = base + ".log"
    with open(paths["log"], "w") as f:
        f.write("\n".join(ctx.log_lines))
    return paths


def plot_telemetry(ctx: Context, path: str | None, show: bool) -> str | None:
    """Optional matplotlib chart of the whole run with step boundaries."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        ctx.log("matplotlib not installed - skipping telemetry plot")
        return None
    rows = ctx.monitor.to_rows()
    if len(rows) < 2:
        return None
    t = [r["t"] for r in rows]
    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(f"GPU stress telemetry - {ctx.device.get('name', '')}")
    series = [("util", "GPU util %", (0, 100)), ("mem_pct", "VRAM %", (0, 100)),
              ("temp", "Temp °C", None), ("power_w", "Power W", None)]
    for ax, (key, label, ylim) in zip(axs, series):
        ax.plot(t, [r[key] for r in rows])
        ax.set_ylabel(label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    ax2 = axs[3].twinx()
    ax2.plot(t, [r["sm_clock"] for r in rows], color="tab:gray", alpha=0.6)
    ax2.set_ylabel("SM MHz")
    starts = {label[:-6]: mt for mt, label in ctx.monitor.marks if label.endswith(":start")}
    ends = {label[:-4]: mt for mt, label in ctx.monitor.marks if label.endswith(":end")}
    for name, mt in starts.items():
        for ax in axs:
            ax.axvline(mt, color="k", alpha=0.2, linestyle="--")
        if ends.get(name, mt) - mt >= 1.0:          # label only steps long enough to read
            axs[0].text(mt + 0.2, 95, name, rotation=90, fontsize=7, va="top")
    axs[3].set_xlabel("seconds")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
    return path
