"""Zero-dependency GPU stress & evaluation pipeline (CUDA driver API + ctypes).

    python -m gpu_stress.lite_pipeline --burn 60 --vram 0.9
"""
from __future__ import annotations

import argparse
import sys

from .cuda import driver as cu
from .lite_steps import build_steps, load_kernels
from .monitor import Monitor
from .pipeline import Context, Pipeline
from .report import plot_telemetry, print_summary, write_reports


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU stress test & evaluation (no PyTorch, no toolkit needed)")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--burn", type=float, default=30, help="seconds per sustained burn step (default 30)")
    p.add_argument("--vram", type=float, default=0.9, help="fraction of free VRAM to cover in memtest (default 0.9)")
    p.add_argument("--mem-passes", type=int, default=6, help="memtest patterns (6 fixed + random)")
    p.add_argument("--matmul-size", type=int, default=4096, help="SGEMM N (auto-reduced to fit)")
    p.add_argument("--verify-samples", type=int, default=32, help="rows sampled for SGEMM verification")
    p.add_argument("--reps", type=int, default=20, help="repetitions for timing bandwidth/matmul")
    p.add_argument("--steps", default="", help="comma list of steps to run (default: all)")
    p.add_argument("--skip", default="", help="comma list of steps to skip")
    p.add_argument("--nvrtc", action="store_true", help="compile kernels with NVRTC for this GPU if a toolkit is present")
    p.add_argument("--interval", type=float, default=0.25, help="telemetry sample interval in seconds")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--gui", action="store_true", help="show telemetry plot at the end (needs matplotlib)")
    p.add_argument("--list", action="store_true", help="list steps and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    steps = build_steps()
    if args.list:
        for s in steps:
            print(f"{s.name:14s} {s.description}")
        return 0

    drv = cu.Driver()
    cctx = cu.Context(drv, args.device)
    mon = Monitor(args.device, args.interval).start()
    device = cctx.info.as_dict()
    device.update({"cc_major": cctx.info.cc_major, "cc_minor": cctx.info.cc_minor,
                   "cores_per_sm": cu.cores_per_sm(cctx.info.cc_major, cctx.info.cc_minor),
                   "cuda_driver_api": drv.driver_version()})
    nv = mon.nvml.static_info()
    device.update({k: v for k, v in nv.items() if v is not None})
    device.setdefault("driver_version", drv.driver_version())

    ctx = Context(config=args, monitor=mon, device=device, oom_types=(cu.CudaOutOfMemory,))
    ctx.extra["cuda"] = cctx
    ctx.log(f"GPU stress (lite) on {device['name']} - driver {device.get('driver_version')}")
    ctx.extra["kernels"] = load_kernels(ctx, cctx, args.nvrtc)

    only = {s for s in args.steps.split(",") if s}
    skip = {s for s in args.skip.split(",") if s}
    status = "error"
    try:
        Pipeline(steps, ctx).run(only or None, skip or None)
        status = print_summary(ctx, "lite")
    except KeyboardInterrupt:
        ctx.log("\ninterrupted - writing partial report")
        status = print_summary(ctx, "lite") if ctx.results else "error"
    finally:
        mon.stop()
        paths = write_reports(ctx, "lite", args.out)
        png = plot_telemetry(ctx, paths["json"].replace(".json", "_telemetry.png"), args.gui)
        if png:
            paths["plot"] = png
        for k, v in paths.items():
            print(f"  {k:14s}: {v}")
        cctx.close()
    return {"pass": 0, "warn": 0, "fail": 1, "error": 2}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
