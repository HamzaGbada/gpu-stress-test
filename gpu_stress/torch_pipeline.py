"""Full PyTorch GPU stress & evaluation pipeline.

    python -m gpu_stress.torch_pipeline --epochs 3 --batch 64 --precision bf16
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def _configure_allocator() -> None:
    """Must run before torch is imported: lets the caching allocator grow
    segments instead of fragmenting, which is what the OOM message suggests.
    torch >= 2.8 reads PYTORCH_ALLOC_CONF (and warns about the old name)."""
    if os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return
    try:
        from importlib.metadata import version
        major, minor = (int(x) for x in version("torch").split("+")[0].split(".")[:2])
        name = "PYTORCH_ALLOC_CONF" if (major, minor) >= (2, 8) else "PYTORCH_CUDA_ALLOC_CONF"
    except Exception:  # noqa: BLE001
        name = "PYTORCH_CUDA_ALLOC_CONF"
    os.environ[name] = "expandable_segments:True"


_configure_allocator()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch GPU stress test & evaluation pipeline")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3, help="ResNet50 training epochs (default 3)")
    p.add_argument("--batch", type=int, default=64, help="requested batch size (auto-halved until it fits)")
    p.add_argument("--max-steps", type=int, default=0, help="cap steps per epoch (0 = whole dataset)")
    p.add_argument("--vram", type=float, default=0.75,
                   help="max fraction of total VRAM for the synthetic dataset; always capped by what is free")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32",
                   help="autocast precision for training and matmul burn (bf16 halves activation memory)")
    p.add_argument("--burn", type=float, default=30, help="seconds for the sustained matmul burn")
    p.add_argument("--matmul-size", type=int, default=8192, help="matmul N (auto-reduced to fit)")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--steps", default="", help="comma list of steps to run (default: all)")
    p.add_argument("--skip", default="", help="comma list of steps to skip")
    p.add_argument("--interval", type=float, default=0.25, help="telemetry sample interval")
    p.add_argument("--out", default="results")
    p.add_argument("--gui", action="store_true", help="show telemetry plot at the end")
    p.add_argument("--list", action="store_true", help="list steps and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from .torch_steps import OOM, build_steps
    steps = build_steps()
    if args.list:
        for s in steps:
            print(f"{s.name:14s} {s.description}")
        return 0

    import torch

    from .cuda.driver import cores_per_sm
    from .monitor import Monitor
    from .pipeline import Context, Pipeline
    from .report import plot_telemetry, print_summary, write_reports

    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch")
        return 2
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    mon = Monitor(args.device, args.interval).start()
    device = {"name": props.name, "total_mem_bytes": props.total_memory, "sm_count": props.multi_processor_count,
              "compute_capability": f"{props.major}.{props.minor}", "cc_major": props.major, "cc_minor": props.minor,
              "cores_per_sm": cores_per_sm(props.major, props.minor)}
    device.update({k: v for k, v in mon.nvml.static_info().items() if v is not None})
    ctx = Context(config=args, monitor=mon, device=device, oom_types=OOM)
    ctx.log(f"GPU stress (torch {torch.__version__}) on {props.name} - driver {device.get('driver_version')}")

    only = {s for s in args.steps.split(",") if s}
    skip = {s for s in args.skip.split(",") if s}
    status = "error"
    try:
        Pipeline(steps, ctx).run(only or None, skip or None)
        status = print_summary(ctx, "torch")
    except KeyboardInterrupt:
        ctx.log("\ninterrupted - writing partial report")
        status = print_summary(ctx, "torch") if ctx.results else "error"
    finally:
        mon.stop()
        paths = write_reports(ctx, "torch", args.out)
        png = plot_telemetry(ctx, paths["json"].replace(".json", "_telemetry.png"), args.gui)
        if png:
            paths["plot"] = png
        for k, v in paths.items():
            print(f"  {k:14s}: {v}")
    return {"pass": 0, "warn": 0, "fail": 1, "error": 2}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
