"""Backwards-compatible entry point: ``python gpu_stress_cli.py --epochs 10 --batch 64 --vram 0.75``.

The stress test now lives in the ``gpu_stress`` package and runs as a
multi-step pipeline (bandwidth -> matmul -> compile -> matmul burn ->
ResNet50 training -> VRAM fill -> fragmentation) with evaluation and reports.
This wrapper maps the old flags onto ``gpu_stress.torch_pipeline``.

Run ``python -m gpu_stress.torch_pipeline --help`` for the full option set, or
``python -m gpu_stress.lite_pipeline`` for the no-PyTorch edition.
"""
import sys

from gpu_stress.torch_pipeline import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    # old flag: --nogui (graph was opt-out); new flag: --gui (opt-in)
    if "--nogui" in argv:
        argv = [a for a in argv if a != "--nogui"]
    elif "--gui" not in argv:
        argv.append("--gui")
    sys.exit(main(argv))
