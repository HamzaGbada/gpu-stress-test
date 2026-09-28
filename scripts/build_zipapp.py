#!/usr/bin/env python3
"""Build `gpu-stress.pyz`, a single-file executable of the lite pipeline.

The lite pipeline has no third-party dependencies, so the whole tool fits in
one ~90 KB zipapp that runs on any machine with CPython 3.10+ and an NVIDIA
driver - no pip, no virtualenv, no network at run time:

    python3 scripts/build_zipapp.py
    ./dist/gpu-stress.pyz --burn 30
    ./dist/gpu-stress.pyz torch --epochs 3      # only if torch is importable

Usage: build_zipapp.py [output_dir]
"""
from __future__ import annotations

import compileall
import shutil
import sys
import zipapp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "gpu_stress"

MAIN = '''\
"""Entry point for the gpu-stress zipapp."""
import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "torch":            # opt into the PyTorch pipeline
        from gpu_stress.torch_pipeline import main as run
        return run(argv[1:])
    from gpu_stress.lite_pipeline import main as run
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
'''


def build(out_dir: Path) -> Path:
    stage = out_dir / "_zipapp_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # Copy the package, keeping the .cu/.ptx kernels and dropping caches.
    shutil.copytree(PKG, stage / "gpu_stress",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    (stage / "__main__.py").write_text(MAIN)
    compileall.compile_dir(str(stage), quiet=2, force=True)

    out = out_dir / "gpu-stress.pyz"
    zipapp.create_archive(stage, target=out, interpreter="/usr/bin/env python3", compressed=True)
    out.chmod(0o755)
    shutil.rmtree(stage)
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
    dest.mkdir(parents=True, exist_ok=True)
    built = build(dest)
    print(f"{built}  ({built.stat().st_size / 1024:.0f} KiB)")
