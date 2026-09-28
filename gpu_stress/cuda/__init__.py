"""CUDA driver API + NVRTC ctypes bindings and the PTX kernels."""
from __future__ import annotations

from importlib.resources import files


def read_kernel_file(name: str) -> str:
    """Read a packaged kernel file (kernels.cu, kernels_smXX.ptx).

    Goes through importlib.resources rather than __file__ so the kernels are
    still readable when gpu_stress runs from a wheel, a zipapp or any other
    non-directory loader.
    """
    return files(__package__).joinpath(name).read_text(encoding="utf-8")
