"""Minimal ctypes binding for NVRTC (runtime CUDA C -> PTX compiler).

Optional: only used when a CUDA toolkit's libnvrtc is present, to compile the
kernels for the exact GPU architecture. The shipped .ptx files are the
fallback and need nothing beyond the NVIDIA driver.
"""
from __future__ import annotations

import ctypes
import glob
import os

_CANDIDATES = [
    "libnvrtc.so",
    "/opt/cuda/lib64/libnvrtc.so",
    "/usr/local/cuda/lib64/libnvrtc.so",
]


def find_nvrtc() -> ctypes.CDLL | None:
    """Locate libnvrtc. Order: $NVRTC_LIB, $CUDA_HOME, $CUDA_PATH, common paths."""
    names: list[str] = []
    if os.environ.get("NVRTC_LIB"):
        names.append(os.environ["NVRTC_LIB"])
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            names.append(os.path.join(os.environ[env], "lib64", "libnvrtc.so"))
    names += _CANDIDATES
    for base in ("/usr/local", "/opt"):
        names += sorted(glob.glob(f"{base}/cuda*/lib64/libnvrtc.so"), reverse=True)
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class NvrtcError(RuntimeError):
    pass


def compile_to_ptx(source: str, arch: str, name: str = "kernels.cu", lib: ctypes.CDLL | None = None) -> str:
    """Compile CUDA C `source` to PTX for `arch` (e.g. 'compute_89')."""
    lib = lib or find_nvrtc()
    if lib is None:
        raise NvrtcError("libnvrtc not found")

    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    prog = ctypes.c_void_p()

    def check(code: int) -> None:
        if code != 0:
            raise NvrtcError(lib.nvrtcGetErrorString(code).decode())

    check(lib.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), 0, None, None))
    opts = [f"--gpu-architecture={arch}", "-default-device", "--std=c++17"]
    c_opts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    rc = lib.nvrtcCompileProgram(prog, len(opts), c_opts)
    log_size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    lib.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
        raise NvrtcError(f"nvrtc compile failed for {arch}:\n{log.value.decode()}")
    ptx_size = ctypes.c_size_t()
    check(lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    check(lib.nvrtcGetPTX(prog, ptx))
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.value.decode()


def build_shipped_ptx() -> None:
    """Regenerate kernels_sm70.ptx / kernels_sm80.ptx next to kernels.cu.

    Only needed when kernels.cu changes. Prefer an older NVRTC (12.x) so the
    emitted PTX ISA is understood by older drivers.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "kernels.cu")) as f:
        src = f.read()
    lib = find_nvrtc()
    if lib is None:
        raise SystemExit("libnvrtc not found - install a CUDA toolkit or set CUDA_HOME")
    major, minor = ctypes.c_int(), ctypes.c_int()
    lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    print(f"NVRTC {major.value}.{minor.value}")
    for arch, fname in (("compute_70", "kernels_sm70.ptx"), ("compute_80", "kernels_sm80.ptx")):
        ptx = compile_to_ptx(src, arch, lib=lib)
        with open(os.path.join(here, fname), "w") as f:
            f.write(ptx)
        version = next(line for line in ptx.splitlines() if line.startswith(".version"))
        print(f"  {fname}: {len(ptx)} bytes, {version}, {ptx.count('.entry')} kernels")


if __name__ == "__main__":
    build_shipped_ptx()
