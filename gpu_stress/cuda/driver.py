"""ctypes binding for the subset of the CUDA driver API (libcuda.so) that the
lite pipeline needs. libcuda ships with the NVIDIA driver, so this works on any
machine that can run nvidia-smi - no toolkit, no PyTorch.
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys
from dataclasses import dataclass

# CUdevice_attribute values (cuda.h)
ATTR_MULTIPROCESSOR_COUNT = 16
ATTR_CLOCK_RATE = 13                # kHz
ATTR_MEMORY_CLOCK_RATE = 36         # kHz
ATTR_GLOBAL_MEMORY_BUS_WIDTH = 37   # bits
ATTR_COMPUTE_CAPABILITY_MAJOR = 75
ATTR_COMPUTE_CAPABILITY_MINOR = 76
ATTR_MAX_THREADS_PER_BLOCK = 1
ATTR_L2_CACHE_SIZE = 38
ATTR_PCI_BUS_ID = 33
ATTR_PCI_DEVICE_ID = 34

CU_EVENT_DEFAULT = 0
CUDA_ERROR_OUT_OF_MEMORY = 2


class CudaError(RuntimeError):
    def __init__(self, code: int, name: str, msg: str, fn: str):
        super().__init__(f"{fn} failed: {name} ({code}): {msg}")
        self.code = code
        self.name = name


class CudaOutOfMemory(CudaError):
    pass


def _load_libcuda() -> ctypes.CDLL:
    names = ["libcuda.so.1", "libcuda.so"]
    if sys.platform == "win32":
        names = ["nvcuda.dll"]
    elif sys.platform == "darwin":
        names = ["/usr/local/cuda/lib/libcuda.dylib"]
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    raise RuntimeError("libcuda not found - is the NVIDIA driver installed?")


class Driver:
    """Thin checked wrapper over libcuda."""

    def __init__(self) -> None:
        self.lib = _load_libcuda()
        self.lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.check(self.lib.cuInit(0), "cuInit")

    def check(self, rc: int, fn: str) -> None:
        if rc == 0:
            return
        name, msg = ctypes.c_char_p(), ctypes.c_char_p()
        self.lib.cuGetErrorName(rc, ctypes.byref(name))
        self.lib.cuGetErrorString(rc, ctypes.byref(msg))
        n = (name.value or b"?").decode()
        m = (msg.value or b"?").decode()
        if rc == CUDA_ERROR_OUT_OF_MEMORY:
            raise CudaOutOfMemory(rc, n, m, fn)
        raise CudaError(rc, n, m, fn)

    def call(self, fn: str, *args) -> None:
        self.check(getattr(self.lib, fn)(*args), fn)

    # -- versions / devices --------------------------------------------------
    def driver_version(self) -> str:
        v = ctypes.c_int()
        self.call("cuDriverGetVersion", ctypes.byref(v))
        return f"{v.value // 1000}.{(v.value % 1000) // 10}"

    def device_count(self) -> int:
        n = ctypes.c_int()
        self.call("cuDeviceGetCount", ctypes.byref(n))
        return n.value


@dataclass
class DeviceInfo:
    index: int
    name: str
    total_mem: int
    sm_count: int
    cc_major: int
    cc_minor: int
    clock_khz: int
    mem_clock_khz: int
    bus_width: int
    l2_bytes: int
    pci_bus_id: str

    @property
    def cc(self) -> str:
        return f"{self.cc_major}.{self.cc_minor}"

    def as_dict(self) -> dict:
        return {
            "index": self.index, "name": self.name, "total_mem_bytes": self.total_mem,
            "sm_count": self.sm_count, "compute_capability": self.cc,
            "max_sm_clock_mhz": self.clock_khz / 1000, "mem_clock_mhz": self.mem_clock_khz / 1000,
            "bus_width_bits": self.bus_width, "l2_cache_bytes": self.l2_bytes, "pci_bus_id": self.pci_bus_id,
        }


class Context:
    """Primary context on one device + module / memory / launch helpers."""

    def __init__(self, drv: Driver, index: int = 0) -> None:
        self.drv = drv
        self.lib = drv.lib
        self.dev = ctypes.c_int()
        drv.call("cuDeviceGet", ctypes.byref(self.dev), index)
        self.ctx = ctypes.c_void_p()
        drv.call("cuDevicePrimaryCtxRetain", ctypes.byref(self.ctx), self.dev)
        drv.call("cuCtxSetCurrent", self.ctx)
        self.index = index
        self.info = self._query_info()
        self._modules: list[ctypes.c_void_p] = []

    # -- info ------------------------------------------------------------------
    def _attr(self, attr: int) -> int:
        v = ctypes.c_int()
        self.drv.call("cuDeviceGetAttribute", ctypes.byref(v), attr, self.dev)
        return v.value

    def _query_info(self) -> DeviceInfo:
        buf = ctypes.create_string_buffer(256)
        self.drv.call("cuDeviceGetName", buf, 256, self.dev)
        total = ctypes.c_size_t()
        self.drv.call("cuDeviceTotalMem_v2", ctypes.byref(total), self.dev)
        pci = ctypes.create_string_buffer(32)
        self.drv.call("cuDeviceGetPCIBusId", pci, 32, self.dev)
        return DeviceInfo(
            index=self.index, name=buf.value.decode(), total_mem=total.value,
            sm_count=self._attr(ATTR_MULTIPROCESSOR_COUNT),
            cc_major=self._attr(ATTR_COMPUTE_CAPABILITY_MAJOR), cc_minor=self._attr(ATTR_COMPUTE_CAPABILITY_MINOR),
            clock_khz=self._attr(ATTR_CLOCK_RATE), mem_clock_khz=self._attr(ATTR_MEMORY_CLOCK_RATE),
            bus_width=self._attr(ATTR_GLOBAL_MEMORY_BUS_WIDTH), l2_bytes=self._attr(ATTR_L2_CACHE_SIZE),
            pci_bus_id=pci.value.decode(),
        )

    def mem_info(self) -> tuple[int, int]:
        """(free, total) bytes as seen by the driver (includes other processes)."""
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.drv.call("cuMemGetInfo_v2", ctypes.byref(free), ctypes.byref(total))
        return free.value, total.value

    # -- modules ---------------------------------------------------------------
    def load_ptx(self, ptx: str) -> Module:
        mod = ctypes.c_void_p()
        self.drv.call("cuModuleLoadData", ctypes.byref(mod), ptx.encode())
        self._modules.append(mod)
        return Module(self, mod)

    # -- memory ----------------------------------------------------------------
    def alloc(self, nbytes: int) -> DeviceBuffer:
        return DeviceBuffer(self, nbytes)

    def synchronize(self) -> None:
        self.drv.call("cuCtxSynchronize")

    def event(self) -> Event:
        return Event(self)

    def close(self) -> None:
        for m in self._modules:
            self.lib.cuModuleUnload(m)
        self._modules.clear()
        self.lib.cuDevicePrimaryCtxRelease_v2(self.dev)


class DeviceBuffer:
    def __init__(self, ctx: Context, nbytes: int) -> None:
        self.ctx = ctx
        self.nbytes = nbytes
        self.ptr = ctypes.c_uint64()
        ctx.drv.call("cuMemAlloc_v2", ctypes.byref(self.ptr), ctypes.c_size_t(nbytes))

    def free(self) -> None:
        if self.ptr.value:
            self.ctx.lib.cuMemFree_v2(self.ptr)
            self.ptr.value = 0

    def __enter__(self) -> DeviceBuffer:
        return self

    def __exit__(self, *exc) -> None:
        self.free()

    def memset32(self, value: int, count: int | None = None) -> None:
        count = self.nbytes // 4 if count is None else count
        self.ctx.drv.call("cuMemsetD32_v2", self.ptr, ctypes.c_uint(value), ctypes.c_size_t(count))

    def upload(self, data: bytes) -> None:
        self.ctx.drv.call("cuMemcpyHtoD_v2", self.ptr, data, ctypes.c_size_t(len(data)))

    def download(self, nbytes: int | None = None, offset: int = 0) -> bytes:
        nbytes = self.nbytes if nbytes is None else nbytes
        host = ctypes.create_string_buffer(nbytes)
        src = ctypes.c_uint64(self.ptr.value + offset)
        self.ctx.drv.call("cuMemcpyDtoH_v2", host, src, ctypes.c_size_t(nbytes))
        return host.raw

    def download_f32(self, count: int | None = None, offset_elems: int = 0) -> list[float]:
        count = self.nbytes // 4 if count is None else count
        return list(struct.unpack(f"{count}f", self.download(count * 4, offset_elems * 4)))

    def download_u32(self, count: int | None = None) -> list[int]:
        count = self.nbytes // 4 if count is None else count
        return list(struct.unpack(f"{count}I", self.download(count * 4)))

    def copy_from(self, src: DeviceBuffer, nbytes: int | None = None) -> None:
        nbytes = min(self.nbytes, src.nbytes) if nbytes is None else nbytes
        self.ctx.drv.call("cuMemcpyDtoDAsync_v2", self.ptr, src.ptr, ctypes.c_size_t(nbytes), None)


class Event:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.ev = ctypes.c_void_p()
        ctx.drv.call("cuEventCreate", ctypes.byref(self.ev), CU_EVENT_DEFAULT)

    def record(self) -> Event:
        self.ctx.drv.call("cuEventRecord", self.ev, None)
        return self

    def synchronize(self) -> None:
        self.ctx.drv.call("cuEventSynchronize", self.ev)

    def elapsed_ms(self, start: Event) -> float:
        ms = ctypes.c_float()
        self.ctx.drv.call("cuEventElapsedTime", ctypes.byref(ms), start.ev, self.ev)
        return ms.value

    def __del__(self) -> None:
        try:
            if self.ev.value:
                self.ctx.lib.cuEventDestroy_v2(self.ev)
        except Exception:
            pass


class Module:
    def __init__(self, ctx: Context, handle: ctypes.c_void_p) -> None:
        self.ctx = ctx
        self.handle = handle

    def function(self, name: str) -> Kernel:
        fn = ctypes.c_void_p()
        self.ctx.drv.call("cuModuleGetFunction", ctypes.byref(fn), self.handle, name.encode())
        return Kernel(self.ctx, fn, name)


# Argument coercion for cuLaunchKernel: Python values -> ctypes scalars.
def _to_ctype(arg):
    if isinstance(arg, DeviceBuffer):
        return ctypes.c_uint64(arg.ptr.value)
    if isinstance(arg, bool):
        return ctypes.c_int(int(arg))
    if isinstance(arg, float):
        return ctypes.c_float(arg)
    if isinstance(arg, int):
        # ints must be tagged explicitly for width; default to 32-bit signed
        return ctypes.c_int(arg)
    if isinstance(arg, ctypes._SimpleCData):
        return arg
    raise TypeError(f"unsupported kernel arg {arg!r}")


class Kernel:
    def __init__(self, ctx: Context, fn: ctypes.c_void_p, name: str) -> None:
        self.ctx = ctx
        self.fn = fn
        self.name = name

    def launch(self, grid: int | tuple, block: int | tuple, *args, shared: int = 0) -> None:
        gx, gy, gz = (grid, 1, 1) if isinstance(grid, int) else (tuple(grid) + (1, 1))[:3]
        bx, by, bz = (block, 1, 1) if isinstance(block, int) else (tuple(block) + (1, 1))[:3]
        cargs = [_to_ctype(a) for a in args]
        ptrs = (ctypes.c_void_p * len(cargs))(*[ctypes.addressof(a) for a in cargs])
        self.ctx.drv.call(
            "cuLaunchKernel", self.fn, gx, gy, gz, bx, by, bz, shared, None, ptrs, None,
        )


u64 = ctypes.c_uint64
u32 = ctypes.c_uint32
i32 = ctypes.c_int32
f32 = ctypes.c_float


def cores_per_sm(major: int, minor: int) -> int:
    """FP32 CUDA cores per SM, mirroring the helper in CUDA samples."""
    table = {
        (3, 0): 192, (3, 5): 192, (3, 7): 192,
        (5, 0): 128, (5, 2): 128, (5, 3): 128,
        (6, 0): 64, (6, 1): 128, (6, 2): 128,
        (7, 0): 64, (7, 2): 64, (7, 5): 64,
        (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
        (9, 0): 128,
        (10, 0): 128, (10, 1): 128, (10, 3): 128,
        (12, 0): 128, (12, 1): 128,
    }
    return table.get((major, minor), 128)


def ptx_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))
