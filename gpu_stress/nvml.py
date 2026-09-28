"""ctypes binding for NVML (libnvidia-ml.so, ships with the NVIDIA driver).

Replaces the pynvml dependency for the handful of queries the monitor needs.
Every query is best-effort: unsupported fields come back as None so the
pipeline keeps working on laptops / vGPUs that don't expose everything.
"""
from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass, field

NVML_TEMPERATURE_GPU = 0
NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0
NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1
NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2

# nvmlClocksThrottleReasons / nvmlClocksEventReasons bitmask
THROTTLE_REASONS = {
    0x0000000000000001: "gpu_idle",
    0x0000000000000002: "applications_clocks_setting",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000010: "sync_boost",
    0x0000000000000020: "sw_thermal_slowdown",
    0x0000000000000040: "hw_thermal_slowdown",
    0x0000000000000080: "hw_power_brake_slowdown",
    0x0000000000000100: "display_clock_setting",
}
# Reasons that indicate the GPU is being held back by thermals/power/hardware.
BAD_THROTTLE = {"hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown"}
SOFT_THROTTLE = {"sw_power_cap"}


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


@dataclass
class Sample:
    t: float
    util: float | None = None
    mem_used: int | None = None
    mem_total: int | None = None
    temp: int | None = None
    power_w: float | None = None
    sm_clock: int | None = None
    mem_clock: int | None = None
    fan: int | None = None
    throttle: list[str] = field(default_factory=list)

    @property
    def mem_pct(self) -> float | None:
        if self.mem_used is None or not self.mem_total:
            return None
        return 100.0 * self.mem_used / self.mem_total


def _load() -> ctypes.CDLL | None:
    names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    if sys.platform == "win32":
        names = ["nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"]
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    return None


class Nvml:
    """One NVML session bound to a device index."""

    def __init__(self, index: int = 0) -> None:
        self.lib = _load()
        self.ok = self.lib is not None
        self.handle = ctypes.c_void_p()
        if not self.ok:
            return
        if self.lib.nvmlInit_v2() != 0:
            self.ok = False
            return
        if self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)) != 0:
            self.ok = False
            self.lib.nvmlShutdown()

    # -- helpers ------------------------------------------------------------
    def _uint(self, fn: str, *args) -> int | None:
        if not self.ok:
            return None
        v = ctypes.c_uint()
        if getattr(self.lib, fn)(self.handle, *args, ctypes.byref(v)) != 0:
            return None
        return v.value

    def _str(self, fn: str, size: int = 96) -> str | None:
        if not self.ok:
            return None
        buf = ctypes.create_string_buffer(size)
        if getattr(self.lib, fn)(self.handle, buf, size) != 0:
            return None
        return buf.value.decode(errors="replace")

    # -- static info ---------------------------------------------------------
    def name(self) -> str | None:
        return self._str("nvmlDeviceGetName")

    def driver_version(self) -> str | None:
        if not self.ok:
            return None
        buf = ctypes.create_string_buffer(80)
        if self.lib.nvmlSystemGetDriverVersion(buf, 80) != 0:
            return None
        return buf.value.decode()

    def power_limit_w(self) -> float | None:
        v = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        return None if v is None else v / 1000.0

    def max_clock(self, kind: int = NVML_CLOCK_SM) -> int | None:
        return self._uint("nvmlDeviceGetMaxClockInfo", kind)

    def temp_threshold(self, kind: int = NVML_TEMPERATURE_THRESHOLD_SLOWDOWN) -> int | None:
        return self._uint("nvmlDeviceGetTemperatureThreshold", kind)

    def bus_width(self) -> int | None:
        return self._uint("nvmlDeviceGetMemoryBusWidth")

    def pcie_link(self) -> dict:
        """Current and maximum PCIe training. A card sitting in an x4 slot, or a
        link that never leaves gen1, shows up here rather than as a mystery in
        the transfer benchmark."""
        return {
            "pcie_gen": self._uint("nvmlDeviceGetCurrPcieLinkGeneration"),
            "pcie_width": self._uint("nvmlDeviceGetCurrPcieLinkWidth"),
            "pcie_gen_max": self._uint("nvmlDeviceGetMaxPcieLinkGeneration"),
            "pcie_width_max": self._uint("nvmlDeviceGetMaxPcieLinkWidth"),
        }

    def static_info(self) -> dict:
        return {
            "nvml_available": self.ok,
            "driver_version": self.driver_version(),
            "power_limit_w": self.power_limit_w(),
            "max_sm_clock_mhz": self.max_clock(NVML_CLOCK_SM),
            "max_mem_clock_mhz": self.max_clock(NVML_CLOCK_MEM),
            "temp_slowdown_c": self.temp_threshold(NVML_TEMPERATURE_THRESHOLD_SLOWDOWN),
            "temp_shutdown_c": self.temp_threshold(NVML_TEMPERATURE_THRESHOLD_SHUTDOWN),
            "bus_width_bits": self.bus_width(),
            **self.pcie_link(),
        }

    # -- dynamic -------------------------------------------------------------
    def throttle_reasons(self) -> list[str]:
        if not self.ok:
            return []
        mask = ctypes.c_ulonglong()
        for fn in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            f = getattr(self.lib, fn, None)
            if f is not None and f(self.handle, ctypes.byref(mask)) == 0:
                return [n for bit, n in THROTTLE_REASONS.items() if mask.value & bit]
        return []

    def sample(self, t: float) -> Sample:
        s = Sample(t=t)
        if not self.ok:
            return s
        u = _Utilization()
        if self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(u)) == 0:
            s.util = float(u.gpu)
        m = _Memory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(m)) == 0:
            s.mem_used, s.mem_total = int(m.used), int(m.total)
        s.temp = self._uint("nvmlDeviceGetTemperature", NVML_TEMPERATURE_GPU)
        p = self._uint("nvmlDeviceGetPowerUsage")
        s.power_w = None if p is None else p / 1000.0
        s.sm_clock = self._uint("nvmlDeviceGetClockInfo", NVML_CLOCK_SM)
        s.mem_clock = self._uint("nvmlDeviceGetClockInfo", NVML_CLOCK_MEM)
        s.fan = self._uint("nvmlDeviceGetFanSpeed")
        s.throttle = self.throttle_reasons()
        return s

    def close(self) -> None:
        if self.ok:
            self.lib.nvmlShutdown()
            self.ok = False
