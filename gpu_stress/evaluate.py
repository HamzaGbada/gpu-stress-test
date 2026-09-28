"""Evaluation rules: turn raw metrics + telemetry into pass / warn / fail findings.

Thresholds are deliberately conservative - the goal is to flag a GPU that is
throttling, mis-computing or under-performing its own theoretical peak, not
to rank GPUs against each other.
"""
from __future__ import annotations

from dataclasses import dataclass

from .monitor import WindowSummary

LEVELS = {"info": 0, "warn": 1, "fail": 2}


@dataclass
class Finding:
    level: str      # info | warn | fail
    message: str

    def as_dict(self) -> dict:
        return {"level": self.level, "message": self.message}


def worst(findings: list[Finding]) -> str:
    lvl = max((LEVELS[f.level] for f in findings), default=0)
    return {0: "pass", 1: "warn", 2: "fail"}[lvl]


# ---------------------------------------------------------------------------
# Theoretical peaks (used for efficiency ratios)
# ---------------------------------------------------------------------------
def theoretical_fp32_tflops(device: dict, clock_mhz: float | None = None) -> float | None:
    """SMs x FP32 cores x 2 (FMA) x clock. Pass the observed clock to get the
    achievable peak under the current power/thermal limits."""
    sm = device.get("sm_count")
    cores = device.get("cores_per_sm")
    clk = clock_mhz or device.get("max_sm_clock_mhz") or device.get("boost_clock_mhz")
    if not (sm and cores and clk):
        return None
    return sm * cores * 2 * clk * 1e6 / 1e12


def theoretical_bandwidth_gbs(device: dict) -> float | None:
    """GDDR: data rate = 2 x memory clock (NVML/driver report the I/O clock).

    Returns None when the bus width is unknown. Unified-memory parts (Grace
    Blackwell GB10, Jetson, iGPUs) report a width of 0 because there is no
    dedicated GPU bus to describe, so those are measured but not scored.
    """
    clk = device.get("mem_clock_mhz") or device.get("max_mem_clock_mhz")
    bus = device.get("bus_width_bits")
    if not (clk and bus):
        return None
    return clk * 1e6 * 2 * bus / 8 / 1e9


# Usable per-lane throughput after link encoding, GB/s (8b/10b up to gen2,
# 128b/130b from gen3, PAM4 from gen6).
PCIE_LANE_GBS = {1: 0.250, 2: 0.500, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def theoretical_pcie_gbs(device: dict) -> float | None:
    """One-directional peak of the host<->device link at its current training."""
    if device.get("integrated"):
        return None
    gen, width = device.get("pcie_gen"), device.get("pcie_width")
    if not (gen and width) or gen not in PCIE_LANE_GBS:
        return None
    return PCIE_LANE_GBS[gen] * width


# ---------------------------------------------------------------------------
# Generic telemetry rules applied to every heavy step
# ---------------------------------------------------------------------------
def telemetry_findings(t: WindowSummary, device: dict) -> list[Finding]:
    out: list[Finding] = []
    if t.samples == 0:
        return [Finding("info", "no telemetry (NVML unavailable)")]

    slowdown = device.get("temp_slowdown_c")
    if t.temp_max is not None:
        if slowdown and t.temp_max >= slowdown - 2:
            out.append(Finding("fail", f"GPU hit {t.temp_max}°C, at its slowdown threshold ({slowdown}°C)"))
        elif slowdown and t.temp_max >= slowdown - 8:
            out.append(Finding("warn", f"GPU reached {t.temp_max}°C, within 8°C of slowdown ({slowdown}°C)"))
        elif not slowdown and t.temp_max >= 90:
            out.append(Finding("warn", f"GPU reached {t.temp_max}°C"))

    if t.hard_throttle_pct > 0:
        hard = ("hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown")
        reasons = [r for r in t.throttle_reasons if r in hard]
        # Some boards (laptop Ada parts in particular) assert a thermal slowdown
        # flag while sitting 30 C below their own threshold. Believe the
        # thermometer over the flag when the two disagree that badly.
        thermal_only = all("thermal" in r for r in reasons)
        implausible = thermal_only and slowdown and t.temp_max is not None and t.temp_max < slowdown - 15
        if implausible:
            out.append(Finding("info", f"NVML reported a thermal slowdown {t.hard_throttle_pct:.0f}% of the time "
                                       f"({', '.join(reasons)}) but the GPU peaked at {t.temp_max}°C, far below its "
                                       f"{slowdown}°C threshold - treating the flag as spurious"))
        else:
            lvl = "fail" if t.hard_throttle_pct >= 10 else "warn"
            out.append(Finding(lvl, f"thermal/HW slowdown active {t.hard_throttle_pct:.0f}% of the time "
                                    f"({', '.join(reasons)})"))
    elif "sw_power_cap" in t.throttle_reasons and t.throttle_pct >= 50:
        out.append(Finding("info", f"power-capped {t.throttle_pct:.0f}% of the time (normal under full load)"))

    if t.sm_clock_min and t.sm_clock_max and t.duration_s >= 5:
        ratio = t.sm_clock_min / t.sm_clock_max
        if ratio < 0.5:
            out.append(Finding("warn", f"SM clock dipped to {t.sm_clock_min} MHz ({ratio*100:.0f}% of {t.sm_clock_max} MHz peak)"))
        max_clk = device.get("max_sm_clock_mhz")
        if max_clk and t.sm_clock_mean and t.sm_clock_mean < 0.6 * max_clk:
            # expected when the board is power-capped (laptops, low TDP limits) - only warn otherwise
            power_capped = "sw_power_cap" in t.throttle_reasons and t.throttle_pct >= 50
            out.append(Finding("info" if power_capped else "warn",
                               f"mean SM clock {t.sm_clock_mean:.0f} MHz is below 60% of rated {max_clk} MHz"
                               + (" (power limit)" if power_capped else "")))

    limit = device.get("power_limit_w")
    if limit and t.power_mean_w:
        out.append(Finding("info", f"avg power {t.power_mean_w:.0f} W / {limit:.0f} W limit ({100*t.power_mean_w/limit:.0f}%)"))
    return out


# ---------------------------------------------------------------------------
# Reusable metric rules for the steps
# ---------------------------------------------------------------------------
def compute_error_finding(errors: int, what: str) -> list[Finding]:
    if errors:
        return [Finding("fail", f"{errors} incorrect results in {what} - unstable GPU / bad overclock / faulty hardware")]
    return [Finding("info", f"{what}: all results verified")]


def efficiency_finding(measured: float, theoretical: float | None, unit: str, what: str,
                       warn_below: float = 0.6, fail_below: float = 0.3) -> list[Finding]:
    if not theoretical:
        return [Finding("info", f"{what}: {measured:.1f} {unit} (no theoretical peak available)")]
    eff = measured / theoretical
    msg = f"{what}: {measured:.1f} {unit} = {eff*100:.0f}% of theoretical {theoretical:.1f} {unit}"
    if eff < fail_below:
        return [Finding("fail", msg + " - far below expected, check driver/power/PCIe")]
    if eff < warn_below:
        return [Finding("warn", msg)]
    return [Finding("info", msg)]


def stability_finding(first: float, last: float, unit: str, what: str) -> list[Finding]:
    """Compare throughput at start vs end of a sustained run."""
    if not first:
        return []
    drop = 1 - last / first
    msg = f"{what}: {first:.1f} -> {last:.1f} {unit} (start vs end, {drop*100:+.0f}% drop)"
    if drop > 0.25:
        return [Finding("fail", msg + " - severe sustained throttling")]
    if drop > 0.10:
        return [Finding("warn", msg + " - performance degrades under sustained load")]
    return [Finding("info", msg)]
