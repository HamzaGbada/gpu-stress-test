"""Tests for the lite pipeline's device handling, kernels and evaluation rules.

None of these need a GPU: they exercise the pure-Python logic that decides what
a measurement means, plus the packaging of the CUDA kernels.
"""
from __future__ import annotations

import argparse
import re
import struct

from gpu_stress.cuda import read_kernel_file
from gpu_stress.evaluate import PCIE_LANE_GBS, theoretical_bandwidth_gbs, theoretical_pcie_gbs
from gpu_stress.lite_steps import (
    EDGE_BITS,
    EDGE_INPUTS,
    KERNEL_NAMES,
    eval_edge_cases,
    eval_memtest,
    eval_nbody,
    eval_stencil,
    eval_system_info,
)
from gpu_stress.monitor import Monitor, summarize
from gpu_stress.nvml import Sample
from gpu_stress.pipeline import Context

RTX4050 = {"sm_count": 20, "cores_per_sm": 128, "max_sm_clock_mhz": 3105, "mem_clock_mhz": 8001,
           "bus_width_bits": 96, "temp_slowdown_c": 92, "power_limit_w": 60, "integrated": False,
           "total_mem_bytes": 6 * 1024 ** 3, "pcie_gen": 4, "pcie_width": 8,
           "pcie_gen_max": 4, "pcie_width_max": 8}

# NVIDIA GB10 (DGX Spark): unified LPDDR5X, so the driver reports no bus width.
GB10 = {"sm_count": 48, "cores_per_sm": 128, "max_sm_clock_mhz": 3003, "mem_clock_mhz": 8533,
        "bus_width_bits": 0, "temp_slowdown_c": 86, "integrated": True,
        "total_mem_bytes": 130659741696}


def _ctx(device: dict) -> Context:
    return Context(config=argparse.Namespace(), monitor=Monitor(interval=0.05), device=dict(device))


def _window(clock: int = 2500, n: int = 20):
    return summarize([Sample(t=i * 0.25, util=99.0, temp=60, power_w=55.0, sm_clock=clock) for i in range(n)])


# ---------------------------------------------------------------------------
# theoretical peaks
# ---------------------------------------------------------------------------
def test_bandwidth_peak_unknown_without_bus_width():
    """GB10 reports bus_width 0; that must be 'unknown', never 0 GB/s."""
    assert abs(theoretical_bandwidth_gbs(RTX4050) - 192) < 0.1
    assert theoretical_bandwidth_gbs(GB10) is None


def test_pcie_peak():
    assert abs(theoretical_pcie_gbs(RTX4050) - 8 * PCIE_LANE_GBS[4]) < 1e-6   # gen4 x8 ~ 15.8 GB/s
    assert theoretical_pcie_gbs(GB10) is None                                  # integrated: no link
    assert theoretical_pcie_gbs({"pcie_gen": 4}) is None                       # width unknown
    assert theoretical_pcie_gbs({"pcie_gen": 99, "pcie_width": 16}) is None    # unknown generation


def test_system_info_explains_unified_memory():
    m = {"theoretical_bandwidth_gbs": None, "theoretical_fp32_tflops": 36.9, "theoretical_pcie_gbs": None}
    msgs = [f.message for f in eval_system_info(m, _window(), _ctx(GB10))]
    assert any("unified-memory" in s for s in msgs)
    assert any("without an efficiency score" in s for s in msgs)


def test_system_info_flags_downtrained_pcie_link():
    dev = dict(RTX4050, pcie_gen=1, pcie_width=4)
    m = {"theoretical_bandwidth_gbs": 192.0, "theoretical_fp32_tflops": 15.9, "theoretical_pcie_gbs": 1.0}
    findings = eval_system_info(m, _window(), _ctx(dev))
    assert any(f.level == "warn" and "gen1 x4" in f.message for f in findings)


# ---------------------------------------------------------------------------
# memtest
# ---------------------------------------------------------------------------
def _memtest_metrics(**over):
    m = {"errors": 0, "tested_gb": 5.5, "passes": 6, "write_gbs": 170.0, "read_gbs": 180.0,
         "alloc_s": 0.2, "truncated": False}
    m.update(over)
    return m


def test_memtest_reports_throughput_and_flags_slow_allocation():
    ctx = _ctx(RTX4050)
    msgs = [f.message for f in eval_memtest(_memtest_metrics(), _window(), ctx)]
    assert any("170" in s and "GB/s" in s for s in msgs)

    findings = eval_memtest(_memtest_metrics(alloc_s=45.0), _window(), ctx)
    assert any(f.level == "warn" and "first-touch" in f.message for f in findings)


def test_memtest_fails_on_any_mismatch():
    findings = eval_memtest(_memtest_metrics(errors=3), _window(), _ctx(RTX4050))
    assert any(f.level == "fail" for f in findings)


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def _nbody_metrics(**over):
    m = {"finite": True, "momentum_rel": 1e-7, "momentum_drift": 1e-9, "momentum_scale": 1e-2,
         "steps": 2000, "bodies": 20480, "tflops_mean": 9.0, "tflops_first": 9.0, "tflops_last": 9.0}
    m.update(over)
    return m


def test_nbody_momentum_conservation_levels():
    ctx = _ctx(RTX4050)
    def level(rel):
        f = [x for x in eval_nbody(_nbody_metrics(momentum_rel=rel), _window(), ctx)
             if "momentum conservation" in x.message]
        assert len(f) == 1
        return f[0].level

    assert level(1e-7) == "info"
    assert level(3e-4) == "info"      # measured healthy drift on real hardware
    assert level(5e-2) == "warn"
    assert level(0.5) == "fail"


def test_nbody_fails_on_non_finite_state():
    findings = eval_nbody(_nbody_metrics(finite=False), _window(), _ctx(RTX4050))
    assert any(f.level == "fail" and "NaN" in f.message for f in findings)


def _stencil_metrics(**over):
    m = {"finite": True, "bounds_violated": False, "sum_drift_rel": 1e-8, "steps": 5000,
         "min_before": -3.0, "max_before": 3.0, "min_after": -2.9, "max_after": 2.9,
         "gbs_mean": 160.0, "gbs_first": 160.0, "gbs_last": 160.0}
    m.update(over)
    return m


def test_stencil_maximum_principle_violation_is_a_hardware_fault():
    findings = eval_stencil(_stencil_metrics(bounds_violated=True, max_after=7.5), _window(), _ctx(RTX4050))
    assert any(f.level == "fail" and "maximum principle" in f.message for f in findings)

    ok = eval_stencil(_stencil_metrics(), _window(), _ctx(RTX4050))
    assert any(f.level == "info" and "maximum principle holds" in f.message for f in ok)


def test_stencil_heat_conservation_levels():
    ctx = _ctx(RTX4050)
    def level(drift):
        f = [x for x in eval_stencil(_stencil_metrics(sum_drift_rel=drift), _window(), ctx)
             if "heat conservation" in x.message]
        assert len(f) == 1
        return f[0].level

    assert level(1e-8) == "info"
    assert level(1e-4) == "warn"
    assert level(0.2) == "fail"


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def _edge_metrics(mask=0, atomics_ok=True, **over):
    atomics = {"atomicAdd": {"got": 256, "expected": 256}, "atomicMax": {"got": 255, "expected": 255},
               "atomicCAS": {"got": 256, "expected": 256}, "atomicXor": {"got": 0, "expected": 0}}
    if not atomics_ok:
        atomics["atomicAdd"] = {"got": 251, "expected": 256}
    m = {"mask": mask, "threads": 256, "atomics": atomics, "atomics_ok": atomics_ok,
         "failed_checks": [], "errors": 0}
    m.update(over)
    return m


def test_edge_case_bits_decode_to_findings():
    ctx = _ctx(RTX4050)
    assert all(f.level == "info" for f in eval_edge_cases(_edge_metrics(), _window(), ctx))

    nan_bit = next(bit for bit, name, *_ in EDGE_BITS if name == "nan_compare")
    findings = eval_edge_cases(_edge_metrics(mask=nan_bit), _window(), ctx)
    assert any(f.level == "fail" and "NaN" in f.message for f in findings)

    den_bit = next(bit for bit, name, *_ in EDGE_BITS if name == "denormal_flushed")
    findings = eval_edge_cases(_edge_metrics(mask=den_bit), _window(), ctx)
    assert [f.level for f in findings] == ["warn"]        # flushing denormals is odd, not fatal


def test_lost_atomic_updates_fail():
    findings = eval_edge_cases(_edge_metrics(atomics_ok=False), _window(), _ctx(RTX4050))
    assert any(f.level == "fail" and "atomic" in f.message for f in findings)


def test_edge_case_inputs_have_the_exact_bit_patterns_the_kernel_expects():
    bits = [struct.unpack("I", struct.pack("f", v))[0] for v in EDGE_INPUTS]
    assert bits[0] == 0x00000001          # smallest denormal
    assert bits[1] == 0x00000000          # +0
    assert bits[2] == 0x7F800000          # +inf
    assert bits[3] & 0x7FFFFFFF > 0x7F800000   # NaN
    assert bits[6] == 0x3F800001          # 1 + 2^-23
    assert bits[7] == 0x3F7FFFFF          # 1 - 2^-24
    # the kernel asserts these two identities; check the maths holds in fp64 too
    assert struct.unpack("f", struct.pack("f", EDGE_INPUTS[6] * EDGE_INPUTS[7]))[0] == 1.0
    assert struct.unpack("I", struct.pack("f", EDGE_INPUTS[4] + EDGE_INPUTS[5]))[0] == 0x3E99999A


# ---------------------------------------------------------------------------
# packaged kernels
# ---------------------------------------------------------------------------
def test_ptx_ships_with_the_package_and_is_loadable_as_a_resource():
    for name in ("kernels_sm70.ptx", "kernels_sm80.ptx"):
        ptx = read_kernel_file(name)
        assert ptx.startswith("//")
        assert ".version" in ptx
    assert ".entry mma_burn" not in read_kernel_file("kernels_sm70.ptx")
    assert ".entry mma_burn" in read_kernel_file("kernels_sm80.ptx")


def test_every_kernel_the_pipeline_loads_exists_in_the_cuda_source_and_ptx():
    src = read_kernel_file("kernels.cu")
    declared = set(re.findall(r'extern "C" __global__ void (\w+)\(', src))
    assert set(KERNEL_NAMES) <= declared, f"missing from kernels.cu: {set(KERNEL_NAMES) - declared}"
    assert "mma_burn" in declared

    ptx80 = read_kernel_file("kernels_sm80.ptx")
    ptx70 = read_kernel_file("kernels_sm70.ptx")
    for name in KERNEL_NAMES:
        assert f".entry {name}" in ptx80, f"{name} missing from kernels_sm80.ptx - rebuild with nvrtc.py"
        assert f".entry {name}" in ptx70, f"{name} missing from kernels_sm70.ptx - rebuild with nvrtc.py"
