"""GPU-independent tests: evaluation rules, telemetry summaries, runner isolation."""
import argparse
import functools

import pytest

from gpu_stress.evaluate import (
    Finding,
    efficiency_finding,
    stability_finding,
    telemetry_findings,
    theoretical_bandwidth_gbs,
    theoretical_fp32_tflops,
    worst,
)
from gpu_stress.lite_pipeline import main as lite_main
from gpu_stress.monitor import Monitor, summarize
from gpu_stress.nvml import Sample
from gpu_stress.pipeline import Context, Pipeline, Step, StepResult, StepSkipped, overall_status, select_steps
from gpu_stress.report import key_metrics

RTX4050 = {"sm_count": 20, "cores_per_sm": 128, "max_sm_clock_mhz": 3105, "mem_clock_mhz": 8001,
           "bus_width_bits": 96, "temp_slowdown_c": 92, "power_limit_w": 60}


def test_theoretical_peaks():
    assert abs(theoretical_fp32_tflops(RTX4050) - 15.9) < 0.1
    assert abs(theoretical_fp32_tflops(RTX4050, 2565) - 13.13) < 0.05
    assert abs(theoretical_bandwidth_gbs(RTX4050) - 192) < 0.1
    assert theoretical_fp32_tflops({}) is None


def test_efficiency_and_stability_levels():
    assert efficiency_finding(10, 15.9, "TFLOPS", "x", warn_below=0.5)[0].level == "info"
    assert efficiency_finding(6, 15.9, "TFLOPS", "x", warn_below=0.5)[0].level == "warn"
    assert efficiency_finding(2, 15.9, "TFLOPS", "x", fail_below=0.3)[0].level == "fail"
    assert stability_finding(10, 9.5, "T", "x")[0].level == "info"
    assert stability_finding(10, 8.5, "T", "x")[0].level == "warn"
    assert stability_finding(10, 7.0, "T", "x")[0].level == "fail"
    assert worst([Finding("info", ""), Finding("warn", "")]) == "warn"
    assert worst([]) == "pass"


def _samples(n=40, temp=60, throttle=(), util=99.0, clock=2500):
    return [Sample(t=i * 0.25, util=util, temp=temp, power_w=55.0, sm_clock=clock, throttle=list(throttle))
            for i in range(n)]


def test_thermal_rules():
    findings = telemetry_findings(summarize(_samples(temp=91)), RTX4050)
    assert any(f.level == "fail" and "slowdown threshold" in f.message for f in findings)
    findings = telemetry_findings(summarize(_samples(temp=85)), RTX4050)
    assert any(f.level == "warn" for f in findings)
    findings = telemetry_findings(summarize(_samples(temp=60)), RTX4050)
    assert all(f.level == "info" for f in findings)


def test_throttle_only_counted_under_load():
    idle = _samples(util=0.0, throttle=("sw_thermal_slowdown", "sw_power_cap"))
    assert summarize(idle).hard_throttle_pct == 0
    loaded = _samples(throttle=("hw_thermal_slowdown",))
    assert summarize(loaded).hard_throttle_pct == 100


def test_thermal_slowdown_is_believed_only_when_the_temperature_agrees():
    # 85 C against a 92 C threshold: plausible, so it counts as a real fault.
    hot = summarize(_samples(temp=85, throttle=("hw_thermal_slowdown",)))
    assert any(f.level == "fail" and "slowdown active" in f.message
               for f in telemetry_findings(hot, RTX4050))

    # 60 C against the same threshold: the flag contradicts the thermometer.
    cool = summarize(_samples(temp=60, throttle=("hw_thermal_slowdown",)))
    findings = telemetry_findings(cool, RTX4050)
    assert any(f.level == "info" and "spurious" in f.message for f in findings)
    assert not any(f.level == "fail" for f in findings)

    # A non-thermal hard reason is always believed - there is no thermometer to check it against.
    brake = summarize(_samples(temp=60, throttle=("hw_power_brake_slowdown",)))
    assert any(f.level == "fail" for f in telemetry_findings(brake, RTX4050))


def test_clock_dip_rule():
    samples = _samples()
    for smp in samples[20:]:
        smp.sm_clock = 900
    findings = telemetry_findings(summarize(samples), RTX4050)
    assert any("dipped" in f.message for f in findings)


class _OOM(Exception):
    pass


def _ctx():
    mon = Monitor(interval=0.05)   # not started: no thread, works without a GPU
    return Context(config=argparse.Namespace(), monitor=mon, device=dict(RTX4050), oom_types=(_OOM,))


def test_runner_isolates_failures_and_rolls_up_status():
    steps = [
        Step("ok", lambda c: {"v": 1}, evaluate=lambda m, t, c: [Finding("info", "fine")]),
        Step("oom", lambda c: (_ for _ in ()).throw(_OOM("boom"))),
        Step("crash", lambda c: 1 / 0),
        Step("skip", lambda c: (_ for _ in ()).throw(StepSkipped("no hw"))),
        Step("warned", lambda c: {}, evaluate=lambda m, t, c: [Finding("warn", "meh")]),
    ]
    ctx = _ctx()
    results = Pipeline(steps, ctx).run()
    assert [r.status for r in results] == ["pass", "error", "error", "skipped", "warn"]
    assert results[1].findings[0].level == "fail" and "out of memory" in results[1].error
    assert overall_status(results) == "error"
    assert overall_status(results[:1] + results[3:]) == "warn"
    assert overall_status([results[0], results[3]]) == "pass"


def test_runner_step_selection():
    ran = []
    steps = [Step(n, functools.partial(lambda c, n: ran.append(n) or {}, n=n)) for n in ("a", "b", "c")]
    Pipeline(steps, _ctx()).run(only={"a", "c"}, skip={"c"})
    assert ran == ["a"]


def test_step_lists_reject_unknown_names():
    steps = [Step(n, lambda c: {}) for n in ("memtest", "matmul")]
    assert select_steps(steps, "memtest, matmul", "") == ({"memtest", "matmul"}, set())
    assert select_steps(steps, "", "") == (set(), set())
    with pytest.raises(ValueError, match="memtst"):
        select_steps(steps, "memtst", "")
    with pytest.raises(ValueError, match="nope"):
        select_steps(steps, "", "nope")


def test_fail_outranks_error_regardless_of_order():
    ctx = _ctx()
    fail = Step("bad", lambda c: {}, evaluate=lambda m, t, c: [Finding("fail", "wrong result")])
    crash = Step("crash", lambda c: 1 / 0)
    assert overall_status(Pipeline([fail, crash], ctx).run()) == "fail"
    assert overall_status(Pipeline([crash, fail], _ctx()).run()) == "fail"


def test_cli_rejects_a_typo_in_steps_before_touching_the_gpu(capsys):
    assert lite_main(["--steps", "memtst"]) == 2
    assert "unknown step(s): memtst" in capsys.readouterr().err


def test_summary_shows_each_number_once():
    r = StepResult("fp32_burn", "pass", 1.0, {"tflops": 11.0, "tflops_mean": 11.0, "errors": 0})
    assert key_metrics(r) == "tflops=11.000, errors=0"
    r = StepResult("bandwidth", "pass", 1.0, {"gbs": 173.0, "gbs_copy_kernel": 173.0})
    assert key_metrics(r) == "gbs=173.000"
