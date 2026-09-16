"""Stress / benchmark steps for the zero-dependency (driver API) pipeline."""
from __future__ import annotations

import os
import random
import time

from .cuda import driver as cu
from .cuda.driver import CudaOutOfMemory, u32, u64
from .evaluate import (
    Finding,
    compute_error_finding,
    efficiency_finding,
    stability_finding,
    theoretical_bandwidth_gbs,
    theoretical_fp32_tflops,
)
from .pipeline import Context, Step, StepSkipped

MB = 1024 * 1024
GB = 1024 ** 3
BLOCK = 256


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ctx(ctx: Context) -> cu.Context:
    return ctx.extra["cuda"]


def _kernel(ctx: Context, name: str) -> cu.Kernel:
    return ctx.extra["kernels"][name]


def _budget(ctx: Context, fraction: float, cap: int | None = None, headroom: int = 128 * MB) -> int:
    """Bytes we may allocate: fraction of *free* memory minus headroom."""
    free, _ = _ctx(ctx).mem_info()
    n = int(free * fraction) - headroom
    if cap:
        n = min(n, cap)
    return max(n, 0)


def _alloc_largest(ctx: Context, want: int, minimum: int = 64 * MB) -> cu.DeviceBuffer:
    """cuMemAlloc `want` bytes, halving on OOM (other processes may hold VRAM)."""
    size = want
    while size >= minimum:
        try:
            return _ctx(ctx).alloc(size)
        except CudaOutOfMemory:
            size //= 2
    raise CudaOutOfMemory(2, "CUDA_ERROR_OUT_OF_MEMORY", f"could not allocate even {minimum // MB} MiB", "cuMemAlloc")


def _timed(ctx: Context, fn) -> float:
    """Run fn() between two events, return milliseconds."""
    c = _ctx(ctx)
    s = c.event().record()
    fn()
    e = c.event().record()
    e.synchronize()
    return e.elapsed_ms(s)


def _grid_for(ctx: Context, n_threads: int, per_sm: int = 8) -> int:
    """Grid-stride grid: enough blocks to fill the GPU, not one per element."""
    sm = _ctx(ctx).info.sm_count
    return max(1, min((n_threads + BLOCK - 1) // BLOCK, sm * per_sm))


# ---------------------------------------------------------------------------
# 0. system info
# ---------------------------------------------------------------------------
def step_system_info(ctx: Context) -> dict:
    c = _ctx(ctx)
    free, total = c.mem_info()
    d = ctx.device
    for k, v in d.items():
        ctx.log(f"  {k:22s}: {v}")
    ctx.log(f"  {'free_mem_bytes':22s}: {free} ({free / GB:.2f} GiB of {total / GB:.2f})")
    ctx.log(f"  {'theoretical_fp32':22s}: {theoretical_fp32_tflops(d) or 0:.1f} TFLOPS")
    ctx.log(f"  {'theoretical_bandwidth':22s}: {theoretical_bandwidth_gbs(d) or 0:.0f} GB/s")
    return {"free_mem_bytes": free, "total_mem_bytes": total,
            "theoretical_fp32_tflops": theoretical_fp32_tflops(d),
            "theoretical_bandwidth_gbs": theoretical_bandwidth_gbs(d)}


# ---------------------------------------------------------------------------
# 1. VRAM integrity (memtest)
# ---------------------------------------------------------------------------
PATTERNS = [0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555, 0x0F0F0F0F, 0xF0F0F0F0]


def step_memtest(ctx: Context) -> dict:
    c = _ctx(ctx)
    want = _budget(ctx, ctx.config.vram)
    buf = _alloc_largest(ctx, want)
    n_words = buf.nbytes // 4
    fill, check = _kernel(ctx, "mem_fill"), _kernel(ctx, "mem_check")
    err = c.alloc(4)
    grid = _grid_for(ctx, n_words, per_sm=16)
    patterns = PATTERNS + [random.getrandbits(32) for _ in range(ctx.config.mem_passes - len(PATTERNS))]
    patterns = patterns[: max(1, ctx.config.mem_passes)]
    ctx.log(f"  testing {buf.nbytes / GB:.2f} GiB with {len(patterns)} patterns")
    total_err = 0
    w_ms = r_ms = 0.0
    per_pattern = {}
    try:
        for p in patterns:
            err.memset32(0)
            w_ms += _timed(ctx, lambda p=p: fill.launch(grid, BLOCK, buf, u64(n_words), u32(p)))
            r_ms += _timed(ctx, lambda p=p: check.launch(grid, BLOCK, buf, u64(n_words), u32(p), err))
            e = err.download_u32(1)[0]
            per_pattern[f"0x{p:08X}"] = e
            total_err += e
            ctx.log(f"  pattern 0x{p:08X}: {e} errors")
    finally:
        err.free()
        buf.free()
    gb = buf.nbytes / 1e9
    return {"tested_gb": round(buf.nbytes / GB, 3), "passes": len(patterns), "errors": total_err,
            "errors_per_pattern": per_pattern,
            "write_gbs": round(gb * len(patterns) / (w_ms / 1000), 1),
            "read_gbs": round(gb * len(patterns) / (r_ms / 1000), 1)}


def eval_memtest(m: dict, t, ctx: Context) -> list[Finding]:
    out = compute_error_finding(m["errors"], f"VRAM integrity over {m['tested_gb']} GiB x {m['passes']} passes")
    total = ctx.device.get("total_mem_bytes") or 0
    if total and m["tested_gb"] * GB < 0.5 * total:
        out.append(Finding("warn", f"only {m['tested_gb']} GiB of {total / GB:.1f} GiB could be tested (VRAM in use elsewhere?)"))
    return out


# ---------------------------------------------------------------------------
# 2. memory bandwidth
# ---------------------------------------------------------------------------
def step_bandwidth(ctx: Context) -> dict:
    c = _ctx(ctx)
    size = _budget(ctx, 0.4, cap=1 * GB)
    size -= size % 16
    src = _alloc_largest(ctx, size)
    dst = _alloc_largest(ctx, src.nbytes)
    size = min(src.nbytes, dst.nbytes) // 16 * 16
    try:
        src.memset32(0x3F800000)
        n4 = size // 16
        copy = _kernel(ctx, "copy_f4")
        grid = _grid_for(ctx, n4, per_sm=16)
        reps = ctx.config.reps
        # warm-up
        copy.launch(grid, BLOCK, src, dst, u64(n4))
        c.synchronize()
        k_ms = _timed(ctx, lambda: [copy.launch(grid, BLOCK, src, dst, u64(n4)) for _ in range(reps)]) / reps
        d_ms = _timed(ctx, lambda: [dst.copy_from(src, size) for _ in range(reps)]) / reps
        # bytes moved = read + write
        gbs_kernel = 2 * size / (k_ms / 1000) / 1e9
        gbs_dtod = 2 * size / (d_ms / 1000) / 1e9
    finally:
        src.free()
        dst.free()
    ctx.log(f"  buffer {size / MB:.0f} MiB: copy kernel {gbs_kernel:.1f} GB/s, cuMemcpyDtoD {gbs_dtod:.1f} GB/s")
    return {"buffer_mb": round(size / MB, 1), "gbs_copy_kernel": round(gbs_kernel, 1),
            "gbs_dtod": round(gbs_dtod, 1), "gbs": round(max(gbs_kernel, gbs_dtod), 1)}


def eval_bandwidth(m: dict, t, ctx: Context) -> list[Finding]:
    return efficiency_finding(m["gbs"], theoretical_bandwidth_gbs(ctx.device), "GB/s", "device memory bandwidth",
                              warn_below=0.6, fail_below=0.3)


# ---------------------------------------------------------------------------
# 3. verified SGEMM
# ---------------------------------------------------------------------------
def step_matmul(ctx: Context) -> dict:
    c = _ctx(ctx)
    n = ctx.config.matmul_size
    while 3 * n * n * 4 > _budget(ctx, 0.5) and n > 512:
        n //= 2
    a, b, cbuf = c.alloc(n * n * 4), c.alloc(n * n * 4), c.alloc(n * n * 4)
    try:
        fill = _kernel(ctx, "fill_pattern_f32")
        gemm = _kernel(ctx, "sgemm_tiled")
        # small integers in [-3,3] and [-2,2] -> exact fp32 sums for K <= 2^24/6
        fill.launch(_grid_for(ctx, n * n), BLOCK, a, u64(n * n), u32(7919), u32(13), u32(7))
        fill.launch(_grid_for(ctx, n * n), BLOCK, b, u64(n * n), u32(104729), u32(5), u32(5))
        grid = ((n + 63) // 64, (n + 63) // 64)
        gemm.launch(grid, 256, a, b, cbuf, n, n, n)
        c.synchronize()
        reps = max(1, ctx.config.reps // 2)
        ms = _timed(ctx, lambda: [gemm.launch(grid, 256, a, b, cbuf, n, n, n) for _ in range(reps)]) / reps
        gflops = 2 * n ** 3 / (ms / 1000) / 1e9

        # verify a random sample of C against a host computation
        def val(idx, mul, add, rng):
            return ((idx * mul + add) & 0xFFFFFFFF) % rng - rng // 2

        rnd = random.Random(1234)
        errors = 0
        checked = 0
        rows = sorted(set(rnd.randrange(n) for _ in range(ctx.config.verify_samples)))
        for i in rows:
            row_c = cbuf.download_f32(n, i * n)
            arow = [val(i * n + k, 7919, 13, 7) for k in range(n)]
            for j in rnd.sample(range(n), 4):
                expected = sum(arow[k] * val(k * n + j, 104729, 5, 5) for k in range(n))
                checked += 1
                if row_c[j] != expected:
                    errors += 1
    finally:
        a.free()
        b.free()
        cbuf.free()
    ctx.log(f"  {n}x{n} SGEMM (register-tiled, no tensor cores): {gflops / 1000:.2f} TFLOPS, {errors}/{checked} mismatches")
    return {"n": n, "ms": round(ms, 3), "gflops": round(gflops, 1), "tflops": round(gflops / 1000, 3),
            "verified": checked, "errors": errors}


def eval_matmul(m: dict, t, ctx: Context) -> list[Finding]:
    out = compute_error_finding(m["errors"], f"{m['verified']} sampled SGEMM outputs")
    peak = theoretical_fp32_tflops(ctx.device, t.sm_clock_mean)
    # a hand-written register-tiled kernel reaches ~25-50% of peak; warn only if far below
    out += efficiency_finding(m["tflops"], peak, "TFLOPS", "SGEMM at observed clock", warn_below=0.12, fail_below=0.05)
    return out


# ---------------------------------------------------------------------------
# 4/5/6. sustained burns (FP32 / FP16 / tensor cores)
# ---------------------------------------------------------------------------
def _burn(ctx: Context, kernel_name: str, flops_per_thread_iter: float, verify, label: str,
          extra_args=(), max_iters: int = 1 << 30, threads_per_sm: int = 2048) -> dict:
    """Run `kernel` back-to-back for config.burn seconds.

    Each launch is auto-sized to ~100 ms so telemetry is sampled many times,
    then per-launch throughput is recorded to spot degradation over time.
    """
    c = _ctx(ctx)
    k = _kernel(ctx, kernel_name)
    n_threads = c.info.sm_count * threads_per_sm
    grid = n_threads // BLOCK
    out = c.alloc(n_threads * 4)
    duration = ctx.config.burn
    try:
        # calibrate iterations for ~100 ms launches
        iters = 2000
        ms = _timed(ctx, lambda: k.launch(grid, BLOCK, out, iters, *extra_args))
        iters = int(min(max(iters * 100.0 / max(ms, 0.01), 100), max_iters))
        ctx.log(f"  {label}: {grid} blocks x {BLOCK} threads, {iters} iters/launch, {duration:.0f}s")
        samples: list[float] = []
        errors = 0
        launches = 0
        t_end = time.time() + duration
        next_log = time.time() + 5
        while time.time() < t_end:
            ms = _timed(ctx, lambda: k.launch(grid, BLOCK, out, iters, *extra_args))
            samples.append(flops_per_thread_iter * iters * n_threads / (ms / 1000) / 1e12)
            launches += 1
            if launches % 10 == 1:
                errors += verify(out.download_f32(), iters)
            if time.time() >= next_log:
                s = ctx.monitor.latest()
                tele = f" | {s.temp}°C {s.power_w:.0f}W {s.sm_clock}MHz" if s and s.temp is not None else ""
                ctx.log(f"    {t_end - time.time():5.0f}s left  {samples[-1]:6.2f} TFLOPS{tele}")
                next_log += 5
        errors += verify(out.download_f32(), iters)
    finally:
        out.free()
    k5 = max(1, len(samples) // 5)
    first, last = sum(samples[:k5]) / k5, sum(samples[-k5:]) / k5
    return {"seconds": duration, "launches": launches, "iters_per_launch": iters, "threads": n_threads,
            "tflops_mean": round(sum(samples) / len(samples), 3), "tflops_min": round(min(samples), 3),
            "tflops_max": round(max(samples), 3), "tflops_first": round(first, 3), "tflops_last": round(last, 3),
            "tflops": round(sum(samples) / len(samples), 3), "errors": errors}


def _peak_findings(measured: float, ratio_to_fp32: float, t, ctx: Context, what: str) -> list[Finding]:
    """Efficiency vs the peak at the *observed* clock (isolates real problems from
    power capping) plus an informational line against the rated boost clock."""
    rated = theoretical_fp32_tflops(ctx.device)
    observed = theoretical_fp32_tflops(ctx.device, t.sm_clock_mean)
    out = efficiency_finding(measured, observed * ratio_to_fp32 if observed else None, "TFLOPS",
                             f"{what} at observed {t.sm_clock_mean or 0:.0f} MHz", warn_below=0.55, fail_below=0.3)
    if rated and observed and abs(rated - observed) / rated > 0.05:
        out.append(Finding("info", f"{what}: {100 * measured / (rated * ratio_to_fp32):.0f}% of rated boost-clock peak "
                                   f"({rated * ratio_to_fp32:.1f} TFLOPS)"))
    return out


def _verify_converged(values: list[float], _iters: int, tol: float = 2e-3) -> int:
    return sum(1 for v in values if not (abs(v - 1.0) < tol))


def step_fp32_burn(ctx: Context) -> dict:
    return _burn(ctx, "fma_burn", 2 * 8, _verify_converged, "FP32 FMA burn", extra_args=(0.5,))


def eval_fp32_burn(m: dict, t, ctx: Context) -> list[Finding]:
    out = compute_error_finding(m["errors"], f"FP32 FMA chains ({m['threads']} threads x {m['launches']} launches)")
    out += _peak_findings(m["tflops_mean"], 1.0, t, ctx, "sustained FP32")
    out += stability_finding(m["tflops_first"], m["tflops_last"], "TFLOPS", "FP32 throughput")
    return out


def step_fp16_burn(ctx: Context) -> dict:
    # fp16 converges in ~40 iterations; tolerance is loose because fp16 has ~3 decimal digits
    return _burn(ctx, "hfma2_burn", 2 * 2 * 8,
                 lambda v, i: sum(1 for x in v if not (abs(x - 1.0) < 2e-2)), "FP16x2 FMA burn", extra_args=(0.5,))


def eval_fp16_burn(m: dict, t, ctx: Context) -> list[Finding]:
    out = compute_error_finding(m["errors"], "FP16 FMA chains")
    # packed fp16 (non tensor-core) runs at the FP32 rate on Ampere/Ada/Blackwell consumer parts
    out += _peak_findings(m["tflops_mean"], 1.0, t, ctx, "sustained FP16x2")
    out += stability_finding(m["tflops_first"], m["tflops_last"], "TFLOPS", "FP16 throughput")
    return out


def step_tensor_burn(ctx: Context) -> dict:
    if "mma_burn" not in ctx.extra["kernels"]:
        raise StepSkipped("tensor-core mma.sync needs compute capability 8.0+")
    # 4 chains x 4 accumulators x (K=16 per mma) = 256 per iteration -> exact while 256*iters < 2^24
    return _burn(ctx, "mma_burn", 16384 / 32,   # flops per warp-iter spread over 32 threads
                 lambda v, i: sum(1 for x in v if x != 256.0 * i), "tensor-core MMA burn",
                 max_iters=60000, threads_per_sm=1024)


def eval_tensor_burn(m: dict, t, ctx: Context) -> list[Finding]:
    out = compute_error_finding(m["errors"], f"tensor-core accumulators ({m['launches']} launches)")
    # GeForce parts run FP16 MMA with FP32 accumulate at 2x the FP32 rate (RTX 40/50: 165/210 vs 83/105 TFLOPS)
    out += _peak_findings(m["tflops_mean"], 2.0, t, ctx, "sustained FP16 tensor-core (fp32 acc)")
    out += stability_finding(m["tflops_first"], m["tflops_last"], "TFLOPS", "tensor-core throughput")
    return out


# ---------------------------------------------------------------------------
# 7. allocation churn (fragmentation)
# ---------------------------------------------------------------------------
def step_alloc_churn(ctx: Context) -> dict:
    c = _ctx(ctx)
    sizes_mb = [16, 64, 256, 1024]
    res = {}
    for mb in sizes_mb:
        if mb * MB > _budget(ctx, 0.8):
            break
        t_alloc = t_free = 0.0
        n = 20
        for _ in range(n):
            t0 = time.perf_counter()
            b = c.alloc(mb * MB)
            c.synchronize()
            t1 = time.perf_counter()
            b.free()
            c.synchronize()
            t_free += time.perf_counter() - t1
            t_alloc += t1 - t0
        res[f"alloc_{mb}mb_ms"] = round(t_alloc / n * 1000, 3)
        res[f"free_{mb}mb_ms"] = round(t_free / n * 1000, 3)
        ctx.log(f"  {mb:5d} MiB: alloc {res[f'alloc_{mb}mb_ms']:.3f} ms, free {res[f'free_{mb}mb_ms']:.3f} ms")
    # many small live allocations then release - checks the driver copes with churn
    live = []
    try:
        for _ in range(256):
            live.append(c.alloc(4 * MB))
    except CudaOutOfMemory:
        pass
    res["small_live_allocs"] = len(live)
    for b in live:
        b.free()
    free_after, _ = c.mem_info()
    res["free_after_bytes"] = free_after
    return res


def eval_alloc_churn(m: dict, t, ctx: Context) -> list[Finding]:
    slow = [k for k, v in m.items() if k.startswith("alloc_") and isinstance(v, float) and v > 50]
    if slow:
        return [Finding("warn", f"slow allocations (>50 ms): {', '.join(slow)}")]
    return [Finding("info", f"allocation churn OK ({m['small_live_allocs']} x 4 MiB live blocks)")]


# ---------------------------------------------------------------------------
# step list
# ---------------------------------------------------------------------------
def build_steps() -> list[Step]:
    return [
        Step("system_info", step_system_info, "device, driver and theoretical peaks"),
        Step("memtest", step_memtest, "write/read patterns over the free VRAM", evaluate=eval_memtest),
        Step("bandwidth", step_bandwidth, "device-to-device copy bandwidth", evaluate=eval_bandwidth),
        Step("matmul", step_matmul, "verified tiled SGEMM", evaluate=eval_matmul),
        Step("fp32_burn", step_fp32_burn, "sustained FP32 FMA load", heavy=True, evaluate=eval_fp32_burn),
        Step("fp16_burn", step_fp16_burn, "sustained packed-FP16 load", heavy=True, evaluate=eval_fp16_burn),
        Step("tensor_burn", step_tensor_burn, "sustained tensor-core (mma.sync) load", heavy=True, evaluate=eval_tensor_burn),
        Step("alloc_churn", step_alloc_churn, "allocation / free latency and churn", evaluate=eval_alloc_churn),
    ]


def load_kernels(ctx: Context, cctx: cu.Context, prefer_nvrtc: bool) -> dict[str, cu.Kernel]:
    """Load the kernels: NVRTC for the exact arch if asked & available, else shipped PTX."""
    info = cctx.info
    if info.cc_major < 7:
        raise RuntimeError(f"the shipped PTX targets sm_70+ (Volta and newer); this GPU is sm_{info.cc_major}{info.cc_minor}. "
                           "Rebuild gpu_stress/cuda/kernels.cu for an older arch with --nvrtc / nvrtc.py")
    src_dir = cu.ptx_dir()
    ptx = None
    if prefer_nvrtc:
        try:
            from .cuda.nvrtc import compile_to_ptx, find_nvrtc
            lib = find_nvrtc()
            if lib is not None:
                ptx = compile_to_ptx(open(os.path.join(src_dir, "kernels.cu")).read(),
                                     f"compute_{info.cc_major}{info.cc_minor}", lib=lib)
                ctx.log(f"  kernels: compiled with NVRTC for compute_{info.cc_major}{info.cc_minor}")
        except Exception as e:  # noqa: BLE001
            ctx.log(f"  NVRTC unavailable ({e}); using shipped PTX")
    if ptx is None:
        fname = "kernels_sm80.ptx" if info.cc_major >= 8 else "kernels_sm70.ptx"
        ptx = open(os.path.join(src_dir, fname)).read()
        ctx.log(f"  kernels: {fname} (driver JIT for sm_{info.cc_major}{info.cc_minor})")
    mod = cctx.load_ptx(ptx)
    names = ["fma_burn", "hfma2_burn", "sgemm_tiled", "fill_pattern_f32", "mem_fill", "mem_check", "copy_f4"]
    if info.cc_major >= 8:
        names.append("mma_burn")
    return {n: mod.function(n) for n in names}
