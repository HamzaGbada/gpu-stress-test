"""Stress / benchmark steps for the zero-dependency (driver API) pipeline."""
from __future__ import annotations

import math
import random
import struct
import time

from .cuda import driver as cu
from .cuda.driver import CudaOutOfMemory, f32, i32, u32, u64
from .evaluate import (
    Finding,
    compute_error_finding,
    efficiency_finding,
    stability_finding,
    theoretical_bandwidth_gbs,
    theoretical_fp32_tflops,
    theoretical_pcie_gbs,
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


def _reduce_stats(ctx: Context, buf: cu.DeviceBuffer, n: int) -> tuple[float, float, float]:
    """(sum, min, max) of an fp32 device array, accumulated in fp64."""
    k = _kernel(ctx, "reduce_stats_f32")
    blocks = min(k.saturating_grid(BLOCK, n), 1024)
    part = _ctx(ctx).alloc(blocks * 3 * 8)
    try:
        k.launch(blocks, BLOCK, buf, u64(n), part)
        v = part.download_f64(blocks * 3)
    finally:
        part.free()
    return sum(v[0::3]), min(v[1::3]), max(v[2::3])


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


# ---------------------------------------------------------------------------
# 0. system info
# ---------------------------------------------------------------------------
def step_system_info(ctx: Context) -> dict:
    c = _ctx(ctx)
    free, total = c.mem_info()
    d = ctx.device
    for k, v in d.items():
        ctx.log(f"  {k:22s}: {v}")
    fp32 = theoretical_fp32_tflops(d)
    bw = theoretical_bandwidth_gbs(d)
    pcie = theoretical_pcie_gbs(d)
    ctx.log(f"  {'free_mem_bytes':22s}: {free} ({free / GB:.2f} GiB of {total / GB:.2f})")
    ctx.log(f"  {'theoretical_fp32':22s}: {fp32:.1f} TFLOPS" if fp32 else "  theoretical_fp32      : unknown")
    if bw:
        ctx.log(f"  {'theoretical_bandwidth':22s}: {bw:.0f} GB/s")
    else:
        why = "unified/integrated memory" if d.get("integrated") else "bus width not reported by the driver"
        ctx.log(f"  {'theoretical_bandwidth':22s}: unknown ({why}) - bandwidth will be reported, not scored")
    ctx.log(f"  {'theoretical_pcie':22s}: {pcie:.1f} GB/s" if pcie else "  theoretical_pcie      : n/a")
    return {"free_mem_bytes": free, "total_mem_bytes": total,
            "theoretical_fp32_tflops": fp32,
            "theoretical_bandwidth_gbs": bw,
            "theoretical_pcie_gbs": pcie}


def eval_system_info(m: dict, t, ctx: Context) -> list[Finding]:
    out = []
    d = ctx.device
    if d.get("integrated"):
        out.append(Finding("info", "integrated / unified-memory GPU: VRAM is shared with the OS, "
                                   "so free memory and bandwidth depend on system load"))
    if not m["theoretical_bandwidth_gbs"]:
        out.append(Finding("info", "no memory-bandwidth peak available for this part - the bandwidth step "
                                   "reports measured GB/s without an efficiency score"))
    gen, width = d.get("pcie_gen"), d.get("pcie_width")
    max_gen, max_width = d.get("pcie_gen_max"), d.get("pcie_width_max")
    if gen and max_gen and (gen < max_gen or (width and max_width and width < max_width)):
        out.append(Finding("warn", f"PCIe link is running at gen{gen} x{width}, below the card's "
                                   f"gen{max_gen} x{max_width} (idle down-training is normal; "
                                   f"check again in the pcie step)"))
    return out


# ---------------------------------------------------------------------------
# 1. VRAM integrity (memtest)
# ---------------------------------------------------------------------------
PATTERNS = [0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555, 0x0F0F0F0F, 0xF0F0F0F0]


def step_memtest(ctx: Context) -> dict:
    c = _ctx(ctx)
    cap = int(ctx.config.mem_max_gb * GB) if ctx.config.mem_max_gb else None
    want = _budget(ctx, ctx.config.vram, cap=cap)
    t0 = time.perf_counter()
    buf = _alloc_largest(ctx, want)
    alloc_s = time.perf_counter() - t0

    n4 = buf.nbytes // 16                      # the kernels move 16 B per element
    bytes_tested = n4 * 16
    fill, check = _kernel(ctx, "mem_fill"), _kernel(ctx, "mem_check")
    err = c.alloc(4)
    grid = fill.saturating_grid(BLOCK, n4, waves=2)
    patterns = PATTERNS + [random.getrandbits(32) for _ in range(ctx.config.mem_passes - len(PATTERNS))]
    patterns = patterns[: max(1, ctx.config.mem_passes)]
    ctx.log(f"  testing {bytes_tested / GB:.2f} GiB with up to {len(patterns)} patterns "
            f"({grid} blocks x {BLOCK} threads, allocation took {alloc_s:.1f}s)")
    if ctx.device.get("integrated") and bytes_tested > 16 * GB:
        ctx.log("  note: unified memory is shared with the OS - a large span can be slow to first-touch")

    total_err = 0
    w_ms = r_ms = 0.0
    per_pattern = {}
    budget_s = ctx.config.mem_budget
    started = time.perf_counter()
    truncated = False
    try:
        for idx, p in enumerate(patterns):
            err.memset32(0)
            w_ms += _timed(ctx, lambda p=p: fill.launch(grid, BLOCK, buf, u64(n4), u32(p)))
            r_ms += _timed(ctx, lambda p=p: check.launch(grid, BLOCK, buf, u64(n4), u32(p), err))
            e = err.download_u32(1)[0]
            per_pattern[f"0x{p:08X}"] = e
            total_err += e
            done = idx + 1
            gbs = 2 * bytes_tested * done / 1e9 / ((w_ms + r_ms) / 1000)
            ctx.log(f"  pattern 0x{p:08X}: {e} errors  ({gbs:.0f} GB/s write+read)")
            elapsed = time.perf_counter() - started
            if budget_s and done < len(patterns) and elapsed / done * (done + 1) > budget_s:
                ctx.log(f"  stopping after {done} patterns: {elapsed:.0f}s spent, "
                        f"--mem-budget is {budget_s:.0f}s (raise it or lower --vram for a fuller sweep)")
                patterns = patterns[:done]
                truncated = True
                break
    finally:
        err.free()
        buf.free()
    gb = bytes_tested / 1e9
    n = len(patterns)
    return {"tested_gb": round(bytes_tested / GB, 3), "passes": n, "errors": total_err,
            "errors_per_pattern": per_pattern, "alloc_s": round(alloc_s, 2), "truncated": truncated,
            "write_gbs": round(gb * n / (w_ms / 1000), 1),
            "read_gbs": round(gb * n / (r_ms / 1000), 1)}


def eval_memtest(m: dict, t, ctx: Context) -> list[Finding]:
    out = compute_error_finding(m["errors"], f"VRAM integrity over {m['tested_gb']} GiB x {m['passes']} passes")
    out.append(Finding("info", f"memtest throughput: {m['write_gbs']} GB/s write, {m['read_gbs']} GB/s read"))
    total = ctx.device.get("total_mem_bytes") or 0
    if total and m["tested_gb"] * GB < 0.5 * total:
        out.append(Finding("warn", f"only {m['tested_gb']} GiB of {total / GB:.1f} GiB could be tested "
                                   f"(VRAM in use elsewhere, or --vram / --mem-max-gb limiting it)"))
    if m["truncated"]:
        out.append(Finding("info", "pattern sweep cut short by --mem-budget"))
    if m["alloc_s"] > 10:
        out.append(Finding("warn", f"allocating the test buffer took {m['alloc_s']:.0f}s - on unified-memory "
                                   f"parts this is first-touch page setup, not a GPU fault"))
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
        grid = copy.saturating_grid(BLOCK, n4, waves=2)
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
    ctx.log(f"  buffer {size / MB:.0f} MiB, {grid} blocks: copy kernel {gbs_kernel:.1f} GB/s, "
            f"cuMemcpyDtoD {gbs_dtod:.1f} GB/s")
    return {"buffer_mb": round(size / MB, 1), "gbs_copy_kernel": round(gbs_kernel, 1),
            "gbs_dtod": round(gbs_dtod, 1), "gbs": round(max(gbs_kernel, gbs_dtod), 1)}


def eval_bandwidth(m: dict, t, ctx: Context) -> list[Finding]:
    return efficiency_finding(m["gbs"], theoretical_bandwidth_gbs(ctx.device), "GB/s", "device memory bandwidth",
                              warn_below=0.6, fail_below=0.3)


# ---------------------------------------------------------------------------
# 3. host <-> device link (PCIe / NVLink), pinned vs pageable
# ---------------------------------------------------------------------------
def step_pcie(ctx: Context) -> dict:
    c = _ctx(ctx)
    if ctx.device.get("integrated"):
        raise StepSkipped("integrated GPU: host and device share the same memory, there is no transfer link")
    size = min(256 * MB, _budget(ctx, 0.25, cap=256 * MB))
    if size < 16 * MB:
        raise StepSkipped("not enough free VRAM for a transfer test")
    dev = c.alloc(size)
    host = c.alloc_host(size)
    reps = max(3, ctx.config.reps // 4)
    try:
        dev.from_host(host)                      # warm-up / first touch
        c.synchronize()
        h2d = _timed(ctx, lambda: [dev.from_host(host) for _ in range(reps)]) / reps
        d2h = _timed(ctx, lambda: [dev.to_host(host) for _ in range(reps)]) / reps
        pinned_h2d = size / (h2d / 1000) / 1e9
        pinned_d2h = size / (d2h / 1000) / 1e9
        # pageable: a plain Python buffer, which the driver must stage internally
        pageable = bytes(size)
        t0 = time.perf_counter()
        for _ in range(3):
            dev.upload(pageable)
        c.synchronize()
        page_h2d = 3 * size / (time.perf_counter() - t0) / 1e9
    finally:
        host.free()
        dev.free()
    ctx.log(f"  {size / MB:.0f} MiB pinned: H2D {pinned_h2d:.1f} GB/s, D2H {pinned_d2h:.1f} GB/s")
    ctx.log(f"  {size / MB:.0f} MiB pageable: H2D {page_h2d:.1f} GB/s")
    return {"size_mb": round(size / MB, 1), "pinned_h2d_gbs": round(pinned_h2d, 2),
            "pinned_d2h_gbs": round(pinned_d2h, 2), "pageable_h2d_gbs": round(page_h2d, 2),
            "gbs": round(max(pinned_h2d, pinned_d2h), 2),
            "pcie_gen": ctx.device.get("pcie_gen"), "pcie_width": ctx.device.get("pcie_width")}


def eval_pcie(m: dict, t, ctx: Context) -> list[Finding]:
    d = ctx.device
    out = efficiency_finding(m["gbs"], theoretical_pcie_gbs(d), "GB/s",
                             f"host<->device link (gen{d.get('pcie_gen', '?')} x{d.get('pcie_width', '?')})",
                             warn_below=0.55, fail_below=0.3)
    gen, width = d.get("pcie_gen"), d.get("pcie_width")
    max_gen, max_width = d.get("pcie_gen_max"), d.get("pcie_width_max")
    if gen and max_gen and gen < max_gen:
        out.append(Finding("warn", f"PCIe link trained to gen{gen} instead of gen{max_gen} under load - "
                                   f"check the slot, riser and BIOS link settings"))
    if width and max_width and width < max_width:
        out.append(Finding("warn", f"PCIe link is x{width} instead of x{max_width} - the card is in a "
                                   f"narrower slot or sharing lanes"))
    if m["pinned_h2d_gbs"] and m["pageable_h2d_gbs"] > m["pinned_h2d_gbs"] * 1.1:
        out.append(Finding("info", "pageable transfers matched pinned ones - unusual, but harmless"))
    return out


# ---------------------------------------------------------------------------
# 4. verified SGEMM
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
        fgrid = fill.saturating_grid(BLOCK, n * n)
        # small integers in [-3,3] and [-2,2] -> exact fp32 sums for K <= 2^24/6
        fill.launch(fgrid, BLOCK, a, u64(n * n), u32(7919), u32(13), u32(7))
        fill.launch(fgrid, BLOCK, b, u64(n * n), u32(104729), u32(5), u32(5))
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
        rows = sorted({rnd.randrange(n) for _ in range(ctx.config.verify_samples)})
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
# 5/6/7. sustained burns (FP32 / FP16 / tensor cores)
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
# 8. physics: direct N-body gravity (compute-bound, momentum-conserving)
# ---------------------------------------------------------------------------
NB_TILE = 256
NBODY_FLOPS_PER_PAIR = 20          # the usual N-body accounting


def _nbody_init(n: int, seed: int = 7) -> tuple[bytes, bytes]:
    """Uniform sphere of unit masses at rest: total momentum is exactly zero."""
    rnd = random.Random(seed)
    pos = bytearray()
    for _ in range(n):
        while True:
            x, y, z = (rnd.uniform(-1, 1) for _ in range(3))
            if x * x + y * y + z * z <= 1.0:
                break
        pos += struct.pack("4f", x, y, z, 1.0)
    vel = struct.pack("4f", 0.0, 0.0, 0.0, 1.0) * n
    return bytes(pos), vel


def _nbody_momentum(ctx: Context, vel: cu.DeviceBuffer, n: int) -> tuple[float, float, float, float]:
    k = _kernel(ctx, "nbody_reduce")
    blocks = min(k.saturating_grid(BLOCK, n), 1024)
    part = _ctx(ctx).alloc(blocks * 4 * 8)
    try:
        k.launch(blocks, BLOCK, vel, u32(n), part)
        v = part.download_f64(blocks * 4)
    finally:
        part.free()
    return sum(v[0::4]), sum(v[1::4]), sum(v[2::4]), sum(v[3::4])


def step_nbody(ctx: Context) -> dict:
    c = _ctx(ctx)
    n = min(131072, max(8192, c.info.sm_count * 1024))
    n -= n % NB_TILE
    while 4 * n * 16 > _budget(ctx, 0.5) and n > NB_TILE * 8:
        n //= 2
    pos_h, vel_h = _nbody_init(n)
    pos, vel = c.alloc(n * 16), c.alloc(n * 16)
    pos2, vel2 = c.alloc(n * 16), c.alloc(n * 16)
    step = _kernel(ctx, "nbody_step")
    dt, eps2 = 1e-3, 1e-2
    duration = ctx.config.physics
    try:
        pos.upload(pos_h)
        vel.upload(vel_h)
        ms = _timed(ctx, lambda: step.launch(n // NB_TILE, NB_TILE, pos, vel, pos2, vel2, i32(n), f32(dt), f32(eps2)))
        per_batch = max(1, int(100.0 / max(ms, 0.01)))     # ~100 ms of work per timed batch
        ctx.log(f"  {n} bodies, {n // NB_TILE} blocks, {per_batch} steps/batch, {duration:.0f}s "
                f"({ms:.2f} ms/step)")

        def batch():
            for _ in range(per_batch):
                step.launch(n // NB_TILE, NB_TILE, pos, vel, pos2, vel2, i32(n), f32(dt), f32(eps2))
                step.launch(n // NB_TILE, NB_TILE, pos2, vel2, pos, vel, i32(n), f32(dt), f32(eps2))

        samples: list[float] = []
        steps_done = 0
        t_end = time.time() + duration
        next_log = time.time() + 5
        while time.time() < t_end:
            ms = _timed(ctx, batch)
            steps_done += 2 * per_batch
            samples.append(2 * per_batch * n * n * NBODY_FLOPS_PER_PAIR / (ms / 1000) / 1e12)
            if time.time() >= next_log:
                s = ctx.monitor.latest()
                tele = f" | {s.temp}°C {s.power_w:.0f}W {s.sm_clock}MHz" if s and s.temp is not None else ""
                ctx.log(f"    {t_end - time.time():5.0f}s left  {samples[-1]:6.2f} TFLOPS{tele}")
                next_log += 5
        px, py, pz, scale = _nbody_momentum(ctx, vel, n)
        sample_pos = pos.download_f32(4096)
    finally:
        for b in (pos, vel, pos2, vel2):
            b.free()
    drift = math.sqrt(px * px + py * py + pz * pz)
    rel = drift / scale if scale else 0.0
    finite = _finite(px, py, pz, scale) and all(math.isfinite(v) for v in sample_pos)
    k5 = max(1, len(samples) // 5)
    ctx.log(f"  {steps_done} steps, momentum |P| = {drift:.3e} vs scale {scale:.3e} "
            f"(relative {rel:.2e}), positions finite: {finite}")
    return {"bodies": n, "steps": steps_done, "seconds": duration,
            "tflops_mean": round(sum(samples) / len(samples), 3),
            "tflops_first": round(sum(samples[:k5]) / k5, 3), "tflops_last": round(sum(samples[-k5:]) / k5, 3),
            "tflops": round(sum(samples) / len(samples), 3),
            "interactions_per_s": round(steps_done * n * n / duration, 0),
            "momentum_drift": drift, "momentum_scale": scale, "momentum_rel": rel, "finite": finite}


def eval_nbody(m: dict, t, ctx: Context) -> list[Finding]:
    out: list[Finding] = []
    if not m["finite"]:
        out.append(Finding("fail", "N-body state contains NaN/Inf - the simulation diverged, "
                                   "which on fixed inputs means a compute fault"))
    rel = m["momentum_rel"]
    msg = (f"momentum conservation: |P|/scale = {rel:.2e} after {m['steps']} steps "
           f"({m['bodies']} bodies)")
    # Healthy fp32 drift measured on an RTX 4050: 1.9e-4 over 2k steps, 3.4e-4 over
    # 13k - it grows sub-linearly because rsqrtf is approximate and d2(i,j) differs
    # from d2(j,i) by an ulp, so the pairwise cancellation is never exact. A GPU
    # actually computing forces wrong loses the antisymmetry outright and lands
    # near O(1), so these thresholds sit ~30x above the healthy band.
    if rel > 1e-1:
        out.append(Finding("fail", msg + " - Newton's third law broken, forces are being computed wrong"))
    elif rel > 1e-2:
        out.append(Finding("warn", msg + " - far more drift than fp32 rounding explains"))
    else:
        out.append(Finding("info", msg))
    # Unlike the FMA burn, this kernel is not pure FMA: every interaction also runs
    # an rsqrtf on the SFU path and reads shared memory, and the 20-flops-per-pair
    # convention counts arithmetic the hardware does not issue one-per-cycle. Real
    # kernels land near 40-60% of the FMA peak, so only a much lower figure is a fault.
    out += efficiency_finding(m["tflops_mean"], theoretical_fp32_tflops(ctx.device, t.sm_clock_mean),
                              "TFLOPS", "N-body FP32 at observed clock",
                              warn_below=0.25, fail_below=0.10)
    out += stability_finding(m["tflops_first"], m["tflops_last"], "TFLOPS", "N-body throughput")
    return out


# ---------------------------------------------------------------------------
# 9. physics: 2D heat diffusion (memory-bound, conservative)
# ---------------------------------------------------------------------------
def step_stencil(ctx: Context) -> dict:
    c = _ctx(ctx)
    side = 8192
    while 2 * side * side * 4 > _budget(ctx, 0.5) and side > 1024:
        side //= 2
    n = side * side
    u, un = c.alloc(n * 4), c.alloc(n * 4)
    heat = _kernel(ctx, "heat_step")
    fill = _kernel(ctx, "fill_pattern_f32")
    alpha = 0.2                                     # <= 0.25 keeps the scheme stable
    duration = ctx.config.physics
    try:
        fill.launch(fill.saturating_grid(BLOCK, n), BLOCK, u, u64(n), u32(2654435761), u32(11), u32(7))
        c.synchronize()
        sum0, min0, max0 = _reduce_stats(ctx, u, n)
        grid = (side // 32, side // 8)
        ms = _timed(ctx, lambda: heat.launch(grid, (32, 8), u, un, i32(side), i32(side), f32(alpha)))
        per_batch = max(2, int(100.0 / max(ms, 0.01)) // 2 * 2)
        ctx.log(f"  {side}x{side} grid ({n * 4 / MB:.0f} MiB x2), alpha={alpha}, "
                f"{per_batch} steps/batch, {duration:.0f}s ({ms:.2f} ms/step)")

        def batch():
            for _ in range(per_batch // 2):
                heat.launch(grid, (32, 8), u, un, i32(side), i32(side), f32(alpha))
                heat.launch(grid, (32, 8), un, u, i32(side), i32(side), f32(alpha))

        samples: list[float] = []
        steps = 0
        t_end = time.time() + duration
        next_log = time.time() + 5
        while time.time() < t_end:
            ms = _timed(ctx, batch)
            steps += per_batch
            # each step reads the field once and writes it once (neighbours come from cache)
            samples.append(2 * per_batch * n * 4 / (ms / 1000) / 1e9)
            if time.time() >= next_log:
                s = ctx.monitor.latest()
                tele = f" | {s.temp}°C {s.power_w:.0f}W {s.sm_clock}MHz" if s and s.temp is not None else ""
                ctx.log(f"    {t_end - time.time():5.0f}s left  {samples[-1]:6.0f} GB/s{tele}")
                next_log += 5
        sum1, min1, max1 = _reduce_stats(ctx, u, n)
    finally:
        u.free()
        un.free()
    # The initial field averages ~0, so sum0 itself is a useless denominator.
    # Normalise by the largest sum the field could possibly have (n * peak value):
    # fp32 rounding random-walks the total by ~sqrt(steps)*n*eps*peak, which lands
    # around 1e-6 relative on any grid size, while a wrong stencil shifts it far more.
    scale = max(abs(max0), abs(min0), 1e-6)
    drift = abs(sum1 - sum0) / (n * scale)
    slack = 1e-4 * scale
    k5 = max(1, len(samples) // 5)
    ctx.log(f"  {steps} steps: sum {sum0:.6e} -> {sum1:.6e} (relative drift {drift:.2e}), "
            f"range [{min1:.4f}, {max1:.4f}] within initial [{min0:.4f}, {max0:.4f}]")
    return {"side": side, "cells": n, "steps": steps, "seconds": duration, "alpha": alpha,
            "gbs_mean": round(sum(samples) / len(samples), 1), "gbs": round(sum(samples) / len(samples), 1),
            "gbs_first": round(sum(samples[:k5]) / k5, 1), "gbs_last": round(sum(samples[-k5:]) / k5, 1),
            "cells_per_s": round(steps * n / duration, 0),
            "sum_before": sum0, "sum_after": sum1, "sum_drift_rel": drift,
            "min_before": min0, "max_before": max0, "min_after": min1, "max_after": max1,
            "bounds_violated": bool(min1 < min0 - slack or max1 > max0 + slack),
            "finite": _finite(sum1, min1, max1)}


def eval_stencil(m: dict, t, ctx: Context) -> list[Finding]:
    out: list[Finding] = []
    if not m["finite"]:
        out.append(Finding("fail", "heat field contains NaN/Inf"))
    if m["bounds_violated"]:
        out.append(Finding("fail", f"maximum principle violated: field left its initial range "
                                   f"[{m['min_before']:.4f}, {m['max_before']:.4f}] -> "
                                   f"[{m['min_after']:.4f}, {m['max_after']:.4f}] - impossible for a "
                                   f"diffusion step with alpha <= 0.25, so the GPU computed it wrong"))
    else:
        out.append(Finding("info", f"maximum principle holds over {m['steps']} diffusion steps"))
    drift = m["sum_drift_rel"]
    msg = f"heat conservation: total drifted {drift:.2e} (relative) over {m['steps']} steps"
    if drift > 1e-3:
        out.append(Finding("fail", msg + " - a conserved quantity is not being conserved"))
    elif drift > 1e-5:
        out.append(Finding("warn", msg))
    else:
        out.append(Finding("info", msg))
    out += efficiency_finding(m["gbs_mean"], theoretical_bandwidth_gbs(ctx.device), "GB/s",
                              "stencil effective bandwidth", warn_below=0.35, fail_below=0.15)
    out += stability_finding(m["gbs_first"], m["gbs_last"], "GB/s", "stencil throughput")
    return out


# ---------------------------------------------------------------------------
# 10. hardware edge cases (IEEE-754 corners, atomics, warp intrinsics)
# ---------------------------------------------------------------------------
EDGE_BITS = [
    (0x0001, "denormal_flushed", "warn", "fp32 denormals are flushed to zero instead of preserved"),
    (0x0002, "nan_compare", "fail", "NaN comparisons do not follow IEEE-754"),
    (0x0004, "inf_arith", "fail", "infinity arithmetic is wrong (1/0, inf-inf or inf+x)"),
    (0x0008, "int_ops", "fail", "32-bit integer intrinsics returned wrong results"),
    (0x0010, "shfl", "fail", "warp shuffle (__shfl_xor_sync) reduction is wrong"),
    (0x0020, "ballot", "fail", "warp ballot (__ballot_sync) mask is wrong"),
    (0x0040, "shared_reduce", "fail", "shared-memory + __syncthreads reduction is wrong"),
    (0x0080, "fma_not_fused", "fail", "fmaf() is not fused - the product is rounded before the add"),
    (0x0100, "sqrt_special", "fail", "sqrtf of -1 / 0 / inf is wrong"),
    (0x0200, "rounding", "fail", "fp32 round-to-nearest-even is wrong"),
    (0x0400, "int64", "fail", "64-bit integer arithmetic is wrong"),
]

# den, zero, +inf, NaN, 0.1, 0.2, 1+2^-23, 1-2^-24, -1, 2
EDGE_INPUTS = [
    struct.unpack("f", struct.pack("I", 0x00000001))[0],
    0.0,
    struct.unpack("f", struct.pack("I", 0x7F800000))[0],
    struct.unpack("f", struct.pack("I", 0x7FC00000))[0],
    0.1, 0.2,
    struct.unpack("f", struct.pack("I", 0x3F800001))[0],
    struct.unpack("f", struct.pack("I", 0x3F7FFFFF))[0],
    -1.0, 2.0,
]


def _xor_below(n: int) -> int:
    """XOR of 0..n-1, in closed form (the kernel XORs every thread id in)."""
    m = n - 1
    return (m, 1, m + 1, 0)[m % 4]


def step_edge_cases(ctx: Context) -> dict:
    c = _ctx(ctx)
    k = _kernel(ctx, "edge_cases")
    blocks = max(1, min(c.info.sm_count * 8, 4096))
    threads = blocks * BLOCK
    inp = c.alloc(len(EDGE_INPUTS) * 4)
    flags = c.alloc(4)
    counters = c.alloc(16)
    try:
        inp.upload(struct.pack(f"{len(EDGE_INPUTS)}f", *EDGE_INPUTS))
        flags.memset32(0)
        counters.memset32(0)
        ms = _timed(ctx, lambda: k.launch(blocks, BLOCK, inp, flags, counters, i32(1234567)))
        mask = flags.download_u32(1)[0]
        cnt = counters.download_u32(4)
    finally:
        counters.free()
        flags.free()
        inp.free()

    atomics = {
        "atomicAdd": (cnt[0], threads),
        "atomicMax": (cnt[1], threads - 1),
        "atomicCAS": (cnt[2], threads // 32),     # one CAS increment per warp
        "atomicXor": (cnt[3], _xor_below(threads)),
    }
    bad_atomics = {k2: v for k2, v in atomics.items() if v[0] != v[1]}
    failed = [name for bit, name, _lvl, _d in EDGE_BITS if mask & bit]
    ctx.log(f"  {threads} threads x {len(EDGE_BITS)} checks in {ms:.2f} ms: "
            f"{'all passed' if not mask else 'FAILED: ' + ', '.join(failed)}")
    for name, (got, want) in atomics.items():
        ctx.log(f"  {name:10s}: {got} (expected {want}){'' if got == want else '  <-- MISMATCH'}")
    return {"threads": threads, "mask": mask, "failed_checks": failed,
            "atomics": {k2: {"got": v[0], "expected": v[1]} for k2, v in atomics.items()},
            "atomics_ok": not bad_atomics, "errors": len(failed) + len(bad_atomics)}


def eval_edge_cases(m: dict, t, ctx: Context) -> list[Finding]:
    out: list[Finding] = []
    for bit, name, level, desc in EDGE_BITS:
        if m["mask"] & bit:
            out.append(Finding(level, f"{name}: {desc}"))
    if not m["atomics_ok"]:
        bad = ", ".join(f"{k} got {v['got']} expected {v['expected']}"
                        for k, v in m["atomics"].items() if v["got"] != v["expected"])
        out.append(Finding("fail", f"atomic operations lost updates across {m['threads']} threads: {bad}"))
    if not out:
        out.append(Finding("info", f"all {len(EDGE_BITS)} IEEE-754 / integer / warp / atomic edge cases "
                                   f"passed on {m['threads']} threads"))
    return out


# ---------------------------------------------------------------------------
# 11. allocation churn (fragmentation)
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
        Step("system_info", step_system_info, "device, driver and theoretical peaks", evaluate=eval_system_info),
        Step("memtest", step_memtest, "write/read patterns over the free VRAM", evaluate=eval_memtest),
        Step("bandwidth", step_bandwidth, "device-to-device copy bandwidth", evaluate=eval_bandwidth),
        Step("pcie", step_pcie, "host<->device link, pinned vs pageable", evaluate=eval_pcie),
        Step("matmul", step_matmul, "verified tiled SGEMM", evaluate=eval_matmul),
        Step("edge_cases", step_edge_cases, "IEEE-754 corners, atomics, warp intrinsics", evaluate=eval_edge_cases),
        Step("fp32_burn", step_fp32_burn, "sustained FP32 FMA load", heavy=True, evaluate=eval_fp32_burn),
        Step("fp16_burn", step_fp16_burn, "sustained packed-FP16 load", heavy=True, evaluate=eval_fp16_burn),
        Step("tensor_burn", step_tensor_burn, "sustained tensor-core (mma.sync) load", heavy=True,
             evaluate=eval_tensor_burn),
        Step("nbody", step_nbody, "N-body gravity, momentum-conserving", heavy=True, evaluate=eval_nbody),
        Step("stencil", step_stencil, "2D heat diffusion, conservative", heavy=True, evaluate=eval_stencil),
        Step("alloc_churn", step_alloc_churn, "allocation / free latency and churn", evaluate=eval_alloc_churn),
    ]


KERNEL_NAMES = [
    "fma_burn", "hfma2_burn", "sgemm_tiled", "fill_pattern_f32", "mem_fill", "mem_check", "copy_f4",
    "reduce_stats_f32", "nbody_reduce", "nbody_step", "heat_step", "edge_cases",
]


def load_kernels(ctx: Context, cctx: cu.Context, prefer_nvrtc: bool) -> dict[str, cu.Kernel]:
    """Load the kernels: NVRTC for the exact arch if asked & available, else shipped PTX."""
    info = cctx.info
    if info.cc_major < 7:
        raise RuntimeError(f"the shipped PTX targets sm_70+ (Volta and newer); this GPU is sm_{info.cc_major}{info.cc_minor}. "
                           "Rebuild gpu_stress/cuda/kernels.cu for an older arch with --nvrtc / nvrtc.py")
    ptx = None
    if prefer_nvrtc:
        try:
            from .cuda import read_kernel_file
            from .cuda.nvrtc import compile_to_ptx, find_nvrtc
            lib = find_nvrtc()
            if lib is not None:
                ptx = compile_to_ptx(read_kernel_file("kernels.cu"),
                                     f"compute_{info.cc_major}{info.cc_minor}", lib=lib)
                ctx.log(f"  kernels: compiled with NVRTC for compute_{info.cc_major}{info.cc_minor}")
        # Any NVRTC problem is recoverable: the shipped PTX always works.
        except Exception as e:
            ctx.log(f"  NVRTC unavailable ({e}); using shipped PTX")
    if ptx is None:
        from .cuda import read_kernel_file
        fname = "kernels_sm80.ptx" if info.cc_major >= 8 else "kernels_sm70.ptx"
        ptx = read_kernel_file(fname)
        ctx.log(f"  kernels: {fname} (driver JIT for sm_{info.cc_major}{info.cc_minor})")
    mod = cctx.load_ptx(ptx)
    names = list(KERNEL_NAMES)
    if info.cc_major >= 8:
        names.append("mma_burn")
    return {n: mod.function(n) for n in names}
