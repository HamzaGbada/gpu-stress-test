"""PyTorch-based stress / benchmark steps.

The training step is the one that used to OOM on small cards: it now measures
how much memory a training step really needs at the requested batch size
(halving the batch until it fits) and only then sizes the in-VRAM synthetic
dataset from what is *left*, instead of blindly taking 75% of the card.
"""
from __future__ import annotations

import math
import os
import time

import torch
from torch import nn

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
IMG_BYTES = 3 * 224 * 224 * 4      # one fp32 ImageNet-size image
OOM = (torch.cuda.OutOfMemoryError,)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _dev(ctx: Context) -> torch.device:
    return torch.device("cuda", ctx.config.device)


def _free_bytes(ctx: Context) -> int:
    """Free memory as the driver sees it (other processes included)."""
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(_dev(ctx))
    return free


def _timed(fn, repeat: int) -> float:
    """Mean seconds per call using CUDA events."""
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeat):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / 1000 / repeat


def _amp_dtype(ctx: Context) -> torch.dtype | None:
    p = ctx.config.precision
    if p == "fp32":
        return None
    if p == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise StepSkipped("bf16 not supported on this GPU")
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------------
# 0. system info
# ---------------------------------------------------------------------------
def step_system_info(ctx: Context) -> dict:
    d = ctx.device
    free, total = torch.cuda.mem_get_info(_dev(ctx))
    info = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "free_mem_bytes": free, "total_mem_bytes": total,
            "theoretical_fp32_tflops": theoretical_fp32_tflops(d),
            "theoretical_bandwidth_gbs": theoretical_bandwidth_gbs(d),
            "alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
    for k, v in {**d, **info}.items():
        ctx.log(f"  {k:24s}: {v}")
    return info


# ---------------------------------------------------------------------------
# 1. bandwidth (device-to-device)
# ---------------------------------------------------------------------------
def step_bandwidth(ctx: Context) -> dict:
    size = min(1 * GB, int(_free_bytes(ctx) * 0.4))
    a = torch.empty(size // 4, dtype=torch.float32, device=_dev(ctx))
    b = torch.empty_like(a)
    t = _timed(lambda: b.copy_(a), ctx.config.reps)
    gbs = 2 * size / t / 1e9
    ctx.log(f"  {size / MB:.0f} MiB copy: {gbs:.1f} GB/s")
    return {"buffer_mb": round(size / MB, 1), "gbs": round(gbs, 1)}


def eval_bandwidth(m, t, ctx) -> list[Finding]:
    return efficiency_finding(m["gbs"], theoretical_bandwidth_gbs(ctx.device), "GB/s", "device memory bandwidth",
                              warn_below=0.6, fail_below=0.3)


# ---------------------------------------------------------------------------
# 2. matmul in every precision (cuBLAS / tensor cores)
# ---------------------------------------------------------------------------
def step_matmul(ctx: Context) -> dict:
    n = ctx.config.matmul_size
    while 3 * n * n * 4 * 1.5 > _free_bytes(ctx) and n > 1024:
        n //= 2
    dev = _dev(ctx)
    res: dict = {"n": n}
    tests = [("fp32", torch.float32, False), ("tf32", torch.float32, True), ("bf16", torch.bfloat16, False),
             ("fp16", torch.float16, False)]
    torch.manual_seed(0)
    a32 = torch.randn(n, n, device=dev)
    b32 = torch.randn(n, n, device=dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = a32[:64] @ b32[:, :64]          # exact fp32 sample, shared by every precision
    for label, dtype, tf32 in tests:
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        torch.backends.cuda.matmul.allow_tf32 = tf32
        a, b = a32.to(dtype), b32.to(dtype)
        try:
            t = _timed(lambda a=a, b=b: a @ b, max(2, ctx.config.reps // 4))
            c = a @ b
        except OOM:
            ctx.log(f"  {label}: OOM at n={n}, skipping")
            continue
        finite = bool(torch.isfinite(c).all())
        sample = c[:64, :64].float()
        rel = float((sample - ref).abs().max() / ref.abs().max()) if finite else float("inf")
        tflops = 2 * n ** 3 / t / 1e12
        res[label] = {"sec": round(t, 5), "tflops": round(tflops, 2), "finite": finite, "max_rel_err": round(rel, 5)}
        ctx.log(f"  {label:5s} {n}x{n}: {tflops:7.1f} TFLOPS  (max rel err vs fp32 {rel:.1e})")
        del a, b, c
        torch.cuda.empty_cache()
    del a32, b32
    torch.backends.cuda.matmul.allow_tf32 = False
    res["tflops"] = res.get("fp32", {}).get("tflops")
    return res


def eval_matmul(m, t, ctx) -> list[Finding]:
    out = []
    tol = {"fp32": 1e-4, "tf32": 5e-2, "bf16": 1e-1, "fp16": 5e-2}
    bad = [k for k in tol if k in m and (not m[k]["finite"] or m[k]["max_rel_err"] > tol[k])]
    if bad:
        out.append(Finding("fail", f"matmul results wrong / non-finite in: {', '.join(bad)}"))
    else:
        out.append(Finding("info", "matmul precision sweep: results consistent with fp32"))
    if m.get("fp32"):
        out += efficiency_finding(m["fp32"]["tflops"], theoretical_fp32_tflops(ctx.device, t.sm_clock_mean), "TFLOPS",
                                  "cuBLAS FP32 at observed clock", warn_below=0.4, fail_below=0.2)
    fast = max((m[k]["tflops"] for k in ("tf32", "bf16", "fp16") if k in m), default=None)
    if fast and m.get("fp32"):
        ratio = fast / m["fp32"]["tflops"]
        lvl = "warn" if ratio < 1.5 and ctx.device.get("cc_major", 0) >= 7 else "info"
        out.append(Finding(lvl, f"tensor-core speedup over FP32: {ratio:.1f}x"))
    return out


# ---------------------------------------------------------------------------
# 3. torch.compile
# ---------------------------------------------------------------------------
def step_compile(ctx: Context) -> dict:
    dev = _dev(ctx)
    model = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 4096)).to(dev)
    x = torch.randn(1024, 4096, device=dev)
    with torch.no_grad():
        t_eager = _timed(lambda: model(x), ctx.config.reps)
        try:
            opt = torch.compile(model)
            t0 = time.time()
            opt(x)
            torch.cuda.synchronize()
            compile_s = time.time() - t0
            t_comp = _timed(lambda: opt(x), ctx.config.reps)
        except Exception as e:  # noqa: BLE001 - missing triton / unsupported GPU
            raise StepSkipped(f"torch.compile unavailable: {type(e).__name__}: {str(e)[:120]}")
    ctx.log(f"  eager {t_eager * 1000:.3f} ms, compiled {t_comp * 1000:.3f} ms (compile took {compile_s:.1f}s)")
    return {"eager_ms": round(t_eager * 1000, 4), "compiled_ms": round(t_comp * 1000, 4),
            "compile_time_s": round(compile_s, 2), "speedup": round(t_eager / t_comp, 3)}


def eval_compile(m, t, ctx) -> list[Finding]:
    if m["speedup"] < 0.8:
        return [Finding("warn", f"torch.compile is slower than eager ({m['speedup']:.2f}x)")]
    return [Finding("info", f"torch.compile speedup {m['speedup']:.2f}x")]


# ---------------------------------------------------------------------------
# 4. sustained matmul burn
# ---------------------------------------------------------------------------
def step_matmul_burn(ctx: Context) -> dict:
    dev = _dev(ctx)
    dtype = _amp_dtype(ctx) or torch.float32
    n = ctx.config.matmul_size
    while 3 * n * n * 4 * 1.5 > _free_bytes(ctx) and n > 1024:
        n //= 2
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    c = torch.empty_like(a)
    def mm():
        torch.matmul(a, b, out=c)

    per_launch = max(1, int(0.1 / _timed(mm, 3)))
    samples: list[float] = []
    duration = ctx.config.burn
    ctx.log(f"  {dtype} {n}x{n} matmul x{per_launch} per launch for {duration:.0f}s")
    t_end = time.time() + duration
    next_log = time.time() + 5
    nonfinite = 0
    while time.time() < t_end:
        t = _timed(mm, per_launch)
        samples.append(2 * n ** 3 / t / 1e12)
        if len(samples) % 20 == 1 and not bool(torch.isfinite(c).all()):
            nonfinite += 1
        if time.time() >= next_log:
            s = ctx.monitor.latest()
            tele = f" | {s.temp}°C {s.power_w:.0f}W {s.sm_clock}MHz" if s and s.temp is not None else ""
            ctx.log(f"    {t_end - time.time():5.0f}s left  {samples[-1]:7.1f} TFLOPS{tele}")
            next_log += 5
    k5 = max(1, len(samples) // 5)
    return {"dtype": str(dtype), "n": n, "seconds": duration, "launches": len(samples),
            "tflops_mean": round(sum(samples) / len(samples), 2), "tflops_min": round(min(samples), 2),
            "tflops_max": round(max(samples), 2), "tflops_first": round(sum(samples[:k5]) / k5, 2),
            "tflops_last": round(sum(samples[-k5:]) / k5, 2), "tflops": round(sum(samples) / len(samples), 2),
            "errors": nonfinite}


def eval_matmul_burn(m, t, ctx) -> list[Finding]:
    out = compute_error_finding(m["errors"], "sustained matmul outputs (finite check)")
    out += stability_finding(m["tflops_first"], m["tflops_last"], "TFLOPS", "matmul throughput")
    return out


# ---------------------------------------------------------------------------
# 5. ResNet50 training with memory budgeting (the OOM fix)
# ---------------------------------------------------------------------------
def _train_step(model, opt, loss_fn, x, y, amp_dtype, scaler):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        loss = loss_fn(model(x), y)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
    return loss


def plan_training_memory(ctx: Context, model, opt, loss_fn, batch: int, amp_dtype) -> dict:
    """Probe real memory use of a training step; shrink batch until it fits.

    Returns the batch that fits, the persistent (params + grads + optimizer
    state) and transient (activations + workspace) bytes measured for it.
    """
    dev = _dev(ctx)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    while batch >= 2:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            x = torch.randn(batch, 3, 224, 224, device=dev)
            y = torch.randint(0, 1000, (batch,), device=dev)
            for _ in range(2):                       # 2nd step has optimizer state allocated
                _train_step(model, opt, loss_fn, x, y, amp_dtype, scaler)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(dev)
            del x, y
            persistent = torch.cuda.memory_allocated(dev)
            return {"batch": batch, "persistent": persistent, "transient": peak - persistent, "peak": peak,
                    "scaler": scaler}
        except OOM:
            ctx.log(f"  batch {batch} does not fit, trying {batch // 2}")
            opt.zero_grad(set_to_none=True)
            x = y = None
            torch.cuda.empty_cache()
            batch //= 2
    raise RuntimeError("ResNet50 does not fit in GPU memory even at batch 2")


def step_training(ctx: Context) -> dict:
    from torchvision.models import resnet50

    dev = _dev(ctx)
    cfg = ctx.config
    amp_dtype = _amp_dtype(ctx)
    total = torch.cuda.get_device_properties(dev).total_memory
    model = resnet50(weights=None).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    plan = plan_training_memory(ctx, model, opt, loss_fn, cfg.batch, amp_dtype)
    batch, scaler = plan["batch"], plan["scaler"]

    # Dataset budget = what is free *now* (model + optimizer already resident)
    # minus the transient activations we measured, minus a safety margin.
    free = _free_bytes(ctx)
    headroom = max(256 * MB, int(total * 0.06))
    budget = min(int(total * cfg.vram), free - plan["transient"] - headroom)
    dataset_size = max(batch * 2, budget // IMG_BYTES) if budget > 0 else batch * 2
    dataset_size -= dataset_size % batch
    ctx.log(f"  batch {batch} ({'requested ' + str(cfg.batch) if batch != cfg.batch else 'as requested'}), "
            f"precision {cfg.precision}")
    ctx.log(f"  model+optimizer {plan['persistent'] / GB:.2f} GiB, activations {plan['transient'] / GB:.2f} GiB, "
            f"free {free / GB:.2f} GiB -> dataset budget {max(budget, 0) / GB:.2f} GiB")
    try:
        x_data = torch.randn(dataset_size, 3, 224, 224, device=dev)
    except OOM:
        dataset_size = max(batch * 2, dataset_size // 2 - dataset_size // 2 % batch)
        torch.cuda.empty_cache()
        x_data = torch.randn(dataset_size, 3, 224, 224, device=dev)
        ctx.log(f"  dataset allocation OOMed, reduced to {dataset_size} images")
    y_data = torch.randint(0, 1000, (dataset_size,), device=dev)
    steps_per_epoch = dataset_size // batch
    ctx.log(f"  dataset {dataset_size:,} images ({dataset_size * IMG_BYTES / GB:.2f} GiB), "
            f"{steps_per_epoch} steps/epoch, {cfg.epochs} epochs")
    if cfg.max_steps:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps)

    epoch_times, losses, oom_events = [], [], 0
    imgs_done = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        i = 0
        while i < steps_per_epoch:
            bx = x_data[i * batch:(i + 1) * batch]
            by = y_data[i * batch:(i + 1) * batch]
            try:
                loss = _train_step(model, opt, loss_fn, bx, by, amp_dtype, scaler)
            except OOM:
                # fragmentation late in a run: shrink the batch and keep going
                oom_events += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if batch <= 2:
                    raise
                batch //= 2
                steps_per_epoch = dataset_size // batch if not cfg.max_steps else min(dataset_size // batch, cfg.max_steps)
                ctx.log(f"  OOM during training, batch reduced to {batch}")
                continue
            if i % 10 == 0:
                losses.append(float(loss))
            imgs_done += batch
            i += 1
        torch.cuda.synchronize()
        dt = time.time() - t0
        epoch_times.append(dt)
        s = ctx.monitor.latest()
        tele = f" | {s.temp}°C {s.power_w:.0f}W" if s and s.temp is not None else ""
        ctx.log(f"  epoch {epoch + 1}/{cfg.epochs}: {dt:.1f}s, {steps_per_epoch / dt:.2f} steps/s, "
                f"{steps_per_epoch * batch / dt:.0f} img/s, loss {float(loss):.3f}{tele}")
    del x_data, y_data, model, opt
    torch.cuda.empty_cache()
    avg = sum(epoch_times) / len(epoch_times)
    return {"batch_size": batch, "requested_batch": cfg.batch, "precision": cfg.precision,
            "dataset_images": dataset_size, "dataset_gb": round(dataset_size * IMG_BYTES / GB, 3),
            "model_optimizer_gb": round(plan["persistent"] / GB, 3), "activation_gb": round(plan["transient"] / GB, 3),
            "epochs": cfg.epochs, "steps_per_epoch": steps_per_epoch, "avg_epoch_sec": round(avg, 3),
            "steps_per_sec": round(steps_per_epoch / avg, 3), "imgs_per_sec": round(steps_per_epoch * batch / avg, 1),
            "epoch_times": [round(t, 3) for t in epoch_times], "loss_curve": losses, "oom_events": oom_events,
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated(dev) / GB, 3)}


def eval_training(m, t, ctx) -> list[Finding]:
    out = []
    if m["batch_size"] != m["requested_batch"]:
        out.append(Finding("warn", f"batch reduced {m['requested_batch']} -> {m['batch_size']} to fit in VRAM "
                                   f"(use --precision bf16 or a smaller --batch)"))
    if m["oom_events"]:
        out.append(Finding("warn", f"{m['oom_events']} OOM events recovered during training"))
    losses = m["loss_curve"]
    if losses and not all(math.isfinite(v) for v in losses):
        out.append(Finding("fail", "training loss became NaN/inf"))
    elif len(losses) >= 4 and losses[-1] > losses[0] * 1.5:
        out.append(Finding("warn", f"loss diverging {losses[0]:.2f} -> {losses[-1]:.2f}"))
    if len(m["epoch_times"]) >= 2:
        per_epoch = m["steps_per_epoch"] * m["batch_size"]
        out += stability_finding(per_epoch / m["epoch_times"][0], per_epoch / m["epoch_times"][-1], "img/s",
                                 "training speed")
    out.append(Finding("info", f"{m['imgs_per_sec']:.0f} img/s at batch {m['batch_size']} ({m['precision']}), "
                               f"peak {m['peak_allocated_gb']} GiB allocated"))
    return out


# ---------------------------------------------------------------------------
# 6. VRAM integrity through the caching allocator
# ---------------------------------------------------------------------------
def step_vram_fill(ctx: Context) -> dict:
    dev = _dev(ctx)
    budget = int(_free_bytes(ctx) * ctx.config.vram) - 256 * MB
    chunk = 256 * MB
    blocks: list[torch.Tensor] = []
    errors = 0
    try:
        while sum(b.numel() * 4 for b in blocks) + chunk <= budget:
            try:
                blocks.append(torch.empty(chunk // 4, dtype=torch.int32, device=dev))
            except OOM:
                break
        tested = sum(b.numel() * 4 for b in blocks)
        for pattern in (0x00000000, -1, 0x55555555, -0x55555556):
            for i, b in enumerate(blocks):
                b.fill_(pattern)
                b.bitwise_xor_(torch.arange(b.numel(), dtype=torch.int32, device=dev) if i == 0 else 0)
            torch.cuda.synchronize()
            for i, b in enumerate(blocks):
                expect = pattern ^ torch.arange(b.numel(), dtype=torch.int32, device=dev) if i == 0 else pattern
                errors += int((b != expect).sum())
    finally:
        blocks.clear()
        torch.cuda.empty_cache()
    ctx.log(f"  tested {tested / GB:.2f} GiB x 4 patterns: {errors} errors")
    return {"tested_gb": round(tested / GB, 3), "errors": errors}


def eval_vram_fill(m, t, ctx) -> list[Finding]:
    return compute_error_finding(m["errors"], f"VRAM fill over {m['tested_gb']} GiB")


# ---------------------------------------------------------------------------
# 7. allocator fragmentation
# ---------------------------------------------------------------------------
def step_fragmentation(ctx: Context) -> dict:
    dev = _dev(ctx)
    res = {}
    for mb in (128, 256, 512, 1024, 2048):
        if mb * MB > _free_bytes(ctx) * 0.8:
            break
        ta = tf = 0.0
        for _ in range(10):
            t0 = time.perf_counter()
            x = torch.empty(mb * MB // 4, device=dev)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            del x
            torch.cuda.synchronize()
            ta += t1 - t0
            tf += time.perf_counter() - t1
        res[f"alloc_{mb}mb_ms"] = round(ta / 10 * 1000, 3)
        res[f"free_{mb}mb_ms"] = round(tf / 10 * 1000, 3)
        ctx.log(f"  {mb:5d} MiB: alloc {res[f'alloc_{mb}mb_ms']:.3f} ms, free {res[f'free_{mb}mb_ms']:.3f} ms")
    stats = torch.cuda.memory_stats(dev)
    res["num_alloc_retries"] = stats.get("num_alloc_retries", 0)
    res["num_ooms"] = stats.get("num_ooms", 0)
    return res


def eval_fragmentation(m, t, ctx) -> list[Finding]:
    out = []
    if m["num_alloc_retries"]:
        out.append(Finding("info", f"caching allocator retried {m['num_alloc_retries']} times during the run"))
    return out


def build_steps() -> list[Step]:
    return [
        Step("system_info", step_system_info, "torch / CUDA / device facts"),
        Step("bandwidth", step_bandwidth, "device-to-device copy bandwidth", evaluate=eval_bandwidth),
        Step("matmul", step_matmul, "cuBLAS matmul in fp32 / tf32 / bf16 / fp16", evaluate=eval_matmul),
        Step("compile", step_compile, "torch.compile speedup", evaluate=eval_compile),
        Step("matmul_burn", step_matmul_burn, "sustained matmul load", heavy=True, evaluate=eval_matmul_burn),
        Step("training", step_training, "ResNet50 training with in-VRAM dataset", heavy=True, evaluate=eval_training),
        Step("vram_fill", step_vram_fill, "fill VRAM and verify patterns", evaluate=eval_vram_fill),
        Step("fragmentation", step_fragmentation, "allocator alloc/free latency", evaluate=eval_fragmentation),
    ]
