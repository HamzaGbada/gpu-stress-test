import torch
import time
import json
import csv
import platform
import os
from torch import nn
from statistics import mean

# ======================================================================
# UTILS
# ======================================================================

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timer(fn, repeat=10):
    synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
        synchronize()
    end = time.time()
    return (end - start) / repeat


def print_header(title):
    print("\n" + "="*70)
    print(title)
    print("="*70 + "\n")


# ======================================================================
# 0. SYSTEM REPORT
# ======================================================================

def system_report():
    print_header("SYSTEM REPORT")

    print(f"Python version        : {platform.python_version()}")
    print(f"PyTorch version       : {torch.__version__}")
    print(f"CUDA available        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version          : {torch.version.cuda}")
        gpu = torch.cuda.get_device_name(0)
        print(f"GPU Name              : {gpu}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total VRAM            : {props.total_memory / (1024**3):.2f} GB")
        print(f"Compute capability    : {props.major}.{props.minor}")
        print(f"SM Count              : {props.multi_processor_count}")
    print()


# ======================================================================
# 1. TORCH.COMPILE BENCHMARK
# ======================================================================

def benchmark_compile():
    print_header("TORCH.COMPILE BENCHMARK")

    M = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096)
    ).cuda()

    x = torch.randn(1024, 4096, device="cuda")

    # Warmup
    for _ in range(10):
        M(x)

    def run_eager():
        M(x)

    t_eager = timer(run_eager)

    # compile
    M_opt = torch.compile(M)

    for _ in range(10):
        M_opt(x)

    def run_compiled():
        M_opt(x)

    t_comp = timer(run_compiled)

    speedup = t_eager / t_comp

    print(f"Eager Avg Time     : {t_eager:.6f} sec")
    print(f"Compiled Avg Time  : {t_comp:.6f} sec")
    print(f"Speedup            : {speedup:.2f}x")

    return {
        "eager_time": t_eager,
        "compiled_time": t_comp,
        "speedup": speedup
    }


# ======================================================================
# 2. MIXED PRECISION BENCHMARK
# ======================================================================

def bm_matmul(dtype, label):
    size = 8192
    A = torch.randn((size, size), device="cuda", dtype=dtype)
    B = torch.randn((size, size), device="cuda", dtype=dtype)

    def run():
        A @ B

    t = timer(run, repeat=5)
    flops = 2 * (size ** 3)
    tflops = (flops / t) / 1e12

    print(f"{label:10s} | {t:.5f} sec | {tflops:.1f} TFLOPS")
    return {"time": t, "tflops": tflops}


def mixed_precision_benchmark():
    print_header("MIXED PRECISION BENCHMARK")
    results = {}

    tests = [
        (torch.float32, "FP32"),
        (torch.bfloat16, "BF16"),
        (torch.float16, "FP16"),
        (torch.float32, "TF32"),  # PyTorch uses TF32 on Ampere+ by default
    ]

    for dtype, label in tests:
        results[label] = bm_matmul(dtype, label)

    return results


# ======================================================================
# 3. FULL TRAINING BENCHMARK (ResNet50)
# ======================================================================

def training_benchmark():
    print_header("RESNET50 TRAINING BENCHMARK")

    # Load ResNet50 safely WITHOUT TorchHub
    from torchvision.models import resnet50

    model = resnet50(weights=None).cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(32, 3, 224, 224, device="cuda")
    y = torch.randint(0, 1000, (32,), device="cuda")
    loss_fn = nn.CrossEntropyLoss()

    # warm-up
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    def train_step():
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    t = timer(train_step)
    steps_per_sec = 1 / t
    imgs_per_sec = 32 / t

    print(f"Time per step    : {t:.6f} sec")
    print(f"Steps per second : {steps_per_sec:.2f}")
    print(f"Images per sec   : {imgs_per_sec:.2f}")

    return {
        "step_time": t,
        "steps_per_sec": steps_per_sec,
        "imgs_per_sec": imgs_per_sec
    }


# ======================================================================
# 4. MEMORY FRAGMENTATION TEST
# ======================================================================

def memory_fragmentation_benchmark():
    print_header("MEMORY FRAGMENTATION TEST")

    block_sizes = [128, 256, 512, 1024, 2048]  # MB
    results = {}

    for size in block_sizes:
        mb = size * 1024 * 1024 // 4  # floats
        t_alloc = []
        t_free = []

        for _ in range(10):
            start = time.time()
            x = torch.empty(mb, device="cuda")
            torch.cuda.synchronize()
            t_alloc.append(time.time() - start)

            start = time.time()
            del x
            torch.cuda.synchronize()
            t_free.append(time.time() - start)

        results[size] = {
            "alloc_time_ms": mean(t_alloc) * 1000,
            "free_time_ms": mean(t_free) * 1000
        }

        print(f"{size} MB | alloc: {results[size]['alloc_time_ms']:.3f} ms | "
              f"free: {results[size]['free_time_ms']:.3f} ms")

    return results


# ======================================================================
# 6. EXPORT RESULTS (JSON + CSV)
# ======================================================================

def export_results(out):
    print_header("EXPORTING RESULTS")

    with open("benchmark_results.json", "w") as f:
        json.dump(out, f, indent=4)

    with open("benchmark_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Test", "Metric", "Value"])
        for k, v in out.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    writer.writerow([k, subk, subv])
            else:
                writer.writerow([k, "", v])

    print("Saved to benchmark_results.json and benchmark_results.csv")


# ======================================================================
# MAIN
# ======================================================================

def main():
    system_report()

    results = {}
    results["compile"] = benchmark_compile()
    results["mixed_precision"] = mixed_precision_benchmark()
    results["training"] = training_benchmark()
    results["memory_fragmentation"] = memory_fragmentation_benchmark()

    export_results(results)

    print("\nBenchmark complete.\n")


if __name__ == "__main__":
    main()

