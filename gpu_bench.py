import torch
import time
import os
import subprocess
import platform

# ---------------------------------------------------------
# 1. Environment & GPU Info
# ---------------------------------------------------------
def system_report():
    print("\n================ SYSTEM REPORT ================\n")
    print(f"Python version  : {platform.python_version()}")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version    : {torch.version.cuda}")
        print(f"GPU count       : {torch.cuda.device_count()}")
        print(f"GPU name        : {torch.cuda.get_device_name(0)}")

        props = torch.cuda.get_device_properties(0)
        print(f"Total VRAM      : {props.total_memory / (1024**3):.2f} GB")
        print(f"SM count        : {props.multi_processor_count}")
        print(f"Max threads/SM  : {props.max_threads_per_multi_processor}")
        print(f"Compute capability : {props.major}.{props.minor}")

    print("\n================================================\n")


# ---------------------------------------------------------
# 2. Benchmark Tools
# ---------------------------------------------------------
def benchmark_step(name, func, repeat=5):
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(repeat):
        func()
        torch.cuda.synchronize()

    end = time.time()
    avg = (end - start) / repeat

    print(f"{name:30s} | Avg: {avg:.6f} sec")
    return avg


# ---------------------------------------------------------
# 3. Huge Matrix Multiplication Benchmark
# ---------------------------------------------------------
def matmul_benchmark():
    print("Running Matrix Multiplication Benchmark...")

    # 8192 x 8192 matrices (≈1.1B elements)
    size = 8192
    A = torch.randn((size, size), device="cuda")
    B = torch.randn((size, size), device="cuda")

    # FLOPS estimation: 2 * N^3
    flops = 2 * (size ** 3)

    def run():
        torch.matmul(A, B)

    t = benchmark_step("MatMul 8192x8192", run, repeat=5)
    tflops = (flops / t) / 1e12

    print(f"Estimated Throughput: {tflops:.2f} TFLOPS")
    return tflops


# ---------------------------------------------------------
# 4. Convolution Benchmark (ResNet-like)
# ---------------------------------------------------------
def conv_benchmark():
    print("\nRunning Convolution Benchmark...")

    batch = 32
    x = torch.randn((batch, 3, 224, 224), device="cuda")
    conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3).cuda()

    def run():
        conv(x)

    t = benchmark_step("Conv2d 7x7 on 32x224x224", run, repeat=10)
    return 1 / t


# ---------------------------------------------------------
# 5. Memory Bandwidth Test
# ---------------------------------------------------------
def memory_bandwidth_benchmark():
    print("\nRunning Memory Bandwidth Test...")

    size = 2_000_000_000  # 2 billion bytes (~2 GB)
    a = torch.empty(size, dtype=torch.float32, device='cuda')

    def run():
        b = a.clone()  # GPU -> GPU memory copy

    t = benchmark_step("2GB GPU memcpy", run, repeat=10)

    bandwidth = (size * 4) / t / 1e9  # GB/s
    print(f"Memory Bandwidth: {bandwidth:.2f} GB/s")

    return bandwidth


# ---------------------------------------------------------
# 6. Full benchmark & report
# ---------------------------------------------------------
def main():
    system_report()

    if not torch.cuda.is_available():
        print("CUDA not available! Cannot run GPU benchmarks.")
        return

    print("Starting full GPU benchmark...\n")

    results = {}

    results["tflops"] = matmul_benchmark()
    results["conv_speed"] = conv_benchmark()
    results["bandwidth"] = memory_bandwidth_benchmark()

    print("\n=================== FINAL REPORT ===================")
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")
    print(f"Compute Capability: {torch.cuda.get_device_properties(0).major}."
          f"{torch.cuda.get_device_properties(0).minor}\n")

    print(f"Matrix Multiply (TFLOPS): {results['tflops']:.2f}")
    print(f"Conv2D speed (1/sec)   : {results['conv_speed']:.3f}")
    print(f"Memory Bandwidth (GB/s): {results['bandwidth']:.1f}")

    print("\nAnalysis:")

    # --- Analysis Section ---
    if results["tflops"] > 200:
        print("• Matrix multiply speed is extremely high → GPU driver & CUDA working.")
    else:
        print("• MatMul below expected range → possible driver or power issue.")

    if results["bandwidth"] > 800:
        print("• Memory bandwidth is strong → confirms PCIe + VRAM configuration OK.")
    else:
        print("• Low bandwidth → PCIe speed or driver installation might be wrong.")

    print("• Your GPU is benchmarking successfully using PyTorch.\n")
    print("====================================================")


# ---------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
