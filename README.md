# RTX 5090 GPU Stress Test & Benchmark Suite

A full GPU stress-testing and benchmarking suite for NVIDIA RTX-class GPUs with PyTorch, torch.compile, real-time monitoring and VRAM load tests.
This suite includes:

* **High-level performance benchmarks** (matmul TFLOPs, mixed precision, torch.compile, memory fragmentation)
* **Deep learning training stress test** (ResNet50)
* **Real-time GPU monitoring graphs**
* **CLI for configuring stress levels**
* **Full VRAM-filling synthetic dataset**
* **Logs + results export to JSON & CSV**

Perfect for:

* GPU stability testing
* Thermal & power load validation
* Benchmarking CUDA & PyTorch performance
* Fan curve tuning
* Hardware validation after overclocking/undervolting

---

# Project Structure

```
.
├── benchmark_results.csv        # Output from gpu_bench.py
├── benchmark_results.json       # Output from gpu_bench.py
├── gpu_bench.py                 # Standard GPU benchmark suite
├── gpu_bench_details.py         # Extended benchmark with detailed metrics
├── gpu_stress_cli.py            # MAIN stress-test CLI with graph & logging
├── logs/                        # Auto-generated training logs
├── results/                     # Auto-generated metrics (JSON)
├── pyproject.toml               # Python project metadata (if using uv/poetry)
└── uv.lock                      # Dependency lockfile
```

---

# Features

### GPU Benchmarks (gpu_bench_details.py)

Includes:

* System & GPU info dump
* `torch.compile` performance
* Matmul TFLOPs in FP32 / BF16 / FP16 / TF32
* Full ResNet50 training benchmark
* Memory fragmentation test
* JSON + CSV output

---

### Advanced Stress Test (gpu_stress_cli.py)

This is the main stress testing system:

#### GPU Stress Training:

* ResNet50 training loop
* Multiple epochs (configurable)
* Huge synthetic dataset allocated directly in VRAM
* Fills **75–95% of GPU memory**
* Pushes SMs, VRAM, Tensor Cores, power circuits

#### Real-Time GPU Monitoring:

Live metrics window showing:

* GPU Utilization %
* VRAM Usage %
* Temperature °C
* Power draw W

Uses **pynvml** + **matplotlib** with background monitor thread.

#### CLI Interface:

```
--epochs N        # Number of training epochs
--batch N         # Batch size
--vram F          # VRAM percentage to allocate (0.0–1.0)
--nogui           # Disable real-time graph (for servers)
```

#### Logs & Results:

Auto-generated:

```
logs/training_log_YYYYMMDD_HHMMSS.txt
results/metrics_YYYYMMDD_HHMMSS.json
```

Contains:

* Step times
* Epoch durations
* Loss curve
* GPU metrics over time
* Dataset size
* Steps/sec
* Full training configuration

---

# Installation

## Install dependencies (pip)

```bash
pip install torch torchvision matplotlib pynvml psutil
```

If using **uv**:

```bash
uv sync
```

---

#  Usage

## 1. Run Standard Benchmark Suite

Produces benchmark_results.json + CSV.


```bash
python gpu_bench_details.py
```

---

## 2. Run the Real Stress Test (Main Tool)

```bash
python gpu_stress_cli.py
```

This launches:

* Full VRAM-filling dataset allocation
* ResNet50 training
* Real-time GPU graphs

---

#  CLI Options

### Example:

```bash
python gpu_stress_cli.py --epochs 20 --batch 96 --vram 0.85
```

### All Options:

| Flag         | Description                                  |
| ------------ | -------------------------------------------- |
| `--epochs N` | Number of epochs to run (default: 10)        |
| `--batch N`  | Batch size (default: 64)                     |
| `--vram F`   | Fraction of VRAM to allocate (default: 0.75) |
| `--nogui`    | Disable real-time graph window               |

---

# Output Files

### Benchmark Suite:

```
benchmark_results.json
benchmark_results.csv
```

### Stress Test:

```
logs/training_log_*.txt
results/metrics_*.json
```

### JSON metrics include:

* epochs
* batch size
* dataset size
* epoch times
* steps/sec
* full loss curve
* GPU util / temp / power / VRAM usage over time

---

# How the Stress Test Works

1. Detects your GPU VRAM
2. Allocates a synthetic dataset to fill X% of it
3. Runs multiple epochs of ResNet50 training
4. Runs a background GPU monitor thread
5. Streams live graphs for GPU metrics
6. Logs everything to disk
7. Saves metrics as JSON for post-analysis

This creates **maximum GPU load**, including:

* FP32/TF32 Tensor Core load
* Gradient accumulation
* VRAM pressure
* High sustained utilization
* High temperature & power usage

Equivalent to:

* Heavy stable diffusion training
* Running a large model
* Memory allocator stress
* Continuous math kernel workloads

---

# Safety Notes

⚠ **This test is extremely demanding.**
Ensure:

* Adequate cooling
* Good PSU health
* Avoid running on laptops
* Disable GPU overclocks if unstable

The test pushes:

* VRAM to near full utilization
* Power draw to 100%
* Tensor cores constantly
* Temperature to thermal limits

---

# Credits

Developed using:

* PyTorch
* TorchVision
* NVIDIA NVML
* Matplotlib
* Python 3.10+

Optimized and validated on **NVIDIA RTX 5090**.
