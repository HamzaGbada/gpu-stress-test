# GPU Stress Test & Evaluation Pipeline

Multi-step stress testing, benchmarking and **pass / warn / fail evaluation** for NVIDIA GPUs,
from a 6 GB laptop RTX 4050 to an RTX 5090. Two editions share one pipeline engine:

| edition | install size | what it needs | what it runs |
|---|---|---|---|
| **lite** (`gpu-stress-lite`) | ~100 KB, **zero pip dependencies** | only the NVIDIA driver (`libcuda`, `libnvidia-ml`) | hand-written CUDA kernels: VRAM memtest, bandwidth, verified SGEMM, sustained FP32 / FP16 / tensor-core burns, allocation churn |
| **torch** (`gpu-stress`) | several GB (`torch`, `torchvision`) | CUDA-enabled PyTorch | cuBLAS matmul in fp32/tf32/bf16/fp16, `torch.compile`, sustained matmul burn, **ResNet50 training with automatic memory budgeting**, VRAM fill, allocator fragmentation |

Every step runs inside a telemetry window (NVML: utilisation, VRAM, temperature, power, SM/mem clocks,
throttle reasons) and is scored against the GPU's own theoretical peaks. Results go to
`results/<backend>_<timestamp>.{json,md,log}` plus `_metrics.csv`, `_telemetry.csv` and an optional
telemetry PNG.

```
$ gpu-stress-lite --burn 60
...
########################################################################
#  FINAL REPORT (lite)  -  overall: PASS
########################################################################
step                   status      time   maxT    avgW  key metrics
system_info            PASS        0.0s      -       -
memtest                PASS        0.4s    44°       4  errors=0, tested_gb=4.773
bandwidth              PASS        0.6s    47°      11  gbs=173.800
matmul                 PASS        0.6s    44°       4  tflops=3.950, errors=0
fp32_burn              PASS       60.0s    60°      56  tflops=10.978, errors=0
fp16_burn              PASS       60.0s    60°      47  tflops=12.958, errors=0
tensor_burn            PASS       60.0s    57°      32  tflops=26.229, errors=0
alloc_churn            PASS        0.5s    50°      30
```

---

## Install

```bash
# lite edition - nothing to download, works anywhere nvidia-smi works
pip install .                  # or: uv sync
gpu-stress-lite

# full PyTorch edition (CUDA 13.0 wheels from the pytorch index)
pip install ".[torch,plot]"    # or: uv sync --extra torch --extra plot
gpu-stress
```

Without installing: `python -m gpu_stress.lite_pipeline` / `python -m gpu_stress.torch_pipeline`.

Optional `matplotlib` (`[plot]` extra) enables the telemetry PNG and `--gui`.

---

## Lite edition (no PyTorch, no CUDA toolkit)

`gpu_stress/cuda/kernels.cu` is compiled once to PTX (`kernels_sm70.ptx`, `kernels_sm80.ptx`, shipped in
the package). At runtime `gpu_stress/cuda/driver.py` talks to `libcuda.so` through `ctypes`: the driver
JIT-compiles the PTX for whatever GPU is present (Volta -> Blackwell), so **no toolkit, nvcc, numba or
PyTorch is needed**. Telemetry comes from `libnvidia-ml.so`, also via ctypes.

| step | what it does | evaluated |
|---|---|---|
| `system_info` | device, driver, theoretical FP32 TFLOPS & bandwidth | - |
| `memtest` | writes/reads 6+ bit patterns over `--vram` of the free VRAM (mini memtest) | any mismatch = FAIL |
| `bandwidth` | device-to-device copy (custom kernel + `cuMemcpyDtoD`) | % of theoretical bus bandwidth |
| `matmul` | register-tiled SGEMM, sampled outputs verified exactly on the host | mismatches, % of FP32 peak |
| `fp32_burn` | `--burn` seconds of FMA chains that must converge to 1.0 | ALU errors, efficiency at observed clock, start-vs-end throughput |
| `fp16_burn` | same with packed `fma.rn.f16x2` | idem |
| `tensor_burn` | `mma.sync.m16n8k16` accumulators with an exactly-known result (sm_80+) | idem |
| `alloc_churn` | alloc/free latency 16 MiB-1 GiB + 256 live blocks | slow allocations |

```
gpu-stress-lite --burn 120 --vram 0.95            # longer burn, test 95% of free VRAM
gpu-stress-lite --steps memtest,tensor_burn       # subset
gpu-stress-lite --skip fp16_burn
gpu-stress-lite --nvrtc                           # compile kernels for this exact GPU if libnvrtc is installed
gpu-stress-lite --list
```

Kernels changed? Rebuild the shipped PTX with any CUDA 12.x toolkit:
`NVRTC_LIB=/usr/local/cuda-12.3/lib64/libnvrtc.so python gpu_stress/cuda/nvrtc.py`
(PTX ISA 8.3 -> works with driver R545+; do not build with CUDA 13 unless every target has an R580+ driver).

---

## Torch edition

| step | what it does | evaluated |
|---|---|---|
| `system_info` | torch / CUDA / cuDNN / device | - |
| `bandwidth` | `copy_` of up to 1 GiB | % of theoretical |
| `matmul` | `--matmul-size` (8192, auto-reduced) in fp32 / tf32 / bf16 / fp16, checked against fp32 | precision errors, cuBLAS efficiency, tensor-core speedup |
| `compile` | `torch.compile` vs eager MLP (skipped if triton is missing) | speedup < 0.8x |
| `matmul_burn` | `--burn` seconds of `--precision` matmul | throughput drop, non-finite outputs |
| `training` | ResNet50 + Adam on a synthetic dataset **kept in VRAM** | batch reductions, OOM recoveries, NaN loss, speed drop |
| `vram_fill` | fills `--vram` of free memory in 256 MiB chunks, 4 patterns verified | mismatches |
| `fragmentation` | alloc/free latency 128 MiB-2 GiB, allocator retries | - |

```
gpu-stress --epochs 3 --batch 64                        # defaults
gpu-stress --precision bf16 --batch 128 --burn 120      # tensor cores, half the activation memory
gpu-stress --steps training --epochs 10 --max-steps 200
python gpu_stress_cli.py --epochs 10 --batch 64 --vram 0.75 --nogui   # old CLI still works
```

### The OOM fix (`CUDA out of memory ... GPU 0 has a total capacity of 5.67 GiB`)

The old script computed the dataset size as `--vram` x *total* VRAM (75% = 4.25 GiB on a 6 GB card) and
allocated it **before** the model. ResNet50 + Adam + fp32 activations at batch 64 need ~5 GiB by themselves,
so anything smaller than ~16 GB crashed; the RTX 5090 only worked because 25% of 32 GB happened to be enough.

`gpu_stress/torch_steps.py::step_training` now:

1. runs two probe training steps at the requested batch, **halving the batch until it fits**, and measures the
   persistent (weights + grads + Adam state) and transient (activations + cuDNN workspace) memory;
2. sizes the dataset from what is **actually free** after that (`free - transient - 6% headroom`), still capped by
   `--vram` x total;
3. recovers from a late OOM (fragmentation) by halving the batch and continuing, and reports it as a warning;
4. sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing torch, as the error message suggests.

`--precision bf16|fp16` runs autocast training, which roughly halves activation memory and exercises the tensor cores.

---

## Evaluation rules (`gpu_stress/evaluate.py`)

Applied to every heavy step from telemetry:

* temperature within 8 °C of the NVML slowdown threshold -> WARN, at the threshold -> FAIL
* `hw_slowdown` / `hw_thermal_slowdown` / `sw_thermal_slowdown` / `hw_power_brake` active -> WARN (FAIL if >=10% of samples); `sw_power_cap` is only informational (normal on laptops)
* SM clock dipping below 50% of its peak in the window, or averaging under 60% of the rated clock -> WARN

Per step: compute-result mismatches -> FAIL; efficiency versus the theoretical peak *at the observed clock*
(SMs x cores x 2 x MHz, or mem-clock x 2 x bus width) -> WARN / FAIL below step-specific thresholds;
throughput dropping >10% (WARN) / >25% (FAIL) between the first and last fifth of a sustained run.

Exit code: 0 = pass/warn, 1 = fail, 2 = error. Overall status is the worst step.

---

## Project layout

```
gpu_stress/
├── pipeline.py        step runner (telemetry windows, OOM/error isolation, status roll-up)
├── evaluate.py        theoretical peaks + pass/warn/fail rules
├── monitor.py         background NVML sampler, per-step summaries
├── nvml.py            ctypes NVML binding (replaces pynvml)
├── report.py          console summary, JSON / Markdown / CSV / PNG
├── lite_steps.py      driver-API steps
├── lite_pipeline.py   `gpu-stress-lite` CLI
├── torch_steps.py     PyTorch steps (memory-budgeted ResNet50 training)
├── torch_pipeline.py  `gpu-stress` CLI
└── cuda/
    ├── driver.py      ctypes binding for libcuda (contexts, modules, memory, events, launches)
    ├── nvrtc.py       optional runtime compilation + PTX rebuild script
    ├── kernels.cu     the CUDA C kernels
    └── kernels_sm70.ptx / kernels_sm80.ptx   shipped, driver-JIT'ed at runtime
gpu_stress_cli.py      legacy CLI (wrapper around the torch pipeline)
gpu_bench.py, gpu_bench_details.py   original single-shot benchmarks
```

## License

MIT
