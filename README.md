# GPU Stress Test & Evaluation Pipeline

Multi-step stress testing, benchmarking and **pass / warn / fail evaluation** for NVIDIA GPUs —
from a 6 GB laptop RTX 4050 to an RTX 5090 or a 128 GB DGX Spark.

Most stress tools tell you a number. This one tells you whether the number is *right*:
every compute step has an exactly known answer or a conserved physical quantity, and every
sustained load is scored against the GPU's own theoretical peak **at the clock it actually ran at**,
so a power-capped laptop isn't reported as broken hardware.

```
$ gpu-stress-lite --burn 10 --physics 10
########################################################################
#  FINAL REPORT (lite)  -  overall: PASS
########################################################################
step                   status      time   maxT    avgW  key metrics
system_info            PASS        0.0s      -       -
memtest                PASS        0.4s    52°       6  errors=0, tested_gb=4.774, write_gbs=163.100
bandwidth              PASS        0.6s    54°      16  gbs=159.700
pcie                   PASS        0.4s    53°      37  gbs=13.370, pinned_h2d_gbs=11.940
matmul                 PASS        0.5s    59°      38  tflops=3.647, errors=0
edge_cases             PASS        0.0s      -       -  errors=0, threads=40960
fp32_burn              PASS       10.1s    66°      56  tflops=10.363, errors=0
fp16_burn              PASS       10.0s    66°      47  tflops=12.269, errors=0
tensor_burn            PASS       10.0s    64°      33  tflops=25.321, errors=0
nbody                  PASS       10.1s    70°      57  tflops=5.351, bodies=20480, momentum_rel=3.16e-04
stencil                PASS       10.0s    70°      60  gbs=175.500, side=8192, sum_drift_rel=2.48e-10
alloc_churn            PASS        0.5s    65°      54
```

(a real run on a 6 GB RTX 4050 Laptop; `--burn`/`--physics` default to 30 s and 15 s)

---

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/HamzaGbada/gpu-stress-test/main/install.sh | sh
```

The installer uses `uv`, `pipx` or `pip` if one is present, and otherwise drops a standalone
**single-file zipapp** (180 KB) into `~/.local/bin/gpu-stress-lite` — no pip, no virtualenv, no
dependencies. Set `GPU_STRESS_METHOD=zipapp` to force that path, or `GPU_STRESS_VERSION=v0.3.0`
to pin a release.

<details>
<summary>Other ways to install</summary>

```bash
# from a checkout
pip install .                    # lite edition, zero dependencies
pip install ".[torch,plot]"      # + PyTorch suite and telemetry plots
uv sync --extra torch --extra plot

# straight from git, no clone
uv tool install "git+https://github.com/HamzaGbada/gpu-stress-test@v0.3.0"

# single file, nothing installed
curl -fsSLO https://github.com/HamzaGbada/gpu-stress-test/releases/latest/download/gpu-stress.pyz
chmod +x gpu-stress.pyz && ./gpu-stress.pyz --burn 30

# no install at all
python -m gpu_stress.lite_pipeline
```
</details>

**Requirements:** Linux, NVIDIA driver R545+, Python 3.10+. No CUDA toolkit. No PyTorch.
See [docs/TESTED_GPUS.md](docs/TESTED_GPUS.md) for the hardware this has been verified on.

---

## The two editions

| | **lite** (`gpu-stress-lite`) | **torch** (`gpu-stress`) |
|---|---|---|
| Install size | ~180 KB, **zero pip dependencies** | several GB (`torch`, `torchvision`) |
| Needs | only the NVIDIA driver | CUDA-enabled PyTorch |
| How | hand-written CUDA kernels shipped as PTX, driver API via `ctypes` | the real ML stack |
| Good for | headless servers, fresh installs, hardware validation, CI | reproducing ML workloads, tensor-core throughput via cuBLAS |

Both share one engine: the same telemetry monitor, evaluator, report writer and exit codes.

---

## Lite edition

`gpu_stress/cuda/kernels.cu` is compiled once to PTX (`kernels_sm70.ptx`, `kernels_sm80.ptx`,
both shipped in the package). At runtime `gpu_stress/cuda/driver.py` talks to `libcuda.so`
through `ctypes` and the driver JIT-compiles the PTX for whatever GPU is present — Volta
through Blackwell, including architectures newer than this release. Telemetry comes from
`libnvidia-ml.so`, also via ctypes.

| Step | What it does | How it is judged |
|---|---|---|
| `system_info` | device, driver, theoretical peaks, PCIe training | flags unified memory, missing peaks, a down-trained link |
| `memtest` | writes/reads bit patterns over `--vram` of free VRAM | any mismatch = **FAIL** |
| `bandwidth` | device-to-device copy (custom kernel + `cuMemcpyDtoD`) | % of theoretical bus bandwidth |
| `pcie` | host↔device transfers, pinned vs pageable | % of the current PCIe training; warns if the link is below the card's max |
| `matmul` | register-tiled SGEMM, sampled outputs verified exactly on the host | mismatches, % of FP32 peak |
| `edge_cases` | 11 IEEE-754 / integer / warp / atomic corner cases | any wrong answer = **FAIL** |
| `fp32_burn` | `--burn` seconds of FMA chains that must converge to exactly 1.0 | ALU errors, efficiency at observed clock, start-vs-end drop |
| `fp16_burn` | same with packed `fma.rn.f16x2` | idem |
| `tensor_burn` | `mma.sync.m16n8k16` with an exactly-known accumulator (sm_80+) | idem |
| `nbody` | N-body gravity, `--physics` seconds | **momentum conservation**, throughput, stability |
| `stencil` | 2D heat diffusion, `--physics` seconds | **heat conservation + maximum principle**, effective bandwidth |
| `alloc_churn` | alloc/free latency 16 MiB–1 GiB, 256 live blocks | slow allocations |

```bash
gpu-stress-lite --list                        # show the steps
gpu-stress-lite --burn 120 --physics 60       # long soak
gpu-stress-lite --steps memtest,edge_cases    # just the correctness checks
gpu-stress-lite --skip tensor_burn
gpu-stress-lite --vram 0.95 --mem-max-gb 32   # big-VRAM machines
gpu-stress-lite --nvrtc                       # compile kernels for this exact GPU if libnvrtc exists
```

### Why physics

A matmul benchmark that returns garbage still returns a number. A simulation governed by a
conservation law does not have that freedom:

* **N-body** — forces between two bodies are equal and opposite, so total momentum is constant.
  The simulation starts from exactly zero net momentum; if the GPU miscomputes a single force,
  the symmetry breaks and momentum drifts away from zero. Compute-bound, hammers the FMA pipes
  and shared memory.
* **Heat diffusion** — with periodic boundaries the total heat is conserved, and for α ≤ 0.25
  each update is a convex combination of neighbours, so the field can *never* leave its initial
  range (the maximum principle). Memory-bound, so it stresses the memory system rather than the ALUs.

Both invariants hold exactly in real arithmetic; in fp32 they hold to rounding. A violation
beyond that is the hardware, not the algorithm.

### Rebuilding the kernels

Only needed if you edit `kernels.cu`:

```bash
NVRTC_LIB=/usr/local/cuda-12.3/lib64/libnvrtc.so python gpu_stress/cuda/nvrtc.py
```

Build with a **CUDA 12.x** NVRTC: that emits PTX ISA 8.3, which any R545+ driver accepts.
CUDA 13 emits a newer ISA that older drivers reject.

---

## Torch edition

| Step | What it does | How it is judged |
|---|---|---|
| `system_info` | torch / CUDA / cuDNN / device | — |
| `bandwidth` | `copy_` of up to 1 GiB | % of theoretical |
| `matmul` | `--matmul-size` in fp32 / tf32 / bf16 / fp16, checked against fp32 | precision errors, cuBLAS efficiency, tensor-core speedup |
| `compile` | `torch.compile` vs eager (skipped without triton) | speedup < 0.8× |
| `matmul_burn` | `--burn` seconds at `--precision` | throughput drop, non-finite outputs |
| `training` | ResNet50 + Adam on a synthetic dataset kept in VRAM | batch reductions, OOM recoveries, NaN loss, speed drop |
| `vram_fill` | fills `--vram` of free memory, 4 patterns verified | mismatches |
| `fragmentation` | alloc/free latency 128 MiB–2 GiB, allocator retries | — |

```bash
gpu-stress --epochs 3 --batch 64
gpu-stress --precision bf16 --batch 128 --burn 120
gpu-stress --steps training --epochs 10 --max-steps 200
python gpu_stress_cli.py --epochs 10 --batch 64 --vram 0.75 --nogui   # legacy CLI still works
```

### The training OOM fix

> `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 196.00 MiB. GPU 0 has a total
> capacity of 5.67 GiB of which 50.38 MiB is free.`

The old script sized the synthetic dataset as `--vram × **total** VRAM` (75% = 4.25 GiB on a 6 GB
card) and allocated it **before** the model. ResNet50 + Adam + fp32 activations at batch 64 need
~5 GiB on their own, so anything below ~16 GB crashed; a 5090 only worked because the remaining
25% of 32 GB happened to be enough.

`step_training` now:

1. runs two probe training steps at the requested batch, **halving it until it fits**, and measures
   persistent (weights + grads + Adam state) versus transient (activations + cuDNN workspace) memory;
2. sizes the dataset from what is **actually free** afterwards (`free − transient − 6% headroom`),
   still capped by `--vram × total`;
3. recovers from a late OOM by halving the batch and continuing, reporting it as a warning;
4. sets `expandable_segments:True` before torch is imported, as the error message suggests.

`--precision bf16|fp16` roughly halves activation memory and exercises the tensor cores.

---

## Evaluation rules

Applied to every sustained step from telemetry:

* temperature within 8 °C of the NVML slowdown threshold → **WARN**; at the threshold → **FAIL**
* `hw_slowdown` / `*_thermal_slowdown` / `hw_power_brake` active → **WARN**, or **FAIL** above 10% of
  samples; `sw_power_cap` is informational (normal on laptops and TDP-limited boards)
* SM clock dipping below 50% of its peak, or averaging under 60% of the rated clock → **WARN**
  (informational when the board is power-capped)
* throttle reasons are only counted while the GPU is actually loaded — a parked GPU reports
  thermal reasons that mean nothing

Per step: any wrong compute result → **FAIL**; efficiency versus the theoretical peak at the
*observed* clock (SMs × cores × 2 × MHz, or mem-clock × 2 × bus width, or PCIe lanes × per-lane rate);
throughput dropping >10% (**WARN**) / >25% (**FAIL**) between the first and last fifth of a sustained run.

**Exit code:** 0 = pass/warn, 1 = fail, 2 = error. Overall status is the worst step.

**Reports:** `results/<backend>_<timestamp>.{json,md,log}` plus `_metrics.csv`, `_telemetry.csv` and,
with `matplotlib` installed, a telemetry PNG (`--gui` to show it live).

---

## Unified-memory parts (GB10 / DGX Spark, Jetson, iGPUs)

These share one memory pool between CPU and GPU, and the driver reports a **0-bit bus width**,
so there is no theoretical bandwidth to score against. The suite detects this and reports measured
throughput without an invented percentage, skips `pcie` (there is no link to test), and warns that
free memory depends on system load. On a 128 GB machine, `--vram 0.9` means a ~103 GiB buffer whose
first touch has to fault in every page — `--mem-max-gb` caps it and `--mem-budget` stops the pattern
sweep when it runs long. Details in [docs/TESTED_GPUS.md](docs/TESTED_GPUS.md).

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
    ├── driver.py      ctypes libcuda binding (contexts, modules, memory, events, occupancy)
    ├── nvrtc.py       optional runtime compilation + PTX rebuild script
    ├── kernels.cu     the CUDA C kernels
    └── kernels_sm70.ptx / kernels_sm80.ptx   shipped, driver-JIT'd at runtime
scripts/build_zipapp.py   builds the single-file gpu-stress.pyz
install.sh                curl | sh installer
gpu_stress_cli.py         legacy CLI (wrapper around the torch pipeline)
gpu_bench.py, gpu_bench_details.py   original single-shot benchmarks
```

## Contributing

Ran it on a GPU that isn't in [the table](docs/TESTED_GPUS.md)? Open an issue or PR with the JSON
report. Bug reports are most useful with `results/<backend>_<timestamp>.json` attached.

## License

MIT — see [LICENSE](LICENSE).
