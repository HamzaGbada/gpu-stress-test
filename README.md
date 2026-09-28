<div align="center">

# GPU Stress Test &amp; Evaluation Pipeline

**Stress-test, benchmark, and actually verify NVIDIA GPUs.**

[![CI](https://github.com/HamzaGbada/gpu-stress-test/actions/workflows/ci.yml/badge.svg)](https://github.com/HamzaGbada/gpu-stress-test/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![NVIDIA Driver R545+](https://img.shields.io/badge/NVIDIA%20driver-R545%2B-76b900.svg)](https://www.nvidia.com/drivers)
[![Version](https://img.shields.io/badge/version-0.3.1-informational.svg)](CHANGELOG.md)

Correctness checks, memory tests, sustained load, physics-based workloads and live telemetry —
reduced to a single verdict: **PASS / WARN / FAIL**.

</div>

---

Most GPU benchmarks answer *"how fast is this GPU?"*. This one also answers
*"is it producing **correct** results while doing it?"*

Every compute workload here has a known result, a numerical invariant, or a physical conservation
law that can be checked. Sustained workloads are scored against the GPU's **observed clock** rather
than its advertised maximum, so a power-capped laptop is not mistaken for failing hardware.

```text
$ gpu-stress-lite --burn 10 --physics 10

########################################################################
#  FINAL REPORT (lite)  -  overall: PASS
########################################################################

step          status    time    maxT    avgW    key metrics
system_info   PASS      0.0s      -       -
memtest       PASS      0.4s     52°       6    errors=0, tested_gb=4.774, write_gbs=163.100
bandwidth     PASS      0.6s     54°      16    gbs=159.700
pcie          PASS      0.4s     53°      37    gbs=13.370, pinned_h2d_gbs=11.940
matmul        PASS      0.5s     59°      38    tflops=3.647, errors=0
edge_cases    PASS      0.0s      -       -     errors=0, threads=40960
fp32_burn     PASS     10.1s     66°      56    tflops=10.363, errors=0
fp16_burn     PASS     10.0s     66°      47    tflops=12.269, errors=0
tensor_burn   PASS     10.0s     64°      33    tflops=25.321, errors=0
nbody         PASS     10.1s     70°      57    momentum_rel=3.16e-04
stencil       PASS     10.0s     70°      60    sum_drift_rel=2.48e-10
alloc_churn   PASS      0.5s     65°      54
```

<sub>Real run on an RTX 4050 Laptop (6 GB, 60 W). `--burn` and `--physics` default to 30 s and 15 s.</sub>

## Contents

- [Install](#install)
- [Hardware support](#hardware-support)
- [Usage](#usage)
- [What it tests](#what-it-tests)
- [Two editions](#two-editions)
- [How results are judged](#how-results-are-judged)
- [Reports](#reports)
- [Using it in CI](#using-it-in-ci)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/HamzaGbada/gpu-stress-test/main/install.sh | sh
gpu-stress-lite --burn 30 --physics 15
```

No CUDA toolkit. No PyTorch. No multi-gigabyte Python environment.

### What you get

The installer sets up the **lite pipeline** by default, because it is dependency-free and runs
anywhere a driver does. The PyTorch pipeline is an opt-in extra worth several GB of wheels:

| Command | Installed by default | What it needs |
|---|:---:|---|
| `gpu-stress-lite` | ✅ | NVIDIA driver only |
| `gpu-stress` | opt-in | `GPU_STRESS_EXTRAS=torch` (pulls `torch` + `torchvision`) |

```bash
# both pipelines, plus telemetry plots
GPU_STRESS_EXTRAS=torch,plot curl -fsSL https://raw.githubusercontent.com/HamzaGbada/gpu-stress-test/main/install.sh | sh
```

Running `gpu-stress` without the extra prints the one-line command to add it rather than a traceback.

### Installer options

| Variable | Default | Purpose |
|---|---|---|
| `GPU_STRESS_EXTRAS` | *(none)* | Extras to include, e.g. `torch,plot` |
| `GPU_STRESS_METHOD` | `auto` | Force `uv`, `pipx`, `pip` or `zipapp` |
| `GPU_STRESS_VERSION` | latest release | Pin a tag, e.g. `v0.3.1`, or `main` |
| `GPU_STRESS_BIN` | `~/.local/bin` | Install directory for the zipapp |

The installer prefers `uv`, then `pipx`, then `pip`. With none of them available it falls back to a
standalone **~180 KB zipapp** — a single executable file with no pip, virtualenv or dependencies.

<details>
<summary>Other ways to install</summary>

```bash
# from a checkout
pip install .                    # lite edition, zero dependencies
pip install ".[torch,plot]"      # + PyTorch pipeline and telemetry plots
uv sync --extra torch --extra plot

# straight from git, no clone
uv tool install "git+https://github.com/HamzaGbada/gpu-stress-test@v0.3.1"
uv tool install "gpu-stress[torch] @ git+https://github.com/HamzaGbada/gpu-stress-test@v0.3.1"

# single file, nothing installed
curl -fsSLO https://github.com/HamzaGbada/gpu-stress-test/releases/latest/download/gpu-stress.pyz
chmod +x gpu-stress.pyz && ./gpu-stress.pyz --burn 30
./gpu-stress.pyz torch --epochs 3     # torch pipeline, if torch is importable

# no install at all
python -m gpu_stress.lite_pipeline
```

</details>

### Requirements

|  | Lite | Torch |
|---|---|---|
| OS | Linux | Linux |
| Driver | NVIDIA R545+ | NVIDIA + CUDA-enabled PyTorch |
| Python | 3.10+ | 3.10+ |
| CUDA toolkit | not required | not required |

## Hardware support

The lite pipeline ships PTX for `sm_70` and `sm_80` and lets the driver JIT-compile it for the GPU
actually present, so the same artifact runs from **Volta through Blackwell** — including
architectures newer than this release.

| GPU / platform | Architecture | Memory | Lite | Torch | Status |
|---|---|---:|:---:|:---:|---|
| RTX 4050 Laptop | Ada (sm_89) | 6 GB | ✅ | ✅ | 🟢 Verified |
| GB10 / NVIDIA DGX Spark | Blackwell (sm_121) | 128 GB unified | ✅ | — | 🟢 Verified (lite) |
| RTX 5090 | Blackwell (sm_120) | 32 GB | — | ✅ | 🔷 Reported working |
| RTX 40 / 50-series | Ada / Blackwell | varies | ✅ | ✅ | 🔵 Supported |
| A100 · H100 · H200 · B200 | Ampere / Hopper / Blackwell | 40–192 GB | ✅ | ✅ | 🔵 Supported |
| Jetson, other unified memory | varies | shared | ✅ | ⚠️ | 🔵 Supported |
| Volta · Turing | sm_70 / sm_75 | varies | ✅¹ | ✅ | 🔵 Supported |
| Pascal and older | ≤ sm_61 | varies | ⚠️² | ✅ | 🟡 Rebuild required |

<sub>
🟢 <b>Verified</b> — a full run is on file in <a href="docs/TESTED_GPUS.md">docs/TESTED_GPUS.md</a> ·
🔷 <b>Reported</b> — reported working, no report filed yet ·
🔵 <b>Supported</b> — compatible with the shipped PTX, not yet exercised<br>
¹ no <code>tensor_burn</code>: <code>mma.sync</code> needs sm_80+ &nbsp;·&nbsp;
² rebuild the kernels with <code>--nvrtc</code>
</sub>

**Unified-memory platforms** (DGX Spark / GB10, Jetson, iGPUs) report a 0-bit memory bus width
because there is no dedicated GPU bus to describe. Rather than invent a percentage against a
nonexistent peak, the suite reports measured throughput, skips `pcie` (host and device are the same
memory), and notes that free memory is shared with the OS. See
[docs/TESTED_GPUS.md](docs/TESTED_GPUS.md) for the specifics.

## Usage

```bash
gpu-stress-lite                                  # full default suite
gpu-stress-lite --list                           # list the steps
gpu-stress-lite --burn 120 --physics 120         # soak test
gpu-stress-lite --steps memtest,edge_cases       # correctness only, fast and cool
gpu-stress-lite --skip tensor_burn               # Volta / Turing
gpu-stress-lite --vram 0.95 --mem-max-gb 80      # large-memory parts
gpu-stress-lite --nvrtc                          # compile kernels for this exact GPU
```

```bash
gpu-stress --epochs 3 --batch 64                 # PyTorch pipeline
gpu-stress --precision bf16 --burn 120           # tensor cores, half the activation memory
```

## What it tests

### Memory and interconnect

| Step | What it does | Fails when |
|---|---|---|
| `system_info` | Device, driver, theoretical peaks, PCIe training | *(informational)* |
| `memtest` | Writes and reads bit patterns across free VRAM | Any single value reads back wrong |
| `bandwidth` | Device-to-device copy, kernel and `cuMemcpyDtoD` | Far below the theoretical bus peak |
| `pcie` | Host↔device, pinned and pageable | Link trained below the card's maximum gen/width |
| `alloc_churn` | Allocation and free latency, 16 MiB → 1 GiB | Allocations become pathologically slow |

### Compute correctness

| Step | What it verifies |
|---|---|
| `matmul` | Register-tiled SGEMM; sampled outputs checked **bit-exactly** against the host |
| `edge_cases` | 11 IEEE-754 / integer / warp / atomic corner cases, each with one right answer |
| `fp32_burn` | FMA chains that must converge to exactly `1.0` |
| `fp16_burn` | Packed `fma.rn.f16x2` under the same invariant |
| `tensor_burn` | `mma.sync.m16n8k16` with an exactly known accumulator (sm_80+) |

`edge_cases` covers denormals, NaN comparison semantics, infinity arithmetic,
round-to-nearest-even, genuinely *fused* FMA, `sqrtf` of -1/0/inf, 32- and 64-bit integer
intrinsics, warp shuffle and ballot, shared-memory reductions, and four kinds of atomics checked
for lost updates across every thread.

### Physics-based stress

Instead of a blind `while True: matmul()`, two steps run workloads where physics supplies something
to verify. Both invariants hold exactly in real arithmetic; in fp32 they hold to rounding. A
violation beyond that is the hardware, not the algorithm.

**`nbody`** — direct O(N²) gravitational simulation. Forces between two bodies are equal and
opposite, so total momentum is conserved. The run starts from exactly zero net momentum, so a single
miscomputed force breaks the symmetry and shows up as drift. Compute-bound.

**`stencil`** — 2D heat diffusion with periodic boundaries. Total heat is conserved, and for
α ≤ 0.25 every update is a convex combination of its neighbours, so the field can *never* leave its
initial range (the maximum principle). Memory-bound, so it stresses the memory system rather than
the ALUs.

> Thresholds are calibrated against healthy hardware: measured drift is ~3×10⁻⁴ for N-body momentum
> and ~10⁻¹⁰ for heat conservation, so the warn/fail levels sit well above the fp32 noise floor.
> Baselines are recorded in [docs/TESTED_GPUS.md](docs/TESTED_GPUS.md).

## Two editions

Both share the same telemetry monitor, evaluator, report writer and exit codes.

|  | `gpu-stress-lite` | `gpu-stress` |
|---|---|---|
| **Install size** | ~180 KB, zero dependencies | several GB (`torch`, `torchvision`) |
| **Needs** | NVIDIA driver only | CUDA-enabled PyTorch |
| **How** | Custom CUDA kernels as PTX, driver API via `ctypes` | The real ML stack |
| Memory / bandwidth / PCIe | ✅ | ✅ |
| Physics workloads | ✅ | — |
| Tensor cores | ✅ (`mma.sync`) | ✅ (cuBLAS) |
| ResNet50 training | — | ✅ |
| Best for | headless servers, CI, hardware validation | reproducing ML workloads |

## How results are judged

The suite evaluates rather than just reporting:

- **Wrong results** — any incorrect value → **FAIL**
- **Thermals** — within 8 °C of the NVML slowdown threshold → **WARN**; at it → **FAIL**
- **Throttling** — thermal or power-brake slowdown → **WARN**, or **FAIL** above 10% of samples;
  `sw_power_cap` is informational, since it is normal on TDP-limited boards
- **Clocks** — SM clock dipping below 50% of its peak → **WARN**
- **Sustained throughput** — >10% drop start-to-end → **WARN**, >25% → **FAIL**
- **Link health** — a PCIe link below the card's maximum → **WARN**

Throttle flags are only counted while the GPU is actually loaded, and a thermal flag asserted far
below the card's own threshold is reported as a spurious NVML flag rather than a fault — both are
common sources of false alarms on laptop parts.

## Reports

Every run writes a timestamped set of artifacts:

```text
results/
├── lite_20260928_154035.json            # full structured report
├── lite_20260928_154035.md              # human-readable summary
├── lite_20260928_154035.log             # console transcript
├── lite_20260928_154035_metrics.csv     # per-step metrics
├── lite_20260928_154035_telemetry.csv   # raw NVML samples
└── lite_20260928_154035_telemetry.png   # telemetry chart (needs matplotlib)
```

Useful for burn-in, server validation, fleet testing, deployment checks and debugging unstable GPUs.

## Using it in CI

Exit codes are meaningful, so the suite drops straight into a pipeline:

```text
0 → PASS or WARN     1 → FAIL     2 → ERROR
```

```yaml
- name: Validate GPU
  run: |
    curl -fsSL https://raw.githubusercontent.com/HamzaGbada/gpu-stress-test/main/install.sh | sh
    gpu-stress-lite --steps memtest,edge_cases,matmul --out results
- uses: actions/upload-artifact@v4
  if: always()
  with:
    name: gpu-report
    path: results/
```

## Architecture

```text
                    gpu-stress-test
                           │
             ┌─────────────┴─────────────┐
       gpu-stress-lite              gpu-stress
             │                           │
       NVIDIA Driver                  PyTorch
   (libcuda / libnvidia-ml)         (CUDA Runtime)
             └─────────────┬─────────────┘
                           │
                     GPU Workloads
          ┌────────────────┼────────────────┐
      Correctness       Stress          Telemetry
          └────────────────┼────────────────┘
                           │
                       Evaluator
                           │
                   PASS / WARN / FAIL
```

```text
gpu_stress/
├── pipeline.py        step runner (telemetry windows, error isolation, status roll-up)
├── evaluate.py        theoretical peaks + pass/warn/fail rules
├── monitor.py         background NVML sampler, per-step summaries
├── nvml.py            ctypes NVML binding (replaces pynvml)
├── report.py          console summary, JSON / Markdown / CSV / PNG
├── lite_steps.py      driver-API steps
├── lite_pipeline.py   `gpu-stress-lite` CLI
├── torch_steps.py     PyTorch steps (memory-budgeted ResNet50 training)
├── torch_pipeline.py  `gpu-stress` CLI
└── cuda/
    ├── driver.py      ctypes libcuda binding (contexts, memory, events, occupancy)
    ├── nvrtc.py       optional runtime compilation + PTX rebuild script
    ├── kernels.cu     the CUDA C kernels
    └── kernels_sm70.ptx / kernels_sm80.ptx   shipped; driver-JIT'd at runtime
scripts/build_zipapp.py   builds the single-file gpu-stress.pyz
install.sh                curl | sh installer
```

## Contributing

Contributions are welcome — especially **hardware reports**, since the verified table can only grow
by people running it on GPUs the author does not own.

**Report a GPU**

```bash
gpu-stress-lite --burn 120 --physics 60 --out results
```

Open an issue or PR with `results/lite_<timestamp>.json` attached. It contains no personal data
beyond the hostname. Bug reports are far more useful with that file included.

**Develop**

```bash
git clone https://github.com/HamzaGbada/gpu-stress-test && cd gpu-stress-test
uv sync --extra dev
uv run pytest tests -q          # GPU-independent tests
uvx ruff check gpu_stress tests scripts
```

The test suite deliberately runs without a GPU: it covers the evaluation rules, telemetry
summaries, the step runner and the kernel packaging, so CI can catch regressions on any runner.

**Rebuilding the kernels** (only needed if you edit `kernels.cu`)

```bash
NVRTC_LIB=/usr/local/cuda-12.x/lib64/libnvrtc.so python gpu_stress/cuda/nvrtc.py
```

Use a **CUDA 12.x** NVRTC: it emits PTX ISA 8.3, which any R545+ driver accepts. CUDA 13 emits a
newer ISA that older drivers reject.

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT — see [LICENSE](LICENSE).