# gpu-stress

Benchmark and stress-test NVIDIA GPUs, and check that they compute correctly while under load.

[![CI](https://github.com/HamzaGbada/gpu-stress-test/actions/workflows/ci.yml/badge.svg)](https://github.com/HamzaGbada/gpu-stress-test/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![NVIDIA driver R545+](https://img.shields.io/badge/NVIDIA%20driver-R545%2B-76b900.svg)](https://www.nvidia.com/drivers)

Most burn-in tools only tell you how fast a GPU is. gpu-stress runs workloads whose answers are
known in advance — exact matrix products, FMA chains that must converge to 1.0, simulations that
conserve momentum and heat — so a GPU that is fast but wrong fails. Every run ends with one verdict
(**PASS**, **WARN** or **FAIL**), a matching exit code and a JSON report that records the version,
driver, device and options, so the same benchmark can be repeated on the next card.

There are two commands:

- **`gpu-stress-lite`** needs only the NVIDIA driver: no CUDA toolkit, no PyTorch, no third-party
  Python packages. It drives the GPU through the CUDA driver API with its own kernels.
- **`gpu-stress`** is an optional PyTorch suite (cuBLAS, `torch.compile`, ResNet50 training) for
  reproducing ML workloads. It needs the `torch` extra, which is several GB of wheels.

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/HamzaGbada/gpu-stress-test/main/install.sh | sh
gpu-stress-lite
```

A default run takes two to three minutes on most GPUs. This is a real run on an RTX 4050 Laptop
GPU (6 GB, 60 W) with shorter sustained loads:

```text
$ gpu-stress-lite --burn 10 --physics 10
...
########################################################################
#  FINAL REPORT (lite)  -  overall: PASS
########################################################################
step                   status      time   maxT    avgW  key metrics
system_info            PASS        0.0s      -       -
memtest                PASS        0.4s    43°      13  errors=0, tested_gb=4.864, write_gbs=176.500
bandwidth              PASS        0.6s    46°      20  gbs=173.000
pcie                   PASS        0.4s    46°      32  gbs=13.160, pinned_h2d_gbs=12.800
matmul                 PASS        0.5s    51°      38  tflops=3.870, errors=0
edge_cases             PASS        0.0s      -       -  errors=0, threads=40960
fp32_burn              PASS       10.1s    59°      56  tflops=11.001, errors=0, threads=40960
fp16_burn              PASS       10.0s    59°      47  tflops=12.984, errors=0, threads=40960
tensor_burn            PASS       10.0s    57°      32  tflops=26.235, errors=0, threads=20480
nbody                  PASS       10.2s    65°      57  tflops=5.504, bodies=20480, momentum_rel=3.22e-04
stencil                PASS       10.1s    65°      60  gbs=182.000, side=8192, sum_drift_rel=2.48e-10
alloc_churn            PASS        0.5s    60°      60
```

Any WARN or FAIL is listed under the table with the reason, for example
`FAIL  [memtest] 3 incorrect results in VRAM integrity over 4.864 GiB x 6 passes`.

## What it checks

### `gpu-stress-lite`

| Step | What it does | Fails when |
|---|---|---|
| `system_info` | Device, driver, theoretical peaks, PCIe link | never (informational) |
| `memtest` | Writes and reads back 6 address-dependent bit patterns over 90% of free VRAM | any word reads back wrong |
| `bandwidth` | Device-to-device copy with a custom kernel and with `cuMemcpyDtoD` | below 30% of the theoretical peak |
| `pcie` | Host↔device copies, pinned and pageable | below 30% of the link's peak (warns if the link trains below the card's maximum) |
| `matmul` | SGEMM on small integers; 128 sampled outputs compared bit-exactly with the host | a sampled output is wrong, or throughput is under 5% of peak |
| `edge_cases` | 11 IEEE-754, integer and warp-intrinsic checks plus 4 atomics, each with one right answer | any check fails (flushed denormals only warn) |
| `fp32_burn` | Sustained FP32 FMA chains that must converge to exactly 1.0 | a wrong value, or throughput collapses |
| `fp16_burn` | The same with packed `fma.rn.f16x2` | a wrong value, or throughput collapses |
| `tensor_burn` | Sustained `mma.sync` with an exactly known accumulator (sm_80+, skipped below) | a wrong value, or throughput collapses |
| `nbody` | Direct O(N²) gravity starting from zero total momentum | momentum drifts: forces are no longer equal and opposite |
| `stencil` | 2D heat diffusion with periodic boundaries | heat is not conserved, or the field leaves its initial range |
| `alloc_churn` | Allocation and free latency from 16 MiB to 1 GiB | never (warns above 50 ms per allocation) |

The two physics steps are there because their invariants hold exactly in real arithmetic and to
rounding in fp32: one miscomputed force breaks the momentum balance, and one bad cell update can
break conservation of heat. `nbody` loads the ALUs; `stencil` loads the memory system.

### `gpu-stress` (PyTorch)

| Step | What it does |
|---|---|
| `system_info` | torch, CUDA runtime, cuDNN and device facts |
| `bandwidth` | Tensor copy bandwidth against the theoretical peak |
| `matmul` | cuBLAS in FP32, TF32, BF16 and FP16, each checked against an FP32 reference |
| `compile` | `torch.compile` speed-up on a small MLP (skipped if it is unavailable) |
| `matmul_burn` | Sustained matmul in `--precision`, outputs checked for NaN/inf |
| `training` | ResNet50 training on an in-VRAM synthetic dataset; the batch is halved until it fits |
| `vram_fill` | Fills free VRAM through PyTorch's caching allocator and checks 4 patterns |
| `fragmentation` | Allocator alloc/free latency from 128 MiB to 2 GiB |

Every threshold and the reasoning behind it is in [docs/STEPS.md](docs/STEPS.md).

## How results are judged

Each step produces findings at three levels: info, warn and fail. A step takes its worst finding,
and the run takes its worst step.

- **Wrong results fail.** One bad memtest word, one wrong SGEMM output or one broken invariant is
  enough.
- **Throughput is compared with the GPU's own peak at the clock it actually ran at**, not with a
  list of other GPUs. A power-capped laptop part therefore scores on what it can do under its cap;
  the comparison with the rated boost clock is reported as info.
- **Sustained loads are also watched through NVML.** Within 8 °C of the slowdown temperature warns,
  within 2 °C fails. Thermal or hardware slowdown warns, and fails when it is active in 10% or more
  of the loaded samples. A throughput drop of more than 10% from start to end warns, more than 25%
  fails. Power capping is info: it is normal on TDP-limited boards.

| Exit code | Meaning |
|---|---|
| `0` | PASS or WARN |
| `1` | FAIL: a wrong result, or a threshold the hardware should meet |
| `2` | ERROR: a step crashed or ran out of memory, or the run could not start (no driver, no GPU, bad `--device`, unknown step name) |

When one step fails and another errors, the run is FAIL.

## Benchmarking a new GPU

For numbers you can compare across cards and reproduce later:

1. **Pin the version** on every machine, so the kernels and thresholds are identical. Take the
   latest tag from the [releases page](https://github.com/HamzaGbada/gpu-stress-test/releases) and
   use the same one everywhere:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/HamzaGbada/gpu-stress-test/main/install.sh | sh -s -- --version <tag> --torch
   ```

2. **Start from an idle GPU.** `nvidia-smi` should show no other processes; anything holding memory
   shrinks the memtest and anything computing skews the throughput.

3. **Run both suites with long sustained loads**, one output directory per card:

   ```bash
   gpu-stress-lite --burn 120 --physics 60 --out results/<card>
   gpu-stress --precision bf16 --burn 120 --out results/<card>
   ```

   Two minutes per load lets clocks and temperatures settle; the 30 s default mostly shows boost
   behaviour.

4. **Read the numbers that describe the card.** In the JSON report, per step: `tflops_mean` for the
   burns and `nbody`, `gbs` for `bandwidth`, `pcie` and `stencil`, `tflops_first` against
   `tflops_last` for how much the card loses once it is hot, and `imgs_per_sec` for ResNet50
   training. The efficiency findings give each figure as a percentage of the card's own peak at the
   clock it held. The PyTorch `matmul` step has the cuBLAS TF32, BF16 and FP16 throughput.

5. **Keep the JSON files.** Each one records the tool version, driver, device facts, every option
   and every metric and finding.

To line up several cards:

```bash
jq -r '.device.name as $gpu | .steps[] | [$gpu, .name, .status, (.metrics.tflops // .metrics.gbs // .metrics.write_gbs // "")] | @tsv' \
   results/*/lite_*.json | column -t -s $'\t'
```

Two caveats for data-centre parts. `tensor_burn` uses `mma.sync`, which Hopper and data-centre
Blackwell run below their full tensor rate (that needs `wgmma` / `tcgen05`), so on those parts treat
its TFLOPS as a stability load and use the cuBLAS `matmul` numbers for peak tensor throughput. And
the evaluator only knows the GeForce FP16 and tensor-core ratios (1× and 2× FP32). When a part beats
them, those steps report the measured TFLOPS without a percentage. The correctness checks are not
affected.

## Usage

```bash
gpu-stress-lite --list                           # list the steps
gpu-stress-lite --burn 120 --physics 120         # longer soak
gpu-stress-lite --steps memtest,edge_cases       # quick correctness check
gpu-stress-lite --skip memtest                   # everything except memtest
gpu-stress-lite --device 1                       # second GPU
gpu-stress-lite --mem-max-gb 32 --mem-budget 300 # large-memory parts
gpu-stress-lite --nvrtc                          # compile the kernels for this GPU (needs a CUDA toolkit)

gpu-stress --epochs 3 --batch 64                 # PyTorch suite
gpu-stress --precision bf16 --burn 120           # tensor cores, half the activation memory
```

The options you are most likely to need (`--help` lists all of them):

| Option | Default | Effect |
|---|---|---|
| `--burn S` | `30` | Seconds per sustained burn step |
| `--physics S` | `15` | Seconds per physics step (`gpu-stress-lite` only) |
| `--steps A,B` | all | Run only these steps |
| `--skip A,B` | none | Skip these steps |
| `--device N` | `0` | CUDA device index |
| `--vram F` | `0.9` lite, `0.75` torch | Share of VRAM the memory tests and the training dataset may use |
| `--mem-max-gb G` | no cap | Cap the memtest buffer, in GiB |
| `--mem-budget S` | `120` | Stop the memtest pattern sweep after about S seconds (`0` = no limit) |
| `--out DIR` | `results` | Where the reports go |
| `--gui` | off | Open the telemetry chart at the end (needs matplotlib) |
| `--epochs`, `--batch`, `--max-steps`, `--precision` | `3`, `64`, `0`, `fp32` | ResNet50 training (`gpu-stress` only) |

## Reports

Every run writes a timestamped set of files to `--out`:

```text
results/
├── lite_20261021_224128.json            full structured report: device, config, every metric and finding
├── lite_20261021_224128.md              human-readable summary
├── lite_20261021_224128.log             console transcript
├── lite_20261021_224128_metrics.csv     one row per step metric
├── lite_20261021_224128_telemetry.csv   raw NVML samples (temperature, power, clocks, throttle reasons)
└── lite_20261021_224128_telemetry.png   telemetry chart, if matplotlib is installed
```

The PyTorch suite writes the same set with a `torch_` prefix.

### In CI

The exit code is the verdict, so a job fails on a bad GPU without parsing anything:

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

## Install

### Installer

```bash
curl -fsSL https://raw.githubusercontent.com/HamzaGbada/gpu-stress-test/main/install.sh | sh                 # gpu-stress-lite
curl -fsSL https://raw.githubusercontent.com/HamzaGbada/gpu-stress-test/main/install.sh | sh -s -- --torch    # + gpu-stress
```

The installer uses `uv` if it finds it, then `pipx`, then `pip --user`. If none of them is
available, or pip refuses because the system Python is externally managed (PEP 668), it downloads
the standalone zipapp into `~/.local/bin` instead. The zipapp is the lite pipeline only, so with
`--torch` the installer stops and asks for uv or pipx rather than falling back to it.

Options go after `sh -s --`. Each one also has an environment variable; when piping, put the
variable on `sh`, not on `curl` (`curl ... | GPU_STRESS_EXTRAS=torch sh`).

| Flag | Variable | Default | Effect |
|---|---|---|---|
| `--torch` | | | Same as `--extras torch,plot` |
| `--extras LIST` | `GPU_STRESS_EXTRAS` | none | Extras to install: `torch`, `plot` |
| `--version REF` | `GPU_STRESS_VERSION` | latest release | A tag such as `v0.4.0`, or `main` |
| `--method M` | `GPU_STRESS_METHOD` | `auto` | `uv`, `pipx`, `pip` or `zipapp` |
| `--bin DIR` | `GPU_STRESS_BIN` | `~/.local/bin` | Where the zipapp goes |

### Other ways

```bash
# uv or pipx, straight from git
uv tool install "git+https://github.com/HamzaGbada/gpu-stress-test@v0.4.0"
uv tool install "gpu-stress[torch,plot] @ git+https://github.com/HamzaGbada/gpu-stress-test@v0.4.0"

# single file, nothing installed (lite pipeline; `torch` as first argument runs the PyTorch suite if torch is importable)
curl -fsSLO https://github.com/HamzaGbada/gpu-stress-test/releases/latest/download/gpu-stress.pyz
python3 gpu-stress.pyz --burn 30

# from a checkout
pip install .                    # gpu-stress-lite
pip install ".[torch,plot]"      # + gpu-stress and the telemetry chart
python -m gpu_stress.lite_pipeline
```

### Requirements

- Linux with an NVIDIA driver, **R545 or newer** for the shipped kernels
- Python 3.10+
- For `gpu-stress`: a CUDA build of PyTorch, installed by the `torch` extra
- `matplotlib` (the `plot` extra) only for the telemetry chart

No CUDA toolkit is needed, except for `--nvrtc`.

## Hardware support

`gpu-stress-lite` ships its kernels as PTX for `compute_70` and `compute_80`. The driver compiles
that PTX for the GPU it finds, so architectures released after this version run without a rebuild:
the GB10 below runs the `compute_80` PTX.

| Architecture | `gpu-stress-lite` | Runs on file |
|---|---|---|
| Blackwell (sm_100, sm_120, sm_121) | all steps | GB10 / DGX Spark (verified) · RTX 5090 (PyTorch suite, reported) |
| Hopper (sm_90) | all steps | |
| Ada (sm_89) | all steps | RTX 4050 Laptop (both suites, verified) |
| Ampere (sm_80, sm_86, sm_87) | all steps | |
| Volta, Turing (sm_70, sm_75) | all except `tensor_burn`, which needs sm_80 | |
| Pascal (sm_60, sm_61) | with `--nvrtc` and a CUDA 12.x toolkit; no `tensor_burn` | |
| Maxwell and older | not supported | |

*Verified* means a full run is recorded in [docs/TESTED_GPUS.md](docs/TESTED_GPUS.md); *reported*
means it was reported working without a run on file. `gpu-stress` runs wherever your PyTorch build
supports the GPU.

**Unified-memory parts** (DGX Spark / GB10, Jetson, integrated GPUs) report a memory bus width of
zero, so there is no theoretical bandwidth to score against: `bandwidth` and `stencil` report the
measured GB/s as info, `pcie` is skipped because host and device share the same memory, and memtest
works on memory the OS is also using. The details are in [docs/TESTED_GPUS.md](docs/TESTED_GPUS.md#nvidia-gb10-dgx-spark).

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `libcuda not found - is the NVIDIA driver installed?` | No driver. Check that `nvidia-smi` works. |
| `cuInit failed: CUDA_ERROR_NO_DEVICE` | No GPU is visible: check `CUDA_VISIBLE_DEVICES`, and in containers pass `--gpus all`. |
| `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` | The driver is older than R545. Update it. |
| `the shipped PTX needs sm_70+` | Pascal GPU. Install a CUDA 12.x toolkit and run with `--nvrtc`. NVRTC is looked up through `NVRTC_LIB`, `CUDA_HOME`, `CUDA_PATH`, the system library path, then `/usr/local/cuda*` and `/opt/cuda*`. |
| `no telemetry (NVML unavailable)` | `libnvidia-ml.so.1` could not be loaded. Results are still checked; the thermal and clock rules are not applied. In containers, include `utility` in `NVIDIA_DRIVER_CAPABILITIES`. |
| `only X GiB of Y GiB could be tested` | Other processes hold VRAM (a desktop session, a browser). Close them for a full memtest. |
| `PyTorch is not installed` | Re-run the installer with `--torch`, or use `gpu-stress-lite`. |
| `power-capped`, `treating the flag as spurious` | Info, not a fault. Laptop GPUs run at their power limit and some report a thermal flag far below their thermal threshold. |

## Development

```bash
git clone https://github.com/HamzaGbada/gpu-stress-test && cd gpu-stress-test
uv sync --extra dev                       # add --extra torch --extra plot for the PyTorch suite
uv run pytest tests -q                    # no GPU needed
uvx ruff check gpu_stress tests scripts gpu_stress_cli.py
uv run gpu-stress-lite --burn 5 --physics 5
```

The tests run without a GPU. They cover the evaluation rules, telemetry summaries, the step runner,
the command line and the kernel packaging; the CUDA steps themselves need real hardware.

```text
gpu_stress/
├── lite_pipeline.py, lite_steps.py    gpu-stress-lite: command line and steps
├── torch_pipeline.py, torch_steps.py  gpu-stress: command line and steps
├── pipeline.py                        step runner: telemetry window per step, error isolation, verdict
├── evaluate.py                        theoretical peaks and the pass/warn/fail rules
├── monitor.py, nvml.py                background NVML sampler (ctypes, no pynvml)
├── report.py                          console summary, JSON, Markdown, CSV, PNG
└── cuda/                              ctypes driver API and NVRTC bindings, kernels.cu, shipped PTX
scripts/build_zipapp.py                builds gpu-stress.pyz
install.sh                             the curl | sh installer
gpu_bench.py, gpu_bench_details.py     standalone PyTorch benchmarks from 0.1 (raw numbers, no verdict)
gpu_stress_cli.py                      0.1 entry point, forwards to gpu-stress
```

### Rebuilding the kernels

Only needed after editing `gpu_stress/cuda/kernels.cu`:

```bash
NVRTC_LIB=/usr/local/cuda-12.x/lib64/libnvrtc.so python gpu_stress/cuda/nvrtc.py
```

Use NVRTC from a **CUDA 12.x** toolkit. Its PTX loads on any R545+ driver; CUDA 13 emits a newer PTX
version that older drivers reject, and it can no longer target `compute_70`.

## Contributing

Hardware reports are the most useful contribution: the support table can only grow through runs on
GPUs the author does not own.

```bash
gpu-stress-lite --burn 120 --physics 60
```

Open an issue with `results/lite_<timestamp>.json` attached. The file includes the machine's
hostname; remove it if you prefer. The same file makes bug reports much easier to act on.

Release history is in [CHANGELOG.md](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE).
