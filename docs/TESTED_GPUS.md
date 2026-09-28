# Tested GPUs

What has actually been run, and what the results looked like. "Verified" means the
whole lite pipeline completed and every correctness check (memtest, SGEMM sampling,
burn-kernel invariants, physics conservation laws, edge cases) passed.

The lite pipeline ships PTX for `compute_70` and `compute_80` and lets the driver
JIT-compile it for the installed GPU, so it runs on architectures newer than
anything listed here without a rebuild — an sm_121 Blackwell part happily runs the
sm_80 PTX.

## Verified

| GPU | Arch (CC) | Memory | Driver | Pipeline | Notes |
|---|---|---|---|---|---|
| GeForce RTX 4050 Laptop | Ada, sm_89 | 6 GB GDDR6, 96-bit | 590.48.01 | lite + torch | Primary development box. 60 W board limit, so it is power-capped under every sustained load — the evaluator reports that as info, not a fault. |
| GeForce RTX 5090 | Blackwell, sm_120 | 32 GB GDDR7 | — | torch | Reported working by the project author. |
| NVIDIA GB10 (DGX Spark) | Blackwell, sm_121 | 128 GB unified LPDDR5X | 580.178 | lite | Runs the shipped sm_80 PTX. Unified memory: the driver reports a 0-bit bus width, so bandwidth is measured but not scored. See the notes below. |

### Measured on the RTX 4050 Laptop (6 GB, 60 W)

A full `gpu-stress-lite --burn 10 --physics 10` run, driver 590.48.01. Useful as a
sanity baseline rather than a leaderboard entry — this part is power-capped at 60 W,
so every sustained figure is bounded by the board, not the silicon.

| Step | Result |
|---|---|
| memtest | 4.77 GiB × 6 patterns, 0 errors, 163 GB/s write / 175 GB/s read |
| bandwidth | 160–174 GB/s (83–91% of the 192 GB/s theoretical peak) |
| pcie | gen4 x8: 13.4 GB/s pinned (84% of the 15.8 GB/s peak), pageable 12.2 GB/s |
| matmul | 3.6–4.0 TFLOPS FP32, all sampled outputs bit-exact |
| edge_cases | 11/11 checks pass on 40960 threads, atomics exact |
| fp32_burn | 10.4–11.0 TFLOPS sustained (~84% of peak at the observed 2200–2550 MHz) |
| fp16_burn | 12.3–13.0 TFLOPS sustained |
| tensor_burn | 25–26 TFLOPS FP16 with FP32 accumulate (~100% of peak at the observed clock) |
| nbody | 5.4 TFLOPS, 20480 bodies; momentum drift 1.9e-4 (2k steps) → 3.4e-4 (13k steps) |
| stencil | 175–180 GB/s (94% of peak); heat conserved to 2.5e-10, maximum principle holds |
| alloc_churn | 0.2–12 ms per allocation, 16 MiB → 1 GiB |

Under `gpu-stress` (PyTorch), the same card trains ResNet50 at 131 img/s in fp32
(batch 32, auto-reduced from 64) and 225 img/s in bf16 (batch 64).

**Healthy-hardware baselines.** The physics invariants are checked against thresholds
calibrated on this GPU: N-body momentum drift stays in the 1e-4 range and grows
sub-linearly with step count, so the warn/fail thresholds sit at 1e-2 / 1e-1. Heat
conservation drifts ~1e-10, with warn/fail at 1e-5 / 1e-3.

## Architecture support

| Architecture | CC | Status |
|---|---|---|
| Volta | 7.0 | Supported (`kernels_sm70.ptx`); no tensor-core step (needs `mma.sync`, CC 8.0+) |
| Turing | 7.5 | Supported; no tensor-core step |
| Ampere | 8.0, 8.6 | Supported, all steps |
| Ada | 8.9 | **Verified** |
| Hopper | 9.0 | Supported, all steps |
| Blackwell | 10.x, 12.0 | Supported, all steps (5090 verified under torch) |
| Blackwell / Grace | 12.1 (GB10) | **Verified** (lite) |
| Pascal and older | ≤ 6.1 | Not supported by the shipped PTX — rebuild with `--nvrtc` or `nvrtc.py` |

Requirements: NVIDIA driver **R545 or newer** (the shipped PTX is ISA 8.3) and
Python 3.10+. No CUDA toolkit, no PyTorch, no numba.

## Notes for unified-memory parts (GB10 / DGX Spark, Jetson, iGPUs)

The GPU and the CPU share one pool of LPDDR5X, which changes what some numbers mean:

* **`bus_width_bits` is reported as 0**, so there is no theoretical bandwidth to
  compare against. `bandwidth` and `stencil` print measured GB/s and the evaluator
  marks them informational instead of inventing a percentage.
* **`pcie` is skipped**: there is no host↔device link to test, because host and
  device memory are the same memory.
* **Free memory is shared with the OS.** `--vram 0.9` of 114 GiB free means a
  ~103 GiB test buffer whose first touch has to fault in that many pages. Use
  `--mem-max-gb` to cap it, and note that `--mem-budget` (120 s by default) stops
  the pattern sweep once the projected time exceeds the budget.
* Both memtest kernels move 16 bytes per thread per step, which matters more on
  wide/unified memory than on a narrow GDDR bus.

If a memtest pass on such a machine still reports low GB/s, the per-pattern line
now prints the achieved write+read rate, and the step records `alloc_s`
separately, so a slow first-touch shows up as allocation time rather than as a
mysteriously long step.

## Adding your GPU

Run the suite and open an issue or PR with the JSON report:

```bash
gpu-stress-lite --burn 30 --out results
# attach results/lite_<timestamp>.json (it contains no personal data beyond the hostname)
```

Useful details to include: GPU model, driver version, `nvidia-smi` output, the
overall status line, and anything the evaluator flagged.
