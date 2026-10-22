# Tested GPUs

Hardware the suite has actually run on, and what healthy results looked like. The numbers are a
sanity baseline for similar parts, not a leaderboard: they depend on the board's power limit,
cooling and driver as much as on the chip.

- **Verified**: run by the author, with the full results recorded below. Every correctness check
  passed (memtest, SGEMM sampling, the burn invariants, the physics conservation laws and the edge
  cases).
- **Tested**: run by the author; the detailed results are not recorded here yet.

| GPU | Architecture | Driver | Status | Notes |
|---|---|---|---|---|
| NVIDIA B200 | Blackwell, sm_100 | | Tested | data-centre, see below |
| NVIDIA H200 | Hopper, sm_90 | | Tested | data-centre, see below |
| NVIDIA H100 | Hopper, sm_90 | | Tested | data-centre, see below |
| NVIDIA A100 | Ampere, sm_80 | | Tested | data-centre, see below |
| NVIDIA GB10 (DGX Spark) | Blackwell, sm_121 | 580.178 | Verified | 128 GB unified LPDDR5X; [results](#nvidia-gb10-dgx-spark) |
| GeForce RTX 5090 | Blackwell, sm_120 | | Tested | |
| GeForce RTX 4090 | Ada, sm_89 | 580.82.09 | Verified | 24 GB, 450 W; both suites; [results](#geforce-rtx-4090-24-gb-450-w) |
| GeForce RTX 4050 Laptop | Ada, sm_89 | 590.48.01 | Verified | 6 GB, 60 W; both suites; [results](#geforce-rtx-4050-laptop-6-gb-60-w) |
| GeForce RTX 3080 | Ampere, sm_86 | | Tested | |
| GeForce RTX 3060 | Ampere, sm_86 | | Tested | |
| Tesla T4 (Google Colab) | Turing, sm_75 | 580.82.07 | Verified | 16 GB, 70 W; both suites; `tensor_burn` skipped (needs sm_80); [results](#tesla-t4-google-colab-16-gb-70-w) |

Every architecture here runs the kernels shipped with the package, with no rebuild: the driver
compiles the `compute_70` or `compute_80` PTX for the GPU it finds. Which architectures are
expected to work beyond this list is in the [README](../README.md#hardware-support).

**Data-centre parts** (A100, H100, H200, B200) have FP16 and tensor-core units much faster than
the GeForce ratios the evaluator assumes, so `fp16_burn` and `tensor_burn` report their TFLOPS
without an efficiency percentage there. On Hopper and Blackwell, `tensor_burn` uses `mma.sync`,
which runs below the full tensor rate (that needs `wgmma` or `tcgen05`); the PyTorch suite's
cuBLAS `matmul` step shows the real tensor-core throughput. Every correctness check applies
unchanged.

## GeForce RTX 4090 (24 GB, 450 W)

gpu-stress 0.4.0, driver 580.82.09, PCIe gen4 x16. The efficiency figures are as the current
evaluator computes them.

`gpu-stress-lite --burn 60`:

| Step | Result |
|---|---|
| memtest | 20.6 GiB × 6 patterns, 0 errors, 811 GB/s write, 935 GB/s read |
| bandwidth | 847 GB/s, 84% of the 1008 GB/s theoretical peak |
| pcie | gen4 x16: 25.5 GB/s pinned, 81% of the 31.5 GB/s peak; 17.2 GB/s pageable |
| matmul | 30.4 TFLOPS FP32, all sampled outputs bit-exact |
| edge_cases | 11/11 checks pass on 262144 threads, atomics exact |
| fp32_burn | 72.0 TFLOPS sustained, 83% of the peak at the observed 2649 MHz |
| fp16_burn | 87.4 TFLOPS sustained, 98% of the peak at the observed 2715 MHz |
| tensor_burn | 177.9 TFLOPS FP16 with FP32 accumulate, 99% of the peak at the observed 2730 MHz |
| nbody | 38.4 TFLOPS, 131072 bodies; momentum drift 1.4×10⁻⁷ after 1680 steps |
| stencil | 919 GB/s, 91% of peak; heat conserved to 2.5×10⁻¹⁰, maximum principle holds |
| alloc_churn | 0.4–2.0 ms per allocation, 16 MiB → 1 GiB |

`gpu-stress --precision bf16 --burn 120` (torch 2.14.0, CUDA 13.0):

| Step | Result |
|---|---|
| bandwidth | 894 GB/s, 89% of peak |
| matmul | 8192³: FP32 55.1, TF32 86.7, BF16 166.9, FP16 168.1 TFLOPS; FP32 is 62% of the peak at the 2715 MHz the step reached |
| compile | `torch.compile` at 0.91× eager on the test MLP (warns below 0.8×) |
| matmul_burn | BF16 157.7 TFLOPS for 120 s at the 450 W limit, 2430–2565 MHz, no drop from start to end |
| training | ResNet50, bf16, batch 64: 1379 img/s, 20.6 GiB peak allocated |
| vram_fill | 16.75 GiB × 4 patterns, 0 errors |

- **The maximum clock is not the boost clock.** The driver reports a 3105 MHz maximum, so the
  info lines "of the peak at the maximum clock" use 101.7 TFLOPS FP32. NVIDIA's specification is
  82.6 TFLOPS at a 2520 MHz boost. Under load the card ran at 2415–2730 MHz.
- **`fp32_burn` is power-capped below the board limit.** NVML reports the software power cap for
  the whole step although the board drew 345 W of 450 W, and the clock settled at 2640 MHz, against
  2715–2730 MHz in the FP16 and tensor-core burns.
- The 0.4.0 report scored the PyTorch FP32 matmul at "150% of theoretical": see the
  [changelog](../CHANGELOG.md).

## Tesla T4 (Google Colab, 16 GB, 70 W)

gpu-stress 0.4.0, driver 580.82.07, PCIe gen3 x16, on a Colab GPU runtime. The shipped
`compute_70` PTX runs on it, compiled by the driver for sm_75. The efficiency figures are as the
current evaluator computes them.

`gpu-stress-lite --burn 60`:

| Step | Result |
|---|---|
| memtest | 12.9 GiB × 6 patterns, 0 errors, 214 GB/s write, 267 GB/s read |
| bandwidth | 233 GB/s, 73% of the 320 GB/s theoretical peak |
| pcie | gen3 x16: 13.1 GB/s pinned, 83% of the 15.8 GB/s peak; 5.0 GB/s pageable |
| matmul | 2.56 TFLOPS FP32, all sampled outputs bit-exact |
| edge_cases | 11/11 checks pass on 81920 threads, atomics exact |
| fp32_burn | 7.3 TFLOPS sustained, 99% of the peak at the observed 1437 MHz; 7.6 → 7.1 TFLOPS from start to end as it heated to 81 °C |
| fp16_burn | 13.4 TFLOPS sustained, 95% of the peak (2× FP32 on Turing) at the observed 1378 MHz |
| tensor_burn | skipped: `mma.sync.m16n8k16` needs sm_80 |
| nbody | 3.6 TFLOPS, 40960 bodies; momentum drift 6.6×10⁻⁸ after 1650 steps |
| stencil | 218 GB/s, 68% of peak; heat conserved to 2.5×10⁻¹⁰, maximum principle holds |
| alloc_churn | 0.4–9.3 ms per allocation, 16 MiB → 1 GiB |

`gpu-stress --precision bf16 --burn 120` (torch 2.14.1, CUDA 13.0):

| Step | Result |
|---|---|
| bandwidth | 240 GB/s, 75% of peak |
| matmul | 8192³: FP32 3.7, TF32 3.6, BF16 2.2, FP16 21.8 TFLOPS. Turing has no TF32, so that row runs in FP32 |
| compile | `torch.compile` at 0.88× eager; Inductor reports too few SMs for its autotuned GEMMs |
| matmul_burn | BF16 1.9 TFLOPS for 120 s, at 615–780 MHz under the 70 W cap |
| training | ResNet50, bf16, batch 64: 44 img/s, 13.3 GiB peak allocated; 21 minutes for 3 epochs |
| vram_fill | 10.5 GiB × 4 patterns, 0 errors |

- **Use `--precision fp16` on this card.** Turing's tensor cores have no bf16, so in bf16 neither
  `matmul_burn` nor `training` loads them: compare the BF16 and FP16 matmul rows. On a slow card,
  `--max-steps` caps the training time.
- **The 70 W limit sets every clock.** The FP32 and FP16 burns averaged 1437 and 1378 MHz against
  the 1590 MHz maximum, the stencil 798 MHz. Each of these is reported as power capping (info), and
  throughput is scored at the clock the card held.
- A Colab GPU is a shared, passively cooled card in a VM, so expect run-to-run spread on the
  sustained steps.

## GeForce RTX 4050 Laptop (6 GB, 60 W)

The development machine. Its board is limited to 60 W, so every sustained load runs at the power
cap; the evaluator reports that as info, and scores throughput at the clock the GPU actually held.

`gpu-stress-lite --burn 10 --physics 10`, driver 590.48.01, ranges over the runs on file:

| Step | Result |
|---|---|
| memtest | 4.8–4.9 GiB × 6 patterns, 0 errors, 163–177 GB/s write, 169–183 GB/s read |
| bandwidth | 160–174 GB/s, 83–91% of the 192 GB/s theoretical peak |
| pcie | gen4 x8: 12.8–13.4 GB/s pinned, 81–85% of the 15.8 GB/s peak; 12.2 GB/s pageable |
| matmul | 3.6–4.0 TFLOPS FP32, all sampled outputs bit-exact |
| edge_cases | 11/11 checks pass on 40960 threads, atomics exact |
| fp32_burn | 10.4–11.0 TFLOPS sustained, about 85% of the peak at the observed 2200–2550 MHz |
| fp16_burn | 12.3–13.0 TFLOPS sustained, about 99% of the peak at the observed clock |
| tensor_burn | 25–26 TFLOPS FP16 with FP32 accumulate, about 100% of the peak at the observed clock |
| nbody | 5.4–5.5 TFLOPS, 20480 bodies; momentum drift 1.9×10⁻⁴ after 2k steps, 3.2–3.4×10⁻⁴ after 7k–13k |
| stencil | 175–182 GB/s, about 94% of peak; heat conserved to 2.5×10⁻¹⁰, maximum principle holds |
| alloc_churn | 0.2–12 ms per allocation, 16 MiB → 1 GiB |

Under `gpu-stress` the same card trains ResNet50 at 131 img/s in fp32 (batch 32, reduced from the
requested 64 to fit) and 225 img/s in bf16 (batch 64).

These runs are where the physics thresholds come from. At this card's 20480 bodies, N-body momentum
drift is about 2×10⁻⁴ after 2k steps and 3.4×10⁻⁴ after 9k–13k, so the warn and fail levels sit
at 10⁻² and 10⁻¹. That is the worst case on file. The drift depends on the body count (1024 per
SM), not on the GPU: with 40960 bodies or more it stays near 10⁻⁷. That holds on the T4 and the
RTX 4090, and on this card too when it is run at 40960 bodies. Heat conservation drifts by
2.5×10⁻¹⁰ on every card, against warn and fail levels of 10⁻⁵ and 10⁻³.

## NVIDIA GB10 (DGX Spark)

Runs the shipped `compute_80` PTX, compiled by the driver for sm_121. The GPU and the CPU share one
pool of LPDDR5X, which changes what some steps mean:

- **The driver reports a memory bus width of 0**, so there is no theoretical bandwidth.
  `bandwidth` and `stencil` report measured GB/s as info instead of a percentage.
- **`pcie` is skipped.** Host and device memory are the same memory, so there is no link to test.
- **Free memory is shared with the OS.** `--vram 0.9` of 114 GiB free is a 103 GiB test buffer, and
  the first touch has to set up that many pages. `memtest` reports that allocation time separately
  (`alloc_s`) so slow page setup is not mistaken for a slow GPU. Use `--mem-max-gb` to cap the
  buffer; `--mem-budget` (120 s by default) stops the pattern sweep once the projected time runs
  past it.

The same applies to Jetson modules and other integrated GPUs.

## Adding your GPU

Run both suites if you can, with longer sustained loads than the defaults:

```bash
gpu-stress-lite --burn 120 --physics 60
gpu-stress --precision bf16                  # if the torch extra is installed
```

Open an issue with the JSON reports from `results/` attached, plus the output of `nvidia-smi`. The
reports record the tool version, driver, device facts, the options used and every metric and
finding. They also include the machine's hostname; remove it if you prefer.
