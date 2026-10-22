# Tested GPUs

Hardware the suite has actually run on, and what healthy results looked like. The numbers are a
sanity baseline for similar parts, not a leaderboard: they depend on the board's power limit,
cooling and driver as much as on the chip.

- **Verified**: a full run completed and every correctness check passed (memtest, SGEMM sampling,
  the burn invariants, the physics conservation laws and the edge cases).
- **Reported**: reported working, but no run is on file yet.

| GPU | Architecture | Memory | Driver | Suite | Status |
|---|---|---|---|---|---|
| GeForce RTX 4050 Laptop | Ada, sm_89 | 6 GB GDDR6, 96-bit | 590.48.01 | lite + PyTorch | Verified |
| NVIDIA GB10 (DGX Spark) | Blackwell, sm_121 | 128 GB unified LPDDR5X | 580.178 | lite | Verified |
| GeForce RTX 5090 | Blackwell, sm_120 | 32 GB GDDR7 | | PyTorch | Reported |

Which architectures are expected to work is in the
[README](../README.md#hardware-support).

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

These runs are where the physics thresholds come from. N-body momentum drift stays around 10⁻⁴ and
grows slower than the step count, so the warn and fail levels sit at 10⁻² and 10⁻¹. Heat
conservation drifts by about 10⁻¹⁰, against warn and fail levels of 10⁻⁵ and 10⁻³.

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

## GeForce RTX 5090

Reported working under the PyTorch suite. No report is on file yet, and the lite suite has not been
run on it.

## Adding your GPU

Run both suites if you can, with longer sustained loads than the defaults:

```bash
gpu-stress-lite --burn 120 --physics 60
gpu-stress --precision bf16                  # if the torch extra is installed
```

Open an issue with the JSON reports from `results/` attached, plus the output of `nvidia-smi`. The
reports record the tool version, driver, device facts, the options used and every metric and
finding. They also include the machine's hostname; remove it if you prefer.
