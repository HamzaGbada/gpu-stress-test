# Steps and thresholds

What every step does, what it measures and exactly when it warns or fails. The rules live in
`gpu_stress/evaluate.py` and in the `eval_*` functions of `lite_steps.py` and `torch_steps.py`;
this page follows them.

## How a verdict is formed

Each step returns metrics, and its rules turn them into findings at three levels: `info`, `warn`
and `fail`. The step's status is its worst finding. Two more statuses come from the runner:
`skipped` (the step does not apply, for example `tensor_burn` below sm_80) and `error` (it
crashed or ran out of GPU memory). A crashing step is recorded and the run continues.

The run's verdict is its worst step, in this order: `fail` > `error` > `warn` > `pass` = `skipped`.
The exit code is `0` for pass or warn, `1` for fail and `2` for error. A run that cannot start at
all (no driver, no GPU, a bad `--device`, an unknown name in `--steps` or `--skip`) also exits `2`.

### Peaks are taken at the observed clock

Throughput rules compare a measurement with what this GPU can do in theory:

- FP32: SMs × FP32 cores per SM × 2 × SM clock. For the sustained steps the clock is the **mean
  clock NVML observed during the step**, so a board held down by its power limit is scored on what
  it can do under that limit. A second, informational line compares with the peak at the maximum
  clock the driver reports. On GeForce boards that is the top of the clock table, well above the
  advertised boost clock: an RTX 4090 reports 3105 MHz (101.7 TFLOPS), where NVIDIA's specification
  quotes 2520 MHz (82.6 TFLOPS).
- The two `matmul` steps finish in about a second, too fast to be sampled reliably: NVML refreshes
  about every half second, and the first samples are taken at idle clock while buffers are set up.
  They are scored at the **highest clock seen in the step** (the maximum clock if no sample landed
  under load). That is at or above the clock the work ran at, so their efficiency is a lower bound.
- Memory: 2 × memory clock × bus width / 8. Unified-memory parts report a bus width of 0, so there
  is no peak and the measurement is reported as info.
- PCIe: per-lane rate × width, for the generation and width the link is trained to while the
  `pcie` step is moving data.

### Telemetry rules

These apply to the sustained steps only: `fp32_burn`, `fp16_burn`, `tensor_burn`, `nbody`,
`stencil`, and in the PyTorch suite `matmul_burn` and `training`. They use the NVML samples taken
during the step (every `--interval`, 0.25 s by default). Throttle reasons are only counted on
samples where the GPU is actually loaded (utilisation of 30% or more, and not flagged idle).

| Condition | Level |
|---|---|
| Temperature at or within 2 °C of the slowdown threshold | fail |
| Within 8 °C of it (or 90 °C when the threshold is unknown) | warn |
| Thermal or hardware slowdown active in ≥ 10% of loaded samples | fail |
| … active at all | warn |
| … but only thermal reasons, while the GPU stayed more than 15 °C below its threshold | info: the flag is treated as spurious |
| Power cap active in ≥ 50% of loaded samples | info |
| Lowest SM clock under 50% of the highest (steps of 5 s or more) | warn |
| Mean SM clock under 60% of the rated maximum (steps of 5 s or more) | warn, or info when the board is power-capped |
| No NVML samples at all | info |

The slowdown reasons are NVML's `hw_slowdown`, `sw_thermal_slowdown`, `hw_thermal_slowdown` and
`hw_power_brake_slowdown`. The spurious-flag rule exists because some laptop parts assert a
thermal slowdown at 60 °C against a 90 °C threshold.

### Stability over a sustained run

Sustained steps record throughput for every launch and compare the first fifth of the run with
the last fifth. A drop of more than 10% warns and more than 25% fails.

## `gpu-stress-lite`

### `system_info`

Prints the device facts, the theoretical peaks and the free memory. It never warns. It notes
unified memory, a missing bandwidth peak, and a PCIe link that is below the card's maximum at
startup, which is normal while the GPU is idle; the `pcie` step checks the link again under load.

### `memtest`

Allocates `--vram` (0.9) of free memory minus 128 MiB, capped at `--mem-max-gb`, halving the
request if the allocation fails. For each pattern it writes `pattern XOR word_index` to every
32-bit word and reads it back, so the value depends on the address as well as the pattern. The
first six patterns are `0x00000000`, `0xFFFFFFFF`, `0xAAAAAAAA`, `0x55555555`, `0x0F0F0F0F` and
`0xF0F0F0F0`; `--mem-passes` above 6 adds random ones. The sweep stops early once the projected
time exceeds `--mem-budget` (120 s).

| Condition | Level |
|---|---|
| Any word reads back wrong | fail |
| Less than half the card's memory could be tested | warn |
| Allocating the buffer took more than 10 s | warn (first-touch page setup on unified memory) |
| Sweep cut short by `--mem-budget` | info |

### `bandwidth`

Copies a buffer of up to 1 GiB with a 16-byte-per-thread kernel and with `cuMemcpyDtoD`, counting
read plus write, and scores the faster of the two. Under 30% of the memory peak fails, under 60%
warns.

### `pcie`

Copies 256 MiB between pinned host memory and the GPU in both directions, and once from pageable
memory. The link generation and width are read from NVML while the transfer is running.

| Condition | Level |
|---|---|
| Faster direction under 30% of the link's peak | fail |
| … under 55% | warn |
| Link trained to a lower generation than the card supports | warn (check the slot, riser and BIOS) |
| Link narrower than the card supports | warn (narrower slot or shared lanes) |

Skipped on integrated and unified-memory GPUs, which have no host-to-device link.

### `matmul`

A register-tiled SGEMM of size `--matmul-size` (4096, halved until three matrices fit in half the
free memory). The inputs are small integers, so every partial sum is exact in fp32 and the host can
compute the right answer: 32 random rows × 4 random columns (`--verify-samples`) are compared
bit for bit.

| Condition | Level |
|---|---|
| Any sampled output differs | fail |
| Under 5% of the FP32 peak at the highest clock seen in the step | fail |
| Under 12% | warn |

The efficiency bar is low on purpose: a hand-written kernel without tensor cores is not cuBLAS.
Healthy GPUs land at 30–35%.

### `edge_cases`

One launch, 8 blocks per SM of 256 threads, in which every thread checks:

| Check | Level if wrong |
|---|---|
| Denormals survive multiplication instead of being flushed to zero | warn |
| NaN compares unequal to everything, itself included | fail |
| `1/0`, `inf - inf`, `inf + x`, `-1 × inf` | fail |
| `(1 + 2⁻²³) × (1 − 2⁻²⁴)` rounds to exactly 1.0, and `0.1f + 0.2f` to `0x3E99999A` | fail |
| `fmaf` is fused: the product's low bits survive the add | fail |
| `sqrtf` of −1, 0 and +inf | fail |
| `__popc`, `__clz`, `__umulhi`, division and modulo by a runtime value | fail |
| 64-bit multiply, divide and shift | fail |
| Warp shuffle butterfly sum and `__ballot_sync` mask | fail |
| Shared-memory tree reduction with `__syncthreads` | fail |
| `atomicAdd`, `atomicMax`, `atomicXor` from every thread and a CAS loop per warp reach their exact totals | fail |

The inputs come from device memory so the compiler cannot fold any of them away.

### `fp32_burn`, `fp16_burn`, `tensor_burn`

Back-to-back launches for `--burn` seconds (30), each sized to about 100 ms so telemetry samples
every launch.

- `fp32_burn`: 8 independent chains per thread of `a = 0.999·a + 0.001`, which converges to 1.0.
  A result more than 2×10⁻³ away from 1.0 is an error.
- `fp16_burn`: the same with packed `fma.rn.f16x2` and `a = 0.9·a + 0.1`, tolerance 2×10⁻².
- `tensor_burn`: `mma.sync.m16n8k16` with all-ones inputs, so every thread's accumulators must sum
  to exactly 256 × iterations. Needs sm_80; skipped below.

Outputs are checked every tenth launch and after the last one.

| Condition | Level |
|---|---|
| Any wrong output | fail |
| Under 30% of the peak at the observed clock | fail |
| Under 55% | warn |
| Start-to-end drop (see [stability](#stability-over-a-sustained-run)) | warn / fail |

The peak is the FP32 peak for `fp32_burn`, and twice the FP32 peak for `tensor_burn`. For
`fp16_burn` it is twice the FP32 peak on Volta and Turing (sm_70, sm_72, sm_75), GP100 and Tegra
Pascal (sm_60, sm_62), 1/64 of it on GP10x Pascal (sm_61), and the FP32 peak on everything else, which
is the rate of Ampere GA10x, Ada and Blackwell GeForce parts. Data-centre
parts have faster FP16 and tensor units (an A100 runs FP16 tensor maths at 16× its FP32 rate), so
when a measurement comes out more than 15% above the assumed peak, the step reports the TFLOPS as
info and does not score it. On Hopper and data-centre Blackwell, `mma.sync` itself runs below the
full tensor rate, which needs `wgmma` / `tcgen05`.

### `nbody`

Direct O(N²) gravity on 1024 bodies per SM (8192 to 131072), unit masses at rest in a sphere, so
the total momentum starts at exactly zero. Forces between two bodies are equal and opposite, so it
must stay there; fp32 rounding and the approximate `rsqrtf` let it drift on healthy hardware. How
far depends on the body count, not the GPU: 20480 bodies drift to 2–3×10⁻⁴ of the total momentum
magnitude, 40960 and more stay near 10⁻⁷. Compare the drift only between cards that ran the same
body count. Runs for `--physics` seconds (15).

| Condition | Level |
|---|---|
| NaN or inf in the state | fail |
| Momentum drift above 10⁻¹ of the momentum scale | fail |
| … above 10⁻² | warn |
| Under 10% of the FP32 peak at the observed clock | fail |
| Under 25% | warn |
| Start-to-end drop | warn / fail |

Each interaction also runs `rsqrtf` on the special-function units, so healthy GPUs land at about
40–60% of the FMA peak.

### `stencil`

2D heat diffusion on an 8192² grid (halved to fit), five-point Jacobi update with α = 0.2 and
periodic boundaries, for `--physics` seconds. With periodic boundaries the total heat is constant,
and with α ≤ 0.25 every new value is a weighted average of old ones, so the field can never leave
its initial range (the maximum principle). Sums are accumulated in fp64.

| Condition | Level |
|---|---|
| NaN or inf in the field | fail |
| A value outside the initial range, by more than 10⁻⁴ of the peak value | fail |
| Total heat drifted more than 10⁻³ (relative to n × peak value) | fail |
| … more than 10⁻⁵ | warn |
| Effective bandwidth under 15% of the memory peak | fail |
| … under 35% | warn |
| Start-to-end drop | warn / fail |

Healthy hardware drifts by about 10⁻¹⁰.

### `alloc_churn`

Allocates and frees 16 MiB, 64 MiB, 256 MiB and 1 GiB buffers 20 times each, then holds up to
256 live 4 MiB buffers. Any mean allocation time above 50 ms warns.

## `gpu-stress` (PyTorch)

| Step | What it does | Rules |
|---|---|---|
| `system_info` | torch, CUDA runtime and cuDNN versions, device facts, allocator settings | none |
| `bandwidth` | `copy_` of up to 1 GiB | fail under 30% of the memory peak, warn under 60% |
| `matmul` | `--matmul-size` (8192, halved to fit) in FP32, TF32, BF16 and FP16; a 64 × 64 corner is compared with an FP32 reference | fail if non-finite or the max relative error exceeds 10⁻⁴ (FP32), 5×10⁻² (TF32, FP16) or 10⁻¹ (BF16); FP32 fail under 10%, warn under 25% of the peak at the highest clock seen in the step; warn if the fastest reduced precision is under 1.5× FP32 on CC 7.0+ |
| `compile` | Eager vs `torch.compile` on a 4096-wide two-layer MLP | warn if compiled is slower than 0.8× eager; skipped when `torch.compile` is unavailable |
| `matmul_burn` | Matmul in `--precision` for `--burn` seconds | fail on NaN/inf output (checked every 20th sample); start-to-end drop |
| `training` | ResNet50 + Adam on an in-VRAM synthetic dataset, `--epochs` epochs | warn if the batch had to shrink or an OOM was recovered; fail on NaN/inf loss; warn if the loss grows by 1.5×; drop between first and last epoch |
| `vram_fill` | 256 MiB blocks up to `--vram` of free memory, 4 patterns | fail on any mismatch |
| `fragmentation` | Alloc/free latency 128 MiB to 2 GiB, allocator statistics | info only |

### How `training` fits the memory

The training step measures a real training step before it sizes anything. It runs two steps at
`--batch` (the second allocates the optimizer state), halving the batch on out-of-memory until it
fits. It then records the persistent memory (weights, gradients, optimizer state) and the
transient peak (activations, workspace), and sizes the synthetic dataset from what is still free
after that, minus a margin of 6% of the card or 256 MiB. An out-of-memory error later in training
halves the batch again instead of ending the step. `--precision bf16` roughly halves the activation
memory, which usually keeps the requested batch on small cards. Before Ampere (sm_80) the tensor
cores have no bf16 support, so use `--precision fp16` on Volta and Turing: on a T4, bf16 matmul
runs at 2.2 TFLOPS against 21.8 in fp16.
