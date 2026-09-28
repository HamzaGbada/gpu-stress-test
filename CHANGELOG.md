# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-09-28

Follow-up to 0.3.0: the installer set up two commands but only one working pipeline.

### Added

* `GPU_STRESS_EXTRAS` for the installer, so the PyTorch pipeline can be installed in the
  same command:

      GPU_STRESS_EXTRAS=torch,plot curl -fsSL .../install.sh | sh

  Extras are passed as a PEP 508 direct reference (`gpu-stress[torch] @ git+...`), which is
  the only form that carries them through uv, pipx and pip alike. The closing message now
  states which commands were actually installed, and the zipapp path warns that a single
  file cannot carry extras.

### Fixed

* `gpu-stress` crashed with a bare `ModuleNotFoundError: No module named 'torch'` when run
  without the optional `torch` extra. The console script is always installed, so this was
  the first thing many users would have hit. It now explains that PyTorch is an opt-in
  extra, prints the command to add it (or to use `gpu-stress-lite` instead), and exits 2.

### Documentation

* README reorganised for an open-source audience: table of contents, an install section
  that states which commands each method gives you, an installer-options table, a
  requirements matrix, a CI section with exit codes and a working workflow snippet, and
  contributing instructions with the development setup.
* Corrected the hardware-support table, which claimed more verified hardware than
  `docs/TESTED_GPUS.md` actually backs. Status is now split into Verified (a report is on
  file), Reported (reported working, no report filed) and Supported (compatible with the
  shipped PTX, not yet exercised).
* Corrected the documented report filenames, which had a `_report` infix the code never wrote.

## [0.3.0] - 2026-09-28

First tagged release.

### Added

* **Physics stress steps**, both with invariants a correct GPU cannot violate:
  * `nbody` — direct O(N²) gravitational N-body simulation. Newton's third law makes
    total momentum a conserved quantity; the run starts at exactly zero net momentum
    and the step fails if it drifts.
  * `stencil` — 2D heat diffusion (5-point Jacobi, periodic boundaries). Checks both
    heat conservation and the maximum principle: with α ≤ 0.25 every update is a convex
    combination, so the field can never leave its initial range.
* **`edge_cases` step** — 11 hardware corner cases with exactly one right answer each:
  denormals, NaN comparison semantics, infinity arithmetic, fp32 round-to-nearest-even,
  genuinely fused FMA, `sqrtf` of -1/0/inf, 32- and 64-bit integer intrinsics, warp
  shuffle and ballot, shared-memory reductions, and four kinds of atomics checked for
  lost updates across every thread.
* **`pcie` step** — host↔device bandwidth, pinned versus pageable, scored against the
  theoretical peak of the *current* PCIe training. Flags a link that trained to a lower
  generation or a narrower width than the card supports — a common and easily missed
  hardware problem. Skipped on unified-memory parts.
* **Installer**: `curl -fsSL .../install.sh | sh`, which uses uv, pipx or pip when
  available and otherwise drops a standalone zipapp into `~/.local/bin`.
* **Single-file zipapp** (`gpu-stress.pyz`, ~180 KB) built by `scripts/build_zipapp.py`:
  the whole lite pipeline with no install step and no dependencies.
* CI and release workflows, `docs/TESTED_GPUS.md`, and this changelog.

### Fixed

* **Unified-memory GPUs (GB10 / DGX Spark, Jetson, iGPUs)**: the driver reports a
  0-bit memory bus width on these parts, which made the theoretical bandwidth come out
  as `0 GB/s`. Bandwidth-style steps now detect the missing peak and report measured
  throughput without a meaningless efficiency percentage, and `system_info` explains why.
* **Memtest on large-VRAM machines** (reported on a 128 GB GB10, where a 103 GiB sweep
  took 160 s with no indication of why): the kernels now move 16 bytes per thread per
  step instead of 4, and the grid is sized from real occupancy rather than a fixed
  blocks-per-SM guess. Measured at 163 GB/s write / 175 GB/s read on an RTX 4050
  (~87% of its theoretical peak). The step is also now diagnosable: every pattern
  prints its achieved GB/s, buffer allocation is timed and reported separately
  (first-touch page setup on unified memory is not a GPU fault), `--mem-max-gb` caps
  the buffer, and `--mem-budget` stops the sweep when the projected time runs long.
* Grid sizing for all grid-stride kernels now uses
  `cuOccupancyMaxActiveBlocksPerMultiprocessor` instead of a hard-coded blocks-per-SM
  guess, so the launch geometry follows the actual GPU rather than the one it was tuned on.
* Evaluation thresholds calibrated against healthy hardware so a good GPU reports PASS:
  N-body efficiency accounts for `rsqrtf` running on the SFU path rather than the FMA
  pipes, and a thermal-slowdown flag asserted 30 °C below the card's own threshold is
  reported as a spurious NVML flag instead of a hardware fault.
* Kernel files are read through `importlib.resources`, so the PTX loads correctly from a
  wheel or zipapp and not just from a source checkout.

### Changed

* `system_info` is now evaluated: it reports integrated/unified memory, a missing
  bandwidth peak, and a PCIe link running below the card's maximum.
* Default step order puts the cheap correctness steps before the long sustained loads.

## [0.2.0] - 2026-09-16

### Added

* `gpu_stress` package: a multi-step pipeline engine (telemetry windows per step,
  per-step OOM/crash isolation, pass/warn/fail roll-up), an evaluator scoring against
  the GPU's own theoretical peaks at the *observed* clock, an NVML telemetry monitor,
  and JSON/Markdown/CSV/PNG reporting.
* **Lite pipeline** (`gpu-stress-lite`): CUDA driver API through ctypes with hand-written
  kernels shipped as PTX — no PyTorch, no CUDA toolkit, no numba. Steps: VRAM memtest,
  bandwidth, verified SGEMM, and sustained FP32 / FP16 / tensor-core burns whose outputs
  have exactly known values.
* **Torch pipeline** (`gpu-stress`): cuBLAS matmul across precisions, `torch.compile`,
  sustained matmul burn, ResNet50 training, VRAM fill and allocator fragmentation.
* ctypes NVML binding replacing `pynvml`.

### Fixed

* **The ResNet50 training OOM on cards smaller than ~16 GB.** The dataset was sized as
  `--vram × total VRAM` and allocated *before* the model, so on a 6 GB card the model,
  optimizer and activations no longer fit. Training now probes real memory use at the
  requested batch size, halves the batch until it fits, sizes the in-VRAM dataset from
  what is actually free afterwards, recovers from late OOM by shrinking the batch, and
  sets `expandable_segments:True` before torch is imported.
* `gpu_bench.py` allocated `torch.empty(2_000_000_000)` — 8 GB, not the "~2 GB" the
  comment claimed — and OOMed on any card below 16 GB.

### Changed

* `torch`/`torchvision` moved to an optional `[torch]` extra; the base install has no
  dependencies. Unused `pandas`, `psutil` and `pynvml` dependencies dropped.

## [0.1.0] - 2025-11-17

* Initial scripts: `gpu_bench.py`, `gpu_bench_details.py`, `gpu_stress_cli.py`.

[0.3.1]: https://github.com/HamzaGbada/gpu-stress-test/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/HamzaGbada/gpu-stress-test/releases/tag/v0.3.0
[0.2.0]: https://github.com/HamzaGbada/gpu-stress-test/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/HamzaGbada/gpu-stress-test/releases/tag/v0.1.0
