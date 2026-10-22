# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-10-22

Bug-fix and documentation release. Two changes can affect scripts that call the tool: a run that
cannot start (no driver, no GPU, bad `--device`) now exits 2 instead of 1, and an unknown name in
`--steps` or `--skip` is an error instead of being ignored.

### Added

* `--version` on both commands, to check which release a machine is running.

### Fixed

* **Exit codes when a run cannot start.** No driver, no visible GPU or a bad `--device` ended in
  a Python traceback and exit code 1, which the CLI documents as a hardware FAIL. Both commands now
  print one line and exit 2 (ERROR).
* **Typos in `--steps` / `--skip`** matched nothing, so the run silently did no work and reported
  SKIPPED. Unknown step names are now rejected with the list of valid ones.
* **The verdict depended on step order** when one step failed and another errored. FAIL now always
  outranks ERROR.
* **`--burn 0` and `--physics 0`** crashed the sustained steps with `ZeroDivisionError`. Every
  sustained step now runs at least one timed launch. `nbody` and `stencil` also computed their
  per-second rates from the requested duration instead of the measured one.
* **PCIe link judged at idle.** GPUs drop the link to gen1 while idle, but the `pcie` step scored
  the startup reading. It now reads the link generation and width while transferring, and
  `system_info` reports the idle reading as info instead of a warning.
* **Telemetry could watch the wrong GPU on multi-GPU machines.** NVML numbers devices in PCI order
  and ignores `CUDA_VISIBLE_DEVICES`, while CUDA defaults to fastest-first. Both pipelines now bind
  NVML to the GPU under test by PCI bus id.
* **`--nvrtc` on Pascal** was refused before NVRTC was even tried, although the docs offered it as
  the way to run there. It now works with a CUDA 12.x toolkit. A PTX that NVRTC produces but the
  driver cannot load (toolkit newer than the driver) falls back to the shipped PTX instead of
  aborting, and a driver too old for the shipped PTX gets a message naming R545.
* **FP16 and tensor-core efficiency on data-centre parts.** Both are scored against the GeForce
  ratios, which produced figures like "800% of theoretical" on an A100. A measurement above the
  assumed peak is now reported unscored, and `fp16_burn` uses the 1/64 FP16 rate of sm_61.
* The summary table printed aliases of the same number (`tflops` and `tflops_mean`, `gbs` and
  `gbs_copy_kernel`, `tflops` and `gflops`).
* `gpu-stress` logged steps per epoch before applying `--max-steps`.
* **Installer:** a failed `uv`, `pipx` or `pip` install printed `installed:` and exited 0, because
  `set -e` ignores a failing command on the left of `&&`. The driver check warned on machines whose
  `libcuda.so.1` was in one of its paths but not all of them. `pip --user` on an externally managed
  Python (PEP 668, the default on current Debian, Ubuntu and Arch) is now followed by the
  zipapp fallback, and the closing message no longer says the PyTorch pipeline was installed when
  the lite-only zipapp was.
* `gpu_bench.py` judged every GPU against fixed 200 TFLOPS / 800 GB/s bars, telling healthy cards
  their driver might be broken; it now prints raw numbers only. The "TF32" row of
  `gpu_bench_details.py` ran in FP32 because TF32 matmul is opt-in in PyTorch.

### Documentation

* README rewritten: shorter, with a real output sample, both step lists, the verdict rules and
  exit codes, a reproducible protocol for benchmarking a new GPU, an options table and a
  troubleshooting table. It also corrects claims the code did not back: verified hardware with no
  run on file, a PCIe step in the PyTorch suite, and Pascal support through `--nvrtc`.
* New `docs/STEPS.md`: every step and every threshold, taken from the evaluation code.
* `docs/TESTED_GPUS.md` reorganised; the RTX 5090 is listed as reported until a run is on file.

## [0.3.2] - 2026-09-28

Documentation and installer usability fix. The lite pipeline, the kernels and the
evaluation rules are unchanged from 0.3.1.

### Fixed

* The documented way to install the PyTorch extra did not work. `VAR=value curl ... | sh`
  sets the variable for `curl`, not for the `sh` that runs the script, so the installer
  never saw `GPU_STRESS_EXTRAS` and silently installed the lite pipeline only. The
  installer now takes command-line options, which cannot be misplaced the same way:

      curl -fsSL .../install.sh | sh -s -- --torch

  `--extras`, `--torch`, `--version`, `--method`, `--bin` and `--help` are accepted, each
  overriding its `GPU_STRESS_*` variable, and the README, the installer's own help and the
  `gpu-stress` missing-torch hint all show the working form.

## [0.3.1] - 2026-09-28

Follow-up to 0.3.0: the installer set up two commands but only one working pipeline.

### Added

* `GPU_STRESS_EXTRAS` for the installer, so the PyTorch pipeline can be installed in the
  same command:

      curl -fsSL .../install.sh | sh -s -- --torch

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

[0.4.0]: https://github.com/HamzaGbada/gpu-stress-test/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/HamzaGbada/gpu-stress-test/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/HamzaGbada/gpu-stress-test/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/HamzaGbada/gpu-stress-test/releases/tag/v0.3.0
[0.2.0]: https://github.com/HamzaGbada/gpu-stress-test/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/HamzaGbada/gpu-stress-test/releases/tag/v0.1.0
