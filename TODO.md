# Status and remaining work

The skill is maintained in git. `SKILL.md`, `references/`, `examples/`, `scripts/`, `templates/` and `tests/` ship; everything else is maintainer material. Build and install with `internal_tools/package_skill.py` (see `internal_tools/package_skill.md`).

## 2026-11-10 rewrite

The paired Haiku pilot (6 of 8 pairs, stopped by expired authentication) showed no benefit from the previous version:

| | Control | With skill |
|---|---|---|
| Correct | 5 | 5 |
| Faster than baseline | 5 | 4 |
| At least 2x faster | 3 | 3 |
| Cost | $3.48 | $5.00 |

Most failures were practical: a forced `-std=c++17`, unnamed kernels under PyTorch's `-fsycl-host-compiler`, `-lonemkl`, and policy rejections from the harness. The fixes:

- **Rewritten docs.** `SKILL.md` now opens with a quick start, the costly errors and routing. The references now state facts and code instead of repeated cautions.
- **New `references/troubleshooting.md`.** Its rows come from pilot logs and from reproductions on B70.
- **Compiled code blocks.** Every code block is a region in `examples/snippets/`, run on B70 by `tests/snippets_test.cpp`. `tests/test_docs.py` keeps the docs and code in sync.
- **Per-generation facts.** Xe-HPG, Xe2 and Xe3P facts are labeled [B70] (measured here), [spec] or [report] (original author, not reproduced).
- **Rules restored.** The original failure-named kernel rules are back, with overstated performance claims reworded as hypotheses.
- **Verified on B70.** `-qmkl` links on 2025.3.3 and 2026.1, while `-qmkl=sycl` fails on 2025.3.3. `icpx` contracts FMAs in device code even with `-fp-model=precise`. `TORCH_XPU_ARCH_LIST` selects PyTorch's AOT targets.
- **Removed.** Harness wording and maintainer paths are gone from shipped text. Maintainer evidence moved to `internal_references/`. `make_table.py` was removed; `accept_performance.py` is the gate.

## Patterns from upstream (2026-11-10)

I mined llama.cpp `ggml-sycl`, torch-xpu-ops, vllm-xpu-kernels, sgl-kernel-xpu, llm-scaler, sycl-tla and SyclKittens, and added six compiled patterns. All of them pass on B70 with oneAPI 2025.3.3 and 2026.1.1:

- alignment-checked vector width;
- sub-group width found by searching the device list;
- Q4_0 reorder with a one-sub-group-per-row GEMV;
- MoE top-k;
- a gated bf16 `joint_matrix` GEMM (the Xe-HPG instantiation is AOT-compiled for `acm-g10`, not run);
- the SPIR-V 2D block load, whose out-of-surface reads were measured as zero.

The Xe3P code paths in the libraries are documented with their sources. None of the libraries supports Xe-HPG.

## L2_030 recheck with Codex gpt-6-luna (high), 2026-11-10

This task regressed most with the old skill: Haiku scored 0.44x with the skill versus 1.04x for control. Runs: `benchmarks/xe-skill-pilot/runs/dev-codex-l2_030-*`.

| Arm | Assessments used | Builds that failed | Fresh final |
|---|---|---|---|
| Skill (new package) | 3 | 1 | correct, 1.042x |
| Control | 6 | 6 | no correct result |

- **Skill arm.** It followed the tested oneMKL example with `-qmkl`, a named kernel and a `.sycl` source. Its one failure was `auto&` bound to the by-value `getCurrentXPUStream()` result; the docs now say this explicitly. It stopped near the ceiling instead of spending its remaining attempts.
- **Control arm.** Every failure is a pitfall the skill covers: a `.cpp` source (three times), `-lonemkl`, and hand-written MKL link lines that failed to load (twice).
- **Ceiling.** The strict-FP32 GEMM runs at about 21 TFLOP/s and dominates the time, so about 1.05x is the cap. `playbook.md` now teaches this headroom check. Large wins should be judged on fusion-heavy tasks such as L2_085 and L2_074.

## Restructure and paired holdout evaluation (2026-11-10)

The skill was reorganized into topic modules:

- analysis (roofline, headroom, doing less work);
- memory;
- registers (spills, GRF, occupancy);
- fusion;
- GEMM/XMX;
- profiling.

Every fact has one home, enforced by `tests/test_docs.py`. Sources: the JAX scaling book, Intel's GPU optimization guide, IGC and compute-runtime master, Intel papers, and upstream kernel repositories. Unverified Xe3P facts are listed in `to_check.md`.

**Method.** Codex `gpt-6-luna` at high effort, skill versus control, one run per arm, on tasks never used to develop the skill. Each task ran on its own GPU lane (renderD128–130), with both arms on the same GPU. Run with `benchmarks/xe-skill-pilot/dev_campaign.sh`; summarize with `dev_compare.py`; logs are in `runs/dev-logs/`.

| Holdout | Package | Skill correct | Control correct | Skill better / control better | Failed builds (S/C) |
|---|---|---|---|---|---|
| 1 (L2_025, L1_042, L2_096, L2_080, L2_014, L2_019) | build-20261110T192839Z | 3/6 | 2/6 | 3 / 1 | 1 / 5 |
| 2 (L2_012, L2_066, L2_070, L1_068, L1_071, L1_091) | build-20261110T195250Z | 6/6 | 3/6 | 4 / 2 | 0 / 14 |

Notable results:

| Task | Skill | Control | What happened |
|---|---|---|---|
| L2_080 | 4.91x | none | Skill agent found the output is identically zero |
| L1_042 | 4.45x | 1.25x | |
| L2_025 | 2.96x | none | |
| L1_071 | 1.31x | 0.31x | |
| L1_091 | 1.55x | 2.54x | Skill agent stopped early |

Across all 12 tasks: correct 9 vs 5, faster than baseline 6 vs 4, failed builds 1 vs 19.

**Fixes taken from these runs.** Each fix came from a failure in a holdout run, so those tasks are now development tasks for later versions:

- `load()` source paths (L2_019);
- hyperparameter binding (L2_096);
- FP32 tolerance on long reductions (L2_014);
- the oneDNN 3.x API and deconvolution weight layout (L2_025, L1_068), verified on B70;
- the "doing less work" lever (L2_014);
- host-container capture in kernels (L1_068);
- the continue-or-stop rule (L1_091).

Holdout 3 uses the last four untouched tasks: L1_002, L1_005, L1_021 and L1_038.

## Remaining

0. **CUDA-source conversion evaluation (the skill's primary job, not yet measured).** The holdouts give the agent a PyTorch model, which measures optimization, not CUDA translation. Plan:
   - **Tasks.** Each task is a PyTorch reference module plus a real CUDA implementation the agent must port. Candidates: NVIDIA cuda-samples (reduction, transpose, scan, histogram; BSD), llama.cpp `ggml-cuda` kernels with exact PyTorch semantics (softmax with scale/mask, RMSNorm, RoPE-NeoX; MIT), and one Hopper-style kernel using `cp.async` or `mma.sync` to exercise `references/cuda-modern-idioms.md`.
   - **Harness.** Stage the tasks in a dev sample next to the frozen one, with `dev_adapter.py` patching the controller's `SAMPLE`, `SAMPLE_SHA256` and `TASKS`. `dev_episode.py` copies `cuda/*.cu` read-only into the workdir and appends "Port the supplied CUDA implementation" to the prompt.
   - **Metrics.** Correctness first, then speed against the PyTorch XPU baseline. Also check whether the port preserved the CUDA algorithm where that was the right call.
1. **Larger paired evaluation.** The 12-task result is one run per arm. Repeat with at least three runs per arm on new task sets, and with a second model, before claiming a general effect. All 24 pilot tasks have now been seen, so new tasks are needed, for example more KernelBench L2/L3 tasks staged into the evaluator.
2. **Hardware coverage.** Run the probe, `scripts/smoke.sh` and `tests/snippets_test.cpp` on Xe-HPG and Xe3P hardware, then turn [report] rows in `references/xe-architectures.md` into measured ones.
3. **CUDA/HIP source-device comparison.** This needs NVIDIA/AMD hardware; until then, ports are validated against host references only.
