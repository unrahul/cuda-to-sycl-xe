# cuda-to-sycl-xe

An agent skill for converting CUDA (and HIP) kernels to SYCL and making them fast on Intel GPUs:

- Xe-HPG: Arc A-series;
- Xe2: Arc B-series, Arc Pro B50/B60/B70, Lunar Lake;
- Xe3: Panther Lake;
- Xe3P: GPU D270, "Crescent Island".

It works with Claude Code, Codex and opencode. The entry point is `SKILL.md`, which routes to topic modules in `references/`:

- analysis (roofline and headroom);
- memory and data movement;
- registers, spills and occupancy;
- fusion;
- GEMM and XMX;
- modern CUDA idioms (TMA, `cp.async`, WGMMA, warp specialization, FP8/FP4, FlashAttention);
- the CUDA-to-SYCL mapping;
- PyTorch XPU extensions;
- testing, performance and troubleshooting.

Every code block in the references is compiled and run by `tests/snippets_test.cpp` and `tests/block2d_test.cpp`. `tests/test_docs.py` keeps the documentation honest:

- every code block comes from a compiled snippet;
- links resolve, and every reference is reachable from `SKILL.md`;
- each fact is written once;
- no provenance or maintainer text ships.

## Add the skill to another repository

1. Build the package:

   ```sh
   python3 internal_tools/package_skill.py        # writes dist/build-<UTC>/cuda-to-sycl-xe/ and a .zip
   ```

   The build fails unless the package passes its own docs tests. It contains `SKILL.md`, `references/`, `examples/` (sources only), `scripts/`, `templates/`, `tests/` and `agents/openai.yaml` (Codex UI metadata).

2. Install it in one of two ways:
   - **Project-level, for Claude Code:** copy the package directory to `<repo>/.claude/skills/cuda-to-sycl-xe/` and commit it.
   - **User-level:** `python3 internal_tools/package_skill.py --install codex,claude,opencode`. This installs to `~/.codex/skills`, `~/.claude/skills` and `~/.config/opencode/skills`; add `--replace` to move an existing install aside.

3. Requirements on the target machine:
   - **Required:** the oneAPI DPC++ compiler (`icpx`) and a Level Zero GPU.
   - **Optional:**
     - PyTorch XPU with `cpp_extension.load(with_sycl=True)`;
     - oneMKL (link with `-qmkl`);
     - a standalone oneDNN built with the SYCL GPU runtime;
     - `unitrace`, VTune or Advisor for profiling.
   - **Matrix-engine examples:** need oneAPI 2026.1 or later on Xe2. Earlier stacks expose no matrix combinations.

4. Self-check:

   ```sh
   source /opt/intel/oneapi/setvars.sh
   scripts/xe_env_check.sh                                  # compiler, GPUs, sub-groups, matrix tiles, torch
   ONEAPI_DEVICE_SELECTOR=level_zero:0 scripts/smoke.sh     # builds and runs every example and snippet test
   python3 -m unittest discover -s tests                    # CPU and documentation tests
   ```

## Evaluation results

**Method.**

- Codex `gpt-6-luna` at high effort solves KernelBench-derived tasks, with and without the skill.
- Each arm gets the same prompt, six assessments and a fresh final assessment by the harness.
- Both arms of a task share one Arc Pro B70.
- Tasks are holdouts never used to develop the skill. Each holdout used the package current at that time.
- Speedup is relative to PyTorch XPU on the same GPU. "None" means no correct result.

**Holdout 1** (package `build-20261110T192839Z`):

| Task | Skill | Control |
|---|---|---|
| L2_025 conv + min + tanh | **2.96x** | none |
| L1_042 max-pool | **4.45x** | 1.25x |
| L2_080 GEMM + max + GELU | **4.91x** | none |
| L2_019 transposed conv + GroupNorm + GELU | none | **1.11x** |
| L2_096 3D transposed conv + pooling | none | none |
| L2_014 GEMM + sum | none | none |

**Holdout 2** (package `build-20261110T195250Z`):

| Task | Skill | Control |
|---|---|---|
| L2_012 GEMM + LeakyReLU | **1.13x** | none |
| L2_066 matmul + softmax | 0.98x | **1.02x** |
| L2_070 GEMM + sigmoid | **0.99x** | none |
| L1_068 3D transposed conv | **0.10x** | none |
| L1_071 transposed conv | **1.31x** | 0.31x |
| L1_091 reverse cumsum | 1.55x | **2.54x** |

**Holdout 3** (package `build-20261110T204734Z`):

| Task | Skill | Control |
|---|---|---|
| L1_002 matmul | **1.21x** | 1.18x |
| L1_005 matrix-scalar multiply | **2.70x** | none |
| L1_021 sigmoid | **2.70x** | none |
| L1_038 L1 normalization | **2.53x** | none |

**Totals over all 16 tasks:**

| | Skill | Control |
|---|---|---|
| Correct | 13 | 6 |
| Faster than baseline | 10 | 5 |
| Failed builds | 2 | 24 |
| Incorrect assessments | 22 | 70 |
| Geomean speedup of correct results | 1.55x | 1.05x |
| Better arm (2 ties) | 11 tasks | 3 tasks |

**Before the rewrite.** In a 6-task Haiku pilot (5 correct per arm, $5.00 versus $3.48), the previous version showed no benefit.

**What the skill changed:**

- **Correctness.** It removes the build and binding mistakes that sink most attempts: `.cpp` instead of `.sycl` sources, `-std=c++17`, `-lonemkl`, `requires_grad` guards, argument types, `load()` paths and oneDNN 3.x descriptors.
- **Large wins.** These come from analysis: doing less work (an identically zero output), fusing memory-bound chains, and using the right library.
- **Limits.** Each result is one run per arm, and every task is a PyTorch module, not a CUDA source to translate. A CUDA-source conversion evaluation is the next step (`TODO.md`).

Reproduce:

```sh
benchmarks/xe-skill-pilot/dev_campaign.sh <package> <tag> renderD128:TASK,... renderD129:TASK,...
benchmarks/xe-skill-pilot/dev_compare.py TASK ...
```

## Xe3P (D270)

D270 is pre-production hardware, so this repository records its capabilities and limits, not its performance. [`xe3p.md`](xe3p.md) is the single file for Xe3P. Each row gives the value, its source, its status and how to check it.

**Checked on a D270** (oneAPI 2026.1, compute runtime 26.44.40863):

| Item | Value |
|---|---|
| Device id; AOT target | `0x674C`; `-device cri` |
| Architecture enum | not named by oneAPI 2026.1; detected through the device id |
| Vector engines | 256 (4 slices × 8 sub-slices × 8); `max_compute_units` counts these, not Xe-cores |
| Sub-group sizes | 16, 32 |
| Max work-group | 1024 |
| Threads per vector engine | 8 at 128 and at 256 GRF |
| `grf_size<512>` | rejected by oneAPI 2026.1 (static assertion) |
| Local memory per work-group | 384 KB |
| L2; cache line | 32 MB; 256 bytes |
| Aspects | fp16, fp64, atomic64 |
| Matrix aspect / `joint_matrix` combinations | 0 / none on oneAPI 2026.1 |
| DPAS through ESIMD and vISA | bf16, fp16 and int8 with depth 8, sub-group 16, N = 16 |
| 2D block loads | Xe2 forms run; reads outside the surface return zero; base 64-byte aligned, pitch a multiple of 16 bytes |
| ESIMD | `xmx::dpas`, SLM block load and store, `block_load`, gathers and barriers run |
| Clock range | 400–2500 MHz |

**Still to obtain or confirm** (enter values in `xe3p.md`):

1. **512-register kernels:** the default GRF ceiling and threads per vector engine at 512, with a compiler that accepts `grf_size<512>`.
2. **DPAS:**
   - whether the array is 16 deep;
   - what `matrix_combinations` reports on oneAPI 2026.2 or later;
   - fp8 and e2m1 support, block-scaled DPAS with group-32 E8M0 scales, and whether NVFP4 (group-16 E4M3 scales) has a native path.
3. **Published SKU figures:** peak FP32, FP16/BF16 XMX, FP8, INT8 and FP64 rates, and memory bandwidth.
4. **2D block loads:** the Xe3P-only widths (transposed 32-bit width 16, 64-bit elements) and the prefetch limits.
5. **Toolchain:** the oneAPI version that names `intel_gpu_cri`, and oneMKL/oneDNN coverage.

On a D270, start with the commands at the top of `xe3p.md`.

## Repository layout

| Path | What it is |
|---|---|
| `SKILL.md`, `references/`, `examples/`, `scripts/`, `templates/`, `tests/` | the shipped skill |
| `internal_tools/` | packaging (`package_skill.py`) and maintainer diagnostics |
| `evals/` | trigger queries and task definitions |
| `benchmarks/xe-skill-pilot/` | paired evaluation harness; `dev_*.py` run development campaigns |
| `TODO.md` | status, evaluation history, remaining work |
| `to_check.md` | non-Xe3P facts awaiting hardware confirmation |
| `xe3p.md` | everything the skill needs about Xe3P |
