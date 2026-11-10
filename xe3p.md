# Xe3P (Intel GPU D270, "Crescent Island") — everything the skill needs

The single home for Xe3P facts. The shipped skill states these values plainly. This file records where each came from, whether it has been confirmed on hardware, and how to confirm it. When a value is confirmed or corrected, update the skill page named in the last column and mark the row confirmed here.

Status:

- **confirmed:** checked on a D270.
- **source:** from Intel documentation or driver/compiler sources, not yet run on hardware.
- **open:** sources conflict or are missing.

"Checked on a D270" means: two D270 cards, oneAPI DPC++ 2026.1.0.20260617, compute runtime 26.44.40863 (Level Zero driver 1.18.40863,
`sycl-ls` reports IP 35.11.0), ocloc 26.35 and 26.44 for `-device cri`, between 2026-11-04 and 2026-11-10. The values come from
`scripts/xe_env_check.sh`, `examples/bin/snippets_test`, `tests/block2d_test.cpp` and `xe_probe --capabilities` on that stack, sycl-tla
example 50 (built for `intel_gpu_cri`), small inline-vISA probes of the DPAS depth and the 2D block-load forms (each form either fails to
assemble or loads a block whose cells hold their own row and column, checked for every cell exactly once), and a oneMKL `gemm` check.
This is pre-production hardware: the file records capabilities and limits, not performance.

On a D270, start with:

```sh
source /opt/intel/oneapi/setvars.sh          # oneAPI 2026.2 or later names the architecture
ONEAPI_DEVICE_SELECTOR=level_zero:0 scripts/xe_env_check.sh
ONEAPI_DEVICE_SELECTOR=level_zero:0 scripts/smoke.sh
unitrace -d examples/bin/snippets_test
```

## Identity and toolchain

| Parameter | Value used | Source | Status | Check | Skill page |
|---|---|---|---|---|---|
| PCI device ids | `0x674C`–`0x6751` | compute-runtime `xe3p_core/cri/device_ids_configs_cri.h:13` | `0x674C` confirmed (both cards); the rest of the range source | `xe_probe --capabilities` device id | xe-architectures (snippet `generation`) |
| `architecture` enum | `intel_gpu_cri`, oneAPI 2026.2+ | sgl-kernel-xpu `Device.cpp` comment; intel/llvm device-architecture spec | source for 2026.2; **confirmed** that 2026.1 does not name it: the device-id fallback detects the card (`snippets_test` prints `generation: Xe3P`), and `ext_intel_device_id` reads `0x674c` | compile `dispatch.hpp` with 2026.2; run `snippets_test` (prints the generation) | xe-architectures |
| AOT device names | `cri`, also `xe3p-xpc` | ocloc accepts `cri`; compute-runtime `platforms.h:65,142` | `cri` **confirmed**: binaries built with ocloc 26.35 and 26.44 run on the D270; `xe3p-xpc` source | `icpx -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device cri"` | xe-architectures |
| Matrix combinations exposed | Xe2 set plus fp8 and e2m1; needs a recent stack | IGC; sycl-tla `mma_xe.hpp` | **corrected for 2026.1**: the matrix aspect is 0 and `joint_matrix` offers no combination (no bf16 8x16x16, no int8 8x16x32; `snippets_test` skips its `joint_matrix` case, `examples/gemv_int8` falls back to the portable GEMV). DPAS itself runs from ESIMD `xmx::dpas` and inline vISA `dpas` (see Matrix engine). 2026.2+ open | `xe_probe --capabilities` matrix list on 2026.2+ | xe-architectures, gemm |
| oneMKL support | "initial support" from oneMKL 2026.1 | oneMKL 2026.1 release notes | SYCL BLAS `gemm` **confirmed** with oneMKL 2026.1: fp32, fp16 and bf16 inputs with fp32 output match a host reference (column-major, 256 x 192 x 320, JIT build with `-qmkl`); other domains not exercised | run `examples/pytorch_xpu` mkl check | backend-selection |
| Driver | compute runtime 26.44.40863 runs the checks clean | the D270 runs above | **confirmed** | `sycl-ls` driver version; `scripts/fault_count.sh` per run | troubleshooting |

## Compute and registers

| Parameter | Value used | Source | Status | Check | Skill page |
|---|---|---|---|---|---|
| Xe-cores / XMX engines | 32 / 256 | Intel D-series product page | **confirmed**: `ext_intel_gpu_slices` 4 x `gpu_subslices_per_slice` 8 x `gpu_eu_count_per_subslice` 8; `max_compute_units` reports 256, the vector engines, not the Xe-cores | `max_compute_units`, `gpu_eu_count` | analysis |
| Sub-group sizes | 16, 32; native SIMD32 for all shaders | compute-runtime `gfx_core_helper_xe3p_core.cpp:27`; IGC `Platform.hpp:1563-1600` | 16 and 32 **confirmed** (`sub_group_sizes`; kernels pinned to either run); the native width of an unpinned kernel open | `sub_group_sizes`; `unitrace -d` SIMD column for an unpinned kernel | xe-architectures |
| GRF register size | 64 B | IGC `Platform.hpp:1012` | source | — | registers |
| GRF counts → threads per vector engine | 8 at 128 and at 256 | compute-runtime `release_helper_cri.cpp:22-24` (IGC `visa/G4_Kernel.cpp` `GRFMode` says 4 at 256) | **confirmed by the runtime**: `gpu_hw_threads_per_eu` = 8, and `max_num_work_groups` for a one-sub-group kernel is 2048 = 8 x 256 with `grf_size<128>` and with `grf_size<256>`. 512 not testable on 2026.1 | `unitrace -q -g ComputeBasic` occupancy for 128/256/512 GRF kernels | registers, xe-architectures |
| Default GRF ceiling | 512 (`EnableCRIDefault512GRF=1`) | IGC `Platform.hpp:103-108`, `igc_flags.h:2155` | source | `unitrace -d` "Register File Size Per Thread" for a register-heavy kernel with no request | registers |
| `grf_size<512>` property | accepted for CRI by the spec | intel/llvm `sycl_ext_intel_grf_size` spec | **corrected for 2026.1**: the compiler rejects it (`grf_size_properties.hpp:56`: static assertion `512U == 128 \|\| 512U == 256`, "Unsupported GRF size"); `grf_size<128>` and `<256>` build and run | compile a kernel with `grf_size<512>`; confirm with `unitrace -d` | registers |
| Register file per Xe-core | 1 MB | Intel Hot Chips 2026 deck | source | — | registers |
| Max work-group size | 1024 | compute-runtime `gfx_core_helper_xe3p_core.cpp:31-33` allows up to 2048 | **corrected**: the device reports 1024, and so does a kernel's own maximum (`snippets_test`: saxpy kernel max work-group 1024) | `max_work_group_size` | xe-architectures |
| Threads per work-group cap | 512 GRF → 16, 256 or SIMD32 → 32, 192 → 40, 160 → 48, else 64 | compute-runtime `gfx_core_helper_xe3p_core.cpp:37-60` | source | kernel `work_group_size` query per GRF mode | registers |
| FP64 | supported, full rate (64 FMA per Xe-core) | compute-runtime `ftrSupportsFP64`; Hot Chips | support **confirmed** (`aspect::fp64` = 1, also fp16 and atomic64); rate source | `aspect::fp64`; published rate | xe-architectures |
| Native tanh, sigmoid | yes | IGC `Platform.hpp:1630-1640` | source | ISA dump of a kernel using `sycl::tanh` | — |

## Matrix engine (DPAS)

| Parameter | Value used | Source | Status | Check | Skill page |
|---|---|---|---|---|---|
| Width / depth | SIMD16, N = 16; Intel describes a 16-deep systolic array | IGC; Hot Chips | the depth-8 instruction **runs**: vISA `dpas.bf.bf.8.M` and `dpas.s8.s8.8.M` (M ≤ 8, sub-group 16, N = 16) and ESIMD `xmx::dpas<8, 8, float, float, half, half>` in kernels built for `cri`; on 2026.1 the vISA parser **refuses** `dpas.bf.bf.16.8`, so the exposed instruction is depth 8 (`dpas.bf.bf.8.8` with all-ones inputs gives 16); whether the array itself is 16 deep stays **open** | ISA and capability query on a newer stack | xe-architectures |
| Input types | Xe2 set plus fp8 E4M3/E5M2 (any mix) and e2m1 | IGC `hasFP8Dpas`, `hasFP4DPAS`; sycl-tla `mma_xe.hpp:254-283` | bf16 x bf16, fp16 x fp16 and s8 x s8 with fp32 / int32 accumulation **confirmed** (ESIMD and inline vISA, parity against FP64 references); fp8 E4M3, fp8 E5M2 and e2m1 inputs **confirmed** through sycl-tla example 50 (`_e4m3`, `_e5m2`, `_e2m1`, and the fp16- and fp32-scale variants `_fp16_e4m3`, `_fp32_e5m2`): each passes its own reference check on a D270 | `matrix_combinations` | gemm, cuda-modern-idioms |
| int8 × int4 DPAS | dropped | sycl-tla comment "the dpas is removed" | source | `matrix_combinations` | xe-architectures |
| Block-scaled DPAS (`bdpas`) | M = 8, group-32 E8M0 (MX) scales | sycl-tla `mma_xe.hpp:147-168`, example 50 | **confirmed**: sycl-tla example 50's hardware block-scaled path (ue8m0 scales, group size 32) passes for e2m1, e4m3 and e5m2 inputs, and the bf16 variant passes | sycl-tla example 50 on D270 | cuda-modern-idioms |
| NVFP4 (group-16 E4M3 scales) | no native path assumed | none found; sycl-tla's xe35 `bdpas` atoms take E8M0 byte scales, and example 50 has no group-16 E4M3 variant | **open** | sycl-tla xe35 atoms; run an NVFP4 GEMM | cuda-modern-idioms, quantized-compute |
| FP64 DPAS | behind a compiler flag, off by default | IGC `EnableFP64Dpas` | source | — | — |
| fp16 → E4M3 conversion | native instruction | sycl-tla `quantize_xe.hpp:67-110` | source | ISA dump | production-patterns |

## Memory

| Parameter | Value used | Source | Status | Check | Skill page |
|---|---|---|---|---|---|
| SLM per work-group | 384 KB | compute-runtime `hw_info_cri.cpp:53` (`maxProgrammableSlmSize`) | **confirmed** (`local_mem_size`) | `local_mem_size` | xe-architectures, memory |
| L1 + SLM per Xe-core | 512 KB | Hot Chips (consistent with 384 KB SLM plus L1) | source | — | memory |
| L2 | 32 MB, 256-byte lines | Hot Chips | **confirmed** (`global_mem_cache_size`, `global_mem_cache_line_size`) | `global_mem_cache_size` | analysis, memory |
| Memory | 160–480 GB LPDDR5X, ≈1.5 TB/s | Intel OCP 2026 announcement (as reported) | source | `global_mem_size`; published figure | analysis |
| FP8 peak | ≈1.3 PFLOPS | same announcement | source | published figure | analysis |
| 2D block loads | as Xe2, plus transposed 32-bit width ≤ 16 and 64-bit width ≤ 8 at height 8, prefetch rows ≤ 256 bytes and ≤ 64 rows | IGC `Subgroup2DBlockIoResolution.cpp:408-456`, `IsaVerification.cpp:4701-4818`; sycl-tla `copy_xe_2d.hpp` (`XE_LOAD_2D_TRANSPOSE` limits for target 35) | the Xe2 forms **run**: `tests/block2d_test.cpp`, and `__builtin_IB_subgroup_block_read_flat_transpose_u32_k8` / `_k1` and `_u16_m8k16v1` feeding int8 and bf16 DPAS, with the Xe2 surface rules (64-byte aligned base, pitch a multiple of 16 bytes: a matrix whose rows are 680 or 360 bytes needs a padded copy or a split layout). The Xe3P-only forms, vISA `lsc_load_block2d` on a payload from `__builtin_IB_subgroup_createBlock2DAddressPayload`: transposed `d32.16x16tn` and `d64.8x8tn` **confirmed** (every cell of the block once; lane l holds row l), as are the Xe2 `d32.8x16tn` and `d64.4x8tn`; `d64.8x16tn` assembles but returns **wrong data** (64-bit transposes need height 8, so the source's "height 8–16" is **corrected**); `d32.32x16tn` crashes the compiler; prefetch `d8.256x64nn` runs, and `d8.512x32nn` and `d8.256x128nn` are refused by the vISA parser | the vISA probes above; extend `tests/block2d_test.cpp` with these forms | memory |
| Reads outside a 2D surface | zero (as on Xe2) | `tests/block2d_test.cpp` | **confirmed** on a D270 (tiles straddling the right and bottom edges read zero outside, no memory outside) | `tests/block2d_test.cpp` | memory |
| Scratch per thread | up to 16 MB | IGC `Platform.hpp:470` | source | — | — |
| Out-of-bounds GRF access | unsupported on Xe3 and later | IGC `Platform.hpp:933` | source | — | — |

## Behavior and software

| Item | Expectation | Source | Status | Check |
|---|---|---|---|---|
| Kernel shapes in Intel's code | same structure as Xe2: register-resident tiles, 2D loads, prefetch, split barriers; FA head-dim-128 tile 512×64×64 with 32 sub-groups, 512 GRF | vllm-xpu-kernels `fmha_utils.hpp:96-105`; sycl-tla FA example | source | run sycl-tla FA example |
| FP8 flash attention | softmax exponent biased by +8, P packed to FP8 | vllm-xpu-kernels xe_3 mainloop | source | — |
| ESIMD | paths that run on BMG may fail on CRI | the skill's original author | ESIMD kernels with `xmx::dpas`, `slm_init` / `slm_block_load` / `slm_block_store`, `block_load`, gathers and `barrier`, built for `cri`, **run** clean; untested ESIMD features stay open | run one ESIMD sample |
| Preemption | thread-group (Xe2: mid-thread) | compute-runtime `hw_info_cri.cpp:49` | source | — |
| Early-silicon faults | apply the fault protocol (kernel rules 29–34) | — | — | `scripts/fault_count.sh` per run |
| Clocks | the cards came up at 400–2300 MHz with RP0 2500 MHz | `xpu-smi` | **confirmed** on both cards; `sudo xpu-smi config -d N -t 0 --frequencyrange 400,2500` sets the range and does not persist across a reboot | `xpu-smi discovery`, `xpu-smi stats` |
| Profiling tools | `unitrace -d` writes nothing for a process that leaves with `std::_Exit`; a PTI view collector loaded with `LD_PRELOAD`, writing its table on a timer and at a normal exit, works; `unitrace` stall sampling gave no data on this driver | the D270 runs above | **confirmed** | profile one short program both ways |
