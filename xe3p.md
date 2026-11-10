# Xe3P (Intel GPU D270, "Crescent Island") — everything the skill needs

The single home for Xe3P facts. The shipped skill states these values plainly. This file records where each came from, whether it has been confirmed on hardware, and how to confirm it. When a value is confirmed or corrected, update the skill page named in the last column and mark the row confirmed here.

Status:

- **confirmed:** checked on a D270.
- **source:** from Intel documentation or driver/compiler sources, not yet run on hardware.
- **open:** sources conflict or are missing.

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
| PCI device ids | `0x674C`–`0x6751` | compute-runtime `xe3p_core/cri/device_ids_configs_cri.h:13` | source | `xe_probe --capabilities` device id | xe-architectures (snippet `generation`) |
| `architecture` enum | `intel_gpu_cri`, oneAPI 2026.2+ | sgl-kernel-xpu `Device.cpp` comment; intel/llvm device-architecture spec | source | compile `dispatch.hpp` with 2026.2; run `snippets_test` (prints the generation) | xe-architectures |
| AOT device names | `cri`, also `xe3p-xpc` | ocloc accepts `cri` (checked on the B70 host); compute-runtime `platforms.h:65,142` | source | `icpx -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device cri"` | xe-architectures |
| Matrix combinations exposed | Xe2 set plus fp8 and e2m1; needs a recent stack | IGC; sycl-tla `mma_xe.hpp` | open | `xe_probe --capabilities` matrix list on 2026.2+ | xe-architectures, gemm |
| oneMKL support | "initial support" from oneMKL 2026.1 | oneMKL 2026.1 release notes | source | run `examples/pytorch_xpu` mkl check | backend-selection |

## Compute and registers

| Parameter | Value used | Source | Status | Check | Skill page |
|---|---|---|---|---|---|
| Xe-cores / XMX engines | 32 / 256 | Intel D-series product page | source | `max_compute_units`, `gpu_eu_count` | analysis |
| Sub-group sizes | 16, 32; native SIMD32 for all shaders | compute-runtime `gfx_core_helper_xe3p_core.cpp:27`; IGC `Platform.hpp:1563-1600` | source | `sub_group_sizes`; `unitrace -d` SIMD column for an unpinned kernel | xe-architectures |
| GRF register size | 64 B | IGC `Platform.hpp:1012` | source | — | registers |
| GRF counts → threads per vector engine | 32–192 → 8, 256 → 4, 512 → 4 | IGC `visa/G4_Kernel.cpp` `GRFMode` | **open**: compute-runtime `release_helper_cri.cpp:22-24` gives 8 threads at 256 | `unitrace -q -g ComputeBasic` occupancy for 128/256/512 GRF kernels | registers, xe-architectures |
| Default GRF ceiling | 512 (`EnableCRIDefault512GRF=1`) | IGC `Platform.hpp:103-108`, `igc_flags.h:2155` | source | `unitrace -d` "Register File Size Per Thread" for a register-heavy kernel with no request | registers |
| `grf_size<512>` property | accepted for CRI | intel/llvm `sycl_ext_intel_grf_size` spec | source | compile a kernel with `grf_size<512>`; confirm with `unitrace -d` | registers |
| Register file per Xe-core | 1 MB | Intel Hot Chips 2026 deck | source | — | registers |
| Max work-group size | up to 2048 | compute-runtime `gfx_core_helper_xe3p_core.cpp:31-33` | source | `max_work_group_size` | xe-architectures |
| Threads per work-group cap | 512 GRF → 16, 256 or SIMD32 → 32, 192 → 40, 160 → 48, else 64 | compute-runtime `gfx_core_helper_xe3p_core.cpp:37-60` | source | kernel `work_group_size` query per GRF mode | registers |
| FP64 | supported, full rate (64 FMA per Xe-core) | compute-runtime `ftrSupportsFP64`; Hot Chips | source | `aspect::fp64`; FP64 FMA microbenchmark | xe-architectures |
| Native tanh, sigmoid | yes | IGC `Platform.hpp:1630-1640` | source | ISA dump of a kernel using `sycl::tanh` | — |

## Matrix engine (DPAS)

| Parameter | Value used | Source | Status | Check | Skill page |
|---|---|---|---|---|---|
| Width / depth | SIMD16, N = 16; Intel describes a 16-deep systolic array | IGC; Hot Chips | **open**: IGC encodes depth 8 with repeat count ≤ 8 for the exposed instruction | DPAS throughput microbenchmark | xe-architectures |
| Input types | Xe2 set plus fp8 E4M3/E5M2 (any mix) and e2m1 | IGC `hasFP8Dpas`, `hasFP4DPAS`; sycl-tla `mma_xe.hpp:254-283` | source | `matrix_combinations` | gemm, cuda-modern-idioms |
| int8 × int4 DPAS | dropped | sycl-tla comment "the dpas is removed" | source | `matrix_combinations` | xe-architectures |
| Block-scaled DPAS (`bdpas`) | M = 8, group-32 E8M0 (MX) scales | sycl-tla `mma_xe.hpp:147-168`, example 50 | source | sycl-tla example 50 on D270 | cuda-modern-idioms |
| NVFP4 (group-16 E4M3 scales) | no native path assumed | none found | **open** | sycl-tla xe35 atoms; run an NVFP4 GEMM | cuda-modern-idioms, quantized-compute |
| FP64 DPAS | behind a compiler flag, off by default | IGC `EnableFP64Dpas` | source | — | — |
| fp16 → E4M3 conversion | native instruction | sycl-tla `quantize_xe.hpp:67-110` | source | ISA dump | production-patterns |

## Memory

| Parameter | Value used | Source | Status | Check | Skill page |
|---|---|---|---|---|---|
| SLM per work-group | 384 KB | compute-runtime `hw_info_cri.cpp:53` (`maxProgrammableSlmSize`) | source | `local_mem_size` | xe-architectures, memory |
| L1 + SLM per Xe-core | 512 KB | Hot Chips (consistent with 384 KB SLM plus L1) | source | — | memory |
| L2 | 32 MB | Hot Chips | source | `global_mem_cache_size` | analysis, memory |
| Memory | 160–480 GB LPDDR5X, ≈1.5 TB/s | Intel OCP 2026 announcement (as reported) | source | `global_mem_size`; `xe_probe` stream rate | analysis |
| FP8 peak | ≈1.3 PFLOPS | same announcement | source | microbenchmark | analysis |
| 2D block loads | as Xe2, plus transposed 32-bit width ≤ 16, 64-bit height 8–16 / width ≤ 8, 256-byte prefetch rows, rows ≤ 64 | IGC `Subgroup2DBlockIoResolution.cpp:408-456`, `IsaVerification.cpp:4701-4818` | source | extend `tests/block2d_test.cpp` with a width-16 transpose | memory |
| Reads outside a 2D surface | zero (as on Xe2) | confirmed on Xe2 only | **open** | `tests/block2d_test.cpp` | memory |
| Scratch per thread | up to 16 MB | IGC `Platform.hpp:470` | source | — | — |
| Out-of-bounds GRF access | unsupported on Xe3 and later | IGC `Platform.hpp:933` | source | — | — |

## Behavior and software

| Item | Expectation | Source | Status | Check |
|---|---|---|---|---|
| Kernel shapes in Intel's code | same structure as Xe2: register-resident tiles, 2D loads, prefetch, split barriers; FA head-dim-128 tile 512×64×64 with 32 sub-groups, 512 GRF | vllm-xpu-kernels `fmha_utils.hpp:96-105`; sycl-tla FA example | source | run sycl-tla FA example |
| FP8 flash attention | softmax exponent biased by +8, P packed to FP8 | vllm-xpu-kernels xe_3 mainloop | source | — |
| ESIMD | paths that run on BMG may fail on CRI | the skill's original author | open | run one ESIMD sample |
| Preemption | thread-group (Xe2: mid-thread) | compute-runtime `hw_info_cri.cpp:49` | source | — |
| Early-silicon faults | apply the fault protocol (kernel rules 29–34) | — | — | `scripts/fault_count.sh` per run |
