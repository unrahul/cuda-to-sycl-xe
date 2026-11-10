---
name: cuda-to-sycl-xe
description: Port CUDA or HIP kernels to SYCL and make SYCL kernels fast on Intel GPUs - Xe-HPG (Arc A-series), Xe2 (Arc B-series, Arc Pro B50/B60/B70, Lunar Lake), Xe3 (Panther Lake) and Xe3P (D270 / Crescent Island). Use for converting .cu or HIP code, writing icpx -fsycl kernels or PyTorch XPU SYCL extensions, roofline and headroom analysis, register spills, GRF modes and occupancy, memory coalescing, SLM, 2D block loads and prefetch, joint_matrix / DPAS / XMX GEMMs, kernel fusion, oneMKL and oneDNN calls, profiling with unitrace, VTune or Advisor, and SYCL build, link or device errors. Not for whole-framework migration or model serving.
---

# CUDA/HIP to SYCL on Intel GPUs

Write one SYCL source that runs on every Intel generation. Find the bottleneck with numbers, then apply the change that moves the most bytes or FLOPs out of the critical path. The generations share the programming model (work-groups, sub-groups, local memory, XMX through `joint_matrix`). They differ in sub-group widths, matrix tiles, register modes, caches and limits, so keep those differences in queried capabilities and per-device tuning, not in separate algorithms.

## Start here

```sh
source /opt/intel/oneapi/setvars.sh        # before `set -u`: the script reads unset variables
scripts/xe_env_check.sh                     # compiler, GPUs, sub-groups, matrix tiles, torch
export ONEAPI_DEVICE_SELECTOR=level_zero:0  # exactly one GPU, picked from the listing
icpx -fsycl -O2 -std=c++20 kernel.cpp -o kernel && ./kernel
unitrace -d ./kernel                        # per-kernel time, spills, GRF, SLM, SIMD width
```

- AOT builds: `-fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device <target>"` with `acm-g10` (Xe-HPG), `bmg` (Xe2), `cri` (Xe3P). An AOT build also prints register spills for that target, even without the hardware.
- PyTorch extensions: copy the closest tested example, and go through the pre-build checklist in [PyTorch integration](references/pytorch-xpu.md) before the first build. Most failed builds are binding mistakes, not kernel bugs.

## Workflow

1. **Contract.** Write down shapes, strides, dtypes, aliasing, accumulation type and tolerance. Write the host reference first: FP64, or wide integers for integer math ([testing](references/testing.md)).
2. **Analysis, written down before any code.** For the whole operation chain, write in your notes:
   - FLOPs;
   - minimum bytes, and the bytes of every intermediate tensor;
   - the time floor on *this* device;
   - whether the math lets you skip work.

   These numbers decide whether to restructure the algorithm, fuse, call a library, move to XMX, fix registers or stop ([analysis](references/analysis.md)).
3. **Correct port.** Translate semantics with the [CUDA/HIP mapping](references/cuda-to-sycl-mapping.md), structurally close to the source. Fix build errors from the [error table](references/troubleshooting.md) before changing the algorithm.
4. **Optimize until near the floor.** One hypothesis per variant, checked against the reference, then timed interleaved with the baseline. Keep going while the best result is far from the floor and attempts remain ([optimization loop](references/playbook.md), [measurement](references/performance.md)).
5. **Dispatch.** Gate specialized paths on queried capabilities and tested shapes. Unknown devices and shapes keep the portable path ([architectures](references/xe-architectures.md)).

## Route by bottleneck

| What limits the kernel, or the task | Read |
|---|---|
| Not yet known; how much faster could it get? | [Analysis](references/analysis.md), then [profiling](references/profiling.md) |
| Bytes: coalescing, vector or 2D block loads, SLM, caches, layout, prefetch | [Memory](references/memory.md) |
| Registers: spills, GRF mode, occupancy, too few work-groups | [Registers](references/registers.md) |
| Round trips between kernels, norms, epilogues, launch overhead | [Fusion](references/fusion.md) |
| GEMM, XMX / `joint_matrix`, convolution | [GEMM](references/gemm.md), [backend selection](references/backend-selection.md) |
| INT8, INT4, FP8, MXFP4 formats | [Quantized compute](references/quantized-compute.md) |
| Vector loads, quantized GEMV, MoE top-k, KV cache, which library supports which generation | [Production patterns](references/production-patterns.md) |
| Generation facts, detection, feature gates, Xe3P paths | [Architectures](references/xe-architectures.md) |
| Converting CUDA or HIP code | [Mapping](references/cuda-to-sycl-mapping.md), [kernel rules](references/kernel-rules.md) |
| Hopper/Blackwell idioms (TMA, `cp.async`, WGMMA, warp specialization, FP8/FP4), FlashAttention, SYCLomatic output | [Modern CUDA idioms](references/cuda-modern-idioms.md) |
| Build, link or runtime error | [Troubleshooting](references/troubleshooting.md) |
| PyTorch XPU extension | [PyTorch integration](references/pytorch-xpu.md) |
| Numerical tests; timing and accepting a speedup | [Testing](references/testing.md), [performance](references/performance.md) |

## Pitfalls

Pitfalls live next to the technique they affect, each stated once:

- PyTorch extensions: the numbered rules in [PyTorch integration](references/pytorch-xpu.md);
- kernels: [kernel rules](references/kernel-rules.md);
- each topic module: its own body and *Gotchas* section;
- by error message: [troubleshooting](references/troubleshooting.md).

Deliver what ran: device, driver, compiler, commands, pass/fail/skip per path, the distance from the time floor, and what was not tested.

## Bundled files

- `scripts/xe_env_check.sh`: toolchain, devices and capabilities in one report.
- `examples/xe_probe.cpp`: `--capabilities` prints device facts; without arguments it measures launch cost and read-pattern rates.
- `examples/snippets/`: the compiled code quoted in the references. `tests/snippets_test.cpp` and `tests/block2d_test.cpp` check each snippet against a host reference.
- `examples/portable_gemv.cpp`, `elementwise/`, `gemv_int8/`: CUDA sources with SYCL ports, from portable baseline to a capability-gated DPAS path.
- `examples/pytorch_xpu/`: named-kernel, oneMKL and oneDNN extensions with a reference checker.
- `scripts/parity_template.cpp`, `scripts/accept_performance.py`: parity-test skeleton and paired-timing acceptance gate.
- `scripts/fault_count.sh`, `scripts/switch_bisect.sh`: kernel-journal fault counts and interleaved bisection for device faults.
- `templates/AGENTS.md`: project instructions for a kernel repository.
