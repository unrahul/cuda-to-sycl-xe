---
name: cuda-to-sycl-xe
description: Port CUDA or ROCm/HIP kernels to SYCL, write new Intel GPU kernels, or optimize existing SYCL kernels with capability-gated variants and reproducible numerical and performance tests. Use for Xe, Xe2 and Xe3P kernel work, including Arc Pro B70. Does not cover whole-framework migration or serving configuration.
---

# Intel Xe kernel engineering

Build one correct native path, then optimize measured bottlenecks. Keep a correct fallback and reproducible evidence. Read only the route relevant to the current decision.

## Route

| Task | Read |
|---|---|
| CUDA or ROCm/HIP conversion | [Mapping](references/cuda-to-sycl-mapping.md), preserving the original mathematical and memory contract |
| New kernel | [Validation](references/testing.md), starting with an independent host reference and portable baseline |
| PyTorch caller | [Extension integration](references/pytorch-xpu.md) and its complete examples before writing the binding |
| GEMM or convolution | [Backend selection](references/backend-selection.md), then [dense dataflow](references/dense-compute.md) |
| Quantized or block-scaled arithmetic | [Quantization contracts](references/quantized-compute.md) |
| Subgroups, matrix instructions or specialization | [Kernel rules](references/kernel-rules.md) and [architecture gates](references/xe-architectures.md) |
| Fusion, grouped or stateful inference | [Production patterns](references/production-patterns.md) |

## Work loop

1. Recover shapes, strides, aliasing, dtypes, state and the numerical contract. Fix tolerances before tuning. Use independent arithmetic on represented inputs. Host FP64 does not require device FP64.
2. Read project instructions and select the permitted backend and execution interface. Preserve compiler/runtime/device identities. In a standalone project, configure one device and compile a small example. In a managed assessment, use its supplied environment, policy check and request interface. Record unavailable probes or profiling explicitly. Use workspace-qualified paths when shell working directories do not persist.
3. Obtain a correct build before changing algorithms. For PyTorch, start from the supplied named-kernel example. Let its loader own the C++ standard and ABI. For standard dense work, compare a qualified native-library path when permitted. Preserve full compiler diagnostics. Classify policy, compile, link, runtime and numerical failures before choosing a repair.
4. Follow the [optimization workflow](references/playbook.md): one hypothesis per variant, numerical checks before timing, then fresh interleaved confirmation. Include allocation, packing, workspace and synchronization costs relevant to the caller. Keep earlier correct snapshots.

## Acceptance

- Apply [validation requirements](references/testing.md) to changed arithmetic, boundaries, repeated calls, streams and dispatch. Exercise the actual specialized path and report unavailable paths as SKIP.
- Use [performance measurement](references/performance.md) for work counts, raw samples and noise. Require gains over the best correct control, then preserve a fallback for unseen contexts. Record unprofiled explanations as hypotheses.
- Gate features on the exact compiler, runtime and device capabilities. A generation name cannot establish support. Preserve full-FP32 arithmetic when required. Investigate missing matrix support using the linked architecture policy.
- Use finite timeouts. Stop on device loss and save diagnostics. Do not stop unrelated services or change driver settings to obtain a measurement.
- Deliver code, independent checks, commands, raw results and exact coverage. Keep measured numbers, local execution status and tuning records outside the shared skill. Specifications require exact product, datatype and source citations.

[Standalone examples](examples/README.md), [project contract template](templates/AGENTS.md), and [public sources](references/sources.md) are optional supporting material. Inspect source and license before reuse.
