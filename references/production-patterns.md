# Patterns from XPU inference kernels

Use the pinned [source catalog](sources.md) to inspect an implementation together with its dispatcher, build requirements and tests. Transfer the operation and dataflow principles. Keep upstream benchmark numbers, local execution status and fixed tuning thresholds outside shared instructions.

## Selective specialization

SGLang's unquantized XPU linear path uses framework linear arithmetic, while routed MoE can dispatch to custom grouped/fused kernels or an explicitly selected Triton backend. The reusable pattern is to preserve a strong supported dense backend and specialize work that benefits from grouping, routing or fusion. Apply the integrating project's backend policy before copying a framework call.

Source: `sglang/python/sglang/srt/layers/quantization/unquant.py`; grouped dispatch and fusion policy: `sgl-kernel-xpu/src/sycl/kernels/moe/xe20/bf16/grouped_gemm_dispatch.h` and `sgl-kernel-xpu/python/sgl_kernel/moe.py`.

## Resident fragments and explicit precision

SYCL-TLA and SyclKittens make copy layouts, register fragments, arithmetic tiles and epilogues explicit. Keep accumulators resident across the reduction, reuse operand fragments and minimize transformations between copies and arithmetic. These ideas also apply to full-FP32 arithmetic, but a BF16/FP16 matrix instruction cannot be substituted into an FP32 contract.

Read actual types, not names: an `sgemm`-named wrapper can dispatch only FP16/BF16, and FP32 output does not identify input multiplication precision. Check adapter eligibility and workspace before launch. Generic examples can require integration work even when their source is present. Confirm that build targets are enabled.

Sources: `sycl-tla/examples/00_bmg_gemm/00_bmg_gemm.cpp`, `sycl-tla/examples/generics/device_agnostic/device_agnostic_gemm.cpp` and its CMake file, `SyclKittens/kernels/gemm/bf16_gemm.dp.cpp`, `sgl-kernel-xpu/src/sycl/kernels/moe/xe20/common/xe_gemm.hpp` and `sgl-kernel-xpu/src/sycl/SGEMMLoraAFwd.cpp`.

## Descriptor, stride and lifecycle ownership

vLLM's native oneDNN paths preserve operand strides/transposes and scale granularity, construct primitive attributes, own scratch and execute on the current XPU stream. SGLang's expert-weight preparation separates logical shape from padded physical stride. Carry these contracts into cache keys and kernel arguments. Prepare reusable weights at the documented lifecycle boundary, and charge dynamic preparation to the call that needs it.

Do not adopt process-wide scratch or architecture caches without device/context/stream ownership. Outputs retained by a caller must survive subsequent calls and scratch growth. Enqueue without a forced wait only when the caller's dependencies and storage lifetime establish completion correctly. Otherwise return/bridge an event or synchronize at the required boundary.

Sources: `vllm-xpu-kernels/csrc/xpu/onednn/onednn_runtime.h` and `vllm-xpu-kernels/csrc/xpu/onednn/fp8_gemm_w8a8.h`; `sglang/python/sglang/srt/layers/quantization/unquant.py`; `sgl-kernel-xpu/tests/test_moe_fused_experts_workspace.py`; `SyclKittens/kernels/gemm/bf16_gemm.dp.cpp`.

## Feature gates and fallbacks

Keep build target, compiler support, runtime device identity and per-operation dtype/layout gates distinct. Share dispatch policy between AOT and JIT implementations to prevent drift. Include target and compiler identity in JIT cache keys. Recheck the actual input device rather than assuming the first process device. A backend selector or force flag cannot create missing hardware support.

Sources: `vllm-xpu-kernels/csrc/xpu/grouped_gemm/grouped_gemm_interface.cpp`; `sgl-kernel-xpu/src/sycl/Device.cpp`, `sgl-kernel-xpu/python/sgl_kernel/_arch.py` and `sgl-kernel-xpu/src/jit/sycl_template_jit.cpp`. Their target-specific branches are implementation references, not permission to enable the same instructions on every Xe, Xe2 or Xe3P product.

## Test the optimization's assumptions

- State convolution: verify old-state reads precede overwrites, padding remains unchanged, sliced/aligned and misaligned storage select valid paths, and every dtype/dilation/shape rejection is meaningful.
- GEMM/MoE: exercise awkward dimensions, groups, transpose/stride, scale/bias regimes, fallback dispatch, scratch reuse/growth and output non-aliasing across calls.
- Attention/cache kernels: specify layout, page/routing indices, causality, sequence tails and accumulation/output precision. Validate producer/consumer dependencies and all queues contributing to output readiness.
- Inspect upstream references and tolerances. Sampled-output checks, whole-tensor RMS criteria, low-precision tolerances and batched-enqueue timing do not replace the project's per-output numerical contract and completion-bounded measurement.

Useful test references: `sgl-kernel-xpu/tests/test_qwen4_short_conv.py`, `sgl-kernel-xpu/tests/test_moe_fused_experts_workspace.py`, `sgl-kernel-xpu/tests/test_sgemm_lora_a_fwd.py`; `vllm-xpu-kernels/tests/test_fp8_gemm_onednn.py` and `vllm-xpu-kernels/tests/fused_moe/test_grouped_gemm_xe2.py`; `SyclKittens/tests/warp/register/tile/mma.dp.hpp`. A test file, simulator configuration or architecture enum is a test resource, not an execution result.
