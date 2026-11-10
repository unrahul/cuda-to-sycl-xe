# Select the compute backend

Choose an implementation family before tuning tiles, and choose it from the byte and FLOP counts in [analysis](analysis.md), not by habit:

- **Large, compute-bound GEMM or convolution with a light epilogue:** library first. oneMKL or oneDNN is near peak, and a custom kernel rarely wins.
- **Chain whose intermediates dwarf its inputs and outputs, or whose first operation is small (few channels, small K):** fused custom kernel first. A library call must write its whole output to memory and cannot win ([fusion](fusion.md), worked example).
- **Unfamiliar library operation** (3D or transposed convolution, unusual strides or dtypes): get the descriptors right before concluding it is unsupported; see the oneDNN rows in [troubleshooting](troubleshooting.md). Keep a custom path ready.

Specify operation, precisions, physical strides, transposes, batching, workspace and the caller's queue. Respect the project's dependency rules.

| Backend | Good candidate workloads | Required checks |
|---|---|---|
| oneMKL SYCL | Standard dense/batched GEMM, GEMV and BLAS operations | Installed GPU BLAS development headers and linkable backend, supported dtype/layout, leading dimensions, strides, queue, dependency events and compute mode |
| oneDNN SYCL | Matmul, dense/grouped/depthwise convolution, supported attention and primitive post-ops | Supported primitive/engine, exact descriptors and math mode, reorders, scratchpad, primitive lifetime, post-op ordering and current-queue interop |
| SYCL-TLA | Tile-composable GEMM, grouped/quantized GEMM and custom mainloop/epilogue combinations | Pinned template revision, compiler and target support, instruction/fragment layouts, dtype combinations, adapter constraints and workspace |
| SyclKittens | Register-tile arithmetic, matrix/attention kernels and custom fused dataflow | Supported primitive and target, explicit tile layout/dtype, register ownership, copy alignment, subgroup/instruction contract and dependencies |
| Custom SYCL or ESIMD | Small reductions, normalization, activation, indexing/cache transforms, short depthwise/state convolution, or fusion a library cannot express efficiently | Proven indexing/synchronization, vector alignment and tails, sufficient reuse, register/resource limits and supported fallback |
| Existing framework or Triton XPU path | An allowed, already integrated operator/backend with a suitable contract | Project policy, actual XPU backend support and dispatch, precision, layouts and complete invocation costs |

Some projects require the arithmetic to run in your own SYCL or a named library and forbid calling back into Torch/ATen; check before choosing the framework row. Label library and custom results separately.

## Prove the backend is usable

1. Compile, link and execute a tiny call in the actual environment before designing around a library. Header presence or an import alone does not qualify it. oneMKL links with `-qmkl`; oneDNN needs a standalone SYCL-runtime build. For PyTorch, start with the [tested integration examples](pytorch-xpu.md); for link errors see [troubleshooting](troubleshooting.md).
2. A framework can embed a private library whose headers are visible but whose symbols are hidden. Do not depend on those private symbols or reconstruct their ABI. Use a supported standalone SDK or an explicitly exported integration API.
3. Stage development headers, libraries, compiler, template dependencies and any JIT requirements before an offline run. A dependency used for RNG does not establish that GPU BLAS is available. Do not spend algorithm-search attempts repeatedly guessing missing include paths.
4. Use the caller's device/context and queue. For oneDNN, construct the engine from the device/context and the stream from that queue. Wrap compatible USM storage with correct memory descriptors. On an out-of-order queue, pass producer dependencies through `sycl_interop::execute` and propagate its completion event. Otherwise establish a proven in-order stream contract. Merely sharing a queue is insufficient. For oneMKL USM, propagate dependency events and completion to subsequent consumers. Follow [framework stream ownership](pytorch-xpu.md).
5. Request the required precision explicitly. oneDNN's strict floating-point mode disables down-conversion. FP32 output or accumulation does not establish full-FP32 input multiplication. Quantized APIs additionally require exact scale granularity, zero-point, packing and output-conversion contracts.
6. Cache descriptors/primitives or packed weights only across a valid lifecycle. Include device/context, operation, strides, dtype, math mode, scales, post-ops and other semantic attributes in cache identity. Own scratch and outputs safely across streams and repeated calls.

Compare the complete required path: allocation, packing/reorder, primitive initialization when uncached, scratch, dispatch and synchronization. Persistent model-weight preparation and dynamic per-call input packing have different lifetimes. A fused epilogue may save traffic while increasing register pressure or reducing parallelism.

## Architecture is a separate gate

Use the exact compiler/runtime target and instruction combinations, not just a generation label. A package's build target, runtime architecture predicate and per-operation dtype/layout constraints must all match. A function called `is_xe3` can identify a particular Xe3P target rather than all Xe3 devices. Avoid default-target fallbacks and process-wide device caches when inputs can use different devices.

For Xe, Xe2 and Xe3P, retain a supported portable/library fallback and validate the actual dispatched branch. Keep product-specific tile choices, GRF settings and thresholds in per-device tuning tables. An environment override cannot enable an unavailable instruction. See [architectures](xe-architectures.md).

Primary references: [oneMKL GEMM API](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-2/gemm.html), [oneDNN SYCL interop](https://uxlfoundation.github.io/oneDNN/dev_guide_dpcpp_interoperability.html), [oneDNN math modes](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_fpmath_mode.html). Inspect the installed API version before adapting a call.
