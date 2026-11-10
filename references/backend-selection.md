# Select the compute backend

Choose an implementation family before tuning tiles. Specify the operation, input/accumulator/output precision, physical strides, transpose, batching, fusion, workspace and caller queue. Respect the project's dependency and benchmark rules. A library route is a candidate to compare, not an automatic winner.

| Backend | Good candidate workloads | Required checks |
|---|---|---|
| oneMKL SYCL | Standard dense/batched GEMM, GEMV and BLAS operations | Installed GPU BLAS development headers and linkable backend, supported dtype/layout, leading dimensions, strides, queue, dependency events and compute mode |
| oneDNN SYCL | Matmul, dense/grouped/depthwise convolution, supported attention and primitive post-ops | Supported primitive/engine, exact descriptors and math mode, reorders, scratchpad, primitive lifetime, post-op ordering and current-queue interop |
| SYCL-TLA | Tile-composable GEMM, grouped/quantized GEMM and custom mainloop/epilogue combinations | Pinned template revision, compiler and target support, instruction/fragment layouts, dtype combinations, adapter constraints and workspace |
| SyclKittens | Register-tile arithmetic, matrix/attention kernels and custom fused dataflow | Supported primitive and target, explicit tile layout/dtype, register ownership, copy alignment, subgroup/instruction contract and dependencies |
| Custom SYCL or ESIMD | Small reductions, normalization, activation, indexing/cache transforms, short depthwise/state convolution, or fusion a library cannot express efficiently | Proven indexing/synchronization, vector alignment and tails, sufficient reuse, register/resource limits and supported fallback |
| Existing framework or Triton XPU path | An allowed, already integrated operator/backend with a suitable contract | Project policy, actual XPU backend support and dispatch, precision, layouts and complete invocation costs |

Framework arithmetic is valid only when the integrating project permits it. A benchmark requiring native SYCL may forbid delegating numerical work to Torch/ATen even when production code correctly uses a framework linear operator. Label native-library and custom results separately.

## Prove the backend is usable

1. Use supplied qualification when it covers the exact runtime, backend and device. Otherwise compile and execute a tiny call in the actual environment, including linking and runtime adapters. Record identities and diagnostics. Header presence or an import alone does not qualify the numerical path. For PyTorch, start with the [complete integration examples](pytorch-xpu.md).
2. A framework can embed a private library whose headers are visible but whose symbols are hidden. Do not depend on those private symbols or reconstruct their ABI. Use a supported standalone SDK or an explicitly exported integration API.
3. Stage development headers, libraries, compiler, template dependencies and any JIT requirements before an offline run. A dependency used for RNG does not establish that GPU BLAS is available. Do not spend algorithm-search attempts repeatedly guessing missing include paths.
4. Use the caller's device/context and queue. For oneDNN, construct the engine from the device/context and the stream from that queue. Wrap compatible USM storage with correct memory descriptors. On an out-of-order queue, pass producer dependencies through `sycl_interop::execute` and propagate its completion event. Otherwise establish a proven in-order stream contract. Merely sharing a queue is insufficient. For oneMKL USM, propagate dependency events and completion to subsequent consumers. Follow [framework stream ownership](pytorch-xpu.md).
5. Request the required precision explicitly. oneDNN's strict floating-point mode disables down-conversion. FP32 output or accumulation does not establish full-FP32 input multiplication. Quantized APIs additionally require exact scale granularity, zero-point, packing and output-conversion contracts.
6. Cache descriptors/primitives or packed weights only across a valid lifecycle. Include device/context, operation, strides, dtype, math mode, scales, post-ops and other semantic attributes in cache identity. Own scratch and outputs safely across streams and repeated calls.

Compare the complete required path: allocation, packing/reorder, primitive initialization when uncached, scratch, dispatch and synchronization. Persistent model-weight preparation and dynamic per-call input packing have different lifetimes. A fused epilogue may save traffic while increasing register pressure or reducing parallelism.

## Architecture is a separate gate

Use the exact compiler/runtime target and instruction combinations, not just a generation label. A package's build target, runtime architecture predicate and per-operation dtype/layout constraints must all match. A function called `is_xe3` can identify a particular Xe3P target rather than all Xe3 devices. Avoid default-target fallbacks and process-wide device caches when inputs can use different devices.

For Xe, Xe2 and Xe3P, retain a supported portable/library fallback and validate the actual dispatched branch. Keep product-specific tile choices, GRF settings and thresholds in external tuning records. An environment override cannot authorize an unavailable instruction. See [architecture policy](xe-architectures.md).

Primary references: [oneMKL GEMM API](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-2/gemm.html), [oneDNN SYCL interop](https://uxlfoundation.github.io/oneDNN/dev_guide_dpcpp_interoperability.html), [oneDNN math modes](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_fpmath_mode.html), and the pinned [implementation catalog](sources.md). Inspect the installed API version before adapting a call.
