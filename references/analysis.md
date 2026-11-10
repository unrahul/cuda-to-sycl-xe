# Analysis before optimization

Decide what limits the kernel, and how much faster it can possibly get, before writing a variant. The method is roofline analysis, applied to Intel GPUs.

## The three numbers

For the operation as the caller sees it:

- **Work** `W`: FLOPs or integer ops. A matmul `[M,K] x [K,N]` is `2*M*N*K`.
- **Bytes** `Q`: the minimum traffic to device memory, meaning every input read once and every output written once, at their stored dtypes. Include scales, zero points and indices for quantized data.
- **Intensity** `I = W / Q`, in FLOP/byte.

Bounds on a device with peak rate `P` (FLOP/s, for the unit and dtype actually used) and memory bandwidth `B` (bytes/s):

```
T_math = W / P        T_mem = Q / B
T_lower = max(T_math, T_mem)        T_upper = T_math + T_mem
critical intensity I_crit = P / B   (compute-bound when I > I_crit)
```

Good kernels land 10-30% above `T_lower`. If the baseline already sits there, report it and stop; the remaining gain cannot exceed `baseline / T_lower`.

Rules of thumb that follow:

- **Matmul.** Intensity is about the smallest dimension (`BDF/(BD+DF+BF) ≈ B` when `B ≪ D, F`). A decode GEMV (`B = 1`) is memory-bound on every GPU. A GEMM becomes compute-bound only once `M` (tokens) exceeds roughly `I_crit / (bytes per element)`.
- **Quantized weights.** Lower-bit weights lower the batch needed for compute-bound execution. Lower-bit arithmetic (int8/fp8 XMX) raises `P` and so `I_crit`.
- **Memory-bound kernels.** Elementwise ops, norms, softmax, decode attention, pooling and reductions are memory-bound. Speed them up by moving fewer bytes: fusion, smaller dtypes, no round trips. Better arithmetic does not help.
- **Vector versus XMX.** The vector units (FP32 FMA) are many times slower than XMX for fp16/bf16/int8. For a compute-bound op whose contract allows lower-precision inputs, using XMX is the first-order decision. When the contract demands FP32, the FP32 vector peak is the ceiling.
- **Cache-resident working sets.** If the working set fits in L2 (query `global_mem_cache_size`), repeated benchmark calls hit the cache and report bandwidth above `B`. Rotate buffers larger than the cache for streaming numbers ([performance](performance.md)).

## Device numbers

Use the numbers for the device you are on, never a neighbor's. Query what you can (`xe_probe --capabilities`). Take peak rates from Intel's published figures for that exact SKU; do not derive missing ones.

| Product (generation) | Xe-cores | Max clock | FP32 vector | INT8 XMX | Memory BW | L2 |
|---|---|---|---|---|---|---|
| Arc A770 16 GB (Xe-HPG) | 32 | 2.1 GHz | not published | 262 TOPS | 560 GB/s | 16 MB |
| Arc B580 (Xe2) | 20 | 2.67 GHz | not published | 233 TOPS | 456 GB/s | 18 MB |
| Arc Pro B60 (Xe2) | | | | 197 TOPS | 456 GB/s | query |
| Arc Pro B50 (Xe2) | | | | 170 TOPS | 224 GB/s | query |
| Arc Pro B70 (Xe2) | 32 | 2.8 GHz | 22.94 TF | 367 TOPS dense | 608 GB/s | not published |
| GPU D270 (Xe3P) | 32 | not published | not published | FP8 ≈1.3 PF | ≈1.5 TB/s | 32 MB |

Values are Intel's published figures for each SKU. Where a figure is not published, including all BF16/FP16 XMX rates, query the device or measure peak with a microbenchmark rather than deriving one.

Critical intensities on B70, with `B` = 608 GB/s:

| Unit | Intensity |
|---|---|
| FP32 vector | ≈38 FLOP/B |
| INT8 XMX | ≈600 OP/B |

On D270 the ratio of compute to bandwidth is different, so recompute the crossover; do not reuse B70's.

## Worked bounds (Arc Pro B70)

| Operation | W | Q | I | Bound | T_lower |
|---|---|---|---|---|---|
| FP32 GEMM 1024×8192×8192 | 1.37e11 | 3.4e8 B | 410 | compute (FP32 vector) | 6.0 ms |
| decode GEMV 1×4096×4096, fp16 weights | 3.4e7 | 3.4e7 B | ≈1 | memory | 55 µs |
| same, int4 weights + group-128 fp16 scales | 3.4e7 | 8.7e6 B | ≈3.9 | memory | 14 µs (if not L2-resident) |
| RMSNorm 4096×4096 fp32 | ≈6.7e7 | 1.34e8 B | ≈0.5 | memory | 221 µs |

The RMSNorm row shows where real wins live:

- Fusing the residual add that precedes the norm removes one full read and write.
- Writing bf16 output halves the write, but only if the contract allows it.

## Where large speedups come from

Ranked by typical gain:

0. **Doing less work.** Before optimizing an operation, check whether the math lets you skip it. Common cases:
   - **A reduction after a linear map** can move inside it. `sum_j (x @ Wᵀ)[i,j] = x[i] · (Σ_j W[j,:])`, so a GEMM followed by a row sum becomes a weight column-sum, done once or per call, plus a GEMV. The FLOPs drop by the hidden size. The same holds for a mean, or a weighted sum with fixed weights.
   - **Constant scales, divisions and biases** fold into weights or into a single epilogue.
   - **A monotone per-element map commutes with max, min and argmax**, so pool first and transform fewer elements ([fusion](fusion.md), worked example).
   - **Work whose result is discarded** need not run, for example rows that a later mask or top-k drops, or softmax before argmax.

   Rewrites change rounding. Accept one when it stays within the task's declared tolerance against an FP64 reference ([testing](testing.md)); the summation order is not part of the contract unless the task says so.
1. **Removing round trips through memory.** Fuse a chain of memory-bound ops into one pass, recompute small intermediates instead of storing them, and keep per-group statistics on chip. This gave 5–10x on B70 for convolution + norm + pool chains ([fusion](fusion.md)).
2. **Getting onto XMX** for compute-bound work whose contract allows fp16/bf16/int8 inputs ([GEMM](gemm.md)).
3. **Fixing registers.** Removing spills, choosing large GRF or a narrower sub-group gave 3–14x on Xe-HPC kernels ported from CUDA ([registers](registers.md)).
4. **Filling the machine.** Small decode kernels need at least as many work-groups as Xe-cores, or a split along K or sequence length ([registers](registers.md), [memory](memory.md)).
5. **Data movement.** Coalesce accesses, use vector or 2D block loads, avoid layout transposes, and reorder quantized weights once ([memory](memory.md)).

Micro-tuning (tile sizes, unrolling) comes last and usually yields tens of percent.

## Write the analysis down

Before the first variant, record the following in the experience notes:

- `W`, `Q` and `I` for the operation;
- the device's `P` and `B`, with their source;
- `T_lower` and the baseline's percentage of it;
- the expected bound and the planned lever.

After each measured variant, update the percentage of `T_lower`. A variant that moves toward the bound is progress. One that does not, while still passing, means the analysis missed something: check spills, occupancy and achieved bandwidth with the [profilers](profiling.md).

## Gotchas

- Spending attempts on a library-bound op: a strict-FP32 GEMM at 6.6 ms against a 6.0 ms floor leaves at most about 10%.
- Skipping the algebra check: a GEMM followed by a row sum collapses to a GEMV, and `x - x.mean()` over a single element is identically zero.
