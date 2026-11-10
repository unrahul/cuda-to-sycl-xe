# GEMM, XMX and convolution

Start with [analysis](analysis.md). For a large GEMM, the question is not how to write a fast GEMM but whether anything can beat the library at all. For strict FP32, oneMKL already runs near the FP32 vector peak. Read [backend selection](backend-selection.md) for how to call oneMKL and oneDNN.

## Decide the path

| Situation | Path |
|---|---|
| Standard GEMM/batched GEMM, any generation | oneMKL (`-qmkl`) or oneDNN matmul. Measure it alone first: it is your ceiling and control |
| GEMM + cheap epilogue (bias, activation, scale) | oneDNN post-ops, or library GEMM followed by one fused elementwise pass |
| GEMM whose epilogue needs whole rows or groups (norm, softmax, top-k) | library GEMM, then a fused epilogue kernel that reads the output once ([fusion](fusion.md)) |
| Small, skinny or many tiny GEMMs (decode, MoE experts, MLP layers that fit on chip) | custom kernel: grouped or persistent, possibly keeping activations in registers across layers |
| Contract allows fp16, bf16 or int8 inputs | XMX path: library in that precision, or `joint_matrix` |
| Strict FP32 | library FP32 or FP32 vector arithmetic. XMX fp16/bf16/tf32 inputs are a contract change |

Two cautions:

- A fully fused kernel can lose to a separate oneMKL call, for example in an MLP backward pass.
- A custom or fused GEMM slower than the vendor GEMM is the most common GEMM failure.

Always A/B against the library.

## Custom GEMM dataflow

- **Layouts.** Make A[M,K], B[K,N], C[M,N] and their physical strides explicit, including weights stored as [N,K]. Cooperative loads must read contiguous addresses in the real layout. A transpose flag does not make strided loads coalesced.
- **Register micro-tiles.** Each work-item or sub-group keeps several output accumulators and reuses loaded A and B values in an outer product. Accumulators stay resident across the whole K loop.
- **GRF budget.** Size the operand tile to the register budget ([registers](registers.md), step 3), and check spills after every tile change.
- **Feeding the tile.** Use 2D block loads (Xe2, Xe3P) or SLM staging (all generations), and prefetch two or three K-steps ahead ([memory](memory.md)). Pack or convert operand layouts once, outside the hot loop. Re-packing per multiply costs 16–24 moves per DPAS.
- **Shape-driven scheduling.**
  - Skinny shapes want asymmetric tiles: clamp each tile edge to the largest power of two not exceeding the dimension.
  - Few output tiles want split-K. It changes the summation order, so check it against the contract.
  - Uneven expert groups want grouped or persistent scheduling.
  - Fewer work-groups than Xe-cores idles the GPU.
- **Tuning order.** algorithm → dtype → fusion → memory access → block loads → persistence → tile and GRF → autotune. Autotune a small set, about a dozen configurations, last.

## XMX through joint_matrix

One source covers all three generations; only the tile parameters and the sub-group width change.

```cpp
// C[M,N] (fp32) = A[M,K] * B[K,N] (bf16, row-major); one sub-group computes one TM x TN tile.
//   Xe2, Xe3P: bf16_gemm<8, 16, 16, 16>      Xe-HPG: bf16_gemm<8, 8, 16, 8>
// Gate with supports_matrix_tile(d, mtype::bf16, mtype::bf16, mtype::fp32, TM, TN, TK).
// Requires M % TM == 0, N % TN == 0, K % TK == 0. A correctness baseline for XMX, not a tuned GEMM:
// it has no operand reuse across tiles; production code adds several tiles per sub-group.
template <std::size_t TM, std::size_t TN, std::size_t TK, std::size_t SG> class Bf16Gemm;

template <std::size_t TM, std::size_t TN, std::size_t TK, std::size_t SG>
sycl::event bf16_gemm(sycl::queue& q, const sycl::ext::oneapi::bfloat16* A, const sycl::ext::oneapi::bfloat16* B,
                      float* C, std::size_t M, std::size_t N, std::size_t K) {
  namespace mx = syclex::matrix;
  using bf16 = sycl::ext::oneapi::bfloat16;
  constexpr auto global = sycl::access::address_space::global_space;
  constexpr auto plain = sycl::access::decorated::no;
  return q.parallel_for<Bf16Gemm<TM, TN, TK, SG>>(
      sycl::nd_range<2>{{M / TM, N / TN * SG}, {1, SG}},
      [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const std::size_t m0 = it.get_group(0) * TM, n0 = it.get_group(1) * TN;
        const auto a = sycl::address_space_cast<global, plain>(A + m0 * K);
        const auto b = sycl::address_space_cast<global, plain>(B + n0);
        const auto c = sycl::address_space_cast<global, plain>(C + m0 * N + n0);
        mx::joint_matrix<sycl::sub_group, bf16, mx::use::a, TM, TK, mx::layout::row_major> ma;
        mx::joint_matrix<sycl::sub_group, bf16, mx::use::b, TK, TN, mx::layout::row_major> mb;
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> mc;
        mx::joint_matrix_fill(sg, mc, 0.0f);
        for (std::size_t k = 0; k < K; k += TK) {
          mx::joint_matrix_load(sg, ma, a + k, K);
          mx::joint_matrix_load(sg, mb, b + k * N, N);
          mx::joint_matrix_mad(sg, mc, ma, mb, mc);
        }
        mx::joint_matrix_store(sg, mc, c, N, mx::layout::row_major);
      });
}
```

- `tests/snippets_test.cpp` checks the Xe2 instantiation against an FP64 reference and skips it on stacks without the combination (oneAPI 2025.3 exposes none on Xe2).
- For speed, give each sub-group several tiles that reuse A and B fragments, pack B in VNNI layout (`layout::ext_intel_packed`), and keep accumulators resident across K.
- Do not use an FP16/BF16 matrix path when the contract requires full FP32 multiplication.

## Convolution choices

| Regime | Candidate organization | Costs and contracts to inspect |
|---|---|---|
| Dense pointwise (1x1) | GEMM over channels and spatial positions, often batched over images | NCHW/NHWC physical layout, groups, stride/subsampling, output reshape and any required reorder |
| Dense spatial convolution | Optimized direct tiling or implicit GEMM, plus a native-library control | Reuse across positions and output channels, input halos, channel blocking, weight layout and boundary masks |
| Explicit im2col + GEMM | Materialize patches when reuse/implementation simplicity justifies it | Expanded workspace and its allocation, write/read traffic, memory budget and packing lifetime |
| Depthwise/grouped short reductions | Direct spatial/vector kernels, or qualified library algorithms | Per-group channel count, spatial coalescing, tails, alignment, activation/epilogue fusion |
| Transposed convolution | Correct gather/scatter or qualified library primitive | Groups, stride, dilation, padding, output padding, overlaps and deterministic/atomic accumulation policy |
| Temporal/state convolution | Bounded direct stencil with explicit state ownership | Old-state snapshot before writes, ring/pool offsets, preserved padding, sequence masks, state/output lifetime and rounding points |

An optimized direct convolution can be highly efficient. Do not infer an algorithm is poor from a scalar implementation, or that implicit GEMM must win. Sharing inputs across output channels and weights across spatial outputs matters in dense convolution. A depthwise kernel has a different reduction and reuse structure.

For each spatial axis, ordinary convolution uses `floor((in + 2*p - d*(k-1) - 1)/s) + 1`; transposed convolution uses `(in-1)*s - 2*p + d*(k-1) + output_padding + 1`. Preserve the source's exact asymmetric-padding and output-padding semantics. Use index types that cover total tensor and workspace sizes.

Keep semantic rounding boundaries during fusion. Collapsing a GEMM followed by a reduction algebraically can change floating-point cancellation and near-zero outputs. Match the declared error contract using independent represented-input references and source-order checks when required. Do not repair a failed comparison by silently relaxing tolerance.

## Gotchas

- `joint_matrix_load` takes the fast path only when the stride is a multiple of 8 bytes and the base is 4-byte aligned; otherwise the compiler uses a slower path.
