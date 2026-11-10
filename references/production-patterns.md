# Patterns from production XPU kernels

Patterns that recur in production Intel GPU inference kernels, written as small kernels. Each code block is in `examples/snippets/patterns.hpp` and is checked by `tests/snippets_test.cpp`.

On Xe-HPG, write portable SYCL plus `joint_matrix`: sycl-tla supports only Xe2 and Xe3P ([architectures](xe-architectures.md)).

## Vector width from alignment (all generations)

Check alignment at run time before choosing a vectorized kernel. A tensor view with an odd offset otherwise breaks vector loads.

```cpp
// Vector loads are legal only if every pointer and every row start is aligned to the
// vector: check base addresses and the row length, then dispatch; never assume.
template <class T>
int max_vector_width(std::size_t cols, std::initializer_list<const T*> ptrs) {
  for (int w : {4, 2}) {
    const auto aligned = [&](const T* p) { return reinterpret_cast<std::uintptr_t>(p) % (w * sizeof(T)) == 0; };
    if (cols % w == 0 && std::all_of(ptrs.begin(), ptrs.end(), aligned)) return w;
  }
  return 1;
}

template <int VW> class RmsNorm;

template <int VW>
sycl::event rms_norm(sycl::queue& q, const float* x, const float* w, float* y, int rows, int cols, float eps) {
  constexpr int wg = 256;
  return q.parallel_for<RmsNorm<VW>>(sycl::nd_range<1>{std::size_t(rows) * wg, wg}, [=](sycl::nd_item<1> it) {
    using V = sycl::vec<float, VW>;
    const int row = it.get_group(0), lid = it.get_local_id(0);
    const V* xr = reinterpret_cast<const V*>(x + std::size_t(row) * cols);
    const V* wv = reinterpret_cast<const V*>(w);
    V* yr = reinterpret_cast<V*>(y + std::size_t(row) * cols);
    float ss = 0.0f;
    for (int i = lid; i < cols / VW; i += wg)
      for (int j = 0; j < VW; ++j) ss += xr[i][j] * xr[i][j];
    const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / cols + eps);
    for (int i = lid; i < cols / VW; i += wg) yr[i] = xr[i] * inv * wv[i];
  });
}

inline sycl::event rms_norm(sycl::queue& q, const float* x, const float* w, float* y, int rows, int cols, float eps) {
  switch (max_vector_width<float>(cols, {x, w, y})) {
    case 4: return rms_norm<4>(q, x, w, y, rows, cols, eps);
    case 2: return rms_norm<2>(q, x, w, y, rows, cols, eps);
    default: return rms_norm<1>(q, x, w, y, rows, cols, eps);
  }
}
```

With very short rows (QK-norm, head_dim 128), one row per work-group idles most lanes. Pack 16 rows into a 2D work-group instead.

## Sub-group width per call (all generations)

```cpp
// Pick the sub-group width per call by searching the device's list. Do not index it:
// Xe-HPG reports {8, 16, 32} and Xe2/Xe3P {16, 32}, so sizes[1] means different widths.
template <int SG> class RowMaxSG;

template <int SG>
sycl::event row_max(sycl::queue& q, const float* x, float* out, int rows, int cols) {
  return q.parallel_for<RowMaxSG<SG>>(
      sycl::nd_range<1>{std::size_t(rows) * SG, SG}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sub = it.get_sub_group();
        const int row = it.get_group(0);
        float m = -INFINITY;
        for (int c = sub.get_local_linear_id(); c < cols; c += SG) m = sycl::fmax(m, x[std::size_t(row) * cols + c]);
        m = sycl::reduce_over_group(sub, m, sycl::maximum<float>());
        if (sub.leader()) out[row] = m;
      });
}

inline sycl::event row_max(sycl::queue& q, const float* x, float* out, int rows, int cols) {
  // Short rows leave lanes idle at 32; long rows can use the wider group. A heuristic to measure.
  if (cols >= 256 && supports_sub_group(q.get_device(), 32)) return row_max<32>(q, x, out, rows, cols);
  return row_max<16>(q, x, out, rows, cols);  // 16 exists on every Xe generation
}
```

Indexing `sub_group_sizes[1]` gives 16 on Xe-HPG and 32 on Xe2. Production code also switches widths per generation at compile time; FP8 quantization uses sub-group 32 on Xe2 and 16 on Xe3P.

## Quantized weights: reorder once, then one sub-group per row (all generations)

```cpp
// GGUF-style Q4_0 block: 32 weights as 4-bit codes, value = (code - 8) * d.
// 18 bytes, so blocks are only 2-byte aligned and wide loads of qs are illegal.
struct BlockQ4 {
  sycl::half d;
  std::uint8_t qs[16];  // byte j: low nibble = weight j, high nibble = weight j + 16
};

// Once, at weight-load time: array of blocks -> all codes, then all scales.
// Each block's 16 code bytes now start on a 16-byte boundary.
class ReorderQ4;
inline sycl::event reorder_q4(sycl::queue& q, const BlockQ4* src, std::uint8_t* qs, sycl::half* d, std::size_t blocks) {
  return q.parallel_for<ReorderQ4>(sycl::range<1>{blocks}, [=](sycl::id<1> b) {
    for (int j = 0; j < 16; ++j) qs[b * 16 + j] = src[b].qs[j];
    d[b] = src[b].d;
  });
}

// y = W x with one 16-lane sub-group per row: lanes take whole blocks, one reduction at the end.
// No local memory and no work-group barrier. k must be a multiple of 32.
class GemvQ4;
inline sycl::event gemv_q4(sycl::queue& q, const std::uint8_t* qs, const sycl::half* d, const float* x, float* y,
                           int rows, int k) {
  constexpr int sg = 16, rows_per_wg = 4;
  const int blocks_per_row = k / 32;
  const std::size_t groups = (rows + rows_per_wg - 1) / rows_per_wg;
  return q.parallel_for<GemvQ4>(
      sycl::nd_range<1>{groups * rows_per_wg * sg, rows_per_wg * sg},
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg)]] {
        const auto sub = it.get_sub_group();
        const int row = it.get_group(0) * rows_per_wg + sub.get_group_linear_id();
        if (row >= rows) return;  // the whole sub-group leaves together; no barrier follows
        float acc = 0.0f;
        for (int b = sub.get_local_linear_id(); b < blocks_per_row; b += sg) {
          const std::size_t blk = std::size_t(row) * blocks_per_row + b;
          const auto* words = reinterpret_cast<const std::uint32_t*>(qs + blk * 16);  // aligned after reorder
          const float* xb = x + b * 32;
          float s = 0.0f;
          for (int w = 0; w < 4; ++w)
            for (int j = 0; j < 4; ++j) {
              const std::uint32_t byte = (words[w] >> (8 * j)) & 0xFF;
              s += float(int(byte & 0xF) - 8) * xb[w * 4 + j] + float(int(byte >> 4) - 8) * xb[w * 4 + j + 16];
            }
          acc += s * float(d[blk]);
        }
        acc = sycl::reduce_over_group(sub, acc, sycl::plus<float>());
        if (sub.leader()) y[row] = acc;
      });
}
```

- This is a dp4a-style GEMV. On Xe2 and Xe3P the next step is int8 `joint_matrix` (`examples/gemv_int8/`), whose cost stays flat up to eight activation columns.
- On Arc A770, Q4_0 was measured faster without the reorder. Measure the reorder on each product rather than assuming it.

## MoE top-k: move (value, index) together (all generations)

```cpp
// MoE routing: the k largest values of each row with their indices, in descending order.
// W lanes per row (16 / W rows per sub-group); each lane owns E strided values (cols <= W * E).
// The (value, index) pair moves through the butterfly together, so every lane agrees on one
// winner; ties go to the lower index. Lanes of missing rows still join every shuffle.
template <int W>
inline void argmax_lanes(sycl::sub_group sub, float& v, int& i) {
  for (int m = W / 2; m > 0; m /= 2) {
    const float ov = sycl::permute_group_by_xor(sub, v, m);
    const int oi = sycl::permute_group_by_xor(sub, i, m);
    if (ov > v || (ov == v && oi < i)) v = ov, i = oi;
  }
}

template <int W, int E> class TopK;

template <int W, int E>
sycl::event topk(sycl::queue& q, const float* x, float* vals, int* idx, int rows, int cols, int k) {
  constexpr int sg = 16, rows_per_sg = sg / W;
  const std::size_t sub_groups = (rows + rows_per_sg - 1) / rows_per_sg;
  return q.parallel_for<TopK<W, E>>(
      sycl::nd_range<1>{sub_groups * sg, sg}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg)]] {
        const auto sub = it.get_sub_group();
        const int lane = sub.get_local_linear_id(), part = lane % W;
        const int row = it.get_group(0) * rows_per_sg + lane / W;
        const bool live = row < rows;
        float own[E];
        for (int e = 0; e < E; ++e) {
          const int c = part + e * W;
          own[e] = live && c < cols ? x[std::size_t(row) * cols + c] : -INFINITY;
        }
        for (int r = 0; r < k; ++r) {
          float v = -INFINITY;
          int i = cols;  // sentinel: loses every tie against a real index
          for (int e = 0; e < E; ++e)
            if (own[e] > v) v = own[e], i = part + e * W;
          argmax_lanes<W>(sub, v, i);
          if (live && part == 0) vals[std::size_t(row) * k + r] = v, idx[std::size_t(row) * k + r] = i;
          if (i < cols && i % W == part) own[i / W] = -INFINITY;  // the owner retires the winner
        }
      });
}
```

Production MoE routers use the same structure with 32-wide sub-groups, and map experts outside the local range to a sentinel index. Reducing values and indices separately can pick a winner whose index came from a different lane.

## Lessons without a snippet

- **Paged KV-cache write.**
  - Use one work-group per token, and return early when `slot < 0` (a padding token).
  - Split the slot into block and offset.
  - Load K and V before storing either.
  - An FP8 cache is `uint8_t`, so store the bit pattern (`sycl::bit_cast`); assigning a float converts it to an integer instead.
  - Keep slot indices `int64_t`; narrowing them to `int` overflows on large caches.
- **FP8 quantization.**
  - Clamp to the format's finite range before converting.
  - Use power-of-two (UE8M0) scales when the consumer expects them.
  - Xe2 emulates the conversion. Xe3P has a native fp16-to-E4M3 conversion that sycl-tla exposes under `SYCL_INTEL_TARGET >= 35`.
- **Register budget.** A per-thread `float` cache of a whole row (16 KB) compiled and ran on BMG, then failed later under load with `UR_RESULT_ERROR_OUT_OF_RESOURCES`. Size private arrays to the GRF budget, or re-read from cache.
- **Occupancy for decode GEMV.** When N is small, split K across several threads per row and combine partials in local memory. Derive the thread target from the device (`ext::intel::info::device::gpu_eu_count` × threads per EU × SIMD width), not from a constant taken from one card.
- **Launch overhead.** Route every launch through one helper that uses `sycl::ext::oneapi::experimental::nd_launch` (no command-group handler) when `SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS` is defined.
- **Reused scratch.** Back-to-back asynchronous kernels that reuse a preallocated buffer produced NaNs, then a collective hang and `DEVICE_LOST`. Scratch shared across calls needs an ordering edge.

## Ownership and gates

- **Prefer a strong dense backend.** Keep a supported dense backend (oneDNN/oneMKL, framework linear) and specialize only the work that gains from grouping, routing or fusion.
- **Read the actual types.** An `sgemm`-named wrapper may dispatch only FP16/BF16, and FP32 output does not identify the input precision.
- **Carry layout and lifetime contracts.** Pass operand strides, transposes and scale granularity into cache keys and kernel arguments. Own scratch per device, context and stream. Outputs must survive later calls and scratch growth.
- **Keep the gates distinct.** Build target, compiler support, runtime device and per-operation dtype/layout gates are separate checks, and AOT and JIT builds should share one dispatch policy. A build can detect whether the compiler knows `intel_gpu_cri` by scanning its headers, instead of checking the version ([architectures](xe-architectures.md)).
- **Force flags don't create support.** A force flag can select a path but never creates hardware support.

## Test what the optimization assumes

- **State convolution.** Check that old-state reads precede overwrites, that padding is unchanged, and that aligned and misaligned storage each pick a valid path.
- **GEMM and MoE.** Cover awkward dimensions, empty experts, transposes and strides, the fallback dispatch, scratch growth, and outputs that do not alias.
- **Attention and caches.** Specify the layout, page indices, causality, sequence tails and accumulation precision. Verify every queue that contributes to the output.
- **Library tests aren't contracts.** Their sampled checks and loose low-precision tolerances do not replace your per-element contract. 
