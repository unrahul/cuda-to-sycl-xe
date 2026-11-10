// patterns.hpp - production kernel patterns for all Xe generations, quoted in
// references/production-patterns.md.
#pragma once
#include "dispatch.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace snippets {

// snippet: vector-width
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
// end snippet

// snippet: subgroup-choice
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
// end snippet

// snippet: q4-reorder
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
// end snippet

// snippet: topk
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
// end snippet

}  // namespace snippets
