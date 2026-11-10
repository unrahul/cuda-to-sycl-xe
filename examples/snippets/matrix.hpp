// matrix.hpp - XMX through joint_matrix, quoted in references/xe-architectures.md.
#pragma once
#include "dispatch.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

#include <cstddef>

namespace snippets {

// snippet: joint-matrix-bf16
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
// end snippet

}  // namespace snippets
