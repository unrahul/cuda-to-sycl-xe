// mapping.hpp - CUDA-to-SYCL translations quoted in references/cuda-to-sycl-mapping.md.
// Each region between "snippet:" and "end snippet" is copied verbatim into the docs;
// tests/test_docs.py fails if a documented block drifts from this compiled source.
#pragma once
#include <sycl/sycl.hpp>

#include <cstddef>
#include <new>

namespace snippets {

class Saxpy;
class RowSum;
class RowSumSG;
class Twice;

// snippet: launch
// CUDA: saxpy<<<(n + 255) / 256, 256, 0, stream>>>(n, a, x, y);
inline sycl::event saxpy(sycl::queue& q, std::size_t n, float a, const float* x, float* y) {
  constexpr std::size_t wg = 256;
  const std::size_t global = (n + wg - 1) / wg * wg;  // whole work-groups only
  return q.parallel_for<Saxpy>(sycl::nd_range<1>{global, wg}, [=](sycl::nd_item<1> it) {
    const std::size_t i = it.get_global_id(0);  // blockIdx.x * blockDim.x + threadIdx.x
    if (i < n) y[i] = a * x[i] + y[i];          // guard the padded tail
  });
}
// end snippet

// snippet: block-reduce
// CUDA: __shared__ float s[256]; ... __syncthreads(); tree reduction in shared memory.
inline sycl::event row_sum(sycl::queue& q, const float* x, float* out, int rows, int cols) {
  constexpr int wg = 256;
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> s{sycl::range<1>{wg}, h};  // __shared__
    h.parallel_for<RowSum>(sycl::nd_range<1>{std::size_t(rows) * wg, wg}, [=](sycl::nd_item<1> it) {
      const int row = it.get_group(0), lid = it.get_local_id(0);
      float acc = 0.0f;
      for (int c = lid; c < cols; c += wg) acc += x[std::size_t(row) * cols + c];
      s[lid] = acc;
      sycl::group_barrier(it.get_group());  // __syncthreads(): every work-item reaches it
      for (int stride = wg / 2; stride > 0; stride /= 2) {
        if (lid < stride) s[lid] += s[lid + stride];  // only the destination half writes
        sycl::group_barrier(it.get_group());
      }
      if (lid == 0) out[row] = s[0];
    });
  });
}
// end snippet

// snippet: subgroup
// CUDA: for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffff, v, o);
// One sub-group per row. The width is a compile-time contract; check sub_group_sizes first.
inline sycl::event row_sum_sg(sycl::queue& q, const float* x, float* out, int rows, int cols) {
  constexpr int sg = 16;  // native on Xe2 and Xe3P; Xe-HPG also compiles 8 and 32
  return q.parallel_for<RowSumSG>(
      sycl::nd_range<1>{std::size_t(rows) * sg, sg},
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg)]] {
        const auto sub = it.get_sub_group();
        const int row = it.get_group(0), lane = sub.get_local_linear_id();
        float v = 0.0f;
        for (int c = lane; c < cols; c += sg) v += x[std::size_t(row) * cols + c];
        for (int o = sg / 2; o > 0; o /= 2) v += sycl::permute_group_by_xor(sub, v, o);
        // Same result, less code: v = sycl::reduce_over_group(sub, v, sycl::plus<float>());
        if (lane == 0) out[row] = v;
      });
}
// end snippet

// snippet: atomic
// CUDA: atomicAdd(&hist[bin], 1u);
inline void hist_add(unsigned* hist, int bin) {
  sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref{hist[bin]};
  ref.fetch_add(1u);
}
// end snippet

// snippet: streams
// CUDA: cudaMemcpyAsync(d, h, bytes, H2D, s); twice<<<g, b, 0, s>>>(d); ...; cudaStreamSynchronize(s);
inline void twice_on_device(sycl::queue& q, const float* in, float* out, std::size_t n) {
  // q must be in order, like a CUDA stream: sycl::queue q{dev, sycl::property::queue::in_order{}};
  float* d = sycl::malloc_device<float>(n, q);
  if (!d) throw std::bad_alloc{};
  q.memcpy(d, in, n * sizeof(float));
  q.parallel_for<Twice>(sycl::range<1>{n}, [=](sycl::id<1> i) { d[i] *= 2.0f; });
  q.memcpy(out, d, n * sizeof(float));
  q.wait_and_throw();  // cudaStreamSynchronize, and rethrows asynchronous errors
  sycl::free(d, q);
}
// end snippet

// snippet: events
// CUDA: cudaEventRecord(done, s1); cudaStreamWaitEvent(s2, done);
inline sycl::event after(sycl::queue& q2, sycl::event done, float* y, std::size_t n) {
  return q2.submit([&](sycl::handler& h) {
    h.depends_on(done);  // cross-queue ordering is explicit
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) { y[i] += 1.0f; });
  });
}
// end snippet

}  // namespace snippets
