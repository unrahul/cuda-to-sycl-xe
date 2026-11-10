// moments.hpp - mergeable mean/variance used by fused normalization kernels,
// quoted in references/fusion.md.
#pragma once
#include <sycl/sycl.hpp>

#include <cstddef>

namespace snippets {

// snippet: moments
// Count, mean and sum of squared deviations. Merging partial blocks this way (Chan et al.)
// avoids the cancellation of sum(x*x) - n*mean*mean in a one-pass FP32 variance.
struct Moments {
  float n = 0.0f, mean = 0.0f, m2 = 0.0f;
};

inline Moments merge(Moments a, Moments b) {
  if (b.n == 0.0f) return a;
  if (a.n == 0.0f) return b;
  const float n = a.n + b.n, d = b.mean - a.mean;
  return {n, a.mean + d * (b.n / n), a.m2 + b.m2 + d * d * (a.n * b.n / n)};
}

// Work-group merge through local memory. Every work-item calls it (it contains barriers)
// and every work-item receives the merged result. s must hold one entry per work-item.
inline Moments group_moments(sycl::nd_item<1> it, Moments mine, const sycl::local_accessor<Moments, 1>& s) {
  const int lid = it.get_local_id(0), wg = it.get_local_range(0);
  s[lid] = mine;
  sycl::group_barrier(it.get_group());
  for (int stride = wg / 2; stride > 0; stride /= 2) {  // wg must be a power of two
    if (lid < stride) s[lid] = merge(s[lid], s[lid + stride]);
    sycl::group_barrier(it.get_group());
  }
  return s[0];
}
// end snippet

class GroupStats;

// One work-group per row: mean and population variance of each row.
inline sycl::event row_moments(sycl::queue& q, const float* x, float* mean, float* var, int rows, int cols) {
  constexpr int wg = 256;
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<Moments, 1> s{sycl::range<1>{wg}, h};
    h.parallel_for<GroupStats>(sycl::nd_range<1>{std::size_t(rows) * wg, wg}, [=](sycl::nd_item<1> it) {
      const int row = it.get_group(0);
      Moments m;
      for (int c = it.get_local_id(0); c < cols; c += wg) m = merge(m, {1.0f, x[std::size_t(row) * cols + c], 0.0f});
      const Moments total = group_moments(it, m, s);
      if (it.get_local_id(0) == 0) {
        mean[row] = total.mean;
        var[row] = total.n > 0.0f ? total.m2 / total.n : 0.0f;
      }
    });
  });
}

}  // namespace snippets
