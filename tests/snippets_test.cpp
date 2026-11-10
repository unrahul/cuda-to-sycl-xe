// snippets_test.cpp - executes every documented snippet against an independent host reference.
// Build: icpx -fsycl -O2 -std=c++20 -I examples tests/snippets_test.cpp -o snippets_test
// Exit 0 pass, 1 fail, 77 skip (no GPU).
#include "snippets/dispatch.hpp"
#include "snippets/mapping.hpp"
#include "snippets/matrix.hpp"
#include "snippets/moments.hpp"
#include "snippets/patterns.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace snippets;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
  failures += !ok;
}

bool close(double got, double ref, double atol, double rtol) {
  return std::isfinite(got) && std::fabs(got - ref) <= atol + rtol * std::fabs(ref);
}

template <class T>
struct Device {  // owning device buffer
  sycl::queue& q;
  T* p;
  Device(sycl::queue& q, const std::vector<T>& h) : q(q), p(sycl::malloc_device<T>(h.size(), q)) {
    if (!p) throw std::bad_alloc{};
    q.memcpy(p, h.data(), h.size() * sizeof(T)).wait();
  }
  ~Device() { sycl::free(p, q); }
  std::vector<T> get(std::size_t n) {
    std::vector<T> h(n);
    q.memcpy(h.data(), p, n * sizeof(T)).wait_and_throw();
    return h;
  }
};

class Histogram;

}  // namespace

int main() {
  sycl::device dev;
  try {
    dev = sycl::device{sycl::gpu_selector_v};
  } catch (const sycl::exception&) {
    std::puts("SKIP: no GPU");
    return 77;
  }
  sycl::queue q{dev, sycl::property::queue::in_order{}};
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  auto random = [&](std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = u(rng);
    return v;
  };

  for (std::size_t n : {std::size_t{1}, std::size_t{255}, std::size_t{257}, std::size_t{4099}}) {
    auto x = random(n), y = random(n);
    Device<float> dx(q, x), dy(q, y);
    saxpy(q, n, 0.5f, dx.p, dy.p);
    auto got = dy.get(n);
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) ok &= close(got[i], 0.5 * x[i] + y[i], 1e-6, 1e-6);
    check(ok, "launch: saxpy with padded tail");
  }

  for (int cols : {1, 15, 16, 17, 255, 256, 1000}) {
    const int rows = 7;
    auto x = random(std::size_t(rows) * cols);
    std::vector<double> ref(rows, 0.0);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) ref[r] += x[std::size_t(r) * cols + c];
    Device<float> dx(q, x), out(q, std::vector<float>(rows, -1.0f));
    bool ok_wg = true, ok_sg = true;
    row_sum(q, dx.p, out.p, rows, cols);
    for (int r = 0; r < rows; ++r) ok_wg &= close(out.get(rows)[r], ref[r], 1e-5 * cols, 1e-5);
    if (supports_sub_group(dev, 16)) {
      row_sum_sg(q, dx.p, out.p, rows, cols);
      for (int r = 0; r < rows; ++r) ok_sg &= close(out.get(rows)[r], ref[r], 1e-5 * cols, 1e-5);
      check(ok_sg, "subgroup: butterfly row sum");
    } else {
      std::puts("SKIP: subgroup: width 16 unsupported");
    }
    check(ok_wg, "block-reduce: local-memory row sum");
  }

  {
    const int n = 10000, bins = 13;
    std::vector<int> idx(n);
    std::vector<unsigned> ref(bins, 0);
    for (int i = 0; i < n; ++i) ref[idx[i] = int(rng() % bins)]++;
    Device<int> di(q, idx);
    Device<unsigned> hist(q, std::vector<unsigned>(bins, 0));
    int* ip = di.p;
    unsigned* hp = hist.p;
    q.parallel_for<Histogram>(sycl::range<1>{std::size_t(n)}, [=](sycl::id<1> i) { hist_add(hp, ip[i]); });
    check(hist.get(bins) == ref, "atomic: histogram");
  }

  {
    const std::size_t n = 1000;
    auto in = random(n);
    std::vector<float> out(n);
    twice_on_device(q, in.data(), out.data(), n);
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) ok &= out[i] == 2.0f * in[i];
    check(ok, "streams: in-order copy, kernel, copy");
  }

  {
    sycl::queue q1{dev}, q2{dev};  // out-of-order queues: only the event orders them
    const std::size_t n = 4096;
    std::vector<float> zeros(n, 0.0f);
    Device<float> y(q1, zeros);
    float* yp = y.p;
    auto done = q1.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) { yp[i] = float(i[0]); });
    after(q2, done, yp, n).wait_and_throw();
    auto got = y.get(n);
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) ok &= got[i] == float(i) + 1.0f;
    check(ok, "events: cross-queue dependency");
  }

  for (int cols : {1, 2, 255, 256, 257, 5000}) {
    const int rows = 5;
    std::vector<float> x(std::size_t(rows) * cols);
    for (auto& v : x) v = 1000.0f + u(rng);  // large offset: naive E[x^2]-E[x]^2 loses digits
    std::vector<double> rm(rows, 0.0), rv(rows, 0.0);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) rm[r] += x[std::size_t(r) * cols + c];
      rm[r] /= cols;
      for (int c = 0; c < cols; ++c) rv[r] += std::pow(x[std::size_t(r) * cols + c] - rm[r], 2);
      rv[r] /= cols;
    }
    Device<float> dx(q, x), mean(q, std::vector<float>(rows)), var(q, std::vector<float>(rows));
    row_moments(q, dx.p, mean.p, var.p, rows, cols);
    auto gm = mean.get(rows), gv = var.get(rows);
    bool ok = true;
    for (int r = 0; r < rows; ++r) ok &= close(gm[r], rm[r], 1e-3, 1e-6) && close(gv[r], rv[r], 1e-4, 1e-3);
    check(ok, "moments: work-group mean and variance with large offset");
  }

  {
    const std::size_t n = 1024;
    std::vector<float> ones(n, 1.0f);
    Device<float> y(q, ones);
    q.parallel_for(sycl::range<1>{n}, ScaleLargeGrf{y.p, 3.0f}).wait_and_throw();
    auto got = y.get(n);
    check(got.front() == 3.0f && got.back() == 3.0f, "grf-mode: large-GRF kernel runs");
  }

  for (int offset : {0, 1}) {  // offset 1 misaligns every row start: must fall back to scalar
    const int rows = 3, cols = 1000;
    auto x = random(std::size_t(rows) * cols + 1), w = random(cols);
    Device<float> dx(q, x), dw(q, w), dy(q, std::vector<float>(x.size(), 0.0f));
    const float* xp = dx.p + offset;
    float* yp = dy.p + offset;
    const int width = max_vector_width<float>(cols, {xp, dw.p, yp});
    rms_norm(q, xp, dw.p, yp, rows, cols, 1e-6f);
    auto got = dy.get(x.size());
    bool ok = width == (offset ? 1 : 4);
    for (int r = 0; r < rows; ++r) {
      double ss = 0.0;
      for (int c = 0; c < cols; ++c) ss += std::pow(x[offset + r * cols + c], 2);
      const double inv = 1.0 / std::sqrt(ss / cols + 1e-6);
      for (int c = 0; c < cols; ++c) {
        const std::size_t i = offset + std::size_t(r) * cols + c;
        ok &= close(got[i], x[i] * inv * w[c], 1e-5, 1e-4);
      }
    }
    check(ok, offset ? "vector-width: misaligned rows use scalar path" : "vector-width: aligned rows use vec4");
  }

  for (int cols : {5, 300}) {  // short rows take sub-group 16, long rows 32 where supported
    const int rows = 9;
    auto x = random(std::size_t(rows) * cols);
    Device<float> dx(q, x), out(q, std::vector<float>(rows));
    row_max(q, dx.p, out.p, rows, cols);
    auto got = out.get(rows);
    bool ok = true;
    for (int r = 0; r < rows; ++r) ok &= got[r] == *std::max_element(x.begin() + r * cols, x.begin() + (r + 1) * cols);
    check(ok, "subgroup-choice: row max");
  }

  {
    const int rows = 37, k = 32 * 21;  // 21 blocks per row: lanes get uneven block counts
    std::vector<BlockQ4> w(std::size_t(rows) * k / 32);
    for (auto& b : w) {
      b.d = sycl::half(u(rng) * 0.1f);
      for (auto& v : b.qs) v = std::uint8_t(rng());
    }
    auto x = random(k);
    std::vector<double> ref(rows, 0.0);
    for (int r = 0; r < rows; ++r)
      for (int b = 0; b < k / 32; ++b) {
        const BlockQ4& blk = w[std::size_t(r) * (k / 32) + b];
        for (int j = 0; j < 16; ++j)
          ref[r] += double(float(blk.d)) * ((blk.qs[j] & 0xF) - 8) * x[b * 32 + j] +
                    double(float(blk.d)) * ((blk.qs[j] >> 4) - 8) * x[b * 32 + j + 16];
      }
    Device<BlockQ4> dw(q, w);
    Device<std::uint8_t> qs(q, std::vector<std::uint8_t>(w.size() * 16));
    Device<sycl::half> d(q, std::vector<sycl::half>(w.size()));
    Device<float> dx(q, x), y(q, std::vector<float>(rows));
    reorder_q4(q, dw.p, qs.p, d.p, w.size());
    gemv_q4(q, qs.p, d.p, dx.p, y.p, rows, k);
    auto got = y.get(rows);
    bool ok = true;
    for (int r = 0; r < rows; ++r) ok &= close(got[r], ref[r], 1e-3, 1e-4);
    check(ok, "q4-reorder: reorder then sub-group-per-row GEMV");
  }

  {
    const int rows = 11, cols = 60, k = 6;  // W = 8 lanes x E = 8 values covers 64 columns
    std::vector<float> x(std::size_t(rows) * cols);
    for (auto& v : x) v = float(int(rng() % 40)) / 4.0f;  // many ties
    Device<float> dx(q, x), vals(q, std::vector<float>(rows * k));
    Device<int> idx(q, std::vector<int>(rows * k));
    topk<8, 8>(q, dx.p, vals.p, idx.p, rows, cols, k);
    auto gv = vals.get(rows * k);
    auto gi = idx.get(rows * k);
    bool ok = true;
    for (int r = 0; r < rows; ++r) {
      std::vector<int> order(cols);
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return x[r * cols + a] > x[r * cols + b]; });
      for (int j = 0; j < k; ++j) ok &= gi[r * k + j] == order[j] && gv[r * k + j] == x[r * cols + order[j]];
    }
    check(ok, "topk: descending values, lower index wins ties");
  }

  if (supports_matrix_tile(dev, mtype::bf16, mtype::bf16, mtype::fp32, 8, 16, 16)) {
    using bf16 = sycl::ext::oneapi::bfloat16;
    const std::size_t M = 32, N = 48, K = 64;
    std::vector<bf16> a(M * K), b(K * N);
    for (auto& v : a) v = bf16(u(rng));
    for (auto& v : b) v = bf16(u(rng));
    Device<bf16> da(q, a), db(q, b);
    Device<float> dc(q, std::vector<float>(M * N));
    bf16_gemm<8, 16, 16, 16>(q, da.p, db.p, dc.p, M, N, K);
    auto got = dc.get(M * N);
    bool ok = true;
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t j = 0; j < N; ++j) {
        double ref = 0.0;
        for (std::size_t kk = 0; kk < K; ++kk) ref += double(float(a[i * K + kk])) * double(float(b[kk * N + j]));
        ok &= close(got[i * N + j], ref, 1e-4, 1e-4);
      }
    check(ok, "joint-matrix-bf16: 8x16x16 tiles, fp32 accumulate");
  } else {
    std::puts("SKIP: joint-matrix-bf16: no bf16 8x16x16 combination on this stack");
  }

  const char* gen[] = {"unknown", "Xe-HPG", "Xe2", "Xe3", "Xe3P"};
  std::printf("generation: %s; int8 8x16x32 matrix tile: %s; saxpy kernel max work-group: %zu\n",
              gen[int(xe_generation(dev))],
              supports_matrix_tile(dev, mtype::sint8, mtype::sint8, mtype::sint32, 8, 16, 32) ? "yes" : "no",
              kernel_max_work_group<Saxpy>(q));
  return failures ? 1 : 0;
}
