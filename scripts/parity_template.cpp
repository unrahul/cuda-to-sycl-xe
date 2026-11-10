// parity_template.cpp - the shape of a kernel parity program: an FP64 reference of the formula, the production kernel as the bitwise
// reference where the arithmetic order is the same, SKIP on a card that does not take the kernel, OK / FAILED and an exit code for ctest.
// Replace the two kernels and the reference with yours; keep the structure. Build: icpx -fsycl -O2 -std=c++20 -I ../examples/common
// parity_template.cpp -o parity_template
//
// Here: y[r] = sum_k w[r][k] x[k] over fp32 rows, as (A) one work-item per row (the "production" order) and (B) a sub-group per row with a
// butterfly (another order). A against the FP64 reference with a tolerance; B against the FP64 reference and, bitwise, against A: it
// fails that check by design (a float sum in another order), which is the case the tolerance exists for.
#include "bench.hpp"
#include "xe_arch.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

class RowKernel;
class SubGroupKernel;

void rows_a(sycl::queue& q, const float* w, const float* x, float* y, int n_in, int n_out) {
    q.parallel_for<RowKernel>(sycl::range<1>((size_t) n_out), [=](sycl::id<1> id) {
        const int r = (int) id[0];
        float s = 0.0f;
        for (int k = 0; k < n_in; ++k) s = sycl::fma(w[(size_t) r * n_in + k], x[k], s);
        y[r] = s;
    });
}
void rows_b(sycl::queue& q, const float* w, const float* x, float* y, int n_in, int n_out) {
    q.parallel_for<SubGroupKernel>(sycl::nd_range<1>((size_t) ((n_out + 15) / 16) * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const int r = (int) (it.get_global_linear_id() / 16), lane = (int) sg.get_local_linear_id();
        if (r >= n_out) return;
        float s = 0.0f;
        for (int k = lane; k < n_in; k += 16) s = sycl::fma(w[(size_t) r * n_in + k], x[k], s);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s += sycl::permute_group_by_xor(sg, s, o);
        if (lane == 0) y[r] = s;
    });
}
// the condition under which the kernel under test runs at all (a generation, a format, an alignment); SKIP otherwise
bool takes(const sycl::device& d) { const auto sizes = d.get_info<sycl::info::device::sub_group_sizes>(); return d.get_info<sycl::info::device::max_work_group_size>() >= 256 && std::find(sizes.begin(), sizes.end(), 16) != sizes.end(); }

int main() {
    sycl::queue q = bench::make_queue();
    if (!takes(q.get_device())) { std::printf("SKIP: the kernel does not run on %s\n", q.get_device().get_info<sycl::info::device::name>().c_str()); return 77; }
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    bool ok = true;
    for (int n_in : {256, 2560}) for (int n_out : {16, 640}) {   // the shapes the dispatch can take, small ones included
        std::vector<float> hw((size_t) n_out * n_in), hx(n_in);
        for (auto& v : hw) v = nd(rng);
        for (auto& v : hx) v = nd(rng);
        std::vector<double> ref(n_out, 0.0);
        for (int r = 0; r < n_out; ++r) for (int k = 0; k < n_in; ++k) ref[r] += (double) hw[(size_t) r * n_in + k] * hx[k];
        float* w = sycl::malloc_device<float>(hw.size(), q); float* x = sycl::malloc_device<float>(n_in, q);
        float* ya = sycl::malloc_device<float>(n_out, q); float* yb = sycl::malloc_device<float>(n_out, q);
        q.memcpy(w, hw.data(), hw.size() * 4); q.memcpy(x, hx.data(), n_in * 4).wait_and_throw();
        rows_a(q, w, x, ya, n_in, n_out); rows_b(q, w, x, yb, n_in, n_out); q.wait_and_throw();
        std::vector<float> ga(n_out), gb(n_out);
        q.memcpy(ga.data(), ya, n_out * 4); q.memcpy(gb.data(), yb, n_out * 4).wait_and_throw();
        const bench::Parity pa = bench::compare(ga, ref), pb = bench::compare(gb, ref), pab = bench::compare(gb, ga);
        const double tol = 1e-5;   // what a float sum of n_in terms can meet; set per kernel from the FP64 error of the production path
        const bool pass = bench::within_tolerance(ga, ref, tol, tol) && bench::within_tolerance(gb, ref, tol, tol);
        ok = ok && pass;
        std::printf("%5d x %4d: A vs FP64 %.1e, B vs FP64 %.1e, B vs A %s  %s\n", n_in, n_out, pa.rel_err, pb.rel_err,
                    pab.bitwise ? "bitwise" : "not bitwise (another order: the FP64 check decides)", pass ? "ok" : "FAIL");
        sycl::free(w, q); sycl::free(x, q); sycl::free(ya, q); sycl::free(yb, q);
    }
    std::printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
