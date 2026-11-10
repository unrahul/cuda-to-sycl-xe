// silu_mul.sycl.cpp - explicit candidate sizes, a host reference check, and device-time diagnostics.
//
// Build: icpx -fsycl -O2 -std=c++20 -I ../common silu_mul.sycl.cpp -o silu_mul
#include "xe_arch.hpp"
#include "bench.hpp"

#include <cmath>
#include <random>
#include <vector>

template <int WG, int V> class SiluMul;
class StreamRef;

inline float silu(float x) { return x / (1.0f + sycl::native::exp(-x)); }

// one work-group per token row; V floats per lane per step; the loop strides over the row
template <int WG, int V>
void silu_mul(sycl::queue& q, const float* x, float* out, int tokens, int d) {
    q.parallel_for<SiluMul<WG, V>>(sycl::nd_range<1>((size_t) tokens * WG, WG), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_group(0);
        const float* a = x + (size_t) t * 2 * d;
        const float* b = a + d;
        float* o = out + (size_t) t * d;
        for (int i = (int) it.get_local_id(0) * V; i < d; i += WG * V) {
            if (i + V > d) {
                for (int k = 0; k < V && i + k < d; ++k) o[i + k] = silu(a[i + k]) * b[i + k];
                continue;
            }
            sycl::vec<float, V> g, u, r;
            g.load(0, sycl::multi_ptr<const float, sycl::access::address_space::global_space>(a + i));
            u.load(0, sycl::multi_ptr<const float, sycl::access::address_space::global_space>(b + i));
#pragma unroll
            for (int k = 0; k < V; ++k) r[k] = silu(g[k]) * u[k];
            r.store(0, sycl::multi_ptr<float, sycl::access::address_space::global_space>(o + i));
        }
    });
}

// Empirical stream reference: read the same 2 d floats per row and write d.
void stream_ref(sycl::queue& q, const float* x, float* out, int tokens, int d) {
    q.parallel_for<StreamRef>(sycl::nd_range<1>((size_t) tokens * 256, 256), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_group(0);
        const float* a = x + (size_t) t * 2 * d;
        const float* b = a + d;
        float* o = out + (size_t) t * d;
        for (int i = (int) it.get_local_id(0); i < d; i += 256) o[i] = a[i] + b[i];
    });
}

int main(int argc, char** argv) {
    sycl::queue q = bench::make_queue();
    if (q.get_device().get_info<sycl::info::device::max_work_group_size>() < 256) {
        std::fprintf(stderr, "requires work-group size 256\n"); return 2;
    }
    std::printf("device: %s; explicit candidate sizes, no tuned default\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int tokens = argc > 1 ? std::atoi(argv[1]) : 4096, d = argc > 2 ? std::atoi(argv[2]) : 6144;
    if (tokens <= 0 || d <= 0) { std::fprintf(stderr, "positive tokens and d required\n"); return 2; }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t) tokens * 2 * d), ref((size_t) tokens * d), got(ref.size());
    for (auto& v : hx) v = nd(rng);
    for (int t = 0; t < tokens; ++t)
        for (int i = 0; i < d; ++i) {
            const float a = hx[(size_t) t * 2 * d + i], b = hx[(size_t) t * 2 * d + d + i];
            ref[(size_t) t * d + i] = a / (1.0f + std::exp(-a)) * b;
        }
    float* x = sycl::malloc_device<float>(hx.size(), q);
    float* out = sycl::malloc_device<float>(ref.size(), q);
    q.memcpy(x, hx.data(), hx.size() * 4).wait_and_throw();
    auto run = [&](int wg, int v) {
        if (wg == 512 && v == 4) silu_mul<512, 4>(q, x, out, tokens, d);
        else if (wg == 256 && v == 4) silu_mul<256, 4>(q, x, out, tokens, d);
        else if (wg == 512 && v == 8) silu_mul<512, 8>(q, x, out, tokens, d);
        else silu_mul<256, 8>(q, x, out, tokens, d);
    };
    const size_t bytes = hx.size() * 4 + ref.size() * 4;
    const double ceiling = bench::time_us(q, 1, [&](int) { stream_ref(q, x, out, tokens, d); }, 20);
    std::printf("empirical stream reference: %.1f us (%.0f GB/s)\n\n", ceiling, bytes / (ceiling * 1e3));
    bench::header();
    bool ok = true;
    for (int wg : {256, 512})
        for (int v : {4, 8}) {
            if ((size_t) wg > q.get_device().get_info<sycl::info::device::max_work_group_size>()) continue;
            run(wg, v);
            q.memcpy(got.data(), out, got.size() * 4).wait_and_throw();
            const bench::Parity p = bench::compare(got, ref);
            ok = ok && bench::within_tolerance(got, ref, 1e-5, 1e-5);
            char par[32]; std::snprintf(par, sizeof par, "%.1e", p.rel_err);
            const double us = bench::time_us(q, 1, [&](int) { run(wg, v); }, 20);
            char name[32]; std::snprintf(name, sizeof name, "wg%d_v%d", wg, v);
            bench::row(name, 1, us, bytes / (us * 1e3), par, ceiling);
        }
    sycl::free(x, q); sycl::free(out, q);
    return ok ? 0 : 1;
}
