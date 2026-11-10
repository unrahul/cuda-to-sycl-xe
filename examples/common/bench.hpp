// bench.hpp - device-time measurement and parity helpers for kernel variants.
//
// Timing uses queue profiling and empirical access-pattern comparisons. Smoke mode reduces
// repetitions and cache rotation; its measurements are not performance results.
// Parity reports global error and supports per-element absolute-plus-relative checks.
#pragma once
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bench {

inline sycl::queue make_queue() {
    return sycl::queue{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
}

inline double us_between(const sycl::event& a, const sycl::event& b) {
    return (double) (b.get_profiling_info<sycl::info::event_profiling::command_end>() -
                     a.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e3;
}

// how many copies of `bytes` make the rotation exceed three times the L2 (at least 2)
inline int copies_beyond_cache(const sycl::queue& q, size_t bytes) {
    if (!bytes) throw std::invalid_argument("bytes must be positive");
    if (std::getenv("BENCH_SMOKE")) return 2;
    const size_t l2 = q.get_device().get_info<sycl::info::device::global_mem_cache_size>();
    return (int) std::max<size_t>(2, (3 * l2 + bytes - 1) / bytes);
}

// device time per call of fn(copy_index) over reps calls after warm-up, rotating over `copies`
inline double time_us(sycl::queue& q, int copies, const std::function<void(int)>& fn, int reps = 0) {
    if (copies <= 0 || reps < 0) throw std::invalid_argument("invalid timing arguments");
    if (std::getenv("BENCH_SMOKE")) reps = 2;
    if (reps == 0) reps = std::max(4 * copies, 48);
    for (int i = 0; i < copies; ++i) fn(i % copies);
    auto t0 = q.ext_oneapi_submit_barrier();
    for (int i = 0; i < reps; ++i) fn(i % copies);
    auto t1 = q.ext_oneapi_submit_barrier();
    t1.wait_and_throw();
    return us_between(t0, t1) / reps;
}

struct Parity {
    bool bitwise = false;
    double rel_err = 0.0;   // max |got - ref| / max |ref|
};

inline Parity compare(const std::vector<float>& got, const std::vector<float>& ref) {
    Parity p;
    p.bitwise = got.size() == ref.size() && std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0;
    if (got.size() != ref.size()) { p.rel_err = std::numeric_limits<double>::infinity(); return p; }
    double scale = 1e-30, err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) { p.bitwise = false; p.rel_err = std::numeric_limits<double>::infinity(); return p; }
        scale = std::max(scale, (double) std::fabs(ref[i]));
        err = std::max(err, (double) std::fabs((double) got[i] - (double) ref[i]));
    }
    p.rel_err = err / scale;
    return p;
}

inline Parity compare(const std::vector<float>& got, const std::vector<double>& ref) {
    Parity p;
    if (got.size() != ref.size()) { p.rel_err = std::numeric_limits<double>::infinity(); return p; }
    double scale = 1e-30, err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) { p.bitwise = false; p.rel_err = std::numeric_limits<double>::infinity(); return p; }
        scale = std::max(scale, std::fabs(ref[i]));
        err = std::max(err, std::fabs((double) got[i] - ref[i]));
    }
    p.rel_err = err / scale;
    return p;
}

template <class Ref>
inline bool within_tolerance(const std::vector<float>& got, const std::vector<Ref>& ref, double atol, double rtol) {
    if (got.size() != ref.size() || !std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0) return false;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double actual = got[i], expected = ref[i];
        if (!std::isfinite(actual) || !std::isfinite(expected) || std::fabs(actual - expected) > atol + rtol * std::fabs(expected)) return false;
    }
    return true;
}

// Ratios are empirical diagnostics, not optimization verdicts.
inline void row(const char* variant, int cols, double us, double gbps, const char* parity, double ceiling_us) {
    const double ratio = ceiling_us > 0 ? us / ceiling_us : 0;
    std::printf("| %-14s | %4d | %9.2f | %7.0f | %-14s | %8.2f |\n", variant, cols, us, gbps, parity, ratio);
}

inline void header() {
    std::printf("| variant        | cols |   us      | GB/s    | parity         | ratio    |\n");
    std::printf("|----------------|------|-----------|---------|----------------|----------|\n");
}

}  // namespace bench
