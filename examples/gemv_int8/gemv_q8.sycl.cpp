// Constrained int8 GEMV candidates. See examples/README.md for supported shapes.
// Timing is a workload-specific diagnostic; DPAS requires separate hardware validation.
#include "xe_arch.hpp"
#include "bench.hpp"
#include "matrix_shape.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace jm = sycl::ext::oneapi::experimental::matrix;

template <int NC, int SG> class GemvDp4a;
template <int NC, int N, int SG> class GemvDpas;
template <int N> class PackB;
template <int SG> class ProbeRow;
class ProbePacked;

// Check each instantiated kernel's device-specific work-group limit as well as
// device-wide limits. Compilation/query failure means this specialization is unavailable.
template <int NC, int N, int SG>
bool kernel_resources_ok(const sycl::queue& q) {
    try {
        auto id = sycl::get_kernel_id<GemvDpas<NC, N, SG>>();
        auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {q.get_device()}, {id});
        auto kernel = bundle.get_kernel(id);
        const auto max_wg = kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
        const auto max_sg = kernel.template get_info<sycl::info::kernel_device_specific::max_sub_group_size>(q.get_device());
        const auto compiled_sg = kernel.template get_info<sycl::info::kernel_device_specific::compile_sub_group_size>(q.get_device());
        return max_wg >= SG && max_sg >= SG && (compiled_sg == 0 || compiled_sg == SG);
    } catch (const sycl::exception&) { return false; }
}
template <int N, int SG>
bool all_dpas_resources_ok(const sycl::queue& q) {
    return kernel_resources_ok<1,N,SG>(q) && kernel_resources_ok<2,N,SG>(q) &&
           kernel_resources_ok<3,N,SG>(q) && kernel_resources_ok<4,N,SG>(q) &&
           kernel_resources_ok<5,N,SG>(q) && kernel_resources_ok<6,N,SG>(q) &&
           kernel_resources_ok<7,N,SG>(q) && kernel_resources_ok<8,N,SG>(q);
}

// Build a packed word without aliasing int8 storage through an int pointer.
inline int packed4(const int8_t* p) {
    uint32_t bits = 0;
    for (int j=0;j<4;++j) bits |= uint32_t(static_cast<uint8_t>(p[j])) << (8*j);
    return sycl::bit_cast<int>(bits);
}

// four signed-byte products summed into acc (CUDA's __dp4a); the engine this comes from uses dpct::dp4a, the migration helper
inline int dp4a(int a, int b, int acc) {
    const sycl::vec<int8_t, 4> va = sycl::bit_cast<sycl::vec<int8_t, 4>>(a), vb = sycl::bit_cast<sycl::vec<int8_t, 4>>(b);
    return acc + (int) va[0] * vb[0] + (int) va[1] * vb[1] + (int) va[2] * vb[2] + (int) va[3] * vb[3];
}

// --- sg32 / sg16: a sub-group per row, lane-strided dp4a, butterfly ---
template <int NC, int SG>
void gemv_dp4a(sycl::queue& q, const int8_t* W, const float* sw, const int8_t* x, const float* sx, float* y, int n_in, int n_out) {
    constexpr int ROWS = 256 / SG;   // sub-groups (rows) per work-group
    q.parallel_for<GemvDp4a<NC, SG>>(sycl::nd_range<1>((size_t) ((n_out + ROWS - 1) / ROWS) * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int row = (int) it.get_group(0) * ROWS + (int) sg.get_group_linear_id();
        if (row >= n_out) return;
        const int8_t* w4 = W + (size_t) row * n_in;
        int acc[NC] = {};
        for (int k = lane; k < n_in / 4; k += SG) {
            const int w = packed4(w4 + 4*k);
#pragma unroll
            for (int c = 0; c < NC; ++c) acc[c] = dp4a(w, packed4(x + (size_t) c * n_in + 4*k), acc[c]);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            int v = acc[c];
#pragma unroll
            for (int o = SG / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
            if (lane == 0) y[(size_t) c * n_out + row] = sx[c] * sw[row] * (float) v;
        }
    });
}

// --- dpas: joint_matrix with W in the packed layout ---
// packed B for a tile of N rows (columns of B) and 32 K: dword (g, n) = W[n][4g .. 4g + 3]; laid out as [n_in / 4][n_out] dwords
template <int N>
void pack_b(sycl::queue& q, const int8_t* W, int32_t* Wp, int n_in, int n_out) {
    q.parallel_for<PackB<N>>(sycl::range<2>((size_t) n_in / 4, (size_t) n_out), [=](sycl::id<2> id) {
        const int g = (int) id[0], n = (int) id[1];
        Wp[(size_t) g * n_out + n] = packed4(W + (size_t) n * n_in + 4 * g);
    });
}
template <int NC, int N, int SG>
void gemv_dpas(sycl::queue& q, const int32_t* Wp, const float* sw, const int8_t* x, const float* sx, float* y, int n_in, int n_out) {
    constexpr int M = 8;   // x must contain eight allocated rows even when NC is smaller
    static_assert(NC >= 1 && NC <= M);
    const int tiles = n_out / N;
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<int32_t, 1> cbuf(sycl::range<1>(M * N), h);   // the accumulator tile, stored once for the scaled writes
      h.parallel_for<GemvDpas<NC, N, SG>>(sycl::nd_range<1>((size_t) tiles * SG, SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int n0 = (int) it.get_group(0) * N, lane = (int) sg.get_local_linear_id();
        jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, M, N> acc;
        jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, M, 32, jm::layout::row_major> a;
        jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, N, jm::layout::ext_intel_packed> b;
        jm::joint_matrix_fill(sg, acc, 0);
        for (int k0 = 0; k0 < n_in; k0 += 32) {
            // A: M rows of 32 int8 at stride n_in (the window's columns; x always holds 8 rows, so rows >= NC are real data whose
            // results the store below discards: the DPAS runs 8 rows whatever NC is)
            jm::joint_matrix_load(sg, a, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(x + k0), (size_t) n_in);
            // B: the packed tile at (k0 / 4, n0): stride between K groups is the packed row (n_out dwords = 4 n_out bytes)
            jm::joint_matrix_load(sg, b, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(reinterpret_cast<const int8_t*>(Wp + (size_t) (k0 / 4) * n_out + n0)), (size_t) n_out * 4);
            jm::joint_matrix_mad(sg, acc, a, b, acc);
        }
        // the tile to local memory (row c = column c of the window, N ints), then lane n scales and stores column n of the NC real rows
        jm::joint_matrix_store(sg, acc, cbuf.template get_multi_ptr<sycl::access::decorated::no>(), (size_t) N, jm::layout::row_major);
        sycl::group_barrier(sg);
        for (int n = lane; n < N; n += SG)
#pragma unroll
            for (int c = 0; c < NC; ++c) y[(size_t) c * n_out + n0 + n] = sx[c] * sw[n0 + n] * (float) cbuf[c * N + n];
      });
    });
}

// --- load-only probes of the two patterns ---
template <int SG>
void probe_row(sycl::queue& q, const int8_t* W, int n_in, int n_out, float* out) {
    constexpr int ROWS = 256 / SG;
    q.parallel_for<ProbeRow<SG>>(sycl::nd_range<1>((size_t) ((n_out + ROWS - 1) / ROWS) * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int row = (int) it.get_group(0) * ROWS + (int) sg.get_group_linear_id();
        if (row >= n_out) return;
        const int8_t* w4 = W + (size_t) row * n_in;
        int acc = 0;
        for (int k = (int) sg.get_local_linear_id(); k < n_in / 4; k += SG) acc ^= packed4(w4 + 4*k);
        if (acc == 0x12345678) sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(out[0]).store(1.0f);
    });
}
void probe_packed(sycl::queue& q, const int32_t* Wp, size_t dwords, float* out) {
    q.parallel_for<ProbePacked>(sycl::nd_range<1>(65536, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int32_t* v = Wp;
        int acc = 0;
        for (size_t i = it.get_global_linear_id(); i < dwords / 4; i += 65536) { acc ^= v[4*i] ^ v[4*i+1] ^ v[4*i+2] ^ v[4*i+3]; }
        if (acc == 0x12345678) sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(out[0]).store(1.0f);
    });
}

// the column-count switch (1..8) for a variant template
#define NC_SWITCH(nc, CALL) switch (nc) { case 1: CALL(1); break; case 2: CALL(2); break; case 3: CALL(3); break; case 4: CALL(4); break; \
                                          case 5: CALL(5); break; case 6: CALL(6); break; case 7: CALL(7); break; default: CALL(8); break; }

int main(int argc, char** argv) {
    sycl::queue q = bench::make_queue();
    const sycl::device d = q.get_device();
    const bool require_dpas = argc == 4 && std::string(argv[3]) == "--require-dpas";
    if (argc > 4 || (argc == 4 && !require_dpas)) { std::fprintf(stderr,"usage: gemv_q8 [n_in n_out [--require-dpas]]\n"); return 2; }
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 2560, n_out = argc > 2 ? std::atoi(argv[2]) : 10240;
    if (n_in <= 0 || n_in > 131071 || n_in % 32 || n_out <= 0) {
        std::fprintf(stderr, "n_in must be a positive multiple of 32 <= 131071; n_out must be positive\n"); return 2;
    }
    const auto subgroups = d.get_info<sycl::info::device::sub_group_sizes>();
    if (d.get_info<sycl::info::device::max_work_group_size>() < 256 ||
        std::find(subgroups.begin(), subgroups.end(), 16) == subgroups.end() ||
        std::find(subgroups.begin(), subgroups.end(), 32) == subgroups.end()) {
        std::fprintf(stderr, "requires work-group size 256 and sub-group sizes 16 and 32\n"); return 2;
    }
    std::printf("device: %s; %d x %d int8 weights\n", d.get_info<sycl::info::device::name>().c_str(), n_in, n_out);
    // A shape combination does not assert a subgroup pairing: verify the subgroup,
    // per-kernel work-group limits and actual execution separately.
    bool shape8 = false, shape16 = false;
    if (d.has(sycl::aspect::ext_intel_matrix)) {
        try {
            for (const auto& c : d.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>()) {
                if (c.atype != jm::matrix_type::sint8 || c.btype != jm::matrix_type::sint8 ||
                    c.ctype != jm::matrix_type::sint32 || c.dtype != jm::matrix_type::sint32) continue;
                shape8 |= xe::matrix_shape_matches(c, 8, 8, 32);
                shape16 |= xe::matrix_shape_matches(c, 8, 16, 32);
            }
        } catch (const sycl::exception&) { std::fprintf(stderr, "matrix combination query failed; DPAS disabled\n"); }
    }
    const auto max_items = d.get_info<sycl::info::device::max_work_item_sizes<1>>()[0];
    const auto local_mem = d.get_info<sycl::info::device::local_mem_size>();
    auto available = [&](int width, bool shape) {
        return shape && n_out % width == 0 && max_items >= (size_t)width &&
            d.get_info<sycl::info::device::max_work_group_size>() >= (size_t)width &&
            local_mem >= (size_t)(8 * width * sizeof(int32_t)) &&
            std::find(subgroups.begin(), subgroups.end(), width) != subgroups.end();
    };
    int dpas_n = 0;
    if (available(16, shape16) && all_dpas_resources_ok<16,16>(q)) dpas_n = 16;
    else if (available(8, shape8) && all_dpas_resources_ok<8,8>(q)) dpas_n = 8;
    const bool dpas_ok = dpas_n != 0;
    if (require_dpas && !dpas_ok) {
        std::fprintf(stderr,"FAIL: no queried and resource-compatible DPAS variant for this shape\n"); return 1;
    }
    std::printf("DPAS candidate width: %d (0 means unavailable; execution still requires parity)\n", dpas_n);
    // data: int8 weights with a per-row scale, 8 int8 columns with a per-column scale, and the FP64 reference
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> q8(-127, 127);
    std::uniform_real_distribution<float> sc(0.001f, 0.02f);
    const size_t wb = (size_t) n_out * n_in;
    std::vector<int8_t> hw(wb), hx((size_t) 8 * n_in);
    std::vector<float> hsw(n_out), hsx(8);
    for (auto& v : hw) v = (int8_t) q8(rng);
    for (auto& v : hx) v = (int8_t) q8(rng);
    for (auto& v : hsw) v = sc(rng);
    for (auto& v : hsx) v = sc(rng);
    const std::string input_case = std::getenv("GEMV_CASE") ? std::getenv("GEMV_CASE") : "random";
    if (input_case == "zero") { std::fill(hw.begin(), hw.end(), 0); }
    else if (input_case == "extreme") { std::fill(hw.begin(),hw.end(),-128); std::fill(hx.begin(),hx.end(),-128); }
    else if (input_case == "alternating") {
        for(size_t i=0;i<hw.size();++i) hw[i]=(i%2)?127:-128;
        for(size_t i=0;i<hx.size();++i) hx[i]=(i%3)?-128:127;
    } else if (input_case != "random") { std::fprintf(stderr,"unknown GEMV_CASE\n"); return 2; }
    std::printf("input case: %s; required DPAS: %s\n",input_case.c_str(),require_dpas?"yes":"no");
    std::vector<double> ref((size_t) 8 * n_out);
    for (int c = 0; c < 8; ++c)
        for (int n = 0; n < n_out; ++n) {
            long long s = 0;
            for (int k = 0; k < n_in; ++k) s += (long long) hw[(size_t) n * n_in + k] * hx[(size_t) c * n_in + k];
            ref[(size_t) c * n_out + n] = (double) hsx[c] * hsw[n] * (double) s;
        }
    const int copies = bench::copies_beyond_cache(q, wb);
    int8_t* W = sycl::malloc_device<int8_t>(wb * copies, q);
    int32_t* Wp = sycl::malloc_device<int32_t>(wb / 4 * copies, q);
    for (int i = 0; i < copies; ++i) q.memcpy(W + wb * i, hw.data(), wb);
    float* sw = sycl::malloc_device<float>(n_out, q); float* sx = sycl::malloc_device<float>(8, q);
    int8_t* x = sycl::malloc_device<int8_t>(hx.size(), q); float* y = sycl::malloc_device<float>((size_t) 8 * n_out, q);
    float* probe_out = sycl::malloc_device<float>(16, q);
    q.memcpy(sw, hsw.data(), n_out * 4); q.memcpy(sx, hsx.data(), 32); q.memcpy(x, hx.data(), hx.size()).wait_and_throw();
    if (dpas_ok) { for (int i = 0; i < copies; ++i) { if (dpas_n == 16) pack_b<16>(q, W + wb * i, Wp + wb / 4 * i, n_in, n_out); else pack_b<8>(q, W + wb * i, Wp + wb / 4 * i, n_in, n_out); } q.wait_and_throw(); }

    const double ceil_row32 = bench::time_us(q, copies, [&](int i) { probe_row<32>(q, W + wb * i, n_in, n_out, probe_out); });
    const double ceil_row16 = bench::time_us(q, copies, [&](int i) { probe_row<16>(q, W + wb * i, n_in, n_out, probe_out); });
    const double ceil_packed = dpas_ok ? bench::time_us(q, copies, [&](int i) { probe_packed(q, Wp + wb / 4 * i, wb / 4, probe_out); }) : 0.0;
    std::printf("empirical load-only times: row per 32-lane sub-group %.1f us (%.0f GB/s), row per 16-lane sub-group %.1f us (%.0f GB/s)",
                ceil_row32, wb / (ceil_row32 * 1e3), ceil_row16, wb / (ceil_row16 * 1e3));
    if (dpas_ok) std::printf(", packed stream %.1f us (%.0f GB/s)", ceil_packed, wb / (ceil_packed * 1e3));
    std::printf("\n\n");

    std::vector<float> got((size_t) 8 * n_out), refbits;
    bench::header();
    bool ok = true;
    for (int nc = 1; nc <= 8; ++nc) {
        auto run = [&](const char* name, double ceiling, const std::function<void(int)>& fn, bool is_ref) {
            q.memset(y, 0, (size_t) 8 * n_out * 4).wait_and_throw();
            fn(0);
            q.memcpy(got.data(), y, (size_t) nc * n_out * 4).wait_and_throw();
            got.resize((size_t) nc * n_out);
            char par[32];
            if (is_ref) {
                refbits = got;
                const auto pd = bench::compare(got, std::vector<double>(ref.begin(), ref.begin() + (size_t) nc * n_out));
                const bool pass = bench::within_tolerance(got, std::vector<double>(ref.begin(), ref.begin() + (size_t) nc * n_out), 1e-5, 1e-5);
                ok = ok && pass;
                std::snprintf(par, sizeof par, "%s %.1e", pass ? "ref" : "DIFF", pd.rel_err);
            }
            else {
                const bench::Parity pb = bench::compare(got, refbits);
                const bench::Parity pd = bench::compare(got, std::vector<double>(ref.begin(), ref.begin() + (size_t) nc * n_out));
                const bool pass = bench::within_tolerance(got, std::vector<double>(ref.begin(), ref.begin() + (size_t) nc * n_out), 1e-5, 1e-5);
                ok = ok && pass;
                if (pb.bitwise && pass) std::snprintf(par, sizeof par, "EQ");
                else std::snprintf(par, sizeof par, "%s %.1e", pass ? "tol" : "DIFF", pd.rel_err);
            }
            got.resize((size_t) 8 * n_out);
            if (!ok) { std::fprintf(stderr, "parity failed: timing suppressed\n"); return; }
            const double us = bench::time_us(q, copies, fn);
            bench::row(name, nc, us, wb / (us * 1e3), par, ceiling);
        };
#define RUN32(NC) run("sg32", ceil_row32, [&](int i) { gemv_dp4a<NC, 32>(q, W + wb * i, sw, x, sx, y, n_in, n_out); }, true)
#define RUN16(NC) run("sg16", ceil_row16, [&](int i) { gemv_dp4a<NC, 16>(q, W + wb * i, sw, x, sx, y, n_in, n_out); }, false)
#define RUNDP16(NC) run("dpas_n16", ceil_packed, [&](int i) { gemv_dpas<NC, 16, 16>(q, Wp + wb / 4 * i, sw, x, sx, y, n_in, n_out); }, false)
#define RUNDP8(NC) run("dpas_n8", ceil_packed, [&](int i) { gemv_dpas<NC, 8, 8>(q, Wp + wb / 4 * i, sw, x, sx, y, n_in, n_out); }, false)
        NC_SWITCH(nc, RUN32)
        NC_SWITCH(nc, RUN16)
        if (dpas_ok && dpas_n == 16 && n_out % 16 == 0) { NC_SWITCH(nc, RUNDP16) }
        if (dpas_ok && dpas_n == 8 && n_out % 8 == 0) { NC_SWITCH(nc, RUNDP8) }
    }
    std::printf("\nratio: time divided by the measured pattern time; diagnostic only\n");
    if (!dpas_ok) std::printf("No compatible queried int8 8x8x32 or 8x16x32 DPAS variant; use portable GEMV for unsupported devices.\n");
    sycl::free(W, q); sycl::free(Wp, q); sycl::free(sw, q); sycl::free(sx, q); sycl::free(x, q); sycl::free(y, q); sycl::free(probe_out, q);
    return ok ? 0 : 1;
}
