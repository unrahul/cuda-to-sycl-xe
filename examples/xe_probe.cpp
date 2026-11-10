// xe_probe.cpp - runtime-reported properties and optional measurements: what the device reports, and the micro-measurements a kernel design
// needs (kernel boundary cost eager and as a graph, empirical read-pattern measurements). Prints markdown.
// Nothing in the source is a number of any card; every number printed comes from this run.
//
// Build: icpx -fsycl -O2 -std=c++20 -I common xe_probe.cpp -o xe_probe
//        XE_PROBE_2D compiles an experimental pattern but the default probe does not launch it.
#include "xe_arch.hpp"
#include "bench.hpp"

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cstdint>
#include <cstdio>

namespace sx = sycl::ext::oneapi::experimental;

class Spin;
class Stream16;
class LaneRow;
class Probe2D;

template <class E, int K> using vec_t = E __attribute__((ext_vector_type(K)));

// `work` dependent multiply-adds per item, one load and one store
sycl::event spin(sycl::queue& q, float* buf, int groups, int work) {
    return q.parallel_for<Spin>(sycl::nd_range<1>((size_t) groups * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const size_t i = it.get_global_linear_id();
        float v = buf[i];
        for (int k = 0; k < work; ++k) v = sycl::fma(v, 0.999f, 0.001f);
        buf[i] = v;
    });
}
// coalesced stream: lane i of every sub-group reads aligned 16-byte runs i, i + 16, ...; XOR-reduced so nothing is optimized away
void stream16(sycl::queue& q, const uint8_t* w, size_t bytes, float* y) {
    const size_t n16 = bytes / 16;
    q.parallel_for<Stream16>(sycl::nd_range<1>(65536, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::uint4* v = reinterpret_cast<const sycl::uint4*>(w);
        unsigned acc = 0;
        for (size_t i = it.get_global_linear_id(); i < n16; i += 65536) { const sycl::uint4 t = v[i]; acc ^= t.x() ^ t.w(); }
        if (acc == 0x12345678u) sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(y[0]).store(1.0f);
    });
}
// one row per lane (the DPAS B-operand feed without 2D loads): lane = row, 32 bytes per step along its own row
void lane_row(sycl::queue& q, const uint8_t* w, int rows, int row_bytes, float* y) {
    q.parallel_for<LaneRow>(sycl::nd_range<1>((size_t) rows, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const vec_t<uint32_t, 8>* r = reinterpret_cast<const vec_t<uint32_t, 8>*>(w + (size_t) it.get_global_linear_id() * row_bytes);
        unsigned acc = 0;
        for (int i = 0; i < row_bytes / 32; ++i) { const vec_t<uint32_t, 8> v = r[i]; acc ^= v[0] ^ v[7]; }
        if (acc == 0x12345678u) sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(y[0]).store(1.0f);
    });
}
#if XE_PROBE_2D
typedef int coord2 __attribute__((ext_vector_type(2)));
SYCL_EXTERNAL extern "C" vec_t<uint32_t, 8> __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(long base, int w1, int h1, int p1, coord2 c);
// the transposed 32-bit 2D block load: 16 rows x 8 dwords per message, lane n = row n (the DPAS B layout). A work-group of 8 sub-groups
// takes a 16-row tile and each sub-group walks one eighth of the row, so eight loads of the tile are in flight at once (the pattern a
// DPAS matrix-vector kernel uses: the row split across sub-groups, the partial sums joined in local memory)
void probe_2d(sycl::queue& q, const uint8_t* w, int rows, int row_bytes, float* y) {
    q.parallel_for<Probe2D>(sycl::nd_range<1>((size_t) (rows / 16) * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int row0 = (int) it.get_group(0) * 16, s = (int) it.get_sub_group().get_group_linear_id();
        const int dw = row_bytes / 4, x0 = s * dw / 8 / 8 * 8, x1 = (s + 1) * dw / 8 / 8 * 8;
        unsigned acc = 0;
        for (int x = x0; x < x1; x += 8) {
            const vec_t<uint32_t, 8> v = __builtin_IB_subgroup_block_read_flat_transpose_u32_k8((long) w, row_bytes - 1, rows - 1, row_bytes - 1, coord2{x, row0});
            acc ^= v[0] ^ v[7];
        }
        if (acc == 0x12345678u) sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(y[0]).store(1.0f);
    });
}
#endif

int main(int argc, char** argv) {
    sycl::queue q = bench::make_queue();
    const sycl::device d = q.get_device();
    const xe::Device dv = xe::query(d);

    std::printf("# %s\n\n", dv.name.c_str());
    std::printf("| query | value |\n|---|---|\n");
    if (d.has(sycl::aspect::ext_intel_device_id)) std::printf("| device id | 0x%04x |\n", dv.device_id);
    else std::printf("| device id | unavailable |\n");
    std::printf("| driver | %s |\n", dv.driver.c_str());
    std::printf("| compute units (runtime-reported) | %d |\n| max work-group size | %zu |\n| local memory per work-group | %zu KB |\n",
                dv.compute_units, dv.max_wg, dv.local_mem / 1024);
    std::printf("| reported global memory cache | %zu MB |\n| cache line | %zu B |\n", dv.l2 >> 20, dv.cache_line);
    std::printf("| sub-group sizes | ");
    for (size_t s : dv.sub_groups) std::printf("%zu ", s);
    std::printf("|\n| fp16 / fp64 / atomic64 / matrix aspects | %d / %d / %d / %d |\n", dv.fp16, dv.fp64, dv.atomic64, dv.matrix);
    if (dv.matrix) {
        std::printf("| joint_matrix combinations (A x B -> C / D, fixed M N K, max M N K) | ");
        try {
            for (const auto& c : d.get_info<sx::info::device::matrix_combinations>()) {
                const auto t = [](sx::matrix::matrix_type m) {
                    using T = sx::matrix::matrix_type;
                    switch (m) {
                        case T::sint8: return std::string("s8"); case T::uint8: return std::string("u8"); case T::fp16: return std::string("fp16");
                        case T::bf16: return std::string("bf16"); case T::tf32: return std::string("tf32"); case T::fp32: return std::string("fp32");
                        case T::fp64: return std::string("fp64"); case T::sint32: return std::string("s32"); case T::uint32: return std::string("u32");
                        default: return std::string("type") + std::to_string((int) m);
                    }
                };
                std::printf("%sx%s->%s/%s fixed=%zux%zux%zu max=%zux%zux%zu; ", t(c.atype).c_str(), t(c.btype).c_str(), t(c.ctype).c_str(), t(c.dtype).c_str(), c.msize, c.nsize, c.ksize, c.max_msize, c.max_nsize, c.max_ksize);
            }
        } catch (...) { std::printf("query failed"); }
        std::printf("|\n");
    }

    if (d.has(sycl::aspect::ext_intel_pci_address))
        std::printf("| PCI address | %s |\n", d.get_info<sycl::ext::intel::info::device::pci_address>().c_str());
    if (argc == 2 && std::string(argv[1]) == "--capabilities") return 0;
    if (argc != 1) { std::fprintf(stderr, "usage: xe_probe [--capabilities]\n"); return 2; }

    if (!d.has(sycl::aspect::ext_oneapi_graph) || dv.max_wg < 256 ||
        std::find(dv.sub_groups.begin(), dv.sub_groups.end(), 16) == dv.sub_groups.end()) {
        std::fprintf(stderr, "SKIP: microbenchmarks require graph support, WG256 and SG16; use --capabilities\n"); return 77;
    }
    const int maxg = 4096;
    float* buf = sycl::malloc_device<float>((size_t) maxg * 256, q);
    q.memset(buf, 0, (size_t) maxg * 256 * 4).wait_and_throw();
    const int N = 200;

    std::printf("## Kernel boundary (us per kernel or per phase, same work per phase)\n\n");
    std::printf("| groups | work | one kernel | eager chain | graph replay |\n|---|---|---|---|---|\n");
    for (int groups : {1, 32, 128, 512}) {
        for (int work : {0, 64, 512}) {
            spin(q, buf, groups, work * N).wait_and_throw();
            auto f0 = q.ext_oneapi_submit_barrier(); spin(q, buf, groups, work * N); auto f1 = q.ext_oneapi_submit_barrier(); f1.wait_and_throw();
            const double fused = bench::us_between(f0, f1) / N;
            for (int i = 0; i < 10; ++i) spin(q, buf, groups, work);
            auto e0 = q.ext_oneapi_submit_barrier(); for (int i = 0; i < N; ++i) spin(q, buf, groups, work); auto e1 = q.ext_oneapi_submit_barrier(); e1.wait_and_throw();
            const double eager = bench::us_between(e0, e1) / N;
            sx::command_graph gr{q.get_context(), d};
            gr.begin_recording(q); for (int i = 0; i < N; ++i) spin(q, buf, groups, work); gr.end_recording(q);
            auto ge = gr.finalize(); q.ext_oneapi_graph(ge).wait_and_throw();
            auto g0 = q.ext_oneapi_submit_barrier(); q.ext_oneapi_graph(ge); auto g1 = q.ext_oneapi_submit_barrier(); g1.wait_and_throw();
            const double graph = bench::us_between(g0, g1) / N;
            std::printf("| %d | %d | %.3f | %.2f | %.2f |\n", groups, work, fused, eager, graph);
        }
    }

    std::printf("\n## Read-pattern diagnostics (GB/s, rotating buffers; cache misses not proven)\n\n| pattern | rows x bytes | us | GB/s |\n|---|---|---|---|\n");
    const int rows = 16384, row_bytes = 2560;   // a square-ish weight slab; change to your shapes
    const size_t bytes = (size_t) rows * row_bytes;
    const int copies = bench::copies_beyond_cache(q, bytes);
    uint8_t* w = sycl::malloc_device<uint8_t>(bytes * copies, q);
    q.memset(w, 0x5a, bytes * copies).wait_and_throw();
    float* y = sycl::malloc_device<float>(16, q);
    auto report = [&](const char* name, const std::function<void(const uint8_t*)>& fn) {
        const double us = bench::time_us(q, copies, [&](int i) { fn(w + bytes * i); });
        std::printf("| %s | %d x %d | %.1f | %.0f |\n", name, rows, row_bytes, us, bytes / (us * 1e3));
    };
    report("coalesced stream", [&](const uint8_t* b) { stream16(q, b, bytes, y); });
    report("one row per lane", [&](const uint8_t* b) { lane_row(q, b, rows, row_bytes, y); });
#if XE_PROBE_2D
    // Generation alone cannot establish support for this experimental builtin.
#endif
    sycl::free(w, q); sycl::free(y, q); sycl::free(buf, q);
    return 0;
}
