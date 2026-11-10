// dispatch.hpp - generation identification, capability gates and kernel properties,
// quoted in references/xe-architectures.md and references/kernel-rules.md.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace snippets {

namespace syclex = sycl::ext::oneapi::experimental;

// snippet: generation
enum class XeGen { Unknown, XeHpg, Xe2, Xe3, Xe3p };

// Identification only: it selects tuning tables, never permission to use a feature.
inline XeGen xe_generation(const sycl::device& d) {
  using A = syclex::architecture;
  const A a = d.get_info<syclex::info::device::architecture>();
  if (a == A::intel_gpu_acm_g10 || a == A::intel_gpu_acm_g11 || a == A::intel_gpu_acm_g12)
    return XeGen::XeHpg;  // Arc A-series (DG2/Alchemist)
  if (a == A::intel_gpu_bmg_g21 || a == A::intel_gpu_bmg_g31 || a == A::intel_gpu_lnl_m)
    return XeGen::Xe2;  // Arc B-series incl. Pro B50/B60/B70, Lunar Lake
  if (a == A::intel_gpu_ptl_h || a == A::intel_gpu_ptl_u || a == A::intel_gpu_wcl)
    return XeGen::Xe3;  // Panther Lake / Wildcat Lake (not Xe3P)
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20260200
  if (a == A::intel_gpu_cri) return XeGen::Xe3p;  // D270 / Crescent Island; enum needs 2026.2+
#endif
  // Older compilers cannot name CRI: fall back to its PCI device ids.
  if (d.has(sycl::aspect::ext_intel_device_id)) {
    const auto id = d.get_info<sycl::ext::intel::info::device::device_id>();
    if (id >= 0x674C && id <= 0x6751) return XeGen::Xe3p;
  }
  return XeGen::Unknown;  // portable path
}
// end snippet

// snippet: gates
inline bool supports_sub_group(const sycl::device& d, std::size_t width) {
  const auto sizes = d.get_info<sycl::info::device::sub_group_sizes>();
  return std::find(sizes.begin(), sizes.end(), width) != sizes.end();
}

using mtype = syclex::matrix::matrix_type;

// True only if the runtime lists this A x B -> C combination with these N and K and an M
// range covering m. A matrix aspect alone does not establish any combination.
// int8 on Xe2/Xe3P: supports_matrix_tile(d, mtype::sint8, mtype::sint8, mtype::sint32, 8, 16, 32)
inline bool supports_matrix_tile(const sycl::device& d, mtype a, mtype b, mtype c, std::size_t m, std::size_t n,
                                 std::size_t k) {
  if (!d.has(sycl::aspect::ext_intel_matrix)) return false;
  for (const auto& x : d.get_info<syclex::info::device::matrix_combinations>()) {
    const bool m_ok = x.msize ? x.msize == m : m <= x.max_msize;  // msize 0 means "up to max_msize"
    if (x.atype == a && x.btype == b && x.ctype == c && m_ok && x.nsize == n && x.ksize == k) return true;
  }
  return false;
}
// end snippet

// snippet: kernel-limit
// The device maximum is not a per-kernel limit: registers and local memory lower it.
template <class KernelName>
std::size_t kernel_max_work_group(const sycl::queue& q) {
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {q.get_device()},
                                                                        {sycl::get_kernel_id<KernelName>()});
  return bundle.get_kernel(sycl::get_kernel_id<KernelName>())
      .template get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
}
// end snippet

// snippet: grf-mode
// Large GRF: twice the registers per hardware thread, half the resident threads.
// A sweep dimension for register-heavy kernels; measure both modes.
struct ScaleLargeGrf {
  float* y;
  float a;
  void operator()(sycl::id<1> i) const { y[i] *= a; }
  auto get(syclex::properties_tag) const {
    return syclex::properties{sycl::ext::intel::experimental::grf_size<256>};
  }
};
// end snippet

}  // namespace snippets
