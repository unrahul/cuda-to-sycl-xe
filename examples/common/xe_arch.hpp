// xe_arch.hpp - runtime device properties, not generation-derived capability or tuning defaults.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cstdlib>
#include <string>
#include <vector>

namespace xe {

// Runtime-reported properties. Device identity alone is not a capability check.
struct Device {
    std::string name, driver;
    uint32_t device_id = 0;
    int compute_units = 0;
    size_t max_wg = 0, local_mem = 0, l2 = 0, cache_line = 0;
    std::vector<size_t> sub_groups;
    bool fp16 = false, fp64 = false, atomic64 = false, matrix = false;
};

inline Device query(const sycl::device& d) {
    Device r;
    r.name = d.get_info<sycl::info::device::name>();
    r.driver = d.get_info<sycl::info::device::driver_version>();
    if (d.has(sycl::aspect::ext_intel_device_id)) r.device_id = d.get_info<sycl::ext::intel::info::device::device_id>();
    r.compute_units = (int) d.get_info<sycl::info::device::max_compute_units>();
    r.max_wg = d.get_info<sycl::info::device::max_work_group_size>();
    r.local_mem = d.get_info<sycl::info::device::local_mem_size>();
    r.l2 = d.get_info<sycl::info::device::global_mem_cache_size>();
    r.cache_line = d.get_info<sycl::info::device::global_mem_cache_line_size>();
    r.sub_groups = d.get_info<sycl::info::device::sub_group_sizes>();
    r.fp16 = d.has(sycl::aspect::fp16);
    r.fp64 = d.has(sycl::aspect::fp64);
    r.atomic64 = d.has(sycl::aspect::atomic64);
    r.matrix = d.has(sycl::aspect::ext_intel_matrix);
    return r;
}

}  // namespace xe
