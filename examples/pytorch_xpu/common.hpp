#pragma once
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

inline void check_tensor(const at::Tensor& x) {
  TORCH_CHECK(x.is_xpu() && x.scalar_type() == at::kFloat && x.is_contiguous(),
              "expected a contiguous FP32 XPU tensor");
  // Parameters usually have requires_grad=True and callers may run with grad enabled:
  // accept them. These ops are forward-only, so their outputs carry no autograd history.
}

inline sycl::queue& current_queue(const at::Tensor& x) {
  auto stream = c10::xpu::getCurrentXPUStream(x.get_device());  // a handle, returned by value
  sycl::queue& q = stream.queue();  // the queue is owned by PyTorch's stream pool
  TORCH_CHECK(q.is_in_order(), "example requires an in-order framework queue");
  TORCH_CHECK(q.get_device().has(sycl::aspect::usm_device_allocations),
              "device USM is required");
  return q;
}

inline void check_matmul(const at::Tensor& a, const at::Tensor& b) {
  check_tensor(a);
  check_tensor(b);
  TORCH_CHECK(a.device() == b.device(), "inputs must share a device");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected two matrices");
  TORCH_CHECK(a.size(1) == b.size(0), "incompatible matrix dimensions");
  TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(1) > 0,
              "library examples require nonempty matrices");
}
