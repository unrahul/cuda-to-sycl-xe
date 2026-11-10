# PyTorch XPU integration

Start with the complete [small examples](../examples/pytorch_xpu/README.md):
a named SYCL affine kernel, oneMKL GEMM and a oneDNN matmul primitive,
with a shared loader and independent host-reference check.

## Standalone development

Configure the framework's supported compiler and runtime before building.
Build the affine example first. The public `torch.utils.cpp_extension.load`
signature must expose `with_sycl` and `extra_sycl_cflags`; let that loader
choose the C++ standard and ABI flags. Do not force standalone C++17 flags
or depend on access to the loader's Python source. See the
[public loader API](https://docs.pytorch.org/docs/stable/cpp_extension.html).

For libraries, supply include paths and exact link arguments from the
installed SDK's documented configuration. Qualify the header, exported
symbols, backend dependencies and actual execution together. Do not guess
`-lonemkl` or use framework-private oneDNN ABI. Match the installed API and
runtime versions to each example's required overloads and SYCL interop
functions. Preserve standard oneMKL compute mode and strict oneDNN fpmath
for a strict FP32 contract.

## Managed assessment

Use the runner's supplied compiler, public SDK configuration and existing
qualification evidence. Read its exact candidate/import policy before
copying an example. The standalone loader and reference test may belong
outside the candidate. Use the permitted diagnostic channel. Record any
unavailable direct build, stream test or profiling as unavailable.

## Queue ownership and triage

Guard the input device. Reject mixed-device, unsupported dtype/layout and
autograd inputs. Submit on `c10::xpu::getCurrentXPUStream(input.get_device())`
and validate producers and consumers on a non-default stream. These
examples require an in-order queue and wait before returning to preserve
tensor, primitive and scratch lifetimes. A separate queue needs explicit
dependencies and completion. Async returns also need correct storage
lifetime/allocator integration, even when both queues share a device.

Preserve full verbose stdout/stderr on failure. Resolve missing loader
parameters, headers, C++ standard or link symbols before changing the
algorithm. For runtime errors, record selected device, queue ordering,
SDK/runtime versions and the failing API. Stop on device loss. A successful
build or import does not establish numerical support or performance.
