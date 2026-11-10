# PyTorch XPU integration examples

`affine.sycl` computes `2*x+1` with a named SYCL kernel. `mkl.sycl` and
`dnnl.sycl` compute contiguous row-major FP32 `A[M,K] @ B[K,N]`, with
standard oneMKL compute mode and strict oneDNN math mode. Matrix dimensions
must be positive. Affine accepts empty tensors. The ops are forward-only:
they accept parameters that require grad, and their outputs carry no autograd
history. Wrap the call in `torch.autograd.Function` if gradients are needed.

Configure the compiler and runtime in the calling environment, then run:

```sh
python examples/pytorch_xpu/check.py affine --device xpu:0
```

The public PyTorch `load` API must expose `with_sycl` and
`extra_sycl_cflags`. The loader owns the C++ standard and ABI flags. No
source access to its Python implementation is required. The compiler must
provide SYCL headers and the framework must provide `c10/xpu/XPUStream.h`.

Set `TORCH_XPU_ARCH_LIST` to the target device (for example `bmg`) to build
one ahead-of-time target instead of PyTorch's default list. An empty value
builds JIT only.

For Intel oneMKL, link with `-qmkl` through the `icpx` driver. `-qmkl=sycl`
failed to link on oneAPI 2025.3.3; `-qmkl` linked and passed on both 2025.3.3
and 2026.1. Source the installed SDK environment to supply `MKLROOT` and
runtime dependencies:

```sh
export CXX=icpx
export XPU_MKL_INCLUDE="$MKLROOT/include"
export XPU_MKL_LDFLAGS='["-qmkl"]'
python examples/pytorch_xpu/check.py mkl --device xpu:0
```

For a public standalone oneDNN SDK, set `DNNL_ROOT` to its installation
prefix and configure its dependent runtime before running:

```sh
export XPU_DNNL_INCLUDE="$DNNL_ROOT/include"
export XPU_DNNL_LDFLAGS="[\"-L$DNNL_ROOT/lib\",\"-ldnnl\",\"-Wl,-rpath,$DNNL_ROOT/lib\"]"
python examples/pytorch_xpu/check.py dnnl --device xpu:0
```

Use that SDK's actual library directory if it differs from `lib`. The flags
are a JSON array of individual arguments. The shared loader accepts the
same configuration as explicit `build(backend, include=..., ldflags=...)`
arguments; an operator module should pass them directly rather than read
the environment. Do not infer library names from package names or use
framework-private oneDNN symbols.
oneDNN must be built with `DNNL_GPU_RUNTIME=SYCL` and expose
`sycl_interop::make_engine`, `make_stream`, `make_memory` and `execute`.
oneMKL must expose the FP32 USM GEMM overload with `compute_mode::standard`.

All examples guard the input device, use its current framework queue and
require that queue to be in order. They wait before returning so tensors,
USM wrappers and primitives remain alive through completion. This is a
simple synchronous baseline with synchronization and setup overhead. An
asynchronous adaptation must retain every resource and integrate allocator
stream recording plus dependency/completion events correctly. Inputs
produced on another stream require a caller-established stream dependency.

Build the tiny example first and preserve verbose compiler/linker output.
Check the installed framework, compiler, SDK/runtime versions and operation
support. Header presence alone does not qualify a backend. Execute the
small reference check on the selected device before benchmarking. The
check uses a non-default stream and independent CPU FP64 accumulation.
It is a smoke check, not a full numerical or performance qualification.
