# PyTorch XPU extensions

Start from `examples/pytorch_xpu/`, and copy the closest example; don't write the binding from scratch. It contains:

- a named SYCL kernel (`affine.sycl`);
- a oneMKL GEMM (`mkl.sycl`);
- a oneDNN matmul (`dnnl.sycl`);
- a shared loader and a checker against an FP64 host reference.

## Rules: check every one before the first build

Each rule prevents a failure that costs a build or run attempt. Rules 5–7 fail only at run time, where an evaluation harness may report nothing but `worker_error`.

1. **Sources.** Name SYCL sources `*.sycl`: a `.cpp` file is compiled by the host C++ compiler and fails with `sycl/sycl.hpp` not found. Locate them with `Path(__file__).with_name(...)`. A path relative to your editing directory, such as `submission/op.sycl`, breaks when the module is imported from another directory.
2. **Compiler flags.** Set `CXX=icpx` and add no `-std=` flag; the loader compiles with `-std=c++20`, which PyTorch requires. The loader also passes `-fsycl-host-compiler`, so every kernel must be named ([kernel rules](kernel-rules.md), rule 1).
3. **Libraries.**
   - oneMKL: `extra_include_paths=[f"{MKLROOT}/include"]` and `extra_ldflags=["-qmkl"]`.
   - oneDNN: a standalone SDK built with `DNNL_GPU_RUNTIME=SYCL`, linked with `["-L<root>/lib", "-ldnnl", "-Wl,-rpath,<root>/lib"]`, using the 3.x API, where you construct `primitive_desc(engine, ...)` directly.
4. **Queue.** `auto stream = c10::xpu::getCurrentXPUStream(x.get_device()); sycl::queue& q = stream.queue();`. The stream handle is returned by value, so `auto& stream` does not compile. The queue is in order.
5. **Autograd.** Accept tensors with `requires_grad`, or pass `.detach()` from Python. Parameters require grad and callers run `forward` with grad enabled, so a `TORCH_CHECK(!w.requires_grad())` guard fails every call. Wrap the op in `torch.autograd.Function` if gradients are needed.
6. **Argument types.** Hyperparameters such as `kernel_size`, `stride`, `padding` and `dilation` may be an `int` or a tuple, so normalize them with `torch.nn.modules.utils._single`, `_pair` or `_triple`. Bind sequences as `std::vector<int64_t>` (or `c10::IntArrayRef`) and scalars as `double` or `int64_t`. A mismatch raises `incompatible function arguments`, or `'int' object is not subscriptable`, inside `forward`.
7. **Module contract, state and layout.** A replacement module subclasses `torch.nn.Module` and keeps the original constructor and `forward` signatures; a plain class is rejected at load time. Keep the original module's parameter and buffer names, for `state_dict` compatibility. Call `.contiguous()` before `data_ptr`. `TORCH_CHECK` the device, dtype and layout, and that devices match.
8. **Lifetimes.**
   - Place `c10::DeviceGuard guard(x.device());` before allocating outputs, and allocate them with `at::empty` / `empty_like`.
   - Raw `data_ptr` pointers must outlive the kernel. The examples call `q.wait_and_throw()` before returning.
   - To return without waiting, keep every tensor alive through PyTorch's caching allocator on the same stream, and make no host access before completion.
9. **Numerics.** Write every output element, including tails and borders. Under tight tolerances, accumulate long reductions in FP64 ([testing](testing.md)).

## Build

```python
from pathlib import Path
from torch.utils.cpp_extension import load

ext = load(
    name="my_op",
    sources=[str(Path(__file__).with_name("my_op.sycl"))],
    with_sycl=True,
    extra_sycl_cflags=["-O2"],
    extra_ldflags=[],              # ["-qmkl"] for oneMKL (rule 3)
    verbose=True,                  # keep the full compiler output
)
```

- **Faster rebuilds.** While iterating, set `TORCH_XPU_ARCH_LIST` to the target, for example `bmg` for Arc B-series and B70. The default compiles ahead of time for seven GPU families and takes minutes; an empty value builds JIT only.
- **Stale builds.** A changed source rebuilds under the same `name`. If a stale build lingers, change `name` or clear `TORCH_EXTENSIONS_DIR`.

## Check

```sh
python examples/pytorch_xpu/check.py affine --device xpu:0
```

The checker passes real `nn.Parameter` inputs on a non-default stream and compares against FP64 host arithmetic. It also repeats calls and verifies that inputs are unchanged and earlier outputs survive later calls. Write the same checks for a new operator before timing it. A successful build or import proves nothing about numerics or speed. For errors, see [troubleshooting](troubleshooting.md).
