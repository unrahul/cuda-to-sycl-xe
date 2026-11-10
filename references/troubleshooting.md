# Build, link and runtime errors

Fix the build before touching the algorithm, and keep the full compiler output: the first error is the one that matters.

Match on meaning, since message wording changes between compiler releases.

## Compile

| Message | Cause | Fix |
|---|---|---|
| `C++20 or later compatible compiler is required to use PyTorch`, followed by hundreds of errors in `intrusive_ptr.h`, `TensorBase.h` | `-std=c++17` forced in `extra_sycl_cflags` or `extra_cflags` | Rule 2 in [PyTorch integration](pytorch-xpu.md) |
| `unnamed type '(lambda at ...)' is invalid; provide a kernel name, or use '-fsycl-unnamed-lambda'` | PyTorch compiles `.sycl` sources with `-fsycl-host-compiler`, which needs kernel names. Plain `icpx -fsycl` accepts unnamed lambdas, so the same code builds standalone | Rule 1 in [kernel rules](kernel-rules.md) |
| `fatal error: 'sycl/sycl.hpp' file not found` while building a PyTorch extension | The source is named `.cpp`, so PyTorch compiles it with the host C++ compiler, not `icpx -fsycl` | Rule 1 in [PyTorch integration](pytorch-xpu.md); changing the include spelling does not help |
| `non-const lvalue reference to type 'XPUStream' cannot bind to a temporary` | `auto& s = c10::xpu::getCurrentXPUStream(i);`: the stream handle is returned by value | Rule 4 in [PyTorch integration](pytorch-xpu.md) |
| `no member named 'desc' in 'dnnl::convolution_forward'` (or `matmul::desc`, `memory::kind`) | oneDNN 2.x API used against oneDNN 3.x, which removed operation descriptors | Construct `primitive_desc(engine, prop_kind::forward_inference, algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md, strides, padding_l, padding_r)` directly, as `examples/pytorch_xpu/dnnl.sycl` does for matmul. oneDNN dilation is zero-based (0 = none) |
| `kernel name ... must be forward declarable` or a similar kernel-name error | Name class declared locally, or a name repeated for two different kernels | One namespace-scope class per kernel, templated when the kernel is |
| `static assertion failed ... is_deprecated_device_copyable_v<std::vector<...>>` | A kernel lambda captures a host container (`std::vector`, `std::string`, a tensor), which cannot be copied to the device | Unpack into scalars or a `std::array`/`sycl::vec` before the lambda, or pass device memory. This applies to the `std::vector<int64_t>` hyperparameters from rule 6 in [PyTorch integration](pytorch-xpu.md) |
| `SYCL kernel cannot call a recursive function`, `... cannot use exceptions`, `... cannot call a virtual function` | Device-code restrictions | Restructure. Host-side `TORCH_CHECK` stays outside the lambda |
| `Kernel compiled with required subgroup size 8, which is unsupported on this platform` (AOT for `bmg`, `ptl-u` or `cri`) | A kernel with `reqd_sub_group_size(8)`, such as the Xe-HPG XMX variant, is in a translation unit built ahead of time for Xe2, Xe3 or Xe3P | See *Building for several generations* in [architectures](xe-architectures.md) |
| `no member named 'matrix_combinations'`, missing `joint_matrix` | Old compiler, or `<sycl/ext/oneapi/matrix/matrix.hpp>` not included | Include the header; if combinations are still empty, see *Matrix aspect 0* below |
| `architecture::intel_gpu_cri` undeclared | Compiler older than oneAPI 2026.2 | Guard with `#if __INTEL_LLVM_COMPILER >= 20260200`; see [architectures](xe-architectures.md) |
| PyTorch extension takes minutes per build | PyTorch compiles ahead of time for a list of GPUs (`pvc,bmg,arl-h,mtl-h,lnl-m,ptl-h,ptl-u`) plus JIT | `TORCH_XPU_ARCH_LIST` ([PyTorch integration](pytorch-xpu.md), Build) |

## Link

| Message | Cause | Fix |
|---|---|---|
| `cannot find -lonemkl` | The open-source oneMKL interface library name; Intel's oneMKL does not ship it | Link with `-qmkl` through `icpx` (rule 3 in [PyTorch integration](pytorch-xpu.md)) |
| `undefined reference to mkl_*` after `-qmkl=sycl` | `-qmkl=sycl` pulled no threading layer on oneAPI 2025.3.3 | Use `-qmkl` (parallel) |
| `undefined symbol: dnnl_...` at import, or oneDNN headers present but symbols hidden | Using the copy of oneDNN embedded inside PyTorch, whose symbols are private | Rule 3 in [PyTorch integration](pytorch-xpu.md) |
| `error while loading shared libraries: libsycl.so` | Runtime environment not configured | `source /opt/intel/oneapi/setvars.sh` in the shell that runs the binary |

## Environment

| Symptom | Cause | Fix |
|---|---|---|
| Script exits inside `setvars.sh` | Sourced after `set -u` (the script reads unset variables) | Source it first, then enable strict mode |
| `sycl-ls` shows no GPU, or `ONEAPI_DEVICE_SELECTOR` selects nothing | Missing `/dev/dri` access, wrong index, or a container started without the render node | Check `sycl-ls`, the `render` group, and `--device /dev/dri/renderD*` |
| Matrix aspect 0 on an Xe2 or Xe3P card | Stack too old: on Xe2, oneAPI 2025.3 exposes none and 2026.1 does | Use the newer compiler and runtime; do not bypass the capability gate |
| Results differ between machines with the same GPU | Different driver or compiler | Record `icpx --version`, driver version and PCI address with every result |

## Runtime

| Message | Cause | Fix |
|---|---|---|
| Every call fails at run time (a harness may show only `worker_error`), with a `TORCH_CHECK` about autograd or `requires_grad` | The extension rejects `nn.Parameter` inputs, which require grad | Rule 5 in [PyTorch integration](pytorch-xpu.md) |
| `run(): incompatible function arguments`; `TypeError: 'int' object is not subscriptable` | Python arguments don't match the pybind signature; module hyperparameters can be an `int` or a tuple | Rule 6 in [PyTorch integration](pytorch-xpu.md) |
| `No such file or directory: '.../op.sycl'` when the model is constructed | `load(sources=...)` used a path relative to the editing directory; the harness imports the module from another working directory | Rule 1 in [PyTorch integration](pytorch-xpu.md) |
| `could not create a primitive descriptor` from oneDNN | Usually a descriptor mismatch, not a missing implementation. A 3D deconvolution fails when the weight dims are given in PyTorch order `{IC, OC, kD, kH, kW}`; oneDNN's logical weight dims are always `{OC, IC, ...}` | Give logical dims `{OC, IC, ...}` and describe PyTorch's ConvTranspose storage with format tag `iodhw` (`iohw` in 2D). On B70 that selected the JIT implementation (`jit:ir`); `oidhw` fell back to the reference one (`ocl:ref`). Run once with `ONEDNN_VERBOSE=all` to see the reason |
| `UR_RESULT_ERROR_OUT_OF_RESOURCES` (40) at launch | Work-group larger than this kernel can run (registers, spills, large GRF), or local memory beyond `local_mem_size` | Query the kernel's limit (`kernel_max_work_group` in [architectures](xe-architectures.md)); shrink the work-group, private arrays or local-memory tiles |
| `UR_RESULT_ERROR_OUT_OF_RESOURCES` only after running for a while under load | Large per-thread private arrays (a whole row cached in registers) beyond the GRF budget (seen on BMG) | Keep private state to a few KB per work-item; re-read from cache instead |
| `undefined reference to __spirv_Subgroup2DBlockLoadINTEL` at link | The host pass references the device-only intrinsic (oneAPI 2025.3) | Use the declaration with its host stub from [memory](memory.md) (2D block loads) |
| NaN from ESIMD `xmx::dpas` with repeat count 4 and fp16 on BMG | Short repeat counts misbehave for that type | Use repeat count 8 and zero-pad the extra rows |
| `UR_RESULT_ERROR_DEVICE_LOST`, `ZE_RESULT_ERROR_DEVICE_LOST` | Out-of-bounds access, a hang (spinning on a flag, divergent barrier), or a driver/hardware fault | Stop. Save command, seed and logs; check `scripts/fault_count.sh`. Re-run the smallest failing shape under device ASan (`-Xarch_device -fsanitize=address`) |
| Wrong results only for some sizes | Tail not guarded, global size not padded, or index overflow | Test sizes around every tile boundary (minus one, equal, plus one) and primes |
| Correct alone, wrong in a sequence | Missing dependency between queues, or output read before completion | In-order queue or `depends_on`; `wait_and_throw()` before host access |
| Error appears at a later, unrelated call | Asynchronous error surfaced late | Call `q.wait_and_throw()` after the suspect kernel while debugging |
| First call slow, later calls fast | JIT compilation | Warm up before timing, or compile AOT |
| `-nan`/`inf` on one path only | Saturating half conversions or different FMA contraction | See rules 6 and 7 in [kernel rules](kernel-rules.md) |
