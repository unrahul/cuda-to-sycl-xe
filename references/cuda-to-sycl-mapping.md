# CUDA and HIP to SYCL

Translate what the code means before how it is spelled, working from the CUDA source. Treat SYCLomatic output as an unverified draft ([modern CUDA idioms](cuda-modern-idioms.md)). Keep the original source runnable when you can, and say whether a comparison ran on the source GPU or only against a CPU reference.

Every code block here is compiled and executed by `tests/snippets_test.cpp` from `examples/snippets/`.

## Launch and indexing

```cpp
// CUDA: saxpy<<<(n + 255) / 256, 256, 0, stream>>>(n, a, x, y);
inline sycl::event saxpy(sycl::queue& q, std::size_t n, float a, const float* x, float* y) {
  constexpr std::size_t wg = 256;
  const std::size_t global = (n + wg - 1) / wg * wg;  // whole work-groups only
  return q.parallel_for<Saxpy>(sycl::nd_range<1>{global, wg}, [=](sycl::nd_item<1> it) {
    const std::size_t i = it.get_global_id(0);  // blockIdx.x * blockDim.x + threadIdx.x
    if (i < n) y[i] = a * x[i] + y[i];          // guard the padded tail
  });
}
```

| CUDA / HIP | SYCL | Watch for |
|---|---|---|
| `<<<grid, block>>>` | `parallel_for(nd_range{grid * block, block})` | SYCL takes the *global* size, which must be a multiple of the local size |
| `blockIdx`, `threadIdx`, `blockDim` | `it.get_group(d)`, `it.get_local_id(d)`, `it.get_local_range(d)` | In a 3D range, CUDA `x` usually maps to SYCL dimension 2 (the fastest-varying) |
| grid-stride loop | step by `it.get_global_range(0)` | index overflow; use `size_t` for products of dimensions |
| `__launch_bounds__` | `[[sycl::reqd_work_group_size(...)]]` or query the kernel limit | CUDA's maximum is a hint; SYCL's attribute is a requirement |
| kernel template | kernel name class per instantiation, e.g. `parallel_for<K<T>>` | PyTorch builds require names; see [troubleshooting](troubleshooting.md) |

## Shared memory and barriers

```cpp
// CUDA: __shared__ float s[256]; ... __syncthreads(); tree reduction in shared memory.
inline sycl::event row_sum(sycl::queue& q, const float* x, float* out, int rows, int cols) {
  constexpr int wg = 256;
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> s{sycl::range<1>{wg}, h};  // __shared__
    h.parallel_for<RowSum>(sycl::nd_range<1>{std::size_t(rows) * wg, wg}, [=](sycl::nd_item<1> it) {
      const int row = it.get_group(0), lid = it.get_local_id(0);
      float acc = 0.0f;
      for (int c = lid; c < cols; c += wg) acc += x[std::size_t(row) * cols + c];
      s[lid] = acc;
      sycl::group_barrier(it.get_group());  // __syncthreads(): every work-item reaches it
      for (int stride = wg / 2; stride > 0; stride /= 2) {
        if (lid < stride) s[lid] += s[lid + stride];  // only the destination half writes
        sycl::group_barrier(it.get_group());
      }
      if (lid == 0) out[row] = s[0];
    });
  });
}
```

- `extern __shared__` (dynamic size) becomes a `local_accessor` sized at submit time.
- `__syncthreads()` inside a branch that not all threads take is undefined in both models. In SYCL, hoist the barrier out of the branch and predicate the work instead.
- A one-call alternative for the reduction is `sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>())`. Its summation order differs from the CUDA tree, so results are not bitwise identical.
- When local memory replaces register shuffles, each phase must read and write disjoint locations, as above. See [kernel rules](kernel-rules.md).

## Warps and sub-groups

```cpp
// CUDA: for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffff, v, o);
// One sub-group per row. The width is a compile-time contract; check sub_group_sizes first.
inline sycl::event row_sum_sg(sycl::queue& q, const float* x, float* out, int rows, int cols) {
  constexpr int sg = 16;  // native on Xe2 and Xe3P; Xe-HPG also compiles 8 and 32
  return q.parallel_for<RowSumSG>(
      sycl::nd_range<1>{std::size_t(rows) * sg, sg},
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg)]] {
        const auto sub = it.get_sub_group();
        const int row = it.get_group(0), lane = sub.get_local_linear_id();
        float v = 0.0f;
        for (int c = lane; c < cols; c += sg) v += x[std::size_t(row) * cols + c];
        for (int o = sg / 2; o > 0; o /= 2) v += sycl::permute_group_by_xor(sub, v, o);
        // Same result, less code: v = sycl::reduce_over_group(sub, v, sycl::plus<float>());
        if (lane == 0) out[row] = v;
      });
}
```

| CUDA / HIP | SYCL | Semantic difference |
|---|---|---|
| `threadIdx.x % 32`, `__lane_id()` | `sub.get_local_linear_id()` | sub-group width is 8, 16 or 32 and fixed per kernel |
| `__shfl_sync(m, v, src)` | `sycl::select_from_group(sub, v, src)` | no mask: all lanes must participate |
| `__shfl_xor_sync` | `sycl::permute_group_by_xor(sub, v, mask)` | as above |
| `__shfl_down_sync` / `__shfl_up_sync` | `sycl::shift_group_left` / `shift_group_right` | shifted-in lanes hold unspecified values |
| `__ballot_sync`, `__any_sync`, `__all_sync` | `sycl::ext::oneapi::group_ballot`, `any_of_group`, `all_of_group` | ballot returns a mask object, not an integer |
| warp reduction | `sycl::reduce_over_group(sub, v, op)` | order differs from the source tree |
| `__syncwarp()` | `sycl::group_barrier(sub)` | sub-group scope only |

- **Shuffle cost depends on the pattern.** A shuffle whose source lane is not known at compile time (`select_from_group(sub, v, f(lane))` with data-dependent `f`) compiles to indirect register access, about one cycle per element. Butterflies and shifts with constant offsets, and broadcasts, are cheap register moves. Replacing generic shuffles with broadcasts, constant butterflies or an SLM exchange can gain 2–5x per kernel.
- CUDA masked operations have no direct SYCL equivalent. Work out which lanes are active and restructure so the whole sub-group calls the collective, with inactive lanes contributing an identity value.
- HIP code may assume wave64 (64-bit ballots, 64-lane layouts). Replacing a literal 32 or 64 with 16 changes the algorithm. Re-derive the per-lane work.
- A CUDA kernel that maps one row to a 32-lane warp ports first to sub-group 32 (supported on all three generations) for parity. A 16-lane version is a redesign. On Xe2 and Xe3P the 16-wide design is usually the DPAS design. See [kernel rules](kernel-rules.md).

## Atomics

```cpp
// CUDA: atomicAdd(&hist[bin], 1u);
inline void hist_add(unsigned* hist, int bin) {
  sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref{hist[bin]};
  ref.fetch_add(1u);
}
```

- State the scope and order explicitly. CUDA `atomicAdd` is relaxed at device scope.
- 64-bit atomics need `aspect::atomic64`. Floating-point atomic add exists in SYCL, but its accumulation order makes results nondeterministic.
- `__threadfence()` alone is not a handshake. Use an `atomic_ref` with acquire/release order on the flag.
- Host-and-device polling of shared memory needs the USM atomic aspects (`usm_atomic_shared_allocations`). Never spin on a global flag as a grid barrier ([kernel rules](kernel-rules.md), rule 4).

## Streams, events and memory

```cpp
// CUDA: cudaMemcpyAsync(d, h, bytes, H2D, s); twice<<<g, b, 0, s>>>(d); ...; cudaStreamSynchronize(s);
inline void twice_on_device(sycl::queue& q, const float* in, float* out, std::size_t n) {
  // q must be in order, like a CUDA stream: sycl::queue q{dev, sycl::property::queue::in_order{}};
  float* d = sycl::malloc_device<float>(n, q);
  if (!d) throw std::bad_alloc{};
  q.memcpy(d, in, n * sizeof(float));
  q.parallel_for<Twice>(sycl::range<1>{n}, [=](sycl::id<1> i) { d[i] *= 2.0f; });
  q.memcpy(out, d, n * sizeof(float));
  q.wait_and_throw();  // cudaStreamSynchronize, and rethrows asynchronous errors
  sycl::free(d, q);
}
```

```cpp
// CUDA: cudaEventRecord(done, s1); cudaStreamWaitEvent(s2, done);
inline sycl::event after(sycl::queue& q2, sycl::event done, float* y, std::size_t n) {
  return q2.submit([&](sycl::handler& h) {
    h.depends_on(done);  // cross-queue ordering is explicit
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) { y[i] += 1.0f; });
  });
}
```

| CUDA / HIP | SYCL |
|---|---|
| `cudaMalloc` / `cudaMallocManaged` / `cudaMallocHost` | `sycl::malloc_device` / `malloc_shared` / `malloc_host` (returns `nullptr` on failure) |
| default stream | a default `sycl::queue` is out of order; pass `property::queue::in_order{}` for stream semantics |
| `cudaDeviceSynchronize` | `q.wait_and_throw()` on every queue in use; it covers only that queue |
| `cudaEventElapsedTime` | queue with `property::queue::enable_profiling{}`, then `event.get_profiling_info<...>()` |
| `cudaGetLastError` | asynchronous errors surface at `wait_and_throw()` or through an `async_handler` |
| CUDA graphs | `sycl::ext::oneapi::experimental::command_graph` (check `aspect::ext_oneapi_graph`) |

## Arithmetic and special hardware

- `__dp4a`, WMMA/MMA/MFMA, `cp.async`, TMA, `ldmatrix` and other tensor-core or async-copy idioms: see [modern CUDA idioms](cuda-modern-idioms.md).
- FMA contraction: `icpx` fuses `a*b + c` into one FMA in device code by default, and still does with `-fp-model=precise`. To round the product separately, as a CUDA reference compiled with `-fmad=false` does, use `-ffp-contract=off`, `-fp-model=strict`, or `#pragma clang fp contract(off)` in the block. Explicit `__fmaf_rn` maps to `sycl::fma`.
- `__expf`, `-use_fast_math`: `-ffast-math` changes results beyond contraction. Preserve the denormal, rounding and saturation behavior that the contract requires.
- `half` / `__nv_bfloat16`: use `sycl::half` and `sycl::ext::oneapi::bfloat16`, and match the source's conversion rounding.
- cuBLAS, hipBLAS, cuDNN: replace them with oneMKL or oneDNN, and check row/column major, transposes, leading dimensions and batch strides. See [backend selection](backend-selection.md).

## Toolchain

- Default to JIT (`icpx -fsycl`). AOT flags and device names are in [architectures](xe-architectures.md).
- Select one GPU with `ONEAPI_DEVICE_SELECTOR=level_zero:N` after reading `sycl-ls`. Indices are not stable PCI identities; `xe_probe --capabilities` prints the PCI address.
- Memory checking: device AddressSanitizer is `-Xarch_device -fsanitize=address`. Combine it with sentinel-guarded output buffers in tests.
