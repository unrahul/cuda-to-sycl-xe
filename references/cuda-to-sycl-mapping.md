# CUDA and ROCm/HIP conversion

Translate semantics before tuning. CUDA/HIP syntax migration tools can provide a starting point, but generated code still needs numerical, memory and dependency validation. Preserve the original runnable implementation when available and distinguish an executed source-device comparison from CPU reference testing.

## Launch and memory

| Source construct | SYCL starting point | Check |
|---|---|---|
| CUDA triple-chevron or HIP kernel launch | `parallel_for(nd_range)` | global range is group count times local range; pad tails safely |
| Block and thread indices | `nd_item` group/local IDs | choose a consistent axis mapping; CUDA x commonly maps to SYCL dimension 2 in a 3D range |
| Grid-stride loop | increment by global range | index/size overflow, empty inputs |
| Shared memory / LDS | `local_accessor` | per-kernel and device limits, alignment, barriers |
| Block barrier | `group_barrier(item.get_group())` | every participating work-item reaches it convergently |
| Device/shared/host allocation | `malloc_device`, `malloc_shared`, `malloc_host` | supported USM aspects, ownership, lifetime, explicit alignment where required |
| Async copy in a stream | `queue.memcpy` plus ordering/dependencies | host buffer lifetime and cross-queue events |
| Vectorized loads | supported vector operations or safely aligned storage | allocation bounds, aliasing, object representation and scalar tails |
| Launch bounds | select and query launch resources | CUDA maximum/resource hints are not equivalent to a required work-group size |

A device-wide synchronization may involve several queues. `q.wait_and_throw()` waits on that queue and surfaces asynchronous errors; it does not wait on all unrelated queues. In-order queues simplify stream translation. Out-of-order queues require explicit dependencies for USM accesses.

## Warps, waves and collectives

| Source operation | Candidate SYCL operation | Semantic constraint |
|---|---|---|
| Lane index | `sg.get_local_linear_id()` | query/require a supported width per compiled kernel |
| Shuffle from lane | `select_from_group` | source lane valid and participating |
| XOR shuffle | `permute_group_by_xor` | converged subgroup participation, valid mask |
| Down/up shuffle | `shift_group_left` / `shift_group_right` | shifted-out lanes do not supply useful values |
| Ballot / any / all | supported ballot extension / `any_of_group` / `all_of_group` | mask representation and participation differ |
| Reduction | `reduce_over_group` | arithmetic order can differ from the original tree |
| Warp barrier | `group_barrier(sg)` | only subgroup scope, not a work-group or grid barrier |

CUDA masked warp intrinsics do not map directly to unmasked SYCL collectives. Reconstruct partial-warp behavior and convergence. HIP kernels may assume wave32 or wave64, including 64-bit ballots and different lane layout. Replacing a literal 32 or 64 with 16 does not establish correctness. Matching subgroup width alone does not ensure floating-point bitwise parity. When using local memory to emulate register shuffles, apply the snapshot/phase-ordering rule in [kernel design rules](kernel-rules.md).

For grid-wide synchronization, use separate kernels and event dependencies in the portable baseline. Ordinary ND-range kernels must not spin on an atomic counter as a global barrier. Experimental root synchronization needs its extension's launch/resource contract and dedicated validation.

## Arithmetic and accelerator operations

- Preserve signed/unsigned variants of packed integer dot products, accumulator range and scaling order. Scalar widening arithmetic gives a simple baseline. Signed C++ overflow is undefined.
- Preserve FP contraction, denormal, rounding and saturation requirements. Half and bfloat16 conversion choices must match the operation's contract.
- CUDA WMMA/MMA and AMD MFMA/WMMA need a new tile/fragment mapping. Intel `joint_matrix` is an extension with specific types, layouts, subgroup sizes and supported combinations. Query capabilities, verify compiler support, and validate an asymmetric known-pattern tile before integrating it.
- CUDA `cp.async`, TMA and `ldmatrix` include layout and synchronization semantics. A 2D block load alone does not reproduce them. Start with plain loads and local staging, then investigate supported block-memory extensions using their exact alignment, pitch and surface constraints.
- Use `atomic_ref` with an explicit supported type, memory scope and ordering. `atomic64` does not establish every floating atomic operation. A fence alone does not provide a synchronization handshake. Concurrent host/device polling requires supported USM atomic capabilities and a valid memory model.
- Prefer supported library APIs for BLAS operations where suitable. Validate row/column layout, transposition, leading dimensions, batched strides, accumulation and stream dependencies when replacing cuBLAS, hipBLAS or rocBLAS.

## Toolchain and performance

Start with the installed compiler's JIT path, commonly `icpx -fsycl`. Verify AOT device names with the installed backend before selecting them. Record compiler flags and runtime versions. Extension APIs change, so inspect the matching headers/specification instead of assuming a copied snippet compiles.

Select one target with an explicit selector such as `ONEAPI_DEVICE_SELECTOR=level_zero:0`, after inspecting enumeration. A device index is not a stable PCI identity. Use profiling-enabled queue events for device timing and completion-bounded wall timing for application latency. Graph replay and fusion require measured end-to-end evaluation including setup and workspace.

Use supported compiler sanitizers where available plus explicit bounds/sentinel tests. Driver debug logging and allocation leak logging do not replace a memory checker. Treat profiler output as diagnostic evidence and check perturbation against unprofiled runs.

Primary specifications and upstream implementations consulted for this package are indexed in [public sources](sources.md). Read [testing](testing.md) before claiming a port is validated.
