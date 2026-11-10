# Registers, spills and occupancy

The general register file (GRF) is the fastest storage in the GPU and the most common hidden limit in ported kernels. Every hardware thread owns a fixed slice of it. A sub-group of 16 or 32 work-items runs as one hardware thread, so each work-item's private variables are spread across that thread's GRF.

## How much each work-item gets

| GRF mode → threads per vector engine | Xe-HPG | Xe2 | Xe3 (Panther Lake) | Xe3P (CRI) |
|---|---|---|---|---|
| Register size | 32 B | 64 B | 64 B | 64 B |
| Default | 128 → 8 | 128 → 8 | 128 → 8 | 128 → 8 |
| Large | 256 → 4 | 256 → 4 | 160 → 6, 192 → 5, 256 → 4 | 256 → 8; 512 (newer compiler) |
| Small (more threads) | | | 32–96 → 10 | |

Xe3's variable registers per thread make occupancy a gradual function of register use: staying at 96 GRFs or fewer gives 10 threads. Xe3P's register file per Xe-core doubles to 1 MB, so 256 registers per thread still keep 8 threads: request large GRF freely there.

Bytes of GRF per work-item = registers × register size / sub-group size. Three ways to get more per work-item:

- **Large GRF** doubles registers per thread and halves resident threads.
- **Sub-group 16 instead of 32** doubles registers per work-item at the same thread count.
- **Fewer simultaneously live values:** smaller tiles, fewer rows per sub-group, recomputation.

Large GRF combined with sub-group 16 gives 4x the registers per work-item. The best combination differed per kernel, so sweep both.

## Detecting spills

When live values exceed the GRF, the compiler spills them to per-thread scratch memory, which is slow, cached global memory. Four ways to detect it:

1. **AOT build warning (most reliable at build time).** `icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg" ...` prints `warning: in kernel '...': compiled SIMD16 allocated 128 regs and spilled around 276`. It may first print `[RetryManager] Start recompilation of the kernel`, which means the compiler retried and still spilled. Use `-device acm-g10` for Xe-HPG and `-device cri` for Xe3P to check those targets from any machine.
2. **JIT runs.** The warning can appear on stderr the first time a kernel is compiled. Cached binaries do not repeat it, so don't rely on it.
3. **unitrace (most reliable at run time).** `unitrace -d ./app` ends with a *Kernel Properties* table showing SIMD width, SLM, private memory, **spill memory per thread** and **register file size** for every kernel:
   ```
   Kernel, Compiled, SIMD, ..., SLM Per Work Group, Private Memory Per Thread, Spill Memory Per Thread, Register File Size Per Thread
   "Wide",      JIT,   16, ...,                  0,                         0,                   17664,                           128
   ```
4. **IGC dumps.** `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>` (set for the build, or for the run when JIT) writes per-kernel files. The `.zeinfo` file has `grf_count`, `spill_size` and a `per_thread_memory_buffers` scratch entry. The `.asm` file has `//.thread_config numGRF=...` and `//.spill size N`.

The compiler may also pick SIMD32 or SIMD16 on its own (unitrace showed SIMD32 for one kernel). That changes registers per work-item, so set `reqd_sub_group_size` when the kernel depends on it.

## Fixing spills, in order

1. **Remove private memory the compiler cannot keep in registers.**
   - A run-time index into a private array (`a[idx]`), or into an array of pointers in a kernel argument, forces it into memory. Use compile-time indices (`#pragma unroll` over the array), selects, or local memory.
   - `sycl::select_from_group` or shuffle with a lane pattern unknown at compile time compiles to indirect register access. Broadcasts and butterflies with compile-time masks are cheap. That change alone can gain 2–5x per kernel.
2. **Large GRF.** Ways to request it:
   - per kernel: the `grf_size<256>` property ([kernel rules](kernel-rules.md) has the snippet). The extension spec lists PVC and DG2 (128/256) and CRI (128/256/512), but oneAPI 2026.1 accepts only 128 and 256. It is also accepted for BMG;
   - automatic per kernel: `grf_size_automatic`, also accepted for BMG;
   - whole build: `-Xsycl-target-backend=spir64_gen "-device bmg -options -ze-opt-large-register-file"`;
   - compiler-chosen per kernel for the whole build: `-options -ze-intel-enable-auto-large-GRF-mode`.

   `-ftarget-register-alloc-mode=pvc:large` applies only to PVC.

   Each roughly halves the spill of a register-heavy kernel; if it still spills at 256 registers, apply step 3. Large GRF alone has given up to 14x on register-heavy kernels.
3. **Smaller per-work-item state.** Shrink the register tile, process fewer rows per sub-group, or split the kernel. A tile's operand footprint `(BM·BK + BK·BN)·bytes` should fit in about half the GRF budget; roughly 32 KB in normal mode and 64 KB in large mode on Xe2. Shrink `BK` first.
4. **Sub-group 16.** If the kernel was written for 32 lanes, 16 doubles registers per work-item. On Xe2 and Xe3P it is also the native width. In one register-heavy kernel, SIMD16 with large GRF ran without spills at 27.7 ms, against 36.1 ms for SIMD32 with default GRF. Under high register pressure, try SIMD16 first.
5. **Smaller and simpler private data**:
   - split arrays of vectors (`float4 v[8]`) into separate scalar arrays;
   - use narrower types;
   - avoid `volatile` and taking the address of private variables;
   - prefer USM pointers to accessors as kernel arguments;
   - limit unrolling;
   - recompute cheap values;
   - spread a per-work-item array across the sub-group with `group_broadcast`.
6. **ESIMD.** Explicit SIMD gives full control of register allocation and messages, at the cost of portability. Published ports gained another 2–3x over SYCL plus large GRF.

Re-check spills after each change. Large GRF halves occupancy, so it can make a kernel slower when that kernel needs threads to hide memory latency. Measure both modes.

## Occupancy

Occupancy is the fraction of hardware thread slots that hold work. Memory-bound kernels need enough of it to keep loads in flight; compute-bound kernels with high register reuse need much less.

| Per Xe-core | Xe-HPG | Xe2, Xe-HPC |
|---|---|---|
| Vector engines × threads | 16 × 8 = 128 threads | 8 × 8 = 64 threads |
| Max SLM per work-group | 64 KB | 128 KB |
| Sub-group sizes | 8, 16, 32 | 16, 32 |


- **Threads per work-group** = work-group size / sub-group size. A work-group of 256 at sub-group 16 is 16 threads. All of a work-group's threads must fit in one Xe-core.
- **SLM is allocated in buckets** (2, 4, 8, 16, 24, 32, 48, 64, 96, 128 KB). A 65 KB request takes 96 KB and can halve the resident work-groups. Size SLM tiles just under a bucket.
- **Waves.** `work-groups / (resident work-groups per Xe-core × Xe-cores)`. A fractional last wave idles the rest of the GPU. With very few work-groups, say 40 work-groups on a 32- or 64-core GPU, much of the GPU idles. Decode attention reaches only about 72% of bandwidth at batch 1 for that reason, and about 94% at batch 16. Split along K or sequence length, or put more rows per work-group, when work-groups are fewer than Xe-cores.
- **Grid sizing from the device.** Use `max_compute_units` together with `ext::intel::info::device::gpu_eu_count` and `gpu_hw_threads_per_eu` where the device supports those aspects. Never use a count taken from another card.

Measure occupancy with `unitrace -q -g ComputeBasic ./app`, which prints per-kernel `XVE_THREADS_OCCUPANCY_ALL[%]` and `XVE_ACTIVE[%]`. See [profiling](profiling.md).
