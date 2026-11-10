# Data movement and memory

Most kernels outside large GEMMs are limited by bytes, not FLOPs ([analysis](analysis.md)). Move fewer bytes, in fewer and wider messages, from the nearest level that holds them.

## Hierarchy

Registers (GRF, per hardware thread) → shared local memory (SLM, per work-group, carved from each Xe-core's L1) → L1 → L2 (device-wide) → device memory (GDDR/HBM/LPDDR).

| Per Xe-core / device | A770 (Xe-HPG) | B580 (Xe2) | Lunar Lake (Xe2) | Panther Lake (Xe3) | D270 (Xe3P) |
|---|---|---|---|---|---|
| L1 + SLM per Xe-core | 192 KB | 256 KB | 192 KB | 256 KB | 512 KB |
| SLM per Xe-core / max per work-group | 128 / 64 KB | 128 / 128 KB | 128 / 128 KB | query | query |
| L2 | 16 MB | 18 MB | 8 MB | up to 16 MB | 32 MB |

Query `local_mem_size`, `global_mem_cache_size`, `global_mem_cache_line_size` and `max_mem_alloc_size` on the device you run on, rather than reading them from a table.

Approximate latencies from older Intel guidance, useful for reasoning only:

| Level | Latency |
|---|---|
| SLM | ~30 cycles |
| Xe-core cache hit | ~50 cycles |
| L2 hit | ~150–200 cycles |
| Miss to memory | ~300–500 cycles |

Latency, not bandwidth, limits a kernel without enough loads in flight. Prefetch, issue more independent loads, or raise occupancy.
- **Allocate the right USM.** Use `malloc_device` for kernel data. `malloc_shared` migrates pages on demand and can stall the first touch. `malloc_host` is accessed over PCIe from the GPU, so use it only for transfers and small flags.

## Coalescing and message width

A sub-group's load is one message when its lanes read consecutive, aligned addresses. Rules:

- **Lane layout.** Lane `i` should read element `base + i` (or `base + i*vec` for vector loads), and the fast-moving index should follow the memory layout. A transposed or strided access issues many messages: transpose once in a producer, or stage through SLM.
- **Message count.** Reading 16 bytes per lane per message beats four 4-byte messages. Load-message count, not bytes, limits many dequantizing kernels ([kernel rules](kernel-rules.md), rule 16).
- **Sub-group block reads.** `sycl::ext::oneapi::experimental::group_load` and `group_store` emit hardware block reads and writes only when given `properties{full_group}`. Add `contiguous_memory` and `alignment<K>` to remove run-time checks. The older `sg.load<N>(ptr)` needs an explicit `reqd_sub_group_size`.
- **Prefetch.** The `sycl_ext_oneapi_prefetch` extension provides `prefetch(ptr, count, properties{prefetch_hint_L1})` and the group form `joint_prefetch(g, ...)`. `joint_matrix_prefetch` covers matrix tiles. `global_ptr::prefetch` reaches only L3, 32 bytes per work-item. Prefetches are not bounds-checked, so guard them.
- **Cache hints.** The `sycl_ext_intel_cache_controls` extension takes per-level read hints (uncached, cached, streaming) and write hints (uncached, write-back, write-through, streaming) on an `annotated_ptr`. Use streaming for data read exactly once, so it does not evict reused data.
- **Vector loads.** Use them only after checking alignment of every pointer and every row start. See the alignment-checked vector width pattern in [production patterns](production-patterns.md).
- **Packed formats.** Formats with odd block sizes (GGUF Q4_0 is 18 bytes) defeat wide loads. Reorder once at load time into separate code and scale arrays (same page).

## Shared local memory (SLM)

Use SLM when work-items reuse data that other work-items loaded. Typical cases are GEMM and convolution tiles, stencils and work-group reductions. It is not worth it for streaming data read once.

- **Barriers.** Every staging round costs barriers. Amortize them over enough arithmetic, and double-buffer (load tile `k+1` while computing on `k`) once the single-buffer version is correct.
- **Size versus occupancy.** SLM is allocated in buckets ([registers](registers.md), Occupancy); size tiles to fit one and keep enough work-groups resident.
- **Bank conflicts.** SLM has 16 banks of 4 bytes, so 64 consecutive bytes cover all of them. Lanes that hit the same bank at different addresses serialize: `slm[lane * 16]` runs at 1/16 speed. For column access to a row-major tile, pad the row pitch by one element, or skew each column's start row. Measure with and without the pad.
- **Snapshot semantics.** When SLM replaces register shuffles, read and write sets per phase must be disjoint ([kernel rules](kernel-rules.md), rule 3).

## Caches and streaming

- **Reuse distance.** Data reused within a short time stays in L1/L2. Order work so that consecutive work-groups share operands, for example a GEMM tile-order swizzle that groups work-groups along M. Swizzling helps only once there are enough tiles; Leave it off below about 16 tiles.
- **Streaming larger than L2.** A sweep over a large tensor runs at device-memory bandwidth. Don't read the same large tensor twice in a kernel chain; fuse instead ([fusion](fusion.md)).
- **Benchmarks.** A benchmark that repeats one call on small data measures L2, not memory. Rotate buffers larger than L2 (`bench::copies_beyond_cache`) or flush, and label results "cache-resident" or "streaming".

## 2D block loads (Xe2, Xe3P)

Xe2 and Xe3P can load a whole 2D tile into a sub-group's registers in one message, with bounds handled in hardware. Xe-HPG cannot; feed tiles from vector loads or SLM there. Declare the SPIR-V intrinsic and build with `-Xspirv-translator -spirv-ext=+SPV_INTEL_2d_block_io`:

```cpp
using coord2 = int __attribute__((ext_vector_type(2)));  // {x = column in elements, y = row}

// Width and pitch are in bytes, height in rows, all describing the whole surface.
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockLoadINTEL(
    int element_bytes, int block_width, int block_height, int block_count, const void* surface_base,
    int surface_width_bytes, int surface_height, int surface_pitch_bytes, coord2 xy, void* dst);
#else  // the host pass compiles kernel bodies too; without this stub oneAPI 2025.3 fails to link
inline void __spirv_Subgroup2DBlockLoadINTEL(int, int, int, int, const void*, int, int, int, coord2, void*) {}
#endif

namespace snippets {

// Called by all 16 lanes of a sub-group together. Loads a 16-column x 8-row tile of 32-bit values
// whose top-left corner is (col, row): lane i receives column col + i, tile[r] holds row row + r.
// Rules: base 64-byte aligned; width and pitch multiples of 4 bytes; col * 4 a multiple of 4.
inline void load_tile_16x8(const unsigned* base, int cols, int rows, int pitch_elems, int col, int row,
                           unsigned (&tile)[8]) {
  __spirv_Subgroup2DBlockLoadINTEL(4, 16, 8, 1, base, cols * 4, rows, pitch_elems * 4, coord2{col, row}, tile);
}

}  // namespace snippets
```

- **Outside the surface.** Elements outside the declared surface, past the width or past the height, read as zero, never as neighboring memory. `tests/block2d_test.cpp` checks this. Declare the true surface. A surface declared larger than its data returns whatever lies past the data, so pad and mask downstream.
- **Limits:**
  - height at most 32 rows;
  - at most 64 bytes per row (`bits × width ≤ 512`);
  - block count 1, 2 or 4;
  - width and pitch at most 2^24 bytes;
  - stores at most 8 rows;
  - transposed loads take 32- or 64-bit elements with width ≤ 8 on Xe2, and 32-bit width ≤ 16 on Xe3P.
- **Companions.** `__spirv_Subgroup2DBlockLoadTransformINTEL` produces the VNNI layout for the DPAS B operand. `__spirv_Subgroup2DBlockPrefetchINTEL` takes the same arguments without a destination; issue it two or three K-steps ahead. `__spirv_Subgroup2DBlockStoreINTEL` stores.
- **Older builtins.** The `__builtin_IB_subgroup_block_read_flat_*` builtins take width, height and pitch *minus one*.

## Layout and transfers

- **Remove layout ops.** Transposes, `cat` and index-select between kernels are full extra passes. Produce the layout the consumer reads, for example by writing V transposed in the kernel that creates it. A tuned LLM decoder removes every transpose and cat by giving attention batch-first input and output and preallocating the KV cache.
- **Host↔device copies.** Use pinned host memory (`malloc_host`) as the staging buffer, overlap copies with kernels on a separate in-order queue joined by events, and batch small copies.
- **Index width.** Use 64-bit offsets for tensors past 2^31 elements. Use 32-bit indices inside a tile where the range is proven, since 64-bit integer math is slower.
