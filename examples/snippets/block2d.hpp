// block2d.hpp - 2D block load from plain SYCL (Xe2 and Xe3P; Xe-HPG has no 2D block messages),
// quoted in references/memory.md.
// Build with: -Xspirv-translator -spirv-ext=+SPV_INTEL_2d_block_io
#pragma once
#include <sycl/sycl.hpp>

// snippet: block-load-2d
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
// end snippet
