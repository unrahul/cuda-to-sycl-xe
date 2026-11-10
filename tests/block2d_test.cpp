// block2d_test.cpp - checks the 2D block load snippet: lane/row mapping and what is returned
// outside the declared surface. Xe2, Xe3 and Xe3P only; exits 77 elsewhere.
// Build: icpx -fsycl -O2 -std=c++20 -I examples -Xspirv-translator -spirv-ext=+SPV_INTEL_2d_block_io \
//        tests/block2d_test.cpp -o block2d_test
#include "snippets/block2d.hpp"
#include "snippets/dispatch.hpp"

#include <cstdio>
#include <vector>

using namespace snippets;

class Load2D;

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  const auto gen = xe_generation(q.get_device());
  if (gen != XeGen::Xe2 && gen != XeGen::Xe3 && gen != XeGen::Xe3p) {
    std::puts("SKIP: 2D block loads need Xe2, Xe3 or Xe3P");
    return 77;
  }
  // Surface: 40 columns x 13 rows inside a 64-column pitch. Everything outside the surface
  // (pitch padding and rows below it) holds a marker, so leaked reads are visible.
  constexpr int cols = 40, rows = 13, pitch = 64, alloc_rows = 24;
  constexpr unsigned marker = 0xDEADBEEF;
  std::vector<unsigned> host(pitch * alloc_rows, marker);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) host[r * pitch + c] = 1000u * r + c + 1;
  auto* dev = sycl::aligned_alloc_device<unsigned>(64, host.size(), q);
  auto* out = sycl::malloc_device<unsigned>(16 * 8, q);
  q.memcpy(dev, host.data(), host.size() * sizeof(unsigned)).wait();

  int failures = 0;
  // in bounds; columns past the width; rows past the height; both
  for (auto [col, row] : {std::pair{16, 4}, {32, 0}, {0, 8}, {32, 8}}) {
    q.parallel_for<Load2D>(sycl::nd_range<1>{16, 16}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
       unsigned tile[8];
       load_tile_16x8(dev, cols, rows, pitch, col, row, tile);
       const int lane = it.get_sub_group().get_local_linear_id();
       for (int r = 0; r < 8; ++r) out[lane * 8 + r] = tile[r];
     }).wait_and_throw();
    std::vector<unsigned> got(16 * 8);
    q.memcpy(got.data(), out, got.size() * sizeof(unsigned)).wait();
    int exact = 0, zero = 0, leaked = 0, wrong = 0;
    for (int lane = 0; lane < 16; ++lane)
      for (int r = 0; r < 8; ++r) {
        const int c = col + lane, y = row + r;
        const bool inside = c < cols && y < rows;
        const unsigned v = got[lane * 8 + r];
        if (inside) v == host[y * pitch + c] ? ++exact : ++wrong;
        else if (v == 0) ++zero;
        else if (v == marker || v == host[y * pitch + c]) ++leaked;
        else ++wrong;
      }
    std::printf("tile at (%2d,%2d): %3d exact, %3d zero outside, %3d memory outside, %3d wrong\n", col, row, exact,
                zero, leaked, wrong);
    failures += wrong + leaked;
  }
  sycl::free(dev, q);
  sycl::free(out, q);
  std::puts(failures ? "FAIL: block-load-2d" : "PASS: block-load-2d (outside the surface reads as zero)");
  return failures ? 1 : 0;
}
