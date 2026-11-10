# Xe-HPG, Xe2, Xe3 and Xe3P

One programming model, different constants. The kernel is the same SYCL; the sub-group width, matrix tile, register modes, caches and limits change. Read the device's own answers with `xe_probe --capabilities` (or `scripts/xe_env_check.sh`). Use this page as the map of what to expect and what to check.

Query the device (`xe_probe --capabilities`) and trust it over this table.

## Generation table

| | Xe-HPG | Xe2 | Xe3 | Xe3P |
|---|---|---|---|---|
| Products | Arc A, Arc Pro A (DG2) | Arc B, Arc Pro B50/B60/B70 (BMG), Lunar Lake (LNL) | Panther Lake (PTL), Wildcat Lake | D270 (Crescent Island, CRI) |
| PCI device ids | DG2 G10/G11/G12 `0x4F8x`, `0x56xx` | BMG G21 `0xE202`–`0xE216`, G31 `0xE220`–`0xE223`; LNL `0x6420`, `0x64A0`, `0x64B0` | `0xB080`–`0xB0B0`, `0xFD80`, `0xFD81` | `0x674C`–`0x6751` |
| `architecture` value | `intel_gpu_acm_g10/g11/g12` | `intel_gpu_bmg_g21`, `_bmg_g31`, `_lnl_m` | `intel_gpu_ptl_h`, `_ptl_u`, `_wcl` | `intel_gpu_cri` (oneAPI 2026.2+) |
| AOT `-device` | `acm-g10` (alias `dg2`) | `bmg`, `bmg-g31`, `lnl-m` | `ptl-h`, `ptl-u` | `cri` (also `xe3p-xpc`) |
| Sub-group sizes | 8, 16, 32 | 16, 32 | 16, 32 | 16, 32; native SIMD32 for all shaders |
| GRF register size | 32 B | 64 B | 64 B | 64 B |
| GRF counts → threads per vector engine | 128→8, 256→4 | 128→8, 256→4 | 32–96→10, 128→8, 160→6, 192→5, 256→4 (automatic) | 128→8, 256→8; 512 needs a compiler that accepts `grf_size<512>` |
| SLM per Xe-core = max per work-group | 64 KB | 128 KB | 128 KB | 384 KB |
| L1 + SLM per Xe-core / L2 | 192 KB / 16 MB (A770) | 256 KB / 18 MB (B580) | 256 KB / up to 16 MB | 512 KB / 32 MB |
| Max work-group | query | 1024 | query | 1024 |
| XMX (DPAS) width | SIMD8, N = 8 | SIMD16, N = 16 | SIMD16, N = 16 | SIMD16, N = 16; the exposed instruction has depth 8 |
| DPAS input types | int8/uint8, int4, int2, fp16, bf16 | the DG2 set plus tf32 | as Xe2 | Xe2 types plus fp8 (E4M3, E5M2), e2m1 (fp4), block-scaled `bdpas` (M = 8); int8 × int4 dropped (sycl-tla) |
| `joint_matrix` combos at runtime | int8 M≤8×8×32, fp16/bf16 M≤8×8×16 | int8 M≤8×16×32; fp16/bf16 M≤8×16×16 plus 16×16×16, 1/32×64×16/32 | as Xe2 | none on oneAPI 2026.1 (matrix aspect 0); DPAS runs through ESIMD `xmx::dpas` |
| 2D block loads | none | yes; outside the surface reads zero | yes | yes; outside the surface reads zero; wider transposes (32-bit width ≤ 16), 256-byte prefetch rows |
| FP64 | no native FP64 (emulation flag) | yes; `aspect::fp64` = 1 | yes | yes; `aspect::fp64` = 1 |
| Reported compute units | XVE count | 256 (B70) | query | 256 (vector engines, not Xe-cores) |

Matrix support depends on the software stack as well as the hardware. On Xe2, oneAPI 2025.3 exposes no matrix combinations and 2026.1 exposes the ones above. IGC lists tf32 DPAS for Xe2, but the B70 runtime query shows no tf32 combination. Trust the runtime query. If an expected combination is missing, try a newer stack before concluding the hardware lacks it.

## Per-generation notes

**Xe-HPG.**
- XMX works at sub-group 8 with N = 8, so a GEMV's row tiling differs from Xe2.
- There are no 2D block messages; feed tiles from vector loads or SLM.
- 256-GRF mode exists at 4 threads per vector engine.
- sycl-tla supports only Xe2 and Xe3P. On Xe-HPG, use portable SYCL plus `joint_matrix`.
- Host writes to host-visible USM were reported not to be seen by a polling kernel on an Arc A750. Build doorbells from device memory.

**Xe2.**
- SIMD16 is native. A 32-lane kernel runs as two halves, and moving to 16 lanes pays off as a DPAS redesign: rows in lanes for B, columns in A.
- 2D block loads exist. A transposed 32-bit 2D load puts row n of a tile in lane n, the DPAS B layout.
- ESIMD is available and FP64 is native.
- Automatic large-GRF selection is off by default; request it ([registers](registers.md)).
- Work-groups are limited to 64 hardware threads, or 32 with SIMD32 or large GRF.

**Xe3 (Panther Lake).** Not Xe3P.
- Registers per thread vary automatically (VRT): 96 or fewer GRFs give 10 threads per vector engine, so register use affects occupancy gradually.
- DPAS and 2D block loads compile for `-device ptl-u`.
- `xe_generation` reports Xe3 separately from Xe3P.

**Xe3P (D270).**
- **Same surface as Xe2, with more of everything:** 384 KB of SLM per work-group, 32 MB L2, and a register file per Xe-core doubled to 1 MB. Work-groups stay at 1024.
- **New DPAS types:** fp8 and fp4, plus block scaling. sycl-tla exposes them under `SYCL_INTEL_TARGET == 35` ([production patterns](production-patterns.md)).
- **Matrix engine on oneAPI 2026.1.** `joint_matrix` offers no combination, so a `joint_matrix` path falls back to the portable one. bf16, fp16 and int8 DPAS run through ESIMD `xmx::dpas` (repeat count ≤ 8, sub-group 16, N = 16) in kernels built for `cri`.
- **Registers.** `grf_size<128>` and `grf_size<256>` both keep 8 threads per vector engine, so 256 registers cost no occupancy. oneAPI 2026.1 rejects `grf_size<512>` with a static assertion.
- **Naming the architecture.** oneAPI releases before 2026.2 cannot name it, so `xe_generation` below returns Unknown on them. Add a device-id check (`0x674C`–`0x6751`, via `aspect::ext_intel_device_id`) and keep every capability gate.
- **ESIMD.** `xmx::dpas`, SLM block loads and stores, `block_load`, gathers and barriers run when built for `cri`. Treat other ESIMD features as untested.
- **Driver.** Use the newest compute runtime available. Treat Xe3P as early silicon and apply the fault protocol in [kernel rules](kernel-rules.md).

**Building for several generations.** Each detection and XMX instantiation compiles ahead of time for its own target: `acm-g10` with `bf16_gemm<8, 8, 16, 8>`, and `bmg`, `ptl-u` and `cri` with `bf16_gemm<8, 16, 16, 16>` (oneAPI 2025.3). A kernel that requires sub-group 8 breaks AOT builds for Xe2 and later. Split per-generation kernels into separate AOT targets or translation units, or ship SPIR-V (JIT) for the variant that only one generation can run.

## What the constants decide

| Constant | Where it enters the kernel |
|---|---|
| Native SIMD width | `reqd_sub_group_size`. DPAS needs sub-group 16 on Xe2/Xe3P and 8 on Xe-HPG |
| GRF mode | Occupancy: 256 registers per thread halves resident threads. Helps kernels with large per-lane state, hurts latency-bound streams. Sweep it |
| Local memory | Work-groups resident per core at a given footprint, and the staging budget per work-group |
| Max work-group size | Upper bound for rows per work-group. The per-kernel limit can be lower; query it |
| Matrix tile (M, N, K) | Rows per sub-group and the K step of the inner loop. Xe-HPG uses N 8, Xe2/Xe3P use N 16 |
| 2D block loads | On Xe2/Xe3P the weight stream can arrive already in DPAS B layout. On Xe-HPG stage through local memory |
| Cache line | Alignment of packed formats ([kernel rules](kernel-rules.md), rule 9) |
| Compute units | Grid size for persistent kernels. Read it from the device or kernel bundle, never from a constant |

## Identify, then gate

From `examples/snippets/dispatch.hpp`, where `syclex` is `sycl::ext::oneapi::experimental`:

```cpp
enum class XeGen { Unknown, XeHpg, Xe2, Xe3, Xe3p };

// Identification only: it selects tuning tables, never permission to use a feature.
inline XeGen xe_generation(const sycl::device& d) {
  using A = syclex::architecture;
  const A a = d.get_info<syclex::info::device::architecture>();
  if (a == A::intel_gpu_acm_g10 || a == A::intel_gpu_acm_g11 || a == A::intel_gpu_acm_g12)
    return XeGen::XeHpg;  // Arc A-series (DG2/Alchemist)
  if (a == A::intel_gpu_bmg_g21 || a == A::intel_gpu_bmg_g31 || a == A::intel_gpu_lnl_m)
    return XeGen::Xe2;  // Arc B-series incl. Pro B50/B60/B70, Lunar Lake
  if (a == A::intel_gpu_ptl_h || a == A::intel_gpu_ptl_u || a == A::intel_gpu_wcl)
    return XeGen::Xe3;  // Panther Lake / Wildcat Lake (not Xe3P)
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20260200
  if (a == A::intel_gpu_cri) return XeGen::Xe3p;  // D270 / Crescent Island; enum needs 2026.2+
#endif
  // Older compilers cannot name CRI: fall back to its PCI device ids.
  if (d.has(sycl::aspect::ext_intel_device_id)) {
    const auto id = d.get_info<sycl::ext::intel::info::device::device_id>();
    if (id >= 0x674C && id <= 0x6751) return XeGen::Xe3p;
  }
  return XeGen::Unknown;  // portable path
}
```

```cpp
inline bool supports_sub_group(const sycl::device& d, std::size_t width) {
  const auto sizes = d.get_info<sycl::info::device::sub_group_sizes>();
  return std::find(sizes.begin(), sizes.end(), width) != sizes.end();
}

using mtype = syclex::matrix::matrix_type;

// True only if the runtime lists this A x B -> C combination with these N and K and an M
// range covering m. A matrix aspect alone does not establish any combination.
// int8 on Xe2/Xe3P: supports_matrix_tile(d, mtype::sint8, mtype::sint8, mtype::sint32, 8, 16, 32)
inline bool supports_matrix_tile(const sycl::device& d, mtype a, mtype b, mtype c, std::size_t m, std::size_t n,
                                 std::size_t k) {
  if (!d.has(sycl::aspect::ext_intel_matrix)) return false;
  for (const auto& x : d.get_info<syclex::info::device::matrix_combinations>()) {
    const bool m_ok = x.msize ? x.msize == m : m <= x.max_msize;  // msize 0 means "up to max_msize"
    if (x.atype == a && x.btype == b && x.ctype == c && m_ok && x.nsize == n && x.ksize == k) return true;
  }
  return false;
}
```

```cpp
// The device maximum is not a per-kernel limit: registers and local memory lower it.
template <class KernelName>
std::size_t kernel_max_work_group(const sycl::queue& q) {
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {q.get_device()},
                                                                        {sycl::get_kernel_id<KernelName>()});
  return bundle.get_kernel(sycl::get_kernel_id<KernelName>())
      .template get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
}
```

- Dispatch on `XeGen` to pick tuning constants such as tile sizes, rows per sub-group and GRF mode. Dispatch on the capability checks to decide whether a path may run at all. Unknown devices take the portable path.
- `aspect::fp64` says FP64 is exposed, not that it is fast; emulation is possible. Host FP64 references do not need device FP64.
- `max_compute_units` is not the Xe-core count, and its meaning varies by driver.
- A tuning table from one product is not a default for another product of the same generation. Keep tables per device and stack, and leave untested rows on the portable path.
- `examples/common/matrix_shape.hpp` handles combinations that mix bounded and fixed dimensions. `examples/gemv_int8/gemv_q8.sycl.cpp` is a complete gated DPAS path, with its fallback, that passed on B70 with oneAPI 2026.1.1.

## Xe3P versus Xe2 in practice

| Feature | Xe2 | Xe3P |
|---|---|---|
| Compile-time target | `SYCL_INTEL_TARGET == 20` (default) | `SYCL_INTEL_TARGET == 35`, set by `-D__SYCL_TARGET_INTEL_GPU_CRI__` |
| DPAS types | int8, uint8, fp16, bf16; int8 × int4 in sycl-tla | adds FP8 (E4M3, E5M2) and E2M1; drops int8 × int4 |
| Block-scaled DPAS | no | `XE_BDPAS_TT`, M must be 8 |
| fp16 → E4M3 conversion | emulated | native instruction |
| GRF per thread | 128 or 256 | 512 by default for large kernels, with fallback to 256 |
| FP8 quantization sub-group | 32 | 16 |

Gate every Xe3P-only feature on the compiler target and on a runtime capability check, and keep the Xe2-compatible path as the fallback.
