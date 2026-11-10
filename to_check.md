# To check

Facts in the shipped skill that rest on sources, not measurements, or where sources disagree. Each item names its source, where the skill uses it, and the quickest check. When an item is confirmed or corrected, update the skill page and delete the item.

## Xe3P (D270 / Crescent Island)

All Xe3P items are in [`xe3p.md`](xe3p.md).

## Xe3 (Panther Lake)

15. **The architecture values `intel_gpu_ptl_h`, `intel_gpu_ptl_u` and `intel_gpu_wcl`** were used in `dispatch.hpp` and compile with oneAPI 2025.3 and 2026.1. They are not run on hardware.
16. **Variable registers per thread** (10 threads at 96 GRFs or fewer, IGC). Check with `unitrace` on a PTL laptop.
17. **DPAS and 2D loads for `-device ptl-u`** [report: llm-scaler].

## Xe-HPG (Arc A)

18. **`joint_matrix` on DG2.** IGC `JointMatrixFuncsResolutionPass.cpp:676-679` appears to reject the joint-matrix API on DG2. The SYCL matrix spec lists DG2 combinations, and our `bf16_gemm<8, 8, 16, 8>` compiled AOT for `acm-g10`.
    - Check: run `tests/snippets_test.cpp` on an Arc A card, and read the combinations it reports.
19. **No native FP64** (IGC `hasNoFP64Inst` lists DG2; emulation flag).
    - Check: `aspect::fp64`.
20. **Host-visible USM polling not seen by a kernel on A750** [report].

## Xe2 (other than B70)

21. **tf32 DPAS.** IGC lists tf32 for Xe2, but the B70 runtime (oneAPI 2026.1.1) reports no tf32 combination.
    - Check: `matrix_combinations` with a newer stack.
22. **B580, B60 and B50 numbers** come from Intel spec pages. FP32 for A770 and B580 is derived (Xe-cores × engines × lanes × 2 × clock).
    - Check: Intel spec pages or a measured FMA microbenchmark.
23. **The `grf_size` spec does not list BMG,** but `grf_size<256>` and `grf_size_automatic` compiled and gave 256 registers on BMG with oneAPI 2025.3.
    - Check: whether the spec has since been updated.

## From the CUDA-porting research (2026-11-10)

27. **`SYCL_CACHE_PERSISTENT=1` reportedly segfaults on Xe2** (Strata `docs/INTEL.md:140`). This box has a populated `~/.cache/libsycl_cache` without obvious crashes.
    - Check: run `tests/snippets_test.cpp` twice with the variable set.
28. **`-foffload-fp32-prec-sqrt`** (correctly rounded square root) is assumed to behave like the measured `-foffload-fp32-prec-div`, but it was not measured.
29. **ESIMD compiler bugs on oneAPI 2026.1 with bmg-g31**, reported by Strata (`docs/INTEL_INT8_EXPERTS.md:76-80`):
    - a dynamic index into a kernel-argument array;
    - `%` by a run-time `n`;
    - `gather` with VS > 1;
    - a simd captured by reference.

    These are not reproduced and not shipped in the skill. Retest them on the current stack before adding them.
30. **Correction to the original skill text.** The archived original described Strata's `STRATA_INTEL_GPU_GEN` override and device-id-first detection. Neither exists in the Strata sources on this box, so those statements were dropped. The D270 numbers that text supplied (384 KB SLM, `0x674C`) are now confirmed by compute-runtime.

## Tooling

24. **JIT spill warning on stderr** appeared once on a first compile and could not be reproduced with caches disabled (`SYCL_CACHE_PERSISTENT=0`, `NEO_CACHE_PERSISTENT=0`).
    - The skill recommends AOT warnings or `unitrace -d` instead. Investigate whether an IGC or env flag controls it.
25. **`unitrace --stall-sampling`** is documented for Data Center GPU Max and later. Not tried on B70.
