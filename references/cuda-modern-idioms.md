# Porting modern CUDA (Ampere, Hopper, Blackwell) to Xe

Kernels written for recent NVIDIA GPUs lean on hardware that Intel GPUs do not have:

- asynchronous copy engines (`cp.async`, TMA);
- warpgroup MMA reading shared-memory descriptors (`wgmma`);
- producer/consumer warp specialization;
- thread-block clusters;
- Blackwell tensor memory.

Transliterating them produces a slow or impossible kernel. Re-express the *dataflow* in the shape Intel's own Xe kernels use. Intel's own Xe GEMM and flash-attention kernels, including those in sycl-tla, use this shape.

**The Xe kernel shape:**

- Every sub-group owns a register-resident output tile.
- Each sub-group loads its own operand tiles straight from global memory into registers with 2D block loads, already in DPAS layout (VNNI or transposed).
- An L1 prefetch runs a few K-steps ahead instead of a shared-memory ring.
- Accumulators live in registers for the whole K loop, using large GRF (256 on Xe2, up to 512 on Xe3P).
- A split barrier (arrive early, wait late) keeps the work-group's sub-groups loosely in step.
- There are no producer roles, no shared memory for operands, and no async-copy barriers.

Xe-HPG lacks 2D block loads, so stage tiles through SLM or vector loads there ([memory](memory.md)).

## Idiom map

| CUDA idiom | Xe-HPG | Xe2 | Xe3P | Advice |
|---|---|---|---|---|
| `cp.async` multistage pipeline | plain loads into SLM | 2D block prefetch into L1, then 2D block load into registers | as Xe2 | Turn the stage ring into a prefetch distance of about 2 tiles. Keep SLM only for data shared across sub-groups |
| TMA (`cp.async.bulk.tensor`) + `mbarrier` | none | 2D block load: one sub-group, one rectangular tile, synchronous completion, so no byte-count barrier | as Xe2, plus wider transposes and 256-byte prefetch rows | Pad or align strides to the block-load rules ([memory](memory.md)) |
| `ldmatrix` / `stmatrix` | sub-group block reads | not needed: VNNI and transposed 2D loads produce DPAS layout | as Xe2 | Delete them, and choose the matching load variant |
| `mma.sync` (m16n8k16 ...) | DPAS at sub-group 8: M ≤ 8, N = 8 | DPAS at sub-group 16: M ≤ 8, N = 16, K = 256 bits / element width | as Xe2, plus fp8 and fp4 inputs | Cover a warp tile with several DPAS per sub-group ([GEMM](gemm.md)) |
| `wgmma` (async, operands from shared memory) | none | none: DPAS operands live in registers; overlap comes from prefetch and other resident threads | as Xe2, with 512 GRF for larger tiles | Keep a register-resident tile per sub-group |
| Warp specialization + `setmaxnreg` | none | possible, but not used by production kernels | as Xe2 | Make sub-groups symmetric. Choose the GRF mode per kernel instead of moving registers between roles ([registers](registers.md)) |
| Thread-block clusters + distributed shared memory | none | none | none | Use a larger work-group, or exchange through L2/global memory. Each sub-group loads its own tiles instead of multicasting |
| `tcgen05` + tensor memory (Blackwell) | none | none | none; the closest is 512 GRF per thread | Keep accumulators in registers |
| FP8 (E4M3/E5M2) GEMM | none | no FP8 DPAS: convert to fp16/bf16 in registers, then DPAS | native FP8 DPAS, any E4M3/E5M2 mix, f32 or bf16 accumulate | Gate on the target ([architectures](xe-architectures.md)) |
| NVFP4 / MXFP4 block-scaled MMA | none | dequantize in software | block-scaled DPAS (`bdpas`, M = 8) with group-32 E8M0 scales (MX). NVFP4's group-16 E4M3 scales have no `bdpas` evidence | Formats in [quantized compute](quantized-compute.md) |
| FlashAttention-3 ping-pong between warpgroups | none | not done: split barriers plus several resident sub-groups overlap softmax with DPAS | as Xe2 | Remove named-barrier choreography |
| FlashAttention-4 software exp2 and lazy rescale | — | Intel kernels use `sycl::native::exp2` with `log2(e)` folded into the scale, rescaling every iteration | as Xe2, with a +8 exponent bias before packing P to FP8 | Lazy rescaling is a valid optional port; no Intel code does it yet |
| Persistent kernel + tile scheduler / Stream-K | manual loop | sycl-tla Stream-K and group schedulers; persistent FA scheduler with the grid sized to the Xe-core count | as Xe2 | Size the grid from the device, never from a constant |
| `__dp4a` | hardware DP4A | hardware DP4A | hardware DP4A | A plain widening int8 loop, or `sycl::ext::oneapi::dot_acc` (`<sycl/ext/oneapi/dot_product.hpp>`), compiles to `dp4a` for `acm-g10`, `bmg` and `cri` (checked in the ISA dump) |
| CUDA graphs | SYCL graphs (`sycl_ext_oneapi_graph`) | same | same | [Fusion](fusion.md), launch overhead |


## FlashAttention forward, as Intel structures it

**Xe2.**

- One work-group per Q tile, with sub-groups of 16 splitting Q rows only. For head dimension 128, the tile is 256×32×32 with 16 sub-groups (16 Q rows each). For head dimension 64, it is 128×32×32 with 8 sub-groups.
- `grf_size<256>`, no SLM. Q is loaded once into registers.
- Each sub-group loads K with a transposed 2D load and V with a VNNI 2D load, and prefetches K/V two tiles ahead.
- Loop body:
  1. barrier arrive;
  2. S = QKᵀ by DPAS;
  3. mask;
  4. row max by sub-group reduction;
  5. `exp2` with scale × log2(e);
  6. partial row sums, with the horizontal sum deferred;
  7. pack P to fp16/bf16, rescale O, then P·V by DPAS;
  8. prefetch the next K/V;
  9. barrier wait.
- The 1/l normalization happens in the epilogue.

**Xe3P.**

- Same loop with `grf_size<512>`, and tiles roughly doubled (512×64×64 with 32 sub-groups for head dimension 128).
- The FP8 path computes QKᵀ in FP8, biases the softmax exponent by +8, packs P to FP8, and folds K/V descales into the softmax scale or the output.
- MXFP K/V use block-scaled DPAS.

**Xe-HPG.** Same structure with sub-group 8 and DPAS N = 8, staging K/V through SLM because there are no 2D block loads.

## Translate from the CUDA source, not from a converter

Work from the CUDA kernel: its indexing, its synchronization, its numerical contract, and the hardware idiom it relies on. Translate each piece by hand with the [mapping](cuda-to-sycl-mapping.md) and the idiom map above.

SYCLomatic (dpct) output can save typing, but it is an automated, syntax-level translation. Treat every line as unverified. Its usual failure modes in real ports:

- **Sub-group size.** It pins `reqd_sub_group_size(32)` only where it sees a shuffle. Other warp-assuming kernels get the compiler default, 16 on Xe2, and race. Set the width deliberately on every kernel that assumes one.
- **Barriers.** It emits full barriers (DPCT1065) where the CUDA code only synchronized shared memory. Use `group_barrier` / `barrier(local_space)`.
- **Ballots.** A `__ballot_sync` mask becomes a meaningless expression around `any_of_group`/`all_of_group`. Rebuild the logic from the CUDA intent.
- **Helpers.** Generated helpers can be wrong outside the cases they were written for. A generated block reduction that computes `nwarps / WARP_SIZE` returns zero for small work-groups.
- **Hardware idioms.** Tensor-core matmul, `cp.async`/TMA pipelines and cooperative-group grid sync have no automatic translation. Real ports end up with slower substitutes:
  - a quantized matmul disabled for accuracy problems;
  - flash-attention prefill left unported;
  - a grid-synchronized softmax falling back to a slower path.

  Redesign these with the idiom map.
- **Generation gates.** Ports often gate their XMX paths to Xe2 only, with no Xe3P entry, so new hardware silently takes the slow path. Detect generations as in [architectures](xe-architectures.md).

## Build settings that CUDA parity needs

- **Division.** Device FP32 division is not correctly rounded by default: roughly 30% of quotients differ from IEEE division, and `-fp-model=precise` does not change that. `-foffload-fp32-prec-div`, or the AOT backend option `-cl-fp32-correctly-rounded-divide-sqrt`, makes division correctly rounded. CUDA divides with correct rounding by default.
- **Large allocations.** Reads more than 4 GiB into one allocation returned garbage in a JIT build on Arc A770 until `-ze-opt-greater-than-4GB-buffer-required` was added. Precompiled library kernels such as oneMKL's do not get the flag.
- **First-launch cost.** `-fsycl-device-code-split=per_kernel` removed a 245 ms first-launch cost.
