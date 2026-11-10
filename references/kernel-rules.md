# Kernel rules

Each rule names the failure it prevents. No rule carries a device number; the probe gives yours.

## Correctness

1. **Name every kernel.** PyTorch's extension build compiles with `-fsycl-host-compiler`, which rejects unnamed kernel lambdas. Declare `class MyKernel;` at namespace scope (not inside a function) and launch `parallel_for<MyKernel>`. Templated kernels need a templated name: `template <int N> class Tile;`.
2. **Barriers are collective.** Every work-item of the group must reach each `group_barrier`, including the padded tail. Predicate loads and stores, not barriers or collectives.
3. **Local memory replacing shuffles keeps snapshot semantics.** In one phase, the locations read and the locations written must be disjoint, or split into a read phase and a write phase separated by a barrier. In a halving reduction only the destination half writes. A barrier *after* an overlapping update does not order the accesses inside it.
4. **No software grid barrier.** Work-groups are not guaranteed to be co-resident, so a kernel spinning on a global counter can hang. Use two kernels with an event dependency.
5. **Bitwise parity needs the same reduction tree.** Two kernels agree in every bit only when every addition happens in the same order. A DPAS variant sums in K order and is never bitwise equal to a dp4a variant. Give it an FP64 reference and a tolerance instead.
6. **Turn off contraction where the reference does.** `icpx` contracts by default; see *FMA contraction* in the [mapping](cuda-to-sycl-mapping.md).
7. **Keep finite.** Half-precision scales and sums saturate. Clamp to the largest finite value before converting, the same way on every path, or one path yields `inf` where another does not.
8. **No run-time index into an array of pointers in a kernel argument.** `ptrs[token]`, with `token` known only on the device, makes the compiler copy the whole argument into private memory per work-item. Pass a base pointer plus stride, or select with a chain of conditionals. Check private-memory and spill reports.
9. **Packed formats are only as aligned as their block.** Quantized blocks of 18, 34 or 210 bytes are 2-byte aligned. A 16-byte vector load through a cast is undefined and can fault. Load two aligned chunks and shift, and read the second chunk only when the address is unaligned, so the load cannot pass the end of the allocation.
10. **2D block loads need the true surface.** See *Outside the surface* in [memory](memory.md).
11. **The DPAS operand layout is a contract.** For int8: in A, lane k holds four K values of column k for each of the M rows. In B, lane n holds column n, one dword per four K values. In D, each lane holds M values. Copy the layout from working code (`examples/gemv_int8/`, SYCL-TLA `XE_DPAS_TT`) and test a tile with a known asymmetric pattern on its own before using it in a kernel.
12. **Validate shapes on the host.** Check that index products fit the index type and that allocations succeeded. Reject unsupported shapes explicitly rather than dropping tails.
13. **The parity test ships with the kernel.** Register it with CTest. It exits 77 (`SKIP_RETURN_CODE 77`) where the specialized path is unavailable, and the portable path must still run. A suite of skips is not a pass.

## Performance

These are hypotheses to test with the [optimization loop](playbook.md), each with the mechanism behind it.

14. **Loads first, then compute, then stores.** A work-item that interleaves loads and stores through pointers the compiler cannot prove disjoint gets serialized memory operations. Load a phase's inputs into registers, compute, store once. Use `__restrict__` only when callers guarantee no aliasing.
15. **Issue weight loads before staging.** When a kernel stages activations into local memory and then streams weights, request the work-item's first weights before the staging barrier, so they are in flight while it waits.
16. **Load message count, not bytes, can be the limit.** A dequantizing kernel issuing a dozen small loads per 32 weights is bound by message count. Coalesce into one 16-byte run per lane for the quants and one header read per block.
17. **Columns are not free for dp4a; they nearly are for DPAS.** A dp4a GEMV reruns its chain per activation column, so time grows with the column count. A DPAS kernel takes up to M = 8 columns in one instruction. Compare a kernel's time at one column and at eight.
18. **Judge a kernel against its access pattern's ceiling.** Coalesced stream, one row per lane and 2D block load have different ceilings. Write a load-only probe of the pattern the kernel uses (same addresses, XOR-reduced, no arithmetic). A kernel at 90% of its pattern's ceiling needs a different pattern, not more tuning.
19. **Sub-group 32 to 16 is a redesign.** Narrowing a 32-lane row kernel halves the lanes per row and keeps the loads per lane. On a SIMD16 machine the 16-wide design is the DPAS design: rows in lanes (B operand), columns in the A operand.
20. **GRF mode and rows per sub-group are sweep dimensions.** `grf_size<256>` doubles registers and halves resident threads. Reading each activation block once for two or four rows trades registers for loads. The best setting depends on shape and column count.

```cpp
// Large GRF: twice the registers per hardware thread, half the resident threads.
// A sweep dimension for register-heavy kernels; measure both modes.
struct ScaleLargeGrf {
  float* y;
  float a;
  void operator()(sycl::id<1> i) const { y[i] *= a; }
  auto get(syclex::properties_tag) const {
    return syclex::properties{sycl::ext::intel::experimental::grf_size<256>};
  }
};
```

21. **Measure launch cost against barrier cost before fusing.** `xe_probe` measures a kernel boundary eagerly and as a graph. Fusion pays when it removes a dependency or a round trip through memory, for example a producer that also writes its consumer's quantized input, or statistics recomputed inside the work-group that consumes them. Fusing merely to save a launch rarely pays when the kernels do real work.
22. **Graph replay cuts host submission cost.** It helps chains of short kernels. Kernels with real work run no faster from a graph. Measure eager and graph-replayed versions of the actual chain.
23. **Time device work beyond the cache when the workload streams.** Rotate over enough weight copies that their total exceeds the reported cache several times over; `bench::copies_beyond_cache` does this. Label cache-resident and streaming results separately.
24. **Cross-check the profiler.** If a profiler's kernel times per window sum to more than the wall time, it slowed the run: use its shares, not its microseconds. Confirm busy time with an unprofiled run.

## Dispatch

25. **One generated table per device and stack.** Sweep every variant over the real shapes and column counts on the target. Accept a variant only when it beats the production path by more than the noise (`scripts/accept_performance.py`), and write the winners into a generated table. Never edit the table by hand.
26. **New kernels default on only where they were measured.** A kernel tuned for a new card is the default on that card only, behind a switch: `0` off, `1` on anywhere (for tests), unset follows the table. Add a master switch that restores the old defaults on the new card for comparison.
27. **Older generations stay byte-identical.** After adding a path for a new card, the dispatch decisions and parity results on the existing cards must not change. Check this.
28. **Read switches through the real wrapper.** Some container or service wrappers drop variables set to `0` so a default applies. A switch tested as `v[0] == '0'` is then never off. Check the wrapper before an A/B run.

## Faults and early silicon

29. **Stop on device loss.** Keep the command, input seed, device, build identity and driver logs. Diagnose bounds, synchronization, compiler/runtime and hardware causes, and do not declare one from a single reset.
30. **Count hardware errors per run from the kernel journal.** `scripts/fault_count.sh SINCE` counts `Hardware Error`, `Engine reset` and `GPU HANG` lines and prints `UNKNOWN` if the journal is unreadable. Unknown is not zero.
31. **Bisect intermittent faults with interleaved runs.** `scripts/switch_bisect.sh` alternates switch families. Attribute a fault only to a family that faults in every run with it on and in none with it off.
32. **Try the runtime's settings before blaming the kernel.** Command-buffer placement, graph patch preambles, and eager versus graph execution have each explained device losses on early silicon. Driver debug keys such as `NEOReadDebugKeys=1` are diagnostic experiments, never defaults.
33. **A kernel that trips the card is optional, not default.** Keep it behind its switch with its parity test, record the symptom, and retest on another card of the same product.
34. **Keep lab loops gentle.** Thousands of back-to-back launches of one shape can fault a card that runs the same kernel fine inside a model. Note which shapes fault only in the lab loop.
