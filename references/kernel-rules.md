# Kernel design rules

## Correctness constraints

- Preserve the specified operation. Do not add clamps, change overflow behavior, flush denormals or enable fast math merely to pass parity.
- Work-group barriers must be reached convergently by participating work-items. Pad tails and predicate loads/stores without skipping collective operations. Subgroup collectives are not direct replacements for CUDA masked warp operations.
- Preserve snapshot semantics when replacing register shuffles with local memory. Each phase needs disjoint read/write roles, or separate read and write phases with synchronization or separate buffers. In a halving reduction, only the destination half writes while the source half remains unchanged. A barrier after an overlapping update does not order accesses inside that phase. See the [SYCL barrier semantics](https://github.khronos.org/SYCL_Reference/iface/barriers-and-fences.html).
- Use explicit event dependencies or an in-order queue. Host access must wait for completion. USM accessibility does not imply concurrent host/device atomic coherence. Check the relevant USM atomic aspects and use valid scopes and orders.
- Avoid software grid-wide spin barriers. Work-group forward progress and simultaneous residency are not guaranteed by an ordinary launch. Use separate kernels and events for a portable global synchronization boundary. Experimental root synchronization requires its documented launch and resource constraints and a dedicated test.
- Respect alignment, object lifetime, aliasing and allocation bounds. An aligned-down vector load can still read before an allocation. An aligned-up read can cross its end. Prefer scalar tail handling until a vector scheme is proved safe.
- Matrix tiles, packing, accumulator widths and subgroup participation form a single contract. Test a tiny asymmetric known-pattern tile before adding pipelining or quantized layouts. Never assume CUDA WMMA fragments or AMD MFMA layouts transfer to Intel matrix operations.
- Validate all shape and index arithmetic on the host. Check size multiplication overflow and allocation failures. Explicitly reject unsupported shapes instead of silently dropping tails.

## Optimization hypotheses to measure

These are candidate experiments, not architecture-wide performance rules:

| Symptom | Candidate experiment | Evidence needed |
|---|---|---|
| Excess memory traffic | reuse data, coalesce accesses, vectorize legal loads | compiler output or profiler counters and numerical checks |
| Private-memory spills | reduce live state, tile differently, try supported GRF modes | generated code/resource reports and latency |
| Many short kernels | fuse compatible work or try graph replay | end-to-end and event timing including setup |
| Repeated dot products | matrix tiling or library baseline | supported combinations, packing cost and shape sweep |
| Poor subgroup utilization | change width and rows per subgroup | workload-specific measurements and collective correctness |
| SLM staging stalls | change layout, buffering or eliminate staging | barrier/resource measurements and tail tests |

Increasing GRF can reduce spills and also alter occupancy. Narrowing a subgroup can help or hurt depending on the algorithm. DPAS does not make added columns free. Graph replay can benefit useful workloads by reducing submission overhead. A load-only probe changes the instruction mix and cannot prove that a full kernel is optimal.

Use compiler reports to test suspected argument copies or spills rather than banning all dynamic indexing. Apply `restrict` only when callers satisfy the no-alias contract. A profiler can perturb scheduling and relative kernel shares as well as absolute timing, so compare with unprofiled runs.

## Dispatch and faults

Gate every specialized variant on actual features and tested shape/layout conditions. Keep an always-available portable path for supported baseline inputs. An override may request a candidate but cannot bypass those guards. Preserve the baseline for A/B measurements and unseen shapes.

On device loss, retain command, input seed, selected device, build identity and accessible driver logs, then stop the failing path. Diagnose bounds, synchronization, compiler/runtime and hardware causes without declaring one from a single reset. Runtime debug keys are version-specific diagnostic experiments and must never become portable skill defaults.
