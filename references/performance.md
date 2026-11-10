# Measuring performance

## What to time

- **Device time**: a queue built with `property::queue::enable_profiling{}`, then `event.get_profiling_info<info::event_profiling::command_start / command_end>()`, or a barrier event before and after N repetitions on an in-order queue (`bench.hpp`).
- **Caller latency**: host wall time to completion (`wait_and_throw()`), including allocation, packing, transfers and dispatch when the caller pays for them.

Report them separately and never compare one against the other. Warm up first: JIT compilation, allocation and first-touch costs land in the first calls. Label results as cache-resident or streaming, and for streaming rotate over buffers larger than the reported cache.

Record with every result: device name and PCI address, driver, compiler version and flags, dtype, layout, shape, cache regime, timing scope, and source hashes of baseline and candidate.

Benchmark habits:

- Flush or rotate more data than L2 between timed calls when measuring the streaming regime.
- Launch a dummy large kernel first so the host has enqueued timing events before short kernels finish.
- Take many repetitions and trim the extremes.
- When comparing with a CUDA result, match its floating-point flags. `icpx` defaults to a fast floating-point model that contracts FMAs; `nvcc` does not use fast math by default.
- Express each result as a percentage of the device's time floor ([analysis](analysis.md)), not only as a speedup over the baseline.

## Work and rate conventions

| Operation | Work | Rate |
|---|---|---|
| Dense GEMM `C = A*B` | `2*M*N*K` FLOPs (multiply and add counted separately) | GFLOP/s = FLOPs / (µs × 1000) |
| int8 GEMM | `2*M*N*K` integer ops | GOP/s or TOPS, with the accumulator type; not FLOPs |
| Row reduction | `rows*(cols-1)` additions | additions/s, plus bytes/s |
| Softmax, transcendental activations | not FLOP-meaningful | latency and effective bytes/s |

Effective bytes/s counts the logical minimum traffic (inputs read once, outputs written once). A cache-resident run can exceed DRAM bandwidth. Physical traffic needs hardware counters. A vendor peak is a citation (product, precision, dense or sparse), not a measurement.

## Accepting a faster variant

Interleave baseline and candidate on the same device (ABBA order), keep raw samples, and confirm a winner with a fresh run rather than reusing the samples that selected it.

`scripts/accept_performance.py record.json --min-speedup 1.05` applies that policy. The record has a `discovery` and a `confirmation` phase. Each phase holds:

- `run_id`, unique per phase;
- `context`, identical across phases, with `device_pci`, `device_name`, `driver`, `runtime`, `compiler`, `build_flags`, `dtype`, `layout`, `shape`, `operation`, `cache_regime`, `timing_scope` (`device_event` or `end_to_end`), `baseline_sha256`, `candidate_sha256` (lowercase), `baseline_variant`, `candidate_variant`;
- `correctness`: `{"baseline": "PASS", "candidate": "PASS"}`;
- `warmup_complete`, `interleaved`, `uncontended`: booleans that the evidence supports;
- `pairs`: at least 20 entries with distinct `sample_id`, `baseline_us`, `candidate_us`, and `order` alternating `AB`/`BA`.

The tool accepts (exit 0) when the lower bound of a paired bootstrap interval of the median speedup exceeds the threshold in both phases. Exit 1 means the evidence is valid but the gain is too small; exit 2 means invalid evidence. Choose the threshold before tuning.

The tool checks records, not truth. It cannot know whether the GPU was idle, the oracle was independent, or every queue was waited on. Do not mark `uncontended` true because a process list was empty: a short snapshot misses jobs, and a display GPU always has clients. Shared or display GPUs give valid correctness results but only indicative timings.
