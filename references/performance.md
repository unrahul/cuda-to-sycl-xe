# Performance measurement and acceptance

Use the operation's mathematical work and a stated timing boundary. Record precision, shape, layout, cache regime, device PCI identity, driver/runtime/compiler, build flags and baseline/candidate source hashes. Keep raw repeated samples. A cooperative GPU lock excludes only other jobs using that lock. Inspect other device activity before asserting an uncontended run.

## Work and rate conventions

| Operation | Work convention | Rate |
|---|---|---|
| Dense real GEMM C=A*B | `2*M*N*K` FLOPs, counting multiply and add as two operations; no beta/epilogue | GFLOP/s = FLOPs / (`elapsed_us * 1000`) |
| Exact scalar dot work | `K` multiplies and `K-1` additions per dot | State this alternative explicitly if using it instead of the common GEMM convention |
| Row sum | `rows*(columns-1)` mathematical additions | Report additions/s and bytes/s as appropriate; actual tree/padding work can differ |
| Signed-int8 GEMM | `2*M*N*K` integer operations under a stated MAC convention | GOP/s or TOPS, with accumulator precision; never label these floating FLOPs |
| Softmax or transcendental activation | Separate reductions, ordinary arithmetic, exponentials/divisions | Prefer latency and explicitly modeled effective bytes/s; an exponential is not one hardware FLOP |

For bytes/s, distinguish logical minimum input/output bytes from actual memory transactions. For softmax, one read plus one write is a minimum logical traffic convention, not a claim that a multi-pass implementation touches memory only twice. Packing and extra temporary buffers add work. Cache-resident effective bytes/s can exceed external-memory bandwidth. Use measured counters to discuss physical traffic.

Report kernel/device-event latency separately from completion-bounded application latency. Include setup, packing, transfers, temporary workspace and dispatch when they belong to the deployment objective. Do not combine a device-only baseline with an end-to-end candidate. Derived throughput must use the same scope as its latency. Validate FLOP/byte formulas and unit conversions independently of the benchmark kernel.

Measured rates are workload/stack results and stay in run evidence. Vendor peak specifications require a primary source, exact SKU, precision, dense/sparse convention and units. Do not call a measured rate a product specification or extrapolate throughput between products.

## Candidate acceptance

`scripts/accept_performance.py record.json` validates two phases, `discovery` and `confirmation`. Each phase contains:

- `run_id`: distinct, nonempty evidence identity.
- `context`: identical across phases, with `device_pci`, `device_name`, `driver`, `runtime`, `compiler`, `build_flags`, `dtype`, `layout`, `shape`, `operation`, `cache_regime`, `timing_scope`, `baseline_sha256`, `candidate_sha256`, `baseline_variant`, `candidate_variant`.
- `correctness`: exactly `{"baseline":"PASS","candidate":"PASS"}`. Preserve detailed numerical test evidence outside the summary.
- `warmup_complete`, `interleaved`, `uncontended`: true only when supported by the run evidence.
- `pairs`: at least 20 objects with distinct `sample_id`, `baseline_us`, `candidate_us`, and alternating `order` values `AB` and `BA`.

`timing_scope` is `device_event` or `end_to_end`. Source hashes are lowercase SHA-256 values. Source/variant identities must differ; two entrypoints in one source may share a file hash but must have different variant identifiers. Sample identities cannot repeat between phases. Store the raw sample arrays directly, rather than replacing them with medians. Confirmation must come from a fresh measurement of the selected candidate, not renamed discovery samples.

The gate requires the lower endpoint of a paired bootstrap median-speedup interval to exceed the requested threshold in both phases. `--min-speedup 1.05` selects a five-percent threshold; choose a threshold appropriate to the project before tuning. Exit 0 accepts, 1 rejects a valid but insufficient gain, and 2 reports invalid evidence. The tool reports the median and confidence lower bound for each phase.

This gate checks records and applies one statistical policy. It cannot attest that a GPU was idle, an oracle was independent, a sample label is truthful or a process waited on every contributing queue. Keep provenance, logs, numerical results and current activity evidence for review. Bootstrap sampling does not remove thermal drift, correlated samples, multiple-comparison effects or a biased workload. Use separate holdout shapes when generalization matters.

Only after acceptance should a scoped dispatch choice be generated. The legacy `make_table.py` formats summary CSV rows and provides no statistical or context gate. Unseen devices, stacks, layouts and shapes retain the portable baseline.

## Activity-observation limits

An empty process/utilization snapshot does not by itself establish uncontended execution. A short snapshot can miss jobs, and some driver/tool combinations do not expose active processes. Check that the observation method detects a known active workload on the selected device and covers the measurement interval. Preserve that positive-control evidence. Otherwise mark isolation unknown and keep the measured gains diagnostic, with promotion withheld. Do not set `uncontended: true` merely because a query exited zero and returned an empty list.
