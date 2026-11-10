# Examples

Build with Intel oneAPI DPC++ after setting up its environment:

```sh
source /opt/intel/oneapi/setvars.sh
./examples/build.sh                 # portable SPIR-V JIT
# Optional AOT: ./examples/build.sh TARGET
# Obtain TARGET from the installed backend for the actual product.
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./scripts/smoke.sh
```

The smoke runner builds the examples, checks parity helpers and dispatch generation, and runs small tail-shaped workloads on one explicitly selected GPU. It writes a unique `validation/smoke.*.log` (or the caller-selected `SMOKE_LOG`). Its `BENCH_SMOKE` setting reduces timing repetitions and cache rotation; timings in that mode are not performance measurements.

`bin/silu_mul [tokens d]` checks four work-group/vector combinations against a host reference, including partial vectors. `bin/gemv_q8 [n_in n_out]` checks subgroup variants against a double reference for one through eight columns. GEMV requires positive `n_in`, a multiple of 32 no greater than 131071, positive `n_out`, work-groups of 256, and subgroup sizes 16 and 32. These restrictions keep packed accesses aligned and integer accumulation bounded. Unsupported matrix combinations are skipped; generation labels alone do not establish feature support. The sample exits nonzero on parity failure or invalid input.

`bin/xe_probe` measures kernel boundaries and read patterns. Graph execution depends on runtime support. Software grid barriers are excluded because independent work-groups need not all be resident. The optional experimental 2D builtin is not launched on the strength of a generation guess. `common/xe_arch.hpp` reports queried device properties. The matrix example checks actual combinations, subgroup support and device/kernel resources before selecting a path. Device identity never authorizes an instruction. `common/bench.hpp` contains timing and parity helpers, which reject non-finite outputs.

`bin/portable_gemv` is the portable starting point for int8 GEMV: it uses no fixed subgroup or work-group width and checks 15 small shapes, including non-divisible inner dimensions, against exact host int64 arithmetic. The supplied case dimensions keep int32 accumulation in range. Extend its operation contract and overflow checks before accepting arbitrary dimensions. The tuned GEMV example is a separate constrained experiment. Its DPAS helper requires an allocation of eight activation rows even for fewer requested output columns; a caller with only NC rows must pad or redesign the tile load.

`gemv_q8 n_in n_out --require-dpas` fails if the matrix path cannot execute. `GEMV_CASE=random|zero|extreme|alternating` selects deterministic numerical cases. Exercise the required matrix path across its supported tile and output-column cases; a skip must not count as success.

Configure both compiler and runtime before building or running. If a vendor setup script fails under shell strict options (for example `set -u`), diagnose setup in a separate shell before enabling those options for the benchmark runner. Keep setup stderr while diagnosing missing runtime libraries. Do not silently continue after a failed setup.
