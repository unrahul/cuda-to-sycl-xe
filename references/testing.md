# Numerical and performance validation

## Freeze the contract

Specify operation, shapes, strides, alignment, aliasing, input/accumulator/output dtypes, rounding and saturation, overflow, exceptional-value behavior, and acceptable error. Separate exact integer semantics from floating-point tolerance. Signed integer overflow in C++ is not a valid way to implement modular arithmetic.

Use independent CPU arithmetic, usually FP64 for floating reductions, based on the format definition and the *represented input values*. Decode quantized values independently. FP64 is a useful reference, not a proof of exactness: large reductions, transcendental functions and severe cancellation may need compensated summation, arbitrary precision or an analytical oracle. Reference execution must remain outside timed GPU work.

For finite values, require every output to satisfy `abs(got-ref) <= atol + rtol*abs(ref)` with operation-specific, predeclared tolerances. Report maximum absolute and relative error, failing index/count, and optionally normalized RMS error. Global scale or RMS checks alone can hide a bad small output. Handle zero references explicitly. Reject unexpected NaN/Inf before arithmetic comparisons. If the contract permits NaNs or infinities, test their placement and sign policy separately. Do not widen tolerances after a candidate fails without a mathematical justification and a new recorded contract.

Bitwise equality is appropriate only when required and achievable. Reduction order, FMA contraction, denormals, math library behavior and intermediate precision all matter. Integer reductions are reorderable only while the mathematical range fits the accumulator or well-defined modular semantics are preserved.

## Required cases

Scope tests to the declared input contract. In a fixed-workload assessment, keep scored shapes and dtypes unchanged. Small independent host checks can qualify an arithmetic change without claiming coverage of the full workload.

Test tiny sizes, zero-sized host behavior, non-divisible tails, prime dimensions, tile boundaries minus/at/plus one, supported strided layouts and alignment offsets, zero/constant/random values with fixed recorded seeds, alternating signs, cancellation, scale extremes and the operation's exceptional values. Exercise every dispatch branch and its fallback. Initialize outputs with sentinels and check guards around output regions where practical. Verify padding and untouched regions. Repeat invocation to expose stale workspace and missing dependencies.

An unsupported specialized path is SKIP with a reason. The portable baseline must still execute. Use distinct skip reporting (for example exit 77 with CTest `SKIP_RETURN_CODE 77`); a suite of skips is not a passing hardware validation. Surface asynchronous exceptions with `wait_and_throw` and return a failing process status. A migration with no original CUDA/HIP device available can establish reference correctness, but cannot claim source-device equivalence was run.

## Timing and promotion

Validate first. Warm JIT, allocations and caches according to the intended workload, then time sufficient repetitions with profiling-enabled queue events. Also measure wall latency with explicit completion. Report setup, packing and transfers separately and include them in deployment totals where applicable. Measure event duration for device work and completion-bounded wall time for caller latency.

Interleave baseline and candidate measurements on the same selected device. Record sample count, median, spread, thermal/power conditions and concurrent activity. A cooperative file lock does not exclude servers or unrelated GPU users. Busy hardware can support a correctness smoke, but cannot establish an isolated speed claim. Cache-resident and streaming results need separate labels. A rotating allocation larger than the reported cache is an experimental setup, not proof of misses.

Use `scripts/accept_performance.py` with the [performance record contract](performance.md) to check discovery and confirmation evidence. The supplied `make_table.py` is a legacy table formatter, not a complete promotion gate. Its CSV cannot represent device/stack/layout identity, raw samples or noise. Keep one verified context per invocation and perform those checks before calling it. Never feed its output directly into production without that review.

Generate dispatch only from valid finite positive timings and passing correctness records. Refuse missing/failed baselines, malformed rows, ambiguous duplicates or mixed contexts. Preserve raw samples and the exact compiler/device/workload identity outside any legacy CSV format. Revalidate the generated dispatch and an unseen-shape fallback. Gains must exceed both the chosen threshold and observed noise.

## Fault evidence

Capture process exit status, stderr, device identity and journal access status for the time window. Unreadable logs mean UNKNOWN, not zero faults. Journal counts are symptoms, not proof of fault attribution or numerical correctness. Stop on device loss. Retain the reproducer for diagnosis before another attempt.

Package smoke tests demonstrate only their listed cases. Record exact commands, results and missing coverage in the integrating project.

## Inference-kernel integration

Apply TensorFold's useful separation of operator arithmetic and model fidelity: validate every changed projection class against an independently implemented host FP64 reference using the same represented weights and input rows. Exercise prompt and decode paths separately. Include per-column error so an aggregate score cannot hide a failing channel. For stateful kernels, also compare intermediate and persistent state across repeated steps.

After kernel checks, use representative captured activations and teacher-forced token/logit comparisons when the integrating project can provide them. Keep top-token agreement, logit error and top-two margins separate from the kernel error threshold. A reference using quantized weights tests the implementation of that quantized operation. Evaluating quantization quality requires the original higher-precision weights as an additional baseline. Report selected layers, rows and prompt chunks precisely. Sampling them does not establish whole-model equivalence.

Use the [public source catalog](sources.md) to inspect upstream test coverage and requirements. Any use of an upstream validation procedure needs its own execution evidence.

## Validate the evaluator

Following the [public source catalog](sources.md), challenge the harness with deliberately wrong outputs, unchanged sentinel outputs, mutated supposedly read-only inputs and missing completion dependencies. Keep pristine reference inputs separate from candidate buffers. A candidate must not change the oracle's inputs to make a comparison pass. Device-event timing must cover all queues contributing to the output, or be supplemented with a correctly synchronized end-to-end measurement.
