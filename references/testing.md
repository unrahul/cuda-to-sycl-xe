# Numerical validation

## Fix the contract first

Write down, before any kernel exists:

- operation, shapes, strides, alignment, aliasing;
- input, accumulator and output dtypes, rounding points, saturation and overflow behavior;
- NaN/Inf policy;
- the tolerance, and why it is that value.

Keep exact integer semantics separate from floating-point tolerances. Changing the tolerance after a candidate fails needs a mathematical reason and a new recorded contract.

## Build an independent reference

- Compute on the host from the *represented* inputs (the same rounded floats, decoded quantized values and scales the kernel sees), in FP64 for floating point and in an integer type wide enough for the true range.
- Host FP64 does not need device FP64.
- FP64 is not exact. Long reductions, transcendentals and heavy cancellation may need compensated summation (`math.fsum`) or an analytic answer.
- Keep reference inputs separate from the buffers the kernel receives, so a kernel that modifies its inputs cannot make the comparison pass.
- Keep the reference out of timed regions.

## Compare every element

For finite outputs require `abs(got - ref) <= atol + rtol * abs(ref)` for each element. Report the maximum absolute and relative error and the index and count of failures. A global RMS or norm can hide one bad channel. Check for unexpected NaN/Inf before comparing. Handle zero references explicitly.

Bitwise equality is right only when the contract demands it and the arithmetic order matches (see rule 5 in [kernel rules](kernel-rules.md)). Integer reductions may be reordered only while the accumulator cannot overflow.

## Cases to cover

- Sizes: 1, tiny, primes, every tile boundary minus one / equal / plus one, the real workload shapes, and empty inputs where the API allows them.
- Layouts: each supported stride, transpose and alignment offset.
- Values: zeros, constants, random with a recorded seed, alternating signs, cancellation, scale extremes, and the operation's special values (`-0.0`, denormals, NaN, Inf, int8 −128, saturation bounds).
- Every dispatch branch and its fallback. An unavailable path exits 77 (SKIP) with a reason, the portable path still runs, and an all-skip suite is not a pass.
- Output buffers pre-filled with a sentinel, with guard regions checked afterwards. Padding and untouched regions stay unchanged.
- Repeated calls, to expose stale scratch or missing dependencies; non-default and multiple queues; outputs that must survive later calls.
- `wait_and_throw()` so asynchronous errors fail the process.

`scripts/parity_template.cpp` is the program skeleton: an FP64 reference, a bitwise check against the production kernel where the order matches, SKIP, and a CTest exit code. `tests/snippets_test.cpp` applies the same pattern to the documented snippets.

## Tight tolerances on long reductions

Estimate your error before the first attempt. A sum of `N` FP32 terms carries rounding error of roughly `sqrt(N) · 6e-8 · (typical |partial sum|)` for pairwise or tree order, and up to `N · 6e-8 · ...` for sequential order. Compare that with `atol + rtol · |output|`.

When the two are close, as for a GEMM followed by a sum over 8192 outputs checked at `rtol = atol = 1e-4`:

- **Only the candidate's error is in your hands.** An FP32 reference has its own rounding, and the check measures the distance between the two. Drive your own error well below the tolerance.
- **Accumulate long reductions in FP64**, or use pairwise or Kahan summation. Keep FP32 storage when the contract fixes the output dtype.
- **Prefer formulations that do less rounding.** Reducing first and multiplying once, as in the algebraic rewrite in [analysis](analysis.md), also cuts error.
- **A failure just over the tolerance means precision, not logic.** Check this before rewriting the algorithm.

## A mismatch with no details

When an evaluator reports only "mismatch", don't abandon a fast design. Check, in this order:

1. **Coverage.** Every output element is written. Check tails, borders and the last partial tile or work-group. Pre-fill outputs with NaN in your own test to expose holes.
2. **Argument plumbing.** Parameter order, which tensor is weight versus bias, `.contiguous()` before taking `data_ptr`, and dtype.
3. **Reduction domain.** A normalization covers exactly the elements the reference covers. For example, GroupNorm statistics include convolution outputs that a later pool drops.
4. **Formula details.** Biased versus unbiased variance, where `eps` is added, `rsqrt` versus `1/sqrt`, clamp bounds, padding and dilation in index math.
5. **Precision.** FP32 accumulation where the reference uses it; contraction differences ([mapping](cuda-to-sycl-mapping.md)); a one-pass variance on data with a large mean.

To isolate the fault, reimplement the kernel's exact indexing in NumPy on a tiny shape and compare it with the framework reference. That tests the algorithm without a GPU. Bisect by temporarily returning an intermediate, such as statistics or the pre-pool tensor, and comparing it.

## Test the test

Before trusting a harness, feed it a wrong output, an untouched sentinel buffer, a kernel that mutates its input, and a missing dependency. It must reject each one. Device-event timing must cover every queue that writes the output.

## Ported code

A port with no CUDA/HIP device available is validated against the reference, not against the source kernel. Say so. When the source kernel's logical order matters (a 32-lane tree, wave64 ballots), model that order on the host and compare it too.

## Inference kernels in a model

- Validate each changed projection class against an FP64 host product of the same quantized weights and activations. Run prompt and decode paths separately, and report per-column error so one channel cannot hide in an aggregate.
- For stateful kernels, compare intermediate and persistent state across repeated steps.
- After kernel checks, use captured activations and teacher-forced token and logit comparisons, keeping their thresholds separate from kernel tolerances.
- A reference that uses the quantized weights tests the implementation, not the quantization quality; that needs the original weights.

## Gotchas

- NaN and Inf semantics must survive algebraic shortcuts: an output that is identically zero for finite inputs must still propagate NaN rows.
