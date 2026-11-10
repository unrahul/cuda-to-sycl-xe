# Fusion

Fusion removes round trips through device memory and kernel boundaries. For memory-bound chains it is the largest lever available: 5–10x on B70 for convolution + normalization + pooling chains, and a transformer decoder layer shrinks from hundreds of ops to about nine. It can also lose, so decide with [analysis](analysis.md) and verify by measurement.

## Fuse, or call the library? Count the bytes

For a chain of operations, compare two numbers ([analysis](analysis.md)):

- **Unfused bytes:** every intermediate tensor written once and read by each consumer pass, plus inputs and outputs.
- **Fused floor:** inputs read once and outputs written once, plus the time for the compute.

If intermediates dominate, a library call for the first operation followed by separate passes cannot win. The library must write its full output to memory. Write one fused kernel, and keep the library only as a correctness control.

Example: `Conv2d(8→64, 3×3)` on 128 images of 128×128, then GroupNorm(16), scale, MaxPool(4) and clamp, all FP32. On Arc Pro B70 (608 GB/s, 22.9 TFLOPS FP32):

| | Bytes | Time floor |
|---|---|---|
| Input / final output | 67 MB / 31 MB | |
| Convolution output (intermediate) | 520 MB | |
| PyTorch eager: conv, then norm statistics, normalize, scale, pool and clamp as separate passes | ≈3.7 GB | ≈6 ms memory; PyTorch eager takes about 21 ms |
| Library conv + one fused tail | conv writes 520 MB, tail reads it twice: ≈1.7 GB | ≈2.8 ms |
| Fully fused | ≈0.1 GB, 1.9e10 FLOP | ≈0.8 ms (compute) |

Fully fused kernels run 5–9x faster than PyTorch eager on this chain, while library convolution plus tail kernels run slower than eager. A convolution with only 8 input channels has too little reuse for the library to be compute-efficient, and it forces the 520 MB intermediate.

This worked example is where the library-versus-fused rule in [backend selection](backend-selection.md) comes from.

## When fusion pays

- **The intermediate is large and read once.** Fusing a producer and its consumer saves a write and a read of the whole tensor. Count the bytes saved against the kernel's own traffic.
- **Statistics, not intermediates, cross the boundary.** Norms, softmax and pooling need per-row or per-group statistics. Organize work so that the group that owns a statistic is one work-group, and the intermediate never leaves the chip (worked example below).
- **Launch and synchronization dominate.** Chains of tiny kernels pay submission cost per launch. Fusion removes it, and so does graph replay ([kernel rules](kernel-rules.md), rules 21–22).
- **Layout ops between kernels.** A transpose, `cat` or index-select is a whole pass. Fold it into the producer's store or the consumer's load ([memory](memory.md)).

## Gotchas: when fusion loses

- It breaks a library GEMM into a slower custom GEMM. Keep the library call and fuse only the epilogue.
- It cuts parallelism, for example one work-group per normalization group when there are fewer groups than Xe-cores. Split statistics into partials and merge them, or use a second pass.
- It raises register pressure until the kernel spills ([registers](registers.md)).
- It changes rounding points that the contract fixes. Algebraic rewrites must be exact or within the stated tolerance.

## Launch overhead without fusion

When kernels must stay separate:

- **Event-less submission.** Every `submit` returns an event, and that overhead is significant. `sycl::ext::oneapi::experimental::nd_launch` (the `sycl_ext_oneapi_enqueue_functions` extension) submits without one.
- **Graphs.** `sycl_ext_oneapi_graph` records a chain once with `begin_recording`/`end_recording`, finalizes it, then replays the whole chain as one submission.
- **Immediate command lists.** On Xe2 and later, the default Level Zero V2 adapter uses only immediate command lists. Older batching variables such as `SYCL_PI_LEVEL_ZERO_BATCH_SIZE` do nothing there.
- **JIT cost.** JIT compilation dominates first calls. Compile AOT, or warm up before timing. `SYCL_CACHE_PERSISTENT=1` caches binaries across runs, but one engine's Intel docs report it segfaulting on Xe2, so test it before relying on it.
- **No automatic fusion.** The Codeplay kernel-fusion extension has been removed. Fuse by hand.

## Recipes

- **Decoder layer**:
  - one QKV linear;
  - linear + residual-add, linear + SiLU and linear + multiply epilogues;
  - fused RMSNorm and RoPE;
  - fused attention that reads and writes the layout its neighbors use;
  - a preallocated KV cache grown in steps instead of concatenated.
- **Small MLPs**: keep each sub-group's activation rows in registers across all layers, and stream only weights through SLM. Intensity rises from 32 to 512 FLOP/byte per layer. They reached 140 of a predicted 244 TFLOPS; the gap was SLM load latency, not bandwidth.
- **Epilogue after a library GEMM:** one kernel reads the GEMM output once and applies bias, norm and activation . The ceiling is the GEMM time; see [analysis](analysis.md).

## Worked example: fused convolution, GroupNorm and pooling

The chain is `Conv2d(3x3) -> GroupNorm -> per-channel scale -> MaxPool2d(4) -> clamp`, FP32. The unfused version writes the convolution output to memory, reads it back for statistics, again to normalize, and again to pool. The fused design touches the input once and writes only pooled outputs. 

1. **One work-group per (sample, group).** A GroupNorm group is the unit that needs global statistics, so give it a whole work-group and the statistics never leave local memory.
2. **Stage the group's weights once** in local memory as `[ci][kh][kw][c]`, so the inner channel loop reads consecutive addresses.
3. **Register tiles.** Each work-item computes a 4x4 block of convolution outputs for all of the group's channels from a 6x6 input patch per input channel. That is one pool window. Input values are reused across the 3x3 taps and all channels without leaving registers.
4. **Statistics without a second pass.** Each work-item folds its blocks into mergeable moments, then the work-group merges them in local memory. Convolution outputs that the pool drops (right and bottom borders) still count toward the statistics.
5. **Pool before normalizing.** Per channel, `f(v) = ((v - mean) * rstd * gamma + beta) * scale` is monotone: non-decreasing when `gamma` and `scale` have the same sign, non-increasing otherwise. Because round-to-nearest preserves monotonicity, `max f(v) = f(max v)` holds bit for bit (use `min v` when `f` decreases). Each work-item keeps only the window extreme, waits for the final statistics, and applies `f` and the clamp once per pooled output. Propagate NaN explicitly in the extreme.

```cpp
// Count, mean and sum of squared deviations. Merging partial blocks this way (Chan et al.)
// avoids the cancellation of sum(x*x) - n*mean*mean in a one-pass FP32 variance.
struct Moments {
  float n = 0.0f, mean = 0.0f, m2 = 0.0f;
};

inline Moments merge(Moments a, Moments b) {
  if (b.n == 0.0f) return a;
  if (a.n == 0.0f) return b;
  const float n = a.n + b.n, d = b.mean - a.mean;
  return {n, a.mean + d * (b.n / n), a.m2 + b.m2 + d * d * (a.n * b.n / n)};
}

// Work-group merge through local memory. Every work-item calls it (it contains barriers)
// and every work-item receives the merged result. s must hold one entry per work-item.
inline Moments group_moments(sycl::nd_item<1> it, Moments mine, const sycl::local_accessor<Moments, 1>& s) {
  const int lid = it.get_local_id(0), wg = it.get_local_range(0);
  s[lid] = mine;
  sycl::group_barrier(it.get_group());
  for (int stride = wg / 2; stride > 0; stride /= 2) {  // wg must be a power of two
    if (lid < stride) s[lid] = merge(s[lid], s[lid + stride]);
    sycl::group_barrier(it.get_group());
  }
  return s[0];
}
```

`sycl::reduce_over_group` accepts only the standard function objects (`plus`, `maximum`, ...), so a custom merge like this goes through local memory. The general lessons:

- Pick the work decomposition from the reduction that forces a global sync.
- Recompute instead of storing when the input is small.
- Look for algebraic reorderings, such as pooling through a monotone map, that are exact under rounding rather than only approximately equal.
