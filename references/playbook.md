# Optimization loop

Run this loop per kernel family on each target device.

1. **Baseline.** Have the reference passing and the best correct control in hand. For standard dense work, that control is oneMKL or oneDNN ([backend selection](backend-selection.md)). Otherwise it is the portable port.
2. **Headroom before effort.** Write down work, bytes, intensity and the time floor `T_lower` on this device ([analysis](analysis.md)). For a chain of operations, compare unfused bytes with the fused floor ([fusion](fusion.md)). Every later decision compares a measured time with `T_lower`.
3. **Diagnose.** Measure the whole path, then profile the slow stage if a profiler is available (`unitrace`, VTune). Inspect the compiler's spill and private-memory reports. Write down unverified explanations as hypotheses.
4. **One hypothesis per variant.** Predict what the change improves and what it may cost (registers, occupancy, local memory).
5. **Check, then time.** Every variant passes the reference on the full case list before any timing.
6. **Time.** Interleave control and candidate, keep raw samples, and confirm the winner in a fresh run ([performance](performance.md)).
7. **Integrate.** Add the variant behind capability and shape gates, test the dispatched branch and the fallback, and re-measure in the caller (packing, workspace, transfers included).
8. **Re-profile.** The bottleneck moves after a win.
9. **Continue or stop.** Stop when the best correct variant is within about 1.3x of `T_lower`, or when no attempts remain. Otherwise spend the next attempt on the next hypothesis.
   - A failed variant is a diagnosis, not a reason to retreat. `OUT_OF_RESOURCES` means shrink the SLM tile or work-group; spills mean change the tiling.
10. **Record.** Store passing, failing and skipped variants, with the device and stack, beside the code, not in this skill.

## Where to look first

| Symptom | Try | Confirm with |
|---|---|---|
| Far below the read-pattern ceiling | coalesce, vectorize legal loads, fewer load messages per byte | probe of the new pattern, compiler output |
| Private-memory spills | smaller per-lane state, different tiling, `grf_size<256>` | spill report, then timing |
| Many short kernels | fuse where a dependency or memory round trip disappears; graph replay for submission cost | end-to-end timing including setup |
| Dot-product bound | `joint_matrix` (DPAS) tile, or oneMKL/oneDNN | matrix combinations, packing cost, shape sweep |
| Poor sub-group utilization | change rows per sub-group or sub-group width | per-shape timings, collective correctness |
| Local-memory staging stalls | change layout or buffering, or remove staging (2D loads on Xe2/Xe3P) | barrier count, tail tests |

A load-only probe changes the instruction mix: it bounds a pattern, it does not prove a kernel optimal. A speedup on one product is a candidate on another, not a default.
