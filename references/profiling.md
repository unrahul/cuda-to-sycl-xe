# Profiling and inspecting kernels

Measure before explaining. None of these commands needs root. Select one GPU first with `ONEAPI_DEVICE_SELECTOR=level_zero:N`.

## unitrace (pti-gpu): first stop

| Question | Command | What to read |
|---|---|---|
| Which kernels take the time? | `unitrace -d ./app` | Device Timing Summary: calls, total, average, min, max per kernel |
| Spills, SLM, GRF, SIMD width per kernel | `unitrace -d ./app` | Kernel Properties table: `Spill Memory Per Thread`, `Private Memory Per Thread`, `SLM Per Work Group`, `Register File Size Per Thread`, `SIMD` |
| Launch shapes | `unitrace -d -v ./app` | kernel names carry `[SIMD16 {groups} {local size}]` |
| Hardware utilization per kernel | `unitrace -q -g ComputeBasic ./app` | per-kernel CSV on stderr: `XVE_ACTIVE[%]`, `XVE_THREADS_OCCUPANCY_ALL[%]`, `GPU_MEMORY_BYTE_READ/WRITE[bytes]`, `GPU_MEMORY_BYTE_READ_RATE[GBpS]`, `AvgGpuCoreFrequencyMHz` |
| Available metric groups | `unitrace --metric-list` | groups such as ComputeBasic, RenderBasic, DeviceCacheProfile |
| Host submission vs execution gaps | `unitrace -s ./app` | append, submit and execute intervals per kernel |
| Timeline | `unitrace --chrome-kernel-logging ./app` | JSON for chrome://tracing or Perfetto |

How to read the numbers:

- **Achieved bandwidth** = (read + write bytes) / GPU time. Compare it with the device's memory bandwidth ([analysis](analysis.md)). Near it means bandwidth-bound and done; far below it with low `XVE_ACTIVE` usually means latency-bound or too few work-groups.
- **Low `XVE_THREADS_OCCUPANCY_ALL`** points to registers (large GRF), SLM buckets, or too few or too small work-groups ([registers](registers.md)).
- **`AvgGpuCoreFrequencyMHz`.** Short kernels on an idle GPU run at low clocks (about 1.2 GHz on B70 for microsecond kernels). Measure steady-state workloads, not one cold launch.
- **Stall attribution.** `--stall-sampling` gives per-instruction stall reasons on Data Center GPU Max and later (`uniview.py` displays them).

## VTune and Advisor

- `vtune -collect gpu-hotspots -r <dir> -- ./app`, then `vtune -report hotspots -r <dir> -group-by=computing-task -format=csv`. The report has per-kernel time, global and local work size, SIMD width and spill size. "XVE Array Stalled/Idle" near 100% with low GPU time means the GPU mostly waited on the host or on memory.
- `advisor --collect=roofline --profile-gpu --project-dir=<dir> -- ./app` collects. Then `advisor --report=roofline --gpu --project-dir=<dir> --report-output=$PWD/roofline.html` writes an interactive roofline that places each kernel. The report step takes `--gpu`; with `--profile-gpu` it fails. `advisor --report=survey --gpu --project-dir=<dir> --format=csv --report-output=survey.csv` gives per-kernel time, active, stalled and occupancy as CSV.

## Looking at generated code

- **Dumps.** `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>` makes the compiler write, per kernel:
  - `.asm`: final ISA, with `//.thread_config numGRF=...`, `//.spill size` and `//.instCount`;
  - `.visaasm`;
  - `.zeinfo`: `grf_count`, `simd_size`, `spill_size` and scratch buffers.

  Set it for the AOT build, or for the run when JIT. Look for `send` messages, block loads (`load_block2d`) and `dpas` in the `.asm` to confirm the compiler emitted what you intended.
- **Build logs.** An AOT build per target (`-device acm-g10`, `bmg`, `cri`) prints spill warnings for each generation from any machine. That checks register pressure on hardware you don't have.

## Profiling discipline

- **Perturbation.** Profilers slow kernels and change overlap. Use their shares and counters, and take timings from unprofiled runs ([performance](performance.md)).
- **Device selection.** Profile on the device you tune for, with one GPU selected. On a GPU that also drives a display, counters include desktop work.
- **Explanations are hypotheses.** Record an explanation without a counter behind it as a hypothesis, not a finding.
