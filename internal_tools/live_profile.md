# Live host profile for one SYCL process

Run from the repository root. The script needs `sudo -n` for bpftrace and runs
the workload as the invoking user. It sources host oneAPI in the workload process.

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 LIVE_PROFILE_TIMEOUT=60 \
  ./internal_tools/live_profile.sh validation/profiler-my-run -- \
  ./examples/bin/portable_gemv
tail -f validation/profiler-my-run/trace.log
```

The output directory must be new or empty. `trace.log` emits a one-second
aggregate snapshot during execution and final aggregates when the process
exits. It never prints individual syscalls. `target.log` contains workload
output. `provenance.txt` records the command, selector, kernel, tool version,
source hashes, target hash and both exit statuses. `drm-fdinfo.log` contains
100 ms samples of any `drm-*` fields exposed for the traced PID. One Python
sampler reads those files and stops when tracing ends. The run is bounded by
`LIVE_PROFILE_TIMEOUT` (default 60 seconds), and a timeout returns 124.

The eBPF filter includes all threads whose process ID matches the target. It
starts before oneAPI initialization and covers initialization, device discovery
and the workload. `host_drm_ioctl_count` classifies an ioctl by the DRM request
type byte (0x64). `host_drm_ioctl_ns` sums time inside those host syscalls
across threads. `host_futex_ns` is host futex syscall time, including waits
unrelated to the GPU. Sums can overlap across threads and are not wall time.
The histogram shows host DRM ioctl latency. None of these fields measures GPU
kernel duration.

The fdinfo sampler records the driver, PCI device, client ID, engine and memory
fields when the driver exposes them. The SYCL runtime may open DRM fds for
devices other than the selected one during discovery. Group records by
`drm-pdev` and `drm-client-id`; an open fd does not prove that a kernel ran on
that device. Do not sum all fdinfo values as selected GPU utilization. Some
driver counters can remain zero during a short run.

This script profiles one host process and its threads. Forked child processes
and workloads running inside containers are outside its PID filter. Its
results are diagnostic traces from a profiled run, not paired benchmark
comparisons. No profiling overhead percentage has been established.

See [the recorded validation](../validation/profiler-20261106-gemv-level-zero-1-final/REPORT.md) for an actual SYCL GEMV run, raw trace, fdinfo samples and exit statuses.
