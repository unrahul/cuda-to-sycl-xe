# Maintainer hardware evidence

After configuring oneAPI, run `ONEAPI_DEVICE_SELECTOR=level_zero:3 ./internal_tools/run_evidence.sh`, choosing an appropriate single device index. The runner records UTC time, compiler, runtime device and host inventory under a timestamped `validation/` directory, then runs the smoke suite and a larger GEMV correctness case with bounded timeouts. Each stage records its exit status. Device index ordering can change. Inspect the runtime inventory when selecting hardware.

The larger run uses `BENCH_SMOKE=1` to limit repetitions and memory pressure. Its timings are diagnostic only. They are not isolated benchmarks or architecture specifications. Keep measurements with their device and software-stack provenance, and do not generalize them as architecture facts.

The runner creates a unique run directory containing `commands.txt`, source hashes, compiler identity, selected-device capabilities (including PCI address when exposed), stage logs and exit statuses. It stops subsequent stages after failure. `timeout` bounds ordinary processes, but an uninterruptible driver wait may outlive it. Save evidence and diagnose that condition before retrying. The host path is currently exercised. Container execution is allowed but must separately record image ID and runtime versions.

For another Xe device, first inspect availability and choose its index, then run this procedure without copying old tuning tables. The pure portable GEMV baseline has no subgroup requirement. Other examples are constrained candidate experiments, and unsupported paths must remain skips. `xe_probe --capabilities` only queries the device. Its full microbenchmark mode has additional graph and launch requirements.

For controlled performance work, create a separate project run with the intended workload/cache regime, paired baseline/candidate samples, raw timing records and spread. This smoke runner does not select production dispatch or establish isolated performance. `scripts/make_table.py` consumes a legacy summary CSV and cannot validate device context, repeated-sample statistics or application integration.

## Required Xe2 DPAS checks

The available `harness-sycl-dev:p0` image exposes matrix support with oneAPI 2026.1.1. The host 2025.3.3 stack does not expose it on these B70 devices. Record image identity with `docker inspect harness-sycl-dev:p0 --format '{{.Id}}'` before a run. The actual tested image ID is retained under `validation/dpas-fix/container-image.txt`.

```sh
docker run --rm --device=/dev/dri \
  -v /home/rahul/Coding/cuda-to-sycl-xe:/work -w /work \
  --entrypoint bash harness-sycl-dev:p0 -lc \
  'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ONEAPI_DEVICE_SELECTOR=level_zero:0 internal_tools/validate_dpas_xe2.sh'
```

Repeat with card 1. The runner records a unique evidence directory and requires actual DPAS results for every case. It checks deterministic random, zero, extreme and alternating inputs against host references. Use this validation mode to investigate missing support without bypassing capability guards. Container-created evidence is made readable to the host user.

## Live host diagnostics

[The eBPF profiler](live_profile.md) provides one-second process-scoped ioctl/wait summaries plus 100 ms DRM fdinfo samples. It has been used on a SYCL GEMV workload. See its scope and units before interpreting the output as device activity.

## Packaging and measurement observation

[Package builder](package_skill.md) creates the standalone skill without private sources or measurements. `observe_xe_exec.sh NEW_DIR -- COMMAND...` records bounded host-wide Xe EXEC ioctl observations and live PID/FD-to-PCI mappings for a command. It requires the installed Xe UAPI header, C compiler, bpftrace and noninteractive sudo. Its observation scope is ioctl submissions, not GPU kernel duration or proof of all possible device activity. The skill-only evaluation summary records its tested use.
