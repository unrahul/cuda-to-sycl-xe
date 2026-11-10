# Kernel project instructions

Replace the bracketed fields before using this template.

## Contract

- Operation and format definition: [formula and independent reference path]
- Supported inputs: [dtypes, shapes, strides, alignment, aliasing, tails]
- Numerical policy: [accumulation, rounding, overflow, NaN/Inf, predeclared tolerances]
- Objective: [deployment latency/throughput and relevant workload]

## Commands and evidence

- Build: [command, toolchain/container image and flags]
- Device inventory and selection: [command and explicit selector]
- Baseline and candidate correctness: [commands]
- Controlled paired measurement: [command, warmup, samples, cache regime]
- Dispatch/fallback regression: [command]
- Results: [unique run directory containing provenance, raw output, status and samples]
- Hardware access: [cooperative lock and known workloads sharing the device]

For a Bash runner, use an explicit interpreter, for example `flock -w TIMEOUT LOCKFILE bash run.sh`. This avoids relying on implicit shell selection or nested command-string quoting.

## Agent workflow

Read repository instructions and prefer available graph tools for discovery. Implement an independent reference and portable baseline first. Record each optimization hypothesis, validate before timing and confirm improvements with fresh interleaved measurements. Query capabilities before specializing. Preserve a supported fallback and validate the integrated dispatch.

Store measured results separately from reusable instructions. Specifications require exact product and source citations. Mark untested architectures and skipped paths explicitly. Do not copy tuning constants across devices based on a generation label.

Stop a workload after device loss and save diagnostics. Do not stop unrelated services, reset hardware, or change driver configuration as part of an ordinary measurement. Unknown journal access is not a clean fault report. Use bounded runs and surface asynchronous errors.

## Completion

Deliver code, reference tests, reproducible commands, observed pass/fail/skip results and hardware coverage. Report unresolved correctness failures and measurement limitations. Check licensing and provenance before incorporating archived upstream code.
