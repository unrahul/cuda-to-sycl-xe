# Validation status and remaining coverage

Reusable instructions contain methods and sourced specifications. Measured values belong to device-specific evidence under `validation/`, never to architecture-wide defaults. Original supplied claims, including prior Xe3P execution claims, are archived in `internal_references/original/`; they have not been independently reproduced here.

Fresh B70 results and exact limitations are recorded in [the verification report](validation/REPORT.md).

## Current repair

The session audit found incomplete compiler feedback, implicit assessment restrictions and missing framework integration examples. The revised entrypoint routes by task, and the new examples cover named-kernel PyTorch integration plus oneMKL and oneDNN. Component checks and limitations are recorded in `validation/lean-release-20261110/`. No improved agent benchmark score is claimed.

The revised managed interface is in `benchmarks/xe-skill-pilot/repair/`. It supplies the same exact contract and free policy validation to both arms, captures compiler output and distinguishes runtime errors. Frozen pilot inputs and results remain unchanged.

## Earlier component qualification

Three skill-only agent workflows, a standalone-package exercise, independent evidence review, a context/statistics acceptance tool and clean local installation are complete for the recorded B70 scope. See [the evaluation summary](validation/agent-evals-20261108/summary.md). Remaining hardware/application coverage below is explicitly outside that release qualification.

The subsequent [community-tool evaluation](validation/skill-optimization-20261108/REPORT.md) preserves mixed initial agent results and focused knowledge checks. Its original grades, corrected publication evidence and feedback-assisted test repairs have separate labels. The Luna campaign and Opus pilot were stopped at the user's request. The Haiku pilot completed six of eight pairs before authentication expired. Its [session audit](validation/skill-session-audit-20261110/REPORT.md) identifies the defects addressed by the current repair. A fresh paired evaluation is still needed to establish agent improvement.

## Current scope

The package supports conversion planning for CUDA and ROCm/HIP, new portable SYCL kernels, capability-gated optimization, host reference validation and evidence collection. The executable examples cover SiLU multiplication and a specific int8 block GEMV format. They are not a general framework migration suite.

Use `internal_tools/run_evidence.sh` after choosing one GPU and configuring the toolchain. The runner stores selected-device and compiler provenance, logs and exit status separately from the skill. Read the latest run record before making a hardware claim.

## Coverage gaps

- Xe-LP, Xe-HPG, Xe-HPC, Xe3 and Xe3P have no fresh execution evidence in this session. Run inventory, portable baseline and supported feature probes when those devices become available.
- The signed-int8 DPAS example is now qualified on B70 cards 0 and 1 with oneAPI 2026.1. See [the DPAS report](validation/DPAS_REPORT.md). Other matrix types, tiles and stacks need independent qualification.
- CUDA and HIP source-device comparisons need corresponding hardware/toolchains and representative source kernels. CPU oracle checks alone do not establish an executed cross-vendor comparison.
- Full dtype, shape, stride and exceptional-value coverage is operation-specific. Supplied smoke cases do not prove arbitrary kernels correct.
- Device-specific AOT, graph, 2D message, GRF and sanitizer paths require separate build/run qualification on the installed stack.
- Smoke timing is diagnostic. No isolated speedup, bandwidth, TOPS or optimal dispatch claim follows from these runs.
- Whole-model numerical fidelity and deployment latency remain the integrating project's responsibility. TensorFold-style layer checks can complement kernel tests but sampled layers/rows are not complete model coverage.

For a new target, keep existing results intact, create a separate run directory, record unsupported paths as SKIP and failures as FAIL, and retain the portable fallback. Only promote tuning choices after repeated correctness and controlled performance measurements on that target.
