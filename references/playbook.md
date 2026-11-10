# Optimization workflow

Select a permitted [backend](backend-selection.md) and establish its actual build/link/run prerequisites before searching kernel variants. For GEMM/convolution, use the concrete dataflow and algorithm choices in [dense compute](dense-compute.md).

1. **Contract:** write the mathematical operation, dtype/layout/shape domain and numerical tolerance. Identify the deployment latency or throughput objective.
2. **Environment:** select one device, record compiler/runtime/driver and inspect concurrent activity. In a managed runner, use supplied qualification and execution commands. Start a unique evidence directory. Use a cooperative lock and inspect unrelated GPU workloads.
3. **Baseline:** execute an independent reference and the best correct permitted implementation. For standard dense work, start with a qualified native-library control when available. Otherwise establish a portable baseline. Preserve source and build identities.
4. **Diagnosis:** measure the complete path, then profile the suspected stage when access permits. Record unavailable profiling and unverified explanations. Treat microbenchmarks as empirical comparisons. Specification estimates need exact product, precision, dense/sparse convention, units and source, labelled theoretical.
5. **Hypothesis:** predict what one change should improve and which resource it may worsen. Candidate dimensions include tile sizes, subgroup sizes, SLM, GRF, load layout, fusion and matrix instructions.
6. **Correctness:** validate every candidate on the declared domain before timing. Include tails, repeated calls and exceptional values as specified in [testing](testing.md).
7. **Measurement:** interleave control and candidate, retain raw samples, distinguish device time from wall latency and match the intended cache regime. Use fresh runs to confirm a selected winner so search noise does not become a tuning result.
8. **Integration:** when adding dispatch, gate supported features and shape/layout cases, then test the actual branch and fallback. Include packing, workspace and transfers in application measurements.
9. **Evidence:** record passing, failing and skipped paths separately. Report the selected product and software stack. Do not promote a kernel merely because it is faster on a different device.

## Source and evidence separation

The reusable skill contains methods and verified specifications. Keep all measurements and local execution-status claims outside the shared skill. Product specifications need a source and exact SKU, datatype and convention. A measured value belongs in a device-specific run record under `validation/` or an external project results directory.

For each target, run discovery and baseline correctness first, then capability probes, candidate correctness and controlled measurement. Do not carry another product’s timing thresholds, best variants or tuning tables to that hardware as defaults. Public reference projects are indexed in [sources](sources.md); upstream claims do not establish local execution evidence.
