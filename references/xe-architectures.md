# Architecture policy

“Xe” names a family, not a capability level. Record the exact product and runtime-reported capabilities. Xe-LP, Xe-HPG, Xe-HPC, Xe2, Xe3 and Xe3P are distinct targets. Do not substitute Xe3 documentation for Xe3P.

| Target | Starting path | Required specialization evidence |
|---|---|---|
| Xe-LP or unknown Intel GPU | portable scalar/vector SYCL | queried subgroup sizes, work-group/resource limits, USM support |
| Xe-HPG / Arc A | portable baseline | compiler/device matrix combinations and exact subgroup/tile pairing |
| Xe-HPC | portable baseline | its own capability record and measurements; never inherit Arc constants |
| Xe2 / Arc B, including Arc Pro B70 | portable baseline, then measured candidates | actual selected device, subgroup widths, matrix combinations, memory and kernel limits |
| Xe3 | portable baseline | separate sources and runtime tests |
| Xe3P | portable baseline if runtime supports the device | exact hardware/compiler evidence before enabling any specialization |

## Probe rather than infer

Upstream target names can differ from product names. For the D270/Crescent Island ecosystem, inspect CRI, Xe35 and Xe3p-XPC target guards alongside the [Intel D-series product specification](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/d-series.html). Distinguish these from Xe3 and client Xe3P variants. Read the installed library's release support and per-operation restrictions, including [oneMKL release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/onemkl/2026.html). A product announcement, source branch or merged patch alone does not authorize every dtype, scale mode or instruction.

Record `name`, `vendor`, `driver_version`, `sub_group_sizes`, `max_work_group_size`, `max_work_item_sizes`, `local_mem_size`, `global_mem_size`, allocation limits and USM aspects. Record `fp16`, `fp64`, `atomic64` and applicable matrix extension queries. Query kernel-specific work-group/resource limits after building where available. The device maximum alone does not make a particular kernel launch legal.

A supported subgroup width does not establish the best width. A matrix aspect does not establish support for every input type, accumulator, tile, layout or subgroup combination. Check the extension version and combination fields against the installed headers and run a small known-pattern tile test. GRF modes and 2D block memory operations are compiler extensions: use separate compile and run probes and enforce their alignment, pitch and surface constraints.

Do not equate `max_compute_units` with Xe-core count across drivers. Do not infer hardware FP64 throughput from `aspect::fp64`; emulation and runtime configuration may affect exposure. Use host FP64 for reference calculations regardless of device FP64.

## Dispatch

Prefer explicit, sourced architecture/device identification, but keep identification separate from feature permission. Unknown IDs use the portable path. Broad PCI ranges and name substrings are hints, not proof of instruction support. An override may select an experiment only after the same capability guards as ordinary dispatch.

Tuning tables are scoped to the actual tested product and stack. A candidate measured on one product is not automatically optimal on other products in its generation. Preserve existing validated defaults unless a deliberate change passes their regression gates. Record target-specific execution coverage in the integrating project; never fill missing tuning rows from another product.

Consult the pinned public [source catalog](sources.md) for primary specifications and implementation references.

## Investigate missing support on available hardware

A false matrix aspect or empty combination query describes the selected software stack's exposure. When the target is expected to support matrix operations, inspect the compiler, runtime, driver and installed headers, then try an available supported stack and repeat the query and a small required-execution correctness test. Keep capability guards intact. Record both stack outcomes before attributing the result to hardware.

Check bounded and fixed dimensions independently in matrix combinations, including maximum sizes and accumulator/output types. The example helper `examples/common/matrix_shape.hpp` handles the mixed representation (such as bounded M with fixed N and K).
