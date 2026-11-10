# Quantized and low-bit compute

Specify the stored format and the arithmetic path separately. `W4A16` describes operand storage widths, not the multiply instruction or accumulator. Integer W8A8 and floating-point FP8 W8A8 are different contracts. Read [backend selection](backend-selection.md) before choosing an implementation.

## Format contract

| Family | Required distinctions |
|---|---|
| INT8 W8A8 | Signed/unsigned operands, symmetric/asymmetric codes, activation and weight scale axes, zero points, accumulator width and output conversion |
| INT4 W4A16 | Signed two's-complement or unsigned nibble codes, packed word/nibble order, group axis/size, scales/zeros, activation type and dequantization precision |
| INT4 W4A8 | All INT4 packing rules plus activation quantization, activation zero points/scales and compensation through the dot product |
| FP8 | Exact E4M3/E5M2 variant and special values, scale direction/granularity, saturation/rounding, operand/accumulator/output types and any conversion before multiplication |
| OCP MXFP4 / MXFP8 | The MX specification defines 32-element blocks with E8M0 scales. MXFP4 elements are E2M1; MXFP8 uses an explicit FP8 element format. Preserve block axes and physical scale layout. |
| NVFP4 | E2M1 elements with E4M3 block scales and a global FP32 scale. A common recipe has 16-element blocks; recipes can also use 2D weight scaling. Preserve the exact checkpoint/API recipe and scale swizzle. |

MXFP4, NVFP4 and integer INT4 are not interchangeable byte formats. A byte tensor containing two nibbles does not identify signedness or numerical interpretation. Trace packer, kernel and reference together. Storage may additionally be transposed, interleaved, VNNI-packed or permuted by activation-order metadata.

For asymmetric integer codes, the unscaled dot product is `sum((a-z_a)*(w-z_w))`. Expanding it requires `sum(a*w) - z_w*sum(a) - z_a*sum(w) + K*z_a*z_w`, applied with the correct per-group parameters and scales. Bound both dot-product and compensation accumulators. An unsigned four-bit zero point cannot always be folded into a signed four-bit value without overflow.

## Choose the actual path

- Native oneDNN/oneMKL: verify the exact routine, installed implementation, operand descriptors, supported quantization attributes and target. Library-family support is not a promise that every low-bit format or scale mode exists. Include packing, reorders, scratch and output conversion in the appropriate lifecycle costs.
- SYCL-TLA: distinguish native integer/FP8/FP4 matrix atoms, block-scaled atoms, and paths that convert storage to FP16/BF16 fragments before MMA. Match the target-specific atom, copy layout, scale layout, alignment and adapter constraints. A template declaration alone does not enable an instruction on a different device.
- Custom fused dequantization: W4A16 can unpack and dequantize weights into registers at activation precision, then perform higher-precision arithmetic. Reuse decoded fragments across outputs. Compare this with native low-bit arithmetic without calling both paths the same precision.
- Workload regime: small-row decode/GEMV, dense prefill and uneven MoE groups have different reuse and scheduling limits. Account for empty experts, routing, padding and output ownership. Keep format conversion and reusable weight preparation distinct from per-call activation quantization.

Names are not contracts. A flag named `use_fp8_w8a8`, an `sgemm` function name or a Python schema does not identify the computation that runs on the XPU; follow the registered backend, the dispatched kernel and its dtype checks. Grouped MXFP4 W4A16 and dense block-scaled MXFP4 W4A4 use different arithmetic and scale representations.

For every Xe, Xe2 or Xe3P variant, check build/compiler support, exact runtime device, operation/dtype/scale restrictions and the selected instruction path. Retain a checked fallback. Do not copy a target-specific eligibility-check bypass unless all its padding, alignment and feature invariants have been independently established.

## Validate representation and arithmetic separately

1. Use tiny hand-constructed values covering both nibbles, signs, zeros, scale-block boundaries, asymmetric zero points and nontrivial strides. Independently verify pack/unpack, dequantization and every transpose/reorder before testing GEMM.
2. Compare the kernel with an independent reference decoded from the same represented quantized inputs and scale values. For integer accumulation, use a host width that safely covers the mathematical range. For floating accumulation, justify the tolerance and preserve specified rounding points.
3. Separately measure quantization error against original high-precision inputs. Correctly implementing a quantized operation does not establish that the quantization itself preserves model quality. Converting MXFP4 to NVFP4 or another recipe requires an authorized numerical contract.
4. Exercise tails, padded scale surfaces, extreme/zero/nonfinite scales as allowed, saturation boundaries, per-token/channel/group modes, bias, experts with no rows, repeated calls and dependent streams. Keep pristine oracle inputs and check untouched storage/output lifetime.
5. Trace the executed backend when an API may emulate or convert the format. Measure the complete required quantize/pack/compute/epilogue path. Keep numerical gates and timing boundaries appropriate to this operation, not inherited from an unrelated low-precision test.

Primary format definitions: [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), [NVFP4 recipe](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html). 
