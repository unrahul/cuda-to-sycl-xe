#!/usr/bin/env bash
# build.sh [AOT_DEVICE]: build examples with icpx. No argument produces JIT binaries.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
flags=(-fsycl -O2 -std=c++20 -I common -Wno-deprecated-declarations)
if [[ $# -gt 1 ]]; then echo 'usage: build.sh [AOT_DEVICE]' >&2; exit 2; fi
if [[ $# -eq 1 ]]; then flags+=(-fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device $1"); fi
if [[ ${XE_PROBE_2D:-0} == 1 ]]; then flags+=(-DXE_PROBE_2D=1); fi
icpx "${flags[@]}" xe_probe.cpp -o bin/xe_probe
icpx "${flags[@]}" elementwise/silu_mul.sycl.cpp -o bin/silu_mul
icpx "${flags[@]}" gemv_int8/gemv_q8.sycl.cpp -o bin/gemv_q8
icpx "${flags[@]}" portable_gemv.cpp -o bin/portable_gemv
printf '%s\n' 'built: bin/xe_probe bin/silu_mul bin/gemv_q8'
