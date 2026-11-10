#!/usr/bin/env bash
# Run from any directory after oneAPI environment setup; select exactly one GPU.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
: "${ONEAPI_DEVICE_SELECTOR:?Set ONEAPI_DEVICE_SELECTOR to one specific GPU, e.g. level_zero:3}"
case "$ONEAPI_DEVICE_SELECTOR" in level_zero:[0-9]) ;; *) echo 'Select one level_zero GPU by index' >&2; exit 2;; esac
mkdir -p "$root/validation"
log="${SMOKE_LOG:-$(mktemp "$root/validation/smoke.XXXXXXXX.log")}"
exec > >(tee "$log") 2>&1
export BENCH_SMOKE=1
printf 'selector=%s\n' "$ONEAPI_DEVICE_SELECTOR"
timeout 180 "$root/examples/build.sh"
timeout 60 icpx -fsycl -O2 -std=c++20 -I "$root/examples/common" "$root/tests/parity_test.cpp" -o "$root/examples/bin/parity_test"
timeout 30 "$root/examples/bin/parity_test"
timeout 120 icpx -fsycl -O2 -std=c++20 -I "$root/examples" "$root/tests/snippets_test.cpp" -o "$root/examples/bin/snippets_test"
timeout 60 "$root/examples/bin/snippets_test"
timeout 120 icpx -fsycl -O2 -std=c++20 -I "$root/examples" -Xspirv-translator -spirv-ext=+SPV_INTEL_2d_block_io \
    "$root/tests/block2d_test.cpp" -o "$root/examples/bin/block2d_test"
set +e; timeout 60 "$root/examples/bin/block2d_test"; status=$?; set -e
[[ $status == 0 || $status == 77 ]] || { echo "block2d_test failed: $status" >&2; exit 1; }
timeout 60 "$root/examples/bin/portable_gemv"
timeout 60 "$root/examples/bin/silu_mul" 3 67
timeout 120 "$root/examples/bin/gemv_q8" 32 257
set +e
timeout 30 "$root/examples/bin/gemv_q8" 33 16
status=$?
set -e
if [[ $status -ne 2 ]]; then echo "invalid GEMV shape returned $status, expected 2" >&2; exit 1; fi
timeout 30 g++ -std=c++17 -I "$root/examples/common" "$root/tests/matrix_shape_test.cpp" -o "$root/examples/bin/matrix_shape_test"
"$root/examples/bin/matrix_shape_test"
python3 -m unittest discover -s "$root/tests" -v
printf 'smoke passed\n'
