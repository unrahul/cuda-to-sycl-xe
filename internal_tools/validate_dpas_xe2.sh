#!/usr/bin/env bash
# Must execute inside a configured SYCL stack; requires actual DPAS execution, no successful skip.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
: "${ONEAPI_DEVICE_SELECTOR:?Select level_zero:0 or level_zero:1}"
case "$ONEAPI_DEVICE_SELECTOR" in level_zero:0|level_zero:1) ;; *) echo 'Use allocated card 0 or 1' >&2; exit 2;; esac
out=$(mktemp -d "$root/validation/dpas-$(date -u +%Y%m%dT%H%M%SZ)-${ONEAPI_DEVICE_SELECTOR//:/-}.XXXXXXXX")
trap 'chmod -R a+rX "$out"' EXIT
printf 'evidence: %s\n' "$out"
exec > >(tee "$out/run.log") 2>&1
icpx --version > "$out/compiler.txt"
printf '%s\n' "$ONEAPI_DEVICE_SELECTOR" > "$out/selector.txt"
sha256sum "$root/examples/gemv_int8/gemv_q8.sycl.cpp" "$root/examples/common/"*.hpp > "$out/source-hashes.txt"
build=(-fsycl -O2 -std=c++20 -I "$root/examples/common")
printf '%q ' icpx "${build[@]}" "$root/examples/gemv_int8/gemv_q8.sycl.cpp" -o "$out/gemv" > "$out/build-command.txt"
timeout -k 5 180 icpx "${build[@]}" "$root/examples/gemv_int8/gemv_q8.sycl.cpp" -o "$out/gemv"
timeout -k 5 120 icpx "${build[@]}" "$root/examples/xe_probe.cpp" -o "$out/xe_probe"
timeout -k 5 30 "$out/xe_probe" --capabilities > "$out/capabilities.txt"
export BENCH_SMOKE=1
for shape in '32 16' '64 32' '2560 10240'; do
    read -r k n <<< "$shape"
    for input_case in random zero extreme alternating; do
        log="$out/$k-$n-$input_case.log"
        printf 'running %s x %s %s\n' "$k" "$n" "$input_case"
        GEMV_CASE=$input_case timeout -k 5 120 "$out/gemv" "$k" "$n" --require-dpas > "$log" 2>&1
        # Guard the harness too: require all 8 output-column counts to have executed DPAS.
        [[ $(grep -c '| dpas_n16 ' "$log") == 8 ]] || { echo 'missing DPAS results' >&2; exit 1; }
        printf '%s %s %s PASS\n' "$k" "$n" "$input_case" >> "$out/status.txt"
    done
done
printf 'DPAS validation PASS\n' >> "$out/status.txt"
