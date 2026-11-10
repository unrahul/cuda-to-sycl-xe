#!/usr/bin/env bash
# Maintainer-only hardware evidence; set up oneAPI and select exactly one Level Zero device.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
: "${ONEAPI_DEVICE_SELECTOR:?Set ONEAPI_DEVICE_SELECTOR to a single level_zero device index}"
case "$ONEAPI_DEVICE_SELECTOR" in level_zero:[0-9]) ;; *) echo 'Select one level_zero GPU by index' >&2; exit 2;; esac
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$root/validation"
dir=$(mktemp -d "$root/validation/${stamp}-${ONEAPI_DEVICE_SELECTOR//:/-}.XXXXXXXX")
{
    printf 'UTC: %s\nselector: %s\nhost kernel: ' "$(date -u +%FT%TZ)" "$ONEAPI_DEVICE_SELECTOR"
    uname -srmo
    printf '\nCompiler:\n'; timeout 15 icpx --version
    printf '\nBuild: examples/build.sh default JIT flags: -fsycl -O2 -std=c++20 -I common -Wno-deprecated-declarations\n';
    printf '\nSource hashes:\n'; (cd "$root" && find SKILL.md TODO.md references examples scripts tests internal_tools -type f ! -path 'examples/bin/*' ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum);
    printf '\nPCI display devices:\n'; if command -v lspci >/dev/null; then lspci -nn | rg -i 'vga|display|3d'; else echo 'lspci unavailable'; fi
} > "$dir/provenance.log" 2>&1
printf 'capabilities.log records selected runtime device, PCI device ID, driver, and feature queries.\n' >> "$dir/provenance.log"
run() {
    local label=$1 limit=$2 status
    shift 2
    printf '%s\n' "running $label (timeout ${limit}s)"
    printf '%q ' timeout --kill-after=5 "$limit" "$@" >> "$dir/commands.txt"
    printf '\n' >> "$dir/commands.txt"
    timeout --kill-after=5 "$limit" "$@" > "$dir/$label.log" 2>&1
    status=$?
    printf '%s exit=%d\n' "$label" "$status" | tee -a "$dir/status.txt"
    return "$status"
}
status=0
run smoke 300 env SMOKE_LOG="$dir/smoke-detail.log" "$root/scripts/smoke.sh" || status=1
if [[ $status -eq 0 ]]; then
    run capabilities 30 "$root/examples/bin/xe_probe" --capabilities || status=1
fi
if [[ $status -eq 0 ]]; then
    run gemv_original_shape 180 env BENCH_SMOKE=1 "$root/examples/bin/gemv_q8" 2560 10240 || status=1
fi
printf 'Evidence: %s\n' "$dir"
exit "$status"
