#!/usr/bin/env bash
# Bounded host syscall profile plus sampled DRM fdinfo for one SYCL process.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
if [[ $# -lt 3 || $2 != -- ]]; then
    echo "usage: ONEAPI_DEVICE_SELECTOR=level_zero:N $0 OUTPUT_DIR -- COMMAND [ARGS...]" >&2
    exit 2
fi
: "${ONEAPI_DEVICE_SELECTOR:?Select one level_zero device}"
case "$ONEAPI_DEVICE_SELECTOR" in level_zero:[0-9]) ;; *) echo 'Select one level_zero device' >&2; exit 2;; esac
out=$1; shift 2
if [[ -d $out && -n $(ls -A "$out") ]]; then
    echo "output directory already has files: $out" >&2
    exit 2
fi
mkdir -p "$out"
out=$(realpath "$out")
target=$(realpath "$1"); shift
[[ -x $target ]] || { echo "target is not executable: $target" >&2; exit 2; }
command -v bpftrace >/dev/null
sudo -n true
limit=${LIVE_PROFILE_TIMEOUT:-60}
[[ $limit =~ ^[1-9][0-9]*$ ]] || { echo 'LIVE_PROFILE_TIMEOUT must be positive integer seconds' >&2; exit 2; }

printf 'UTC=%s\nselector=%s\ntarget=%s\nbpftrace=%s\nkernel=%s\n' \
    "$(date -u +%FT%TZ)" "$ONEAPI_DEVICE_SELECTOR" "$target" "$(bpftrace --version)" "$(uname -srmo)" > "$out/provenance.txt"
printf 'argv:' >> "$out/provenance.txt"; printf ' %q' "$target" "$@" >> "$out/provenance.txt"; printf '\n' >> "$out/provenance.txt"
printf 'source hashes:\n' >> "$out/provenance.txt"
sha256sum "$root/internal_tools/live_profile.sh" "$root/internal_tools/live_profile.bt" \
    "$root/internal_tools/live_profile_target.sh" "$root/internal_tools/live_profile_fdinfo.py" \
    "$target" >> "$out/provenance.txt"

# The target starts as the invoking user with their existing supplementary groups.
# It stops before oneAPI loads; attach completes before we resume it.
printf '%q ' /usr/bin/bash "$root/internal_tools/live_profile_target.sh" \
    "$out/pid" "$out/setup.log" "$ONEAPI_DEVICE_SELECTOR" "$target" "$@" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
rm -f "$out/pid"
timeout --kill-after=5s "${limit}s" /usr/bin/bash \
    "$root/internal_tools/live_profile_target.sh" "$out/pid" "$out/setup.log" "$ONEAPI_DEVICE_SELECTOR" "$target" "$@" \
    > "$out/target.log" 2>&1 &
target_supervisor=$!
pid=''; tracer=''; sampler=''; active=1
cleanup() {
    if [[ $active -eq 1 ]]; then
        [[ -n $pid ]] && kill -CONT "$pid" 2>/dev/null || true
        [[ -n $pid ]] && kill -TERM "$pid" 2>/dev/null || true
        [[ -n $tracer ]] && kill -TERM "$tracer" 2>/dev/null || true
        [[ -n $sampler ]] && kill -TERM "$sampler" 2>/dev/null || true
        kill -TERM "$target_supervisor" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM
for ((i=0; i<200; i++)); do
    [[ -s $out/pid ]] && { pid=$(<"$out/pid"); break; }
    kill -0 "$target_supervisor" 2>/dev/null || break
    sleep 0.05
done
[[ $pid =~ ^[0-9]+$ ]] || { echo "target did not reach attach point; see $out/target.log" >&2; exit 1; }
printf 'profile_pid=%s\n' "${pid:-unavailable}" >> "$out/provenance.txt"
timeout --kill-after=5s "${limit}s" sudo -n bpftrace -q -B line --no-warnings -p "$pid" \
    "$root/internal_tools/live_profile.bt" -- "$pid" > "$out/trace.log" 2>&1 &
tracer=$!
attached=0
for ((i=0; i<200; i++)); do
    if rg -q "^PROFILE_PID $pid$" "$out/trace.log" 2>/dev/null; then attached=1; break; fi
    kill -0 "$tracer" 2>/dev/null || break
    sleep 0.05
done
[[ $attached -eq 1 ]] || { echo "eBPF attach failed; see $out/trace.log" >&2; exit 1; }
python3 "$root/internal_tools/live_profile_fdinfo.py" "$pid" "$out/drm-fdinfo.log" 600 &
sampler=$!
kill -CONT "$pid"

wait "$tracer" || tracer_status=$?
wait "$target_supervisor" || target_status=$?
kill -TERM "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
active=0
printf 'tracer_exit=%s\ntarget_exit=%s\n' "${tracer_status:-0}" "${target_status:-0}" >> "$out/provenance.txt"
if [[ ! -s $out/drm-fdinfo.log ]]; then
    printf 'No DRM fdinfo counters sampled for the launched process.\n' > "$out/drm-fdinfo.log"
fi
if [[ ${tracer_status:-0} -ne 0 ]]; then
    echo "profiler failed (exit ${tracer_status}); see $out/trace.log" >&2
    exit "${tracer_status}"
fi
if [[ ${target_status:-0} -ne 0 ]]; then
    echo "target failed (exit ${target_status}); see $out/target.log" >&2
    exit "${target_status}"
fi
echo "Profile: $out"
