#!/usr/bin/env bash
# Stop before loading oneAPI so the parent can attach eBPF to this exact PID.
set -euo pipefail
pid_file=$1; setup_log=$2; selector=$3; shift 3
printf '%s\n' "$$" > "$pid_file"
kill -STOP "$$"
set +eu
source /opt/intel/oneapi/setvars.sh > "$setup_log" 2>&1
setup_status=$?
set -eu
if [[ $setup_status -ne 0 ]]; then
    echo "oneAPI setup failed (exit $setup_status); see $setup_log" >&2
    exit "$setup_status"
fi
export ONEAPI_DEVICE_SELECTOR="$selector"
exec "$@"
