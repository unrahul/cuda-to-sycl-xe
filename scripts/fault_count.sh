#!/usr/bin/env bash
# Count journal symptoms in a time window; unavailable journal access is UNKNOWN.
set -uo pipefail
[[ $# == 1 ]] || { echo 'usage: fault_count.sh SINCE' >&2; exit 2; }
log=$(mktemp); trap 'rm -f "$log"' EXIT
if ! timeout --kill-after=2 10 sudo -n journalctl -k --since "$1" --no-pager >"$log" 2>/dev/null; then
    echo UNKNOWN; exit 2
fi
awk '/Hardware Error|Engine reset|GPU HANG/ {n++} END {print n+0}' "$log"
