#!/usr/bin/env bash
# Controlled interleaved diagnostic runs. Stop on any failed command, fault or unknown journal access.
set -uo pipefail
root=$(cd "$(dirname "$0")" && pwd)
[[ $# -ge 4 && $1 =~ ^[1-9][0-9]*$ ]] || { echo 'usage: switch_bisect.sh ROUNDS FAMILY... -- CMD...' >&2; exit 2; }
rounds=$1; shift
families=()
while [[ $# -gt 0 && $1 != -- ]]; do families+=("$1"); shift; done
[[ $# -ge 2 && ${#families[@]} -gt 0 ]] || exit 2
shift
out=$(mktemp -d "${TMPDIR:-/tmp}/switch-bisect.XXXXXXXX")
echo "logs: $out"
for ((r=1;r<=rounds;r++)); do
    for i in "${!families[@]}"; do
        IFS=: read -r label assignments <<< "${families[$i]}"
        vars=(); [[ -z ${assignments:-} ]] || IFS=, read -r -a vars <<< "$assignments"
        for assignment in "${vars[@]}"; do
            [[ $assignment =~ ^[a-zA-Z_][a-zA-Z_0-9]*= ]] || { echo 'invalid environment assignment' >&2; exit 2; }
        done
        start=$(date '+%F %T')
        timeout --kill-after=5 "${BISECT_TIMEOUT_SECONDS:-60}" env "${vars[@]}" "$@" >"$out/$r-$i.log" 2>&1
        rc=$?
        count=$("$root/fault_count.sh" "$start"); journal_rc=$?
        printf 'round %s %s: exit %s, journal symptoms %s\n' "$r" "$label" "$rc" "$count"
        if [[ $rc != 0 || $journal_rc != 0 || $count != 0 ]]; then
            echo 'Stopped; inspect evidence before another diagnostic attempt.' >&2; exit 1
        fi
    done
done
