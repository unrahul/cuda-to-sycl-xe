#!/usr/bin/env bash
# Observe all Xe EXEC ioctl submissions during one bounded command (host namespace).
set -euo pipefail
[[ $# -ge 3 && $2 == -- ]] || { echo 'usage: observe_xe_exec.sh NEW_DIR -- COMMAND...' >&2; exit 2; }
out=$1; shift 2
[[ ! -e $out ]] || { echo 'output must be new' >&2; exit 2; }
mkdir -p "$out"; out=$(realpath "$out")
# Derive the type/number from the installed UAPI, ignoring encoded struct size.
cat > "$out/ioctl-number.c" <<'C'
#include <stdio.h>
#include <drm/xe_drm.h>
int main(void){printf("%lu\n",(unsigned long)DRM_IOCTL_XE_EXEC & 65535UL);}
C
cc "$out/ioctl-number.c" -o "$out/ioctl-number"
number=$("$out/ioctl-number")
[[ $number =~ ^[0-9]+$ ]] || exit 2
cat > "$out/monitor.bt" <<BPF
BEGIN { printf("READY\\n"); }
tracepoint:syscalls:sys_enter_ioctl / (args->cmd & 65535) == $number / {
  @xe_exec[pid, args->fd, comm] = count();
}
interval:s:1 { printf("SNAPSHOT\\n"); print(@xe_exec); }
BPF
printf 'utc=%s\nkernel=%s\nioctl_type_number=%s\n' "$(date -u +%FT%TZ)" "$(uname -sr)" "$number" > "$out/provenance.txt"
sha256sum /usr/include/drm/xe_drm.h "$0" >> "$out/provenance.txt"
printf '%q ' "$@" > "$out/command.txt"; printf '\n' >> "$out/command.txt"
tracer=''; supervisor=''; sampler=''
cleanup() {
 if [[ -n $tracer ]]; then sudo -n kill -INT "$tracer" 2>/dev/null || true; fi
 if [[ -n $supervisor ]]; then wait "$supervisor" 2>/dev/null || true; fi
 touch "$out/stop-sampler"
 if [[ -n $sampler ]]; then wait "$sampler" 2>/dev/null || true; fi
}
trap cleanup EXIT
# Record the actual root tracer PID before exec, rather than guessing via pgrep.
timeout -k 5 150 sudo -n bash -c 'printf "%s\n" "$$" > "$1"; exec bpftrace -q -B line "$2"' bash \
 "$out/tracer.pid" "$out/monitor.bt" > "$out/trace.log" 2>&1 &
supervisor=$!
ready=0
for ((i=0;i<200;i++)); do
 if [[ -s $out/tracer.pid ]]; then tracer=$(cat "$out/tracer.pid"); fi
 if rg -q '^READY$' "$out/trace.log"; then ready=1; break; fi
 kill -0 "$supervisor" 2>/dev/null || break
 sleep .05
done
[[ $ready == 1 && $tracer =~ ^[0-9]+$ ]] || { echo 'tracer failed to attach' >&2; exit 1; }
sudo -n python3 "$(dirname "$0")/xe_exec_fdmap.py" "$out" &
sampler=$!
set +e
timeout -k 5 120 "$@" > "$out/workload.log" 2>&1
status=$?
set -e
printf 'workload_exit=%s\n' "$status" >> "$out/provenance.txt"
cleanup; tracer=''; supervisor=''; sampler=''
echo "Observation: $out"
exit "$status"
