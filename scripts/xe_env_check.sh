#!/usr/bin/env bash
# xe_env_check.sh - one report of the SYCL toolchain, Intel GPUs and the selected device's capabilities.
# Run after: source /opt/intel/oneapi/setvars.sh
# Optional: ONEAPI_DEVICE_SELECTOR=level_zero:N to describe one GPU (default: the runtime's default GPU).
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)

command -v icpx >/dev/null || { echo 'icpx not found: source /opt/intel/oneapi/setvars.sh first (before set -u)' >&2; exit 2; }

echo '## Compiler'
icpx --version | head -1

echo; echo '## GPUs (Level Zero)'
sycl-ls 2>/dev/null | grep 'level_zero:gpu' || echo 'none visible: check /dev/dri access and the render group'
echo "selector: ${ONEAPI_DEVICE_SELECTOR:-unset (export ONEAPI_DEVICE_SELECTOR=level_zero:N to pick one GPU)}"

echo; echo '## Selected device'
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
if icpx -fsycl -O1 -std=c++20 -I "$root/examples/common" "$root/examples/xe_probe.cpp" -o "$tmp/xe_probe" 2>"$tmp/build.log"; then
    "$tmp/xe_probe" --capabilities
else
    echo 'xe_probe failed to build:'; head -5 "$tmp/build.log"
fi

echo; echo '## Libraries'
echo "oneMKL: ${MKLROOT:-not configured}  (link with -qmkl)"
echo "oneDNN: ${DNNLROOT:-${DNNL_ROOT:-not configured}}"
python3 - <<'EOF' 2>/dev/null || echo 'torch: not importable'
import inspect
import torch
from torch.utils import cpp_extension
sycl = 'with_sycl' in inspect.signature(cpp_extension.load).parameters
print(f'torch {torch.__version__}: xpu available={torch.xpu.is_available()}, '
      f'SYCL extensions={sycl}, devices={torch.xpu.device_count()}')
EOF
