#!/usr/bin/env python3
"""Sample DRM fdinfo for one PID; supplemental to the eBPF syscall trace."""
import glob
import os
import sys
import time

pid, output, max_samples = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
base = f"/proc/{pid}/fdinfo"
with open(output, "w", encoding="utf-8", buffering=1) as log:
    for _ in range(max_samples):
        if not os.path.isdir(base):
            break
        stamp = time.monotonic_ns()
        for path in glob.glob(f"{base}/*"):
            try:
                with open(path, encoding="utf-8") as info:
                    for line in info:
                        if line.startswith("drm-"):
                            log.write(f"{stamp} fd={os.path.basename(path)} {line}")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        time.sleep(0.1)
