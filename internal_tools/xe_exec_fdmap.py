#!/usr/bin/env python3
"""Resolve the observed Xe EXEC pid/fd keys while their processes still exist."""
import json,re,sys,time
from pathlib import Path
out=Path(sys.argv[1]); seen=set(); offset=0
with (out/'fd-mapping.jsonl').open('w',buffering=1) as log:
    while True:
        with (out/'trace.log').open() as trace:
            trace.seek(offset);lines=trace.readlines();offset=trace.tell()
        for line in lines:
            m=re.match(r'@xe_exec\[(\d+), (\d+), (.+)\]: (\d+)',line)
            if not m:continue
            pid,fd,name,count=m.groups()
            try:
                text=Path(f'/proc/{pid}/fdinfo/{fd}').read_text()
                fields=dict(line.split(':',1) for line in text.splitlines() if ':' in line)
                pci=fields.get('drm-pdev','').strip();client=fields.get('drm-client-id','').strip()
                key=(pid,fd,pci,client)
                if pci and key not in seen:
                    log.write(json.dumps(dict(pid=int(pid),fd=int(fd),name=name,pci=pci,client=client,observed_exec_count=int(count),monotonic_ns=time.monotonic_ns()))+'\n');seen.add(key)
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
        if (out/'stop-sampler').exists():break
        time.sleep(.1)
