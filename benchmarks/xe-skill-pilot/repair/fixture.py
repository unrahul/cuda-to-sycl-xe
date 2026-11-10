#!/usr/bin/env python3
"""CPU-only real-JIT smoke of build capture and public assessment feedback."""
import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
import adapter

IMAGE=adapter.c.IMAGE

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('verbose','quiet','link','undefined'))
    a=parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        name='xe-repair-cpu-'+uuid.uuid4().hex[:12]
        cmd=['docker','run','--rm','--name',name,'--network=none','--cpus=4','--memory=8g','--memory-swap=8g',
             '--pids-limit=512','--cap-drop=ALL','--security-opt=no-new-privileges',
             '--user',f'{__import__("os").getuid()}:{__import__("os").getgid()}',
             '-e','HOME=/tmp','-e','MAX_JOBS=2','-v',f'{adapter.HERE}:/repair:ro','-v',f'{tmp}:/results:rw',
             '--entrypoint','python',IMAGE,'/repair/fixture_worker.py',a.mode]
        try:
            proc=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
        finally:
            subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
        record={'task_id':'CPU_FIXTURE','status':'failed','correct':False,'errors':[{'category':'worker_error','message':'private sentinel'}]}
        result=adapter.public(record,Path(tmp))
        print(json.dumps(result))
        expected = {'quiet':'missing_fixture_symbol','verbose':'missing_fixture_symbol','link':'cannot find -lrepair_nonexistent_library','undefined':'undefined reference to `missing_link_fixture()'}[a.mode]
        message = result.get('errors',[{}])[0].get('message','')
        return 0 if proc.returncode==0 and result['status']=='candidate_build_failed' and expected in message else 1
if __name__=='__main__': raise SystemExit(main())
