#!/usr/bin/env python3
"""Revised trusted controller and free static-policy check."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parent
EVALUATOR = HERE.parent / 'evaluator'
sys.path.insert(0, str(EVALUATOR))
import controller as c
import worker as w

# Only build-scoped, error-bearing lines are disclosed. Paths are stripped and
# no source excerpts, command lines, Python traceback, or reference output pass.
def diagnostics(text):
    lines = []
    for line in text.splitlines():
        if not re.search(r'(?:\berror:|\bfatal error:|cannot find -l|undefined reference to|library not found for|ninja: build stopped)', line, re.I):
            continue
        line = re.sub(r'(?:/[\w. +@~=-]+)+', '[path]', line)
        line = re.sub(r'\b(?:seed|oracle)\b.*', '[redacted]', line, flags=re.I)
        line = line.strip()[:500]
        if line and line not in lines:
            lines.append(line)
    return '\n'.join(lines[:20])[:6000]

def public(record, output):
    result = c.public_result(record)
    # Candidate import exceptions are untrusted and may embed arbitrary values.
    # Retain only a bare missing-library name, never the exception or traceback.
    for error in result['errors']:
        if error.get('category') == 'candidate_load':
            match = re.search(r'(?<![\w/])([\w.+-]+\.so(?:\.[0-9]+)*): cannot open shared object file', error.get('message',''))
            error['message'] = ('Missing shared library: ' + match.group(1)) if match else 'Candidate failed to load.'
    build = output / 'build-diagnostics.private'
    all_errors = ' '.join(str(e.get('message','')) for e in record.get('errors',[]) if isinstance(e,dict))
    token = re.search(r'\bUR_RESULT_ERROR_[A-Z0-9_]+\b', all_errors)
    if not record.get('correct') and token:
        result['errors'] = [{'category':'candidate_runtime', 'message':'Device runtime error: '+token.group(0)}]
        result['status'] = 'candidate_runtime_failed'
    elif not record.get('correct') and build.exists() and (output / 'build-terminal.flag').exists():
        msg = diagnostics(build.read_text(encoding='utf-8', errors='replace'))
        result['errors'] = [{'category':'candidate_build', 'message':msg or 'Extension compilation failed; no compiler diagnostic was captured.'}]
        result['status'] = 'candidate_build_failed'
    return result

def policy(candidate):
    legacy = w.load_legacy()
    try:
        result = legacy.audit_candidate(candidate)
        return {'status':'passed','backend':result['backend']}
    except Exception as error:
        # The scanner produces candidate-only messages. Remove host paths.
        message = str(error).replace(str(Path(candidate).resolve().parent), '[submission]')
        return {'status':'candidate_policy_rejected','errors':[{'category':'candidate_policy','message':message[:3000]}]}

def launch(task, mode, output, submission=None, phase='baseline', task_source=None):
    old = c.command
    def command(*args, **kwargs):
        cmd = old(*args, **kwargs)
        image_index = cmd.index('--entrypoint')
        cmd[image_index:image_index] = ['-v', f'{HERE}:/repair:ro']
        cmd[cmd.index('/evaluator/worker.py')] = '/repair/worker.py'
        return cmd
    c.command = command
    try:
        return c.launch(task, mode, output, submission, phase, task_source)
    finally:
        c.command = old

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('validate','assess'))
    p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--task-id')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--phase',choices=('assessment','final'),default='assessment')
    a=p.parse_args(argv)
    candidate=a.candidate.absolute()
    if candidate.name!='model_new.py' or candidate.is_symlink() or not candidate.is_file():
        print(json.dumps({'status':'candidate_policy_rejected','errors':[{'category':'candidate_policy','message':'candidate must be a regular file named model_new.py'}]})); return 1
    if a.action=='validate':
        with tempfile.TemporaryDirectory() as tmp:
            try:
                staged=c.snapshot(candidate,Path(tmp)/'submission')
                result=policy(staged/'model_new.py')
            except Exception:
                result={'status':'candidate_policy_rejected','errors':[{'category':'candidate_policy','message':'Submission snapshot failed: check file types, symlinks and size limits.'}]}
        print(json.dumps(result));return 0 if result['status']=='passed' else 1
    if not a.task_id or not a.run_dir: p.error('assess requires --task-id and --run-dir')
    fixture = a.task_id == 'PREFLIGHT_ONLY'
    task_source = EVALUATOR/'fixtures/problem.py' if fixture else None
    task = {'id':'PREFLIGHT_ONLY','sha256':c.digest(task_source)} if fixture else c.get_task(a.task_id)
    run=a.run_dir.resolve()
    c.require_backend_readiness(run)
    output=c.reserve(run,task['id'],a.phase,candidate)
    output.mkdir(parents=True,exist_ok=False)
    submission=c.snapshot(candidate,output/'submission')
    result=launch(task,'evaluate',output,submission,a.phase,task_source=task_source)
    feedback=public(result,output)
    if fixture: feedback.update(scored=False,qualification_fixture=True)
    print(json.dumps(feedback))
    return 0 if result.get('correct') else 1

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status':'controller_error','correct':False,'error_category':type(error).__name__}))
        raise SystemExit(2)
