#!/usr/bin/env python3
"""Run one development episode with the current skill package and the repaired harness.

    python3 benchmarks/xe-skill-pilot/dev_episode.py L2_030 --arm skill --package dist/build-.../cuda-to-sycl-xe

Not part of any frozen protocol: results go to runs/dev-<task>-<arm>-<UTC>/ and are diagnostic.
Reuses the Haiku pilot's prompt, task pins, environment and fresh-final selection so outcomes are
comparable with runs/haiku55-xhigh-8-v1, while swapping in repair/runner.py + repair/adapter.py.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import campaign_haiku as pilot  # noqa: E402  (prompt, pins and selection of the frozen pilot)
sys.path.insert(0, str(ROOT / 'evaluator'))
import controller as evaluator  # noqa: E402  (the fixed 24-task sample and its hashes)

RUNNERS = {  # agent CLI -> (runner, default model, default effort)
    'claude': (ROOT / 'repair/runner.py', pilot.MODEL, pilot.EFFORT),
    'codex': (ROOT / 'repair/codex_runner.py', 'gpt-6-luna', 'high'),
}
CONTROLLER = ROOT / 'dev_adapter.py'  # repair/adapter.py, optionally on another GPU
PROTOCOL = pilot.RUN / 'protocol.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('task_id')
    parser.add_argument('--arm', choices=('skill', 'control'), default='skill')
    parser.add_argument('--package', type=Path, help='built skill package (required for the skill arm)')
    parser.add_argument('--runner', choices=RUNNERS, default='claude')
    parser.add_argument('--model')
    parser.add_argument('--effort')
    parser.add_argument('--gpu', help='render node for assessments, e.g. renderD130 (default: the frozen renderD129)')
    a = parser.parse_args()
    runner, model, effort = RUNNERS[a.runner]
    a.model, a.effort = a.model or model, a.effort or effort
    if a.arm == 'skill' and not (a.package and (a.package / 'SKILL.md').is_file()):
        parser.error('--package must point at a built skill package')

    protocol = json.loads(PROTOCOL.read_text())
    evaluator.get_task(a.task_id)  # rejects IDs outside the sample and changed sources
    source = ROOT / 'tasks' / f'{a.task_id}.py'

    stamp = pilot.now().replace(':', '').replace('-', '').split('.')[0]
    dest = ROOT / 'runs' / f'dev-{a.runner}-{a.task_id.lower()}-{a.arm}-{stamp}'
    trusted = dest / 'trusted'
    trusted.mkdir(parents=True)
    shutil.copy2(pilot.READY, trusted / 'backend-readiness.json')
    if a.gpu:
        pilot.save(trusted / 'gpu.json', {'render': a.gpu})

    work = Path(tempfile.mkdtemp(prefix=f'xe-dev-{a.task_id.lower()}-{a.arm}-'))
    (work / 'submission').mkdir()
    shutil.copy2(source, work / 'task.py')
    pilot.save(work / 'environment.json', protocol['environment'])
    shutil.copytree(pilot.SDK, work / 'sdk', symlinks=True)
    readonly = ['task.py', 'environment.json', 'sdk']
    if a.arm == 'skill':
        shutil.copytree(a.package, work / 'skill', ignore=shutil.ignore_patterns('__pycache__', 'package-manifest.json'))
        readonly.append('skill')
    (dest / 'prompt.txt').write_text(pilot.prompt(a.task_id, a.arm))
    pilot.save(dest / 'episode.json', {'task_id': a.task_id, 'arm': a.arm, 'workdir': str(work), 'model': a.model,
                                       'effort': a.effort, 'runner': a.runner, 'gpu': a.gpu or 'renderD129', 'package': str(a.package), 'started_utc': pilot.now()})
    print(f'run: {dest}\nworkdir: {work}', flush=True)

    cmd = [sys.executable, str(runner), '--workdir', str(work), '--prompt', str(dest / 'prompt.txt'),
           '--result', str(dest / 'agent'), '--task-id', a.task_id, '--controller', str(CONTROLLER),
           '--run-dir', str(trusted), '--model', a.model, '--effort', a.effort]
    for name in readonly:
        cmd += ['--read-only', name]
    with (dest / 'runner.stdout.log').open('w') as out, (dest / 'runner.stderr.log').open('w') as err:
        launched = subprocess.run(cmd, stdout=out, stderr=err)
    status_path = dest / 'agent/status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'exit': launched.returncode}

    best = pilot.select_assessment(trusted, a.task_id)
    if best:
        _, result_path = best
        r = subprocess.run([sys.executable, str(CONTROLLER), 'assess', '--task-id', a.task_id,
                            '--candidate', str(result_path.parent / 'submission/model_new.py'),
                            '--run-dir', str(trusted), '--phase', 'final'], text=True, capture_output=True)
        (dest / 'final.stdout.log').write_text(r.stdout)
        (dest / 'final.stderr.log').write_text(r.stderr)
        final = json.loads(r.stdout or '{}')
        final['selected_assessment'] = result_path.parent.name
    else:
        final = {'task_id': a.task_id, 'status': 'no_correct_assessment', 'correct': False, 'speedup': None}
    final.update(arm=a.arm, agent_status=status, completed_utc=pilot.now())
    pilot.save(dest / 'final.json', final)
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
