#!/usr/bin/env python3
"""One selected kernel, two fresh Luna contexts, repaired assessment interface."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/luna-high-L2_030-lean-v1'
INPUTS = RUN / 'inputs'
RELEASE = ROOT.parents[1] / 'dist/lean-repair-20261110/cuda-to-sycl-xe'
TASK = 'L2_030'


def engine():
    spec = importlib.util.spec_from_file_location('_paired_engine', ROOT / 'campaign_haiku.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_prompt = module.prompt
    module.ROOT, module.RUN = INPUTS, RUN
    module.CONTROLLER = ROOT / 'repair/adapter.py'
    module.RUNNER = ROOT / 'repair/codex_runner.py'
    module.MODEL, module.EFFORT = 'gpt-6-luna', 'high'
    module.prompt = lambda task, arm: original_prompt(task, arm).replace(
        'Read skill/SKILL.md and its linked local references before implementing.',
        'Read skill/SKILL.md and follow its task-relevant routes.')
    return module


E = engine()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (RUN / 'protocol.json').exists():
        return json.loads((RUN / 'protocol.json').read_text())
    sample = json.loads((ROOT / 'sample.json').read_text())
    task = next(x for x in sample['selected'] if x['id'] == TASK)
    (INPUTS / 'tasks').mkdir(parents=True)
    shutil.copy2(ROOT / 'tasks' / (TASK + '.py'), INPUTS / 'tasks' / (TASK + '.py'))
    shutil.copytree(RELEASE, INPUTS / 'bundle/cuda-to-sycl-xe',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'bin'))
    packet = INPUTS / 'bundle/cuda-to-sycl-xe'
    files = {str(p.relative_to(packet)): sha(p) for p in packet.rglob('*') if p.is_file()}
    E.save(INPUTS / 'bundle/manifest.json', {'files': files})
    environment = json.loads(subprocess.check_output(
        [sys.executable, str(ROOT / 'evaluator/controller.py'), 'metadata'], text=True))
    environment.update(task_count=1, selected_task=TASK)
    sources = [Path(__file__).resolve(), ROOT / 'campaign_haiku.py',
               ROOT / 'evaluator/controller.py', ROOT / 'evaluator/worker.py', E.READY,
               E.SDK / 'manifest.json', INPUTS / 'tasks' / (TASK + '.py'),
               INPUTS / 'bundle/manifest.json', Path(shutil.which('codex')).resolve()]
    sources += [ROOT / 'repair' / name for name in
                ('codex_runner.py', 'luna-config-native.toml', 'adapter.py', 'worker.py',
                 'build_capture.py', 'CONTRACT.md')]
    protocol = {'name': 'L2_030 selected-kernel paired diagnostic retest',
                'frozen_utc': E.now(), 'model': E.MODEL, 'effort_requested': E.EFFORT,
                'tasks_per_arm': 1, 'arms': ['control', 'skill'],
                'schedule': [{'task_id': TASK, 'arm': arm} for arm in ('control', 'skill')],
                'sampling': {'selected': [task], 'selection': 'User-selected prior regression; no random-sample claim'},
                'environment': environment, 'skill_files': files,
                'pins': {str(path): sha(path) for path in sources},
                'limits': {'assessments': 6, 'fresh_final': 1, 'wall_seconds': 7200,
                           'inference_completions': 120, 'free_policy_checks': True},
                'isolation': 'Fresh /tmp workdir and Codex home per arm; no web, delegation, other skills or repository access; SDK/task/skill read-only',
                'monitoring': 'Observer reads tools/feedback/results; no coaching or in-run instruction changes',
                'interpretation': 'Single selected task with revised skill and repaired harness; no population or old-model causal claim'}
    for path in sources:
        dest = RUN / 'frozen-source' / path.relative_to(ROOT.parents[1]) if path.is_relative_to(ROOT.parents[1]) else RUN / 'frozen-source/codex-binary-sha256.txt'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_relative_to(ROOT.parents[1]):
            shutil.copy2(path, dest)
        else:
            dest.write_text(sha(path) + '\n')
    E.save(RUN / 'protocol.json', protocol)
    return protocol


def verify(protocol):
    for name, digest in protocol['pins'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Frozen input changed: ' + name)
    for name, digest in protocol['skill_files'].items():
        if sha(INPUTS / 'bundle/cuda-to-sycl-xe' / name) != digest:
            raise ValueError('Frozen skill changed: ' + name)
    sdk = json.loads((E.SDK / 'manifest.json').read_text())
    for name, digest in sdk['files'].items():
        if sha(E.SDK / name) != digest:
            raise ValueError('SDK changed: ' + name)


def monitor(stop):
    seen = set()
    while True:
        for episode in (RUN / 'episodes').glob('*/*'):
            metadata = episode / 'episode.json'
            if not metadata.exists():
                continue
            work = Path(json.loads(metadata.read_text())['workdir'])
            paths = list((work / 'feedback').glob('*.json'))
            paths += list((episode / 'trusted/tasks' / TASK).glob('*/result.json'))
            for path in paths:
                key = str(path)
                if key in seen:
                    continue
                try:
                    record = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                seen.add(key)
                event = {'utc': E.now(), 'arm': episode.name,
                         'kind': 'feedback' if path.parent.name == 'feedback' else 'assessment',
                         'entry': path.name if path.parent.name == 'feedback' else path.parent.name,
                         **{key: record.get(key) for key in ('status', 'correct', 'speedup')}}
                event['errors'] = [x.get('category') for x in record.get('errors', [])]
                with (RUN / 'monitor.jsonl').open('a') as target:
                    target.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
        if stop.wait(1):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    protocol = prepare()
    verify(protocol)
    if args.action == 'prepare':
        print(json.dumps({'protocol': str(RUN / 'protocol.json'), 'sessions': 2}))
        return
    stop = threading.Event()
    watcher = threading.Thread(target=monitor, args=(stop,))
    watcher.start()
    try:
        for item in protocol['schedule']:
            verify(protocol)
            E.episode(item, protocol)
        E.save(RUN / 'progress.json', {'stage': 'complete', 'updated_utc': E.now()})
    finally:
        stop.set()
        watcher.join()


if __name__ == '__main__':
    main()
