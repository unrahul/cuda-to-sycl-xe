#!/usr/bin/env python3
"""Trusted orchestration for the frozen random sample; never agent-visible."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/opus55-high-v2'
CONTROLLER = ROOT / 'evaluator/controller.py'
RUNNER = ROOT / 'runner/claude_pilot.py'
READY = ROOT / 'evaluator/validation/native-sdk-feedback-v2/backend-readiness.json'
SDK = ROOT / 'evaluator/sdk/public-api'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def metadata():
    return json.loads(subprocess.check_output(
        [sys.executable, str(CONTROLLER), 'metadata'], text=True))


def check_readiness():
    spec = importlib.util.spec_from_file_location('_trusted_pilot_controller', CONTROLLER)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    controller.require_backend_readiness(READY.parent)
    public = SDK / 'onednn/include'
    private = controller.ONEDNN_PREFIX / 'include'
    if tree_paths(public) != tree_paths(private):
        raise ValueError('oneDNN header sets differ')
    for name in tree_paths(public)[0]:
        if sha(public / name) != sha(private / name):
            raise ValueError('actual oneDNN headers drifted: ' + name)


def tree_paths(base):
    files, links = set(), {}
    for path in base.rglob('*'):
        name = str(path.relative_to(base))
        if path.is_symlink():
            if not path.resolve(strict=True).is_relative_to(base):
                raise ValueError('external packet link: ' + name)
            links[name] = str(path.readlink())
        elif path.is_file():
            files.add(name)
    return files, links


def freeze():
    if (RUN / 'protocol.json').exists():
        raise ValueError('protocol already frozen')
    sample = json.loads((ROOT / 'sample.json').read_text())
    ready = json.loads(READY.read_text())
    if ready.get('status') != 'passed':
        raise ValueError('backend readiness missing')
    seed = sample['seed_hex']
    rank = lambda label, task: sha_text(seed + '/' + label + '/' + task['id'])
    ordered = sorted(sample['selected'], key=lambda t: rank('execution-order', t))
    first_control = {t['id'] for t in sorted(sample['selected'], key=lambda t: rank('arm-order', t))[:12]}
    schedule = []
    for t in ordered:
        arms = ['control', 'skill'] if t['id'] in first_control else ['skill', 'control']
        schedule.extend({'task_id': t['id'], 'arm': arm} for arm in arms)
    pins = [ROOT / 'sample.json', ROOT / 'bundle/manifest.json', READY,
            ROOT / 'evaluator/sdk/public-api/manifest.json',
            CONTROLLER, ROOT / 'evaluator/worker.py', RUNNER, Path(__file__)]
    protocol = {
        'name': 'KernelBench-derived random L1/L2 skill pilot',
        'frozen_utc': now(), 'model': 'claude-opus-5-5', 'effort_requested': 'high',
        'protocol_revision': 'v2: generic lazy-extension compiler diagnostics exposed; v1 qualification archived and excluded; identical random draw, fresh contexts',
        'runner': 'Claude CLI Bash-only; Codex command sandbox',
        'tasks_per_arm': 24, 'arms': ['control', 'skill'], 'schedule': schedule,
        'control': 'task + public SDK APIs + fixed assessment interface',
        'skill': 'identical environment plus frozen released skill bundle',
        'sampling': sample, 'environment': metadata(),
        'agent_context': 'fresh isolated /tmp workdir and clean CLI home per episode',
        'agent_tools': ['Bash'], 'web_search': False, 'tool_network': False,
        'model_transport': 'trusted CLI provider connection only',
        'repository_access': False, 'other_skills_memory_prior_solutions': False,
        'candidate_assessments': 6, 'fresh_final_assessments': 1,
        'fresh_final_eligibility': 'at most one, only after a correct timed assessment',
        'selection': 'best correct timed assessment snapshot, then fresh final; failures retained',
        'model_wall_timeout_seconds': 7200, 'model_response_limit': 120,
        'correctness': 'original XPU full-output reference + reviewed negate-first/half-first gates; mutation/repeat/current-stream checks',
        'independent_fp64_scope': 'SDK qualification, not every sampled task',
        'timing': 'original workloads; five warmups and 31 alternating completion-bounded wall pairs',
        'score': 'fast_p: correct fresh finals with speedup strictly >p, fixed24 denominator per arm',
        'comparability': 'this paired pilot only; not a CUDA leaderboard or the original270 campaign',
        'security_limit': 'tool boundary enforced; native candidate auditing is a practical policy guard, not a malicious-code security proof',
        'pins': {str(p.relative_to(ROOT)): sha(p) for p in pins},
        'sdk_directory_aliases': tree_paths(SDK)[1],
        'runtime_pins': {str(Path(shutil.which(tool)).resolve()):
                         sha(Path(shutil.which(tool)).resolve()) for tool in ('claude', 'codex')},
    }
    verify(protocol)
    for name in protocol['pins']:
        dest = RUN / 'frozen-source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    save(RUN / 'protocol.json', protocol)
    print(json.dumps({'protocol': str(RUN / 'protocol.json'), 'episodes': len(schedule)}))


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def verify(protocol):
    check_readiness()
    for path, digest in protocol['runtime_pins'].items():
        if sha(path) != digest:
            raise ValueError('frozen runtime changed: ' + path)
    for name, digest in protocol['pins'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('frozen implementation changed: ' + name)
    bundle = json.loads((ROOT / 'bundle/manifest.json').read_text())
    files, links = tree_paths(ROOT / 'bundle/cuda-to-sycl-xe')
    if files != set(bundle['files']) or links:
        raise ValueError('frozen skill file set changed')
    for name, digest in bundle['files'].items():
        if sha(ROOT / 'bundle/cuda-to-sycl-xe' / name) != digest:
            raise ValueError('frozen skill changed: ' + name)
    sdk_manifest = json.loads((SDK / 'manifest.json').read_text())
    files, links = tree_paths(SDK)
    if (files != set(sdk_manifest['files']) | {'manifest.json'} or
            links != protocol['sdk_directory_aliases']):
        raise ValueError('frozen SDK file set changed')
    for name, digest in sdk_manifest['files'].items():
        path = SDK / name
        if path.is_symlink() or sha(path) != digest:
            raise ValueError('frozen SDK header changed: ' + name)


def prompt(task_id, arm):
    guide = ('Read skill/SKILL.md and its linked local references before implementing. '
             'This is your only supplied procedural/optimization guide.\n') if arm == 'skill' else ''
    return f'''Implement task {task_id} in this fresh isolated workspace.
Read task.py for the exact operation, constructor, forward signature and original workload.
{guide}Write submission/model_new.py defining ModelNew, plus any companion native sources under submission/.
Use only this workspace and the supplied read-only SDK API headers. No Internet, web search,
other skills, memories, prior solutions, other workspaces or external repositories.
The SDK headers are under sdk/; environment.json gives actual container API/include/link paths.
Those paths describe the trusted assessment container, not this editing shell.
All GPU compilation/execution goes through the assessment mailbox. Do not use Docker or direct GPU commands.
Keep the original shapes, dtypes, model parameters and inference semantics. Native SYCL computation
and declared direct oneMKL/oneDNN SYCL library calls are permitted. Torch storage, allocation,
views, copies, parameters and extension loading are permitted; Torch/ATen numerical delegation,
CPU numerical fallback, reference imports/calls, reduced-precision substitutions, cached answers,
and evaluator/timing introspection are forbidden. Preserve state_dict names or supply
load_reference_state(state_dict). Use SYCL_BACKEND='library' for native-library computation.
Use the input device's current XPU queue or establish all completion dependencies before return.
For assessment N from1 through6, finish your candidate files, then atomically write
requests/N.json with exactly {{"id": N}}. Wait for feedback/N.json before modifying files or
requesting another assessment. Each request consumes one candidate assessment, including build failures.
Feedback can take many minutes because the original workload is large. Poll the local file;
do not assume a slow test failed. You have at most six assessments and a two-hour model wall budget.
Correctness includes original input seeds and valid-domain inputs beyond the generator values.
After correctness, optimize against the reported same-device PyTorch XPU baseline. Preserve genuine
attempts and report honest failures. No task replacements or shrinking workloads.
When done, leave the chosen implementation in submission/ and write experience.md with decisions,
limitations and meaningful recovered errors. The trusted parent independently selects the best
correct assessed source snapshot and performs one fresh final assessment. No implementation hints
will be supplied by the parent. Do not modify task.py, environment.json, sdk/ or skill/.
'''


def select_assessment(trusted, task_id):
    candidates = []
    for p in (trusted / 'tasks' / task_id).glob('assessment-*/result.json'):
        try:
            r = json.loads(p.read_text())
            speed = float(r.get('speedup') or 0)
            if r.get('correct') is True and r.get('status') == 'success' and math.isfinite(speed) and speed > 0:
                candidates.append((speed, p))
        except (ValueError, OSError):
            continue
    return max(candidates, default=None, key=lambda x: x[0])


def episode(item, protocol):
    task_id, arm = item['task_id'], item['arm']
    dest = RUN / 'episodes' / task_id / arm
    if (dest / 'final.json').exists():
        return
    if dest.exists():
        raise ValueError('unfinished episode requires explicit recovery: ' + str(dest))
    dest.mkdir(parents=True)
    trusted = dest / 'trusted'
    trusted.mkdir()
    shutil.copy2(READY, trusted / 'backend-readiness.json')
    work = Path(tempfile.mkdtemp(prefix=f'xe-pilot-{task_id.lower()}-{arm}-'))
    (work / 'submission').mkdir()
    task = next(t for t in protocol['sampling']['selected'] if t['id'] == task_id)
    source = ROOT / 'tasks' / (task_id + '.py')
    if sha(source) != task['sha256']:
        raise ValueError('task changed')
    shutil.copy2(source, work / 'task.py')
    save(work / 'environment.json', protocol['environment'])
    shutil.copytree(SDK, work / 'sdk', symlinks=True)
    readonly = ['task.py', 'environment.json', 'sdk']
    if arm == 'skill':
        shutil.copytree(ROOT / 'bundle/cuda-to-sycl-xe', work / 'skill')
        readonly.append('skill')
    (dest / 'prompt.txt').write_text(prompt(task_id, arm))
    save(dest / 'episode.json', {'task_id': task_id, 'arm': arm, 'workdir': str(work), 'started_utc': now()})
    cmd = [sys.executable, str(RUNNER), '--workdir', str(work), '--prompt', str(dest / 'prompt.txt'),
           '--result', str(dest / 'agent'), '--task-id', task_id, '--controller', str(CONTROLLER),
           '--run-dir', str(trusted)]
    for name in readonly:
        cmd += ['--read-only', name]
    save(RUN / 'progress.json', {**item, 'stage': 'agent', 'updated_utc': now()})
    with (dest / 'runner.stdout.log').open('w') as out, (dest / 'runner.stderr.log').open('w') as err:
        launched = subprocess.run(cmd, stdout=out, stderr=err)
    status_path = dest / 'agent/status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'exit': launched.returncode, 'violation': 'runner did not establish a completed model episode'}
    if (status.get('violation') or status.get('broker_fault') or
            (status.get('exit') != 0 and not status.get('budget_hit'))):
        raise ValueError('isolation/infrastructure failure: ' + str(status))
    best = select_assessment(trusted, task_id)
    if best:
        _, result_path = best
        selected = result_path.parent / 'submission/model_new.py'
        save(RUN / 'progress.json', {**item, 'stage': 'fresh_final', 'updated_utc': now()})
        cmd = [sys.executable, str(CONTROLLER), 'assess', '--task-id', task_id,
               '--candidate', str(selected), '--run-dir', str(trusted), '--phase', 'final']
        r = subprocess.run(cmd, text=True, capture_output=True)
        (dest / 'final.stdout.log').write_text(r.stdout)
        (dest / 'final.stderr.log').write_text(r.stderr)
        if r.returncode not in (0, 1):
            raise ValueError('final controller failed; inspect final logs before recovery')
        final = json.loads(r.stdout)
        if final.get('status') == 'controller_error':
            raise ValueError('final infrastructure error; inspect final logs before recovery')
        final['selected_assessment'] = result_path.parent.name
    else:
        final = {'task_id': task_id, 'status': 'no_correct_assessment', 'correct': False, 'speedup': None}
    final.update(arm=arm, agent_exit=launched.returncode, completed_utc=now())
    final['agent_budget_hit'] = status.get('budget_hit')
    save(dest / 'final.json', final)
    # Keep actor output and evidence; discard only this run's redundant SDK header copy.
    shutil.rmtree(work / 'sdk')
    score(protocol)


def score(protocol):
    summaries = {}
    for arm in protocol['arms']:
        paths = [RUN / 'episodes' / row['id'] / arm / 'final.json'
                 for row in protocol['sampling']['selected']]
        records = [json.loads(p.read_text()) for p in paths if p.exists()]
        def qualifies(record, threshold):
            speed = float(record.get('speedup') or 0)
            return (record.get('correct') is True and record.get('status') == 'success'
                    and math.isfinite(speed) and speed > threshold)
        counts = {str(p): sum(qualifies(r, p) for r in records) for p in (0, 1, 2)}
        summaries[arm] = {'denominator': 24, 'terminal': len(records), 'counts_fast_p': counts,
                          'fast_p': {p: n / 24 for p, n in counts.items()}}
    result = {'complete': all(x['terminal'] == 24 for x in summaries.values()),
              'name': protocol['name'], 'model': protocol['model'], 'arms': summaries}
    save(RUN / 'score.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run', 'score'])
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
        return
    protocol = json.loads((RUN / 'protocol.json').read_text())
    verify(protocol)
    if args.action == 'score':
        print(json.dumps(score(protocol), indent=2))
        return
    for item in protocol['schedule']:
        verify(protocol)
        episode(item, protocol)
    save(RUN / 'progress.json', {'stage': 'complete', 'updated_utc': now()})


if __name__ == '__main__':
    main()
