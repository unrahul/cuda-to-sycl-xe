#!/usr/bin/env python3
"""Fail-closed Codex CLI pilot; see README before use."""
import argparse, json, os, pathlib, shutil, subprocess, tempfile

p = argparse.ArgumentParser()
p.add_argument('--workdir', required=True, type=pathlib.Path)
p.add_argument('--prompt', required=True, type=pathlib.Path)
p.add_argument('--result', required=True, type=pathlib.Path)
p.add_argument('--probe-only', action='store_true')
p.add_argument('--diagnostic', action='store_true')
p.add_argument('--model', default='gpt-6.1-sol')
p.add_argument('--effort', default='high')
a = p.parse_args()
work = a.workdir.resolve(strict=True)
result = a.result.resolve()
if not work.is_dir() or work == pathlib.Path('/') or work in result.parents:
    p.error('workdir must be a directory and result must be outside it')
# Avoid ancestors carrying project instructions and avoid a shared temporary directory.
if not work.is_relative_to('/tmp') or work == pathlib.Path('/tmp'):
    p.error('workdir must be a dedicated directory below /tmp')
if any((x / 'AGENTS.md').exists() for x in list(work.parents) if x != pathlib.Path('/')):
    p.error('workdir has an ancestor AGENTS.md')
if result.exists():
    p.error('result already exists')
exe = pathlib.Path(shutil.which('codex') or '').resolve(strict=True)
auth = pathlib.Path.home() / '.codex/auth.json'
if not auth.is_file():
    p.error('file-backed Codex auth is unavailable')
result.mkdir(parents=True, mode=0o700)
with tempfile.TemporaryDirectory(prefix='xe-pilot-control-', dir='/tmp') as temp:
    home = pathlib.Path(temp) / 'home'; home.mkdir(mode=0o700)
    shutil.copyfile(auth, home / 'auth.json'); (home / 'auth.json').chmod(0o600)
    sentinel = pathlib.Path(temp) / 'hidden'; sentinel.write_text('hidden')
    # TOML strings encoded by JSON have compatible escaping.
    config = '''default_permissions = "pilot"
approval_policy = "never"
web_search = "disabled"
[skills.bundled]
enabled = false
[skills]
include_instructions = false
[features]
apps = false
plugins = false
hooks = false
browser_use = false
computer_use = false
memories = false
multi_agent = false
image_generation = false
skill_mcp_dependency_install = false
skip_host_skill_discovery = true
[permissions.pilot.filesystem]
":minimal" = "read"
''' + json.dumps(str(exe)) + ' = "read"\n' + ''.join(json.dumps(str(pathlib.Path('/opt/intel/oneapi/compiler/2025.3') / part)) + ' = "read"\n' for part in ('bin', 'lib', 'include')) + '''[permissions.pilot.filesystem.":workspace_roots"]
"." = "write"
[permissions.pilot.network]
enabled = false
'''
    (home / 'config.toml').write_text(config)
    env = {'HOME': str(home), 'CODEX_HOME': str(home), 'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8', 'TMPDIR': str(work)}
    # Separate from task prompt and model; sandbox exactly the active profile.
    probe = '''import os, socket, sys
checks = {}
def denied(path):
 try:
  with open(path, 'rb') as f: f.read(1)
  return False
 except OSError: return True
checks['outside_control_unreadable'] = denied(sys.argv[1])
checks['auth_unreadable'] = denied(sys.argv[2])
try:
 os.listdir(sys.argv[3]); checks['repo_unreadable'] = False
except OSError: checks['repo_unreadable'] = True
checks['workspace_write'] = False
try:
 p = '.pilot-sandbox-probe'; open(p, 'w').write('ok'); os.unlink(p); checks['workspace_write'] = True
except OSError: pass
for family, target, key in [(socket.AF_INET, ('1.1.1.1', 443), 'internet_denied'), (socket.AF_UNIX, '/var/run/docker.sock', 'docker_socket_denied')]:
 try:
  s = socket.socket(family); s.settimeout(2)
  try: s.connect(target); checks[key] = False
  finally: s.close()
 except OSError: checks[key] = True
print(__import__('json').dumps(checks, sort_keys=True))
sys.exit(0 if all(checks.values()) else 1)
'''
    # Probe outside repo path existence first; inaccessible paths must actually exist.
    repo_marker = pathlib.Path('/home/rahul/Coding/cuda-to-sycl-xe')
    cmd = [str(exe), 'sandbox', '-P', 'pilot', '-C', str(work), '--', '/usr/bin/python3', '-c', probe, str(sentinel), str(home/'auth.json'), str(repo_marker)]
    check = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=20)
    (result/'sandbox-probe.json').write_text(json.dumps({'command':cmd, 'exit':check.returncode, 'stdout':check.stdout, 'stderr':check.stderr}, indent=2))
    if check.returncode:
        raise SystemExit('Sandbox probe failed; model not launched. See sandbox-probe.json')
    if a.probe_only:
        raise SystemExit(0)
    if not a.diagnostic:
        raise SystemExit('Benchmark model launch disabled pending isolation verification. See README.')
    prompt = ('Do not execute a shell command. If collaboration.list_agents is an actual callable tool, call it now; if not, state unavailable. If an MCP resource list tool is an actual callable tool, call it now; if not, state unavailable. Report only actual results and do not infer tool availability.')
    cmd = [str(exe), 'exec', '--strict-config', '--ephemeral', '--skip-git-repo-check', '--ignore-rules', '-C', str(work), '-m', a.model, '-c', 'model_reasoning_effort='+json.dumps(a.effort), '--json', '-']
    (result/'invocation.json').write_text(json.dumps({'command':cmd, 'model_requested':a.model, 'effort_requested':a.effort, 'note':'CLI JSON events do not necessarily report provider-returned model or effort.'}, indent=2))
    with (result/'events.jsonl').open('w') as out, (result/'stderr.log').open('w') as err:
        run = subprocess.run(cmd, input=prompt, env=env, text=True, stdout=out, stderr=err)
    (result/'status.json').write_text(json.dumps({'exit':run.returncode}, indent=2))
    raise SystemExit(run.returncode)
