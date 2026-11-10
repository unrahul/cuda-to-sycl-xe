#!/usr/bin/env python3
"""repair/adapter.py on a chosen GPU, for development runs only.

The frozen controller maps renderD129 (card 1) under /tmp/xe-skill-eval-gpu1.lock. A dev run
may place {"render": "renderD130"} in <run-dir>/gpu.json; this wrapper then remaps the docker
device, the lock and the recorded render node to that card. Without gpu.json it is adapter.py.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'repair'))
import adapter  # noqa: E402

c = adapter.c
FROZEN = 'renderD129'


def render_for(argv):
    if '--run-dir' not in argv:
        return None
    config = Path(argv[argv.index('--run-dir') + 1]) / 'gpu.json'
    return json.loads(config.read_text())['render'] if config.exists() else None


def remap(render):
    card = re.fullmatch(r'renderD(\d+)', render)
    if not card or not Path('/dev/dri', render).is_char_device():
        raise ValueError(f'unknown render node {render}')
    lock = f'/tmp/xe-skill-eval-{render}.lock'
    swap = lambda s: s.replace(FROZEN, render).replace('/tmp/xe-skill-eval-gpu1.lock', lock) if isinstance(s, str) else s

    frozen_command = c.command
    c.command = lambda *a, **k: [swap(x) for x in frozen_command(*a, **k)]

    class Subprocess:  # rewrites only the wrapped docker launch; everything else passes through
        def __getattr__(self, name):
            return getattr(subprocess, name)

        @staticmethod
        def run(args, *a, **k):
            return subprocess.run([swap(x) for x in args] if isinstance(args, list) else args, *a, **k)

    c.subprocess = Subprocess()
    deep = lambda v: [deep(x) for x in v] if isinstance(v, list) else (
        {k: deep(x) for k, x in v.items()} if isinstance(v, dict) else swap(v))
    frozen_dump = c.dump
    c.dump = lambda path, value: frozen_dump(path, deep(value))


if __name__ == '__main__':
    render = render_for(sys.argv)
    if render:
        remap(render)
    try:
        raise SystemExit(adapter.main())
    except Exception as error:
        print(json.dumps({'status': 'controller_error', 'correct': False, 'error_category': type(error).__name__}))
        raise SystemExit(2)
