#!/usr/bin/env python3
"""Stream the decision-relevant events of running dev episodes, one line each.

    python3 dev_watch.py 'runs/dev-codex-l2_0*-*-2026111*' --expect 12

Emits skill reads, mailbox requests, assessment/validation feedback, errors, agent messages,
and the final result; exits when every run has final.json.
"""
import json
import re
import sys
import time
from pathlib import Path


def short(text, n=160):
    return re.sub(r'\s+', ' ', str(text)).strip()[:n]


class Run:
    def __init__(self, path):
        self.path = Path(path)
        task = next((part.upper() for part in self.path.name.split('-') if part.startswith('l')), '?')
        self.tag = f"{task} {'SKILL' if '-skill-' in self.path.name else 'CTRL'}"
        self.offset = 0
        self.seen_feedback = set()
        self.done = False
        self.work = Path(json.loads((self.path / 'episode.json').read_text())['workdir'])

    def events(self):
        log = self.path / 'agent/events.jsonl'
        if not log.exists():
            return
        with log.open() as f:
            f.seek(self.offset)
            for line in f:
                if not line.endswith('\n'):
                    break
                self.offset += len(line)
                try:
                    yield json.loads(line)
                except ValueError:
                    continue

    def poll(self):
        out = []
        for e in self.events():
            item = e.get('item') or {}
            kind = item.get('type')
            if e.get('type') == 'item.started' and kind == 'command_execution':
                cmd = item.get('command', '')
                if 'skill/' in cmd:
                    out.append(f'reads {short(re.findall(r"skill/[^ \x27\"|;]+", cmd), 140)}')
                elif 'requests/' in cmd:
                    out.append(f'request: {short(cmd, 140)}')
            elif e.get('type') == 'item.completed' and kind == 'agent_message':
                out.append(f'says: {short(item.get("text"), 220)}')
            elif kind == 'error' or e.get('type') in ('error', 'turn.failed'):
                out.append(f'ERROR event: {short(item or e, 200)}')
        for fb in sorted((self.work / 'feedback').glob('*.json')):
            if fb.name in self.seen_feedback:
                continue
            try:
                d = json.loads(fb.read_text())
            except ValueError:
                continue  # still being written
            self.seen_feedback.add(fb.name)
            errs = '; '.join(short(x.get('message', x), 120) for x in d.get('errors', [])[:2])
            out.append(f'feedback {fb.name}: status={d.get("status")} correct={d.get("correct")} '
                       f'speedup={d.get("speedup")} backend={d.get("sycl_backend")} {errs}')
        final = self.path / 'final.json'
        if final.exists() and not self.done:
            self.done = True
            d = json.loads(final.read_text())
            out.append(f'FINAL: status={d.get("status")} correct={d.get("correct")} speedup={d.get("speedup")} '
                       f'selected={d.get("selected_assessment")}')
        return [f'[{self.tag}] {line}' for line in out]


def main():
    """Args: glob patterns of run directories, then --expect N. New runs are picked up as they start."""
    args = sys.argv[1:]
    expect = int(args[args.index('--expect') + 1]) if '--expect' in args else None
    patterns = [a for i, a in enumerate(args) if a != '--expect' and (i == 0 or args[i - 1] != '--expect')]
    runs = {}
    while True:
        for pattern in patterns:
            for path in sorted(Path('.').glob(pattern)):
                if path not in runs and (path / 'episode.json').exists():
                    runs[path] = Run(path)
                    print(f'[{runs[path].tag}] started', flush=True)
        for r in runs.values():
            for line in r.poll():
                print(line, flush=True)
        if runs and all(r.done for r in runs.values()) and (expect is None or len(runs) >= expect):
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
