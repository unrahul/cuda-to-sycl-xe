#!/usr/bin/env python3
"""Paired skill-vs-control summary of development runs.

    python3 dev_compare.py L2_025 L1_042 L2_096 L2_080 L2_014 L2_019 [--runner codex]

For each task, takes the newest finished skill and control run and reports the fresh-final result,
assessments used, failed builds and the first correct assessment; then aggregate metrics over tasks.
Incorrect finals count as speedup 0 for "faster" and are excluded from the geometric mean.
"""
import argparse
import json
import math
from pathlib import Path

RUNS = Path(__file__).resolve().parent / 'runs'


def newest(task, arm, runner):
    finished = [d for d in RUNS.glob(f'dev-{runner}-{task.lower()}-{arm}-*') if (d / 'final.json').exists()]
    return max(finished, default=None)


def summarize(run):
    final = json.loads((run / 'final.json').read_text())
    results = sorted((run / 'trusted/tasks').glob('*/assessment-*/result.json'))
    records = [json.loads(p.read_text()) for p in results]
    correct = [i for i, r in enumerate(records, 1) if r.get('correct')]
    return {
        'correct': bool(final.get('correct')),
        'speedup': final.get('speedup') if final.get('correct') else None,
        'used': len(records),
        'failed_builds': sum(r.get('status') in ('candidate_build_failed', 'candidate_load_failed') or
                             str(r.get('status', '')).startswith('candidate_build') for r in records),
        'first_correct': correct[0] if correct else None,
        'incorrect': len(records) - len(correct),
    }


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def fmt(x, spec='.2f'):
    return '-' if x is None else format(x, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tasks', nargs='+')
    parser.add_argument('--runner', default='codex')
    a = parser.parse_args()

    rows = {}
    for task in a.tasks:
        pair = {arm: newest(task, arm, a.runner) for arm in ('skill', 'control')}
        if all(pair.values()):
            rows[task] = {arm: summarize(run) for arm, run in pair.items()}

    print('| Task | Skill final | Control final | Skill / control | Assessments (S/C) | Incorrect (S/C) | Failed builds (S/C) | First correct (S/C) |')
    print('|---|---|---|---|---|---|---|---|')
    for task, r in rows.items():
        s, c = r['skill'], r['control']
        ratio = s['speedup'] / c['speedup'] if s['speedup'] and c['speedup'] else None
        print(f"| {task} | {fmt(s['speedup'])}x{'' if s['correct'] else ' (none)'} | "
              f"{fmt(c['speedup'])}x{'' if c['correct'] else ' (none)'} | {fmt(ratio)} | "
              f"{s['used']}/{c['used']} | {s['incorrect']}/{c['incorrect']} | {s['failed_builds']}/{c['failed_builds']} | "
              f"{fmt(s['first_correct'], 'd')}/{fmt(c['first_correct'], 'd')} |")

    print()
    for arm in ('skill', 'control'):
        arm_rows = [r[arm] for r in rows.values()]
        speedups = [r['speedup'] for r in arm_rows if r['speedup']]
        print(f"{arm:8s} correct {sum(r['correct'] for r in arm_rows)}/{len(arm_rows)}, "
              f"faster than baseline {sum((r['speedup'] or 0) > 1 for r in arm_rows)}/{len(arm_rows)}, "
              f"geomean speedup of correct {fmt(geomean(speedups))}x, "
              f"incorrect assessments {sum(r['incorrect'] for r in arm_rows)}, "
              f"failed builds {sum(r['failed_builds'] for r in arm_rows)}")
    both = [(r['skill']['speedup'] or 0, r['control']['speedup'] or 0) for r in rows.values()]
    print(f"skill better on {sum(s > c for s, c in both)}, control better on {sum(c > s for s, c in both)}, "
          f"ties {sum(s == c for s, c in both)} of {len(both)} tasks")


if __name__ == '__main__':
    main()
