#!/usr/bin/env python3
"""Validate paired, context-bound kernel measurements before accepting a candidate."""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

CONTEXT = ('device_pci', 'device_name', 'driver', 'runtime', 'compiler',
           'build_flags', 'dtype', 'layout', 'shape', 'operation', 'cache_regime',
           'timing_scope', 'baseline_sha256', 'candidate_sha256', 'baseline_variant', 'candidate_variant')

def positive(x):
    return type(x) in (int, float) and math.isfinite(x) and x > 0

def check(run):
    context = run['context']
    if any(k not in context or context[k] in (None, '', []) for k in CONTEXT):
        raise ValueError('incomplete device/software/workload context')
    for key in ('baseline_sha256', 'candidate_sha256'):
        value = context[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid source hash')
    if (context['baseline_sha256'], context['baseline_variant']) == (context['candidate_sha256'], context['candidate_variant']):
        raise ValueError('baseline and candidate must have distinct source/variant identities')
    if context['timing_scope'] not in ('device_event', 'end_to_end'):
        raise ValueError('unsupported timing scope')
    if run.get('correctness') != {'baseline': 'PASS', 'candidate': 'PASS'}:
        raise ValueError('both paths must pass correctness (SKIP is not PASS)')
    if run.get('interleaved') is not True or run.get('warmup_complete') is not True:
        raise ValueError('requires interleaving and completed warmup')
    if run.get('uncontended') is not True:
        raise ValueError('uncontended execution has not been established')
    pairs = run['pairs']
    if len(pairs) < 20:
        raise ValueError('at least 20 paired samples per phase required')
    orders = set()
    for i, pair in enumerate(pairs):
        if not positive(pair['baseline_us']) or not positive(pair['candidate_us']):
            raise ValueError('invalid timing')
        order = pair['order']
        if order not in ('AB', 'BA'):
            raise ValueError('invalid pair order')
        if i and order == pairs[i-1]['order']:
            raise ValueError('pair order must alternate AB and BA')
        orders.add(order)
    if len(orders) != 2:
        raise ValueError('both pair orders required')
    return context

def summarize(pairs, minimum_gain, seed):
    # A conservative decision statistic over paired speedups. Bootstrap assumes
    # pairs represent the target steady-state distribution; it cannot fix drift.
    ratios = [p['baseline_us']/p['candidate_us'] for p in pairs]
    rng = random.Random(seed)
    resampled = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(4000))
    lower = resampled[int(0.025 * len(resampled))]
    return {'samples': len(pairs), 'median_speedup': statistics.median(ratios),
            'bootstrap_95pct_lower_speedup': lower,
            'baseline_median_us': statistics.median(p['baseline_us'] for p in pairs),
            'candidate_median_us': statistics.median(p['candidate_us'] for p in pairs),
            'accepted': lower > minimum_gain}

def evaluate(document, minimum_gain=1.05):
    if not positive(minimum_gain) or minimum_gain <= 1:
        raise ValueError('minimum speedup must exceed one')
    first, confirm = document['discovery'], document['confirmation']
    if check(first) != check(confirm):
        raise ValueError('mixed contexts between discovery and confirmation')
    if not first.get('run_id') or not confirm.get('run_id') or first['run_id'] == confirm['run_id']:
        raise ValueError('independent confirmation run identity required')
    # Disjoint pair IDs prevent a duplicate of the discovery samples being
    # relabeled as confirmation. IDs are evidence labels, not trusted execution.
    seen = set()
    for phase in (first, confirm):
        for pair in phase['pairs']:
            identity = pair.get('sample_id')
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError('missing or reused sample identity')
            seen.add(identity)
    results = {name: summarize(document[name]['pairs'], minimum_gain, i)
               for i, name in enumerate(('discovery', 'confirmation'))}
    results['accepted'] = all(r['accepted'] for r in results.values())
    results['minimum_speedup'] = minimum_gain
    results['context_sha256'] = hashlib.sha256(json.dumps(first['context'],sort_keys=True).encode()).hexdigest()
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--min-speedup', type=float, default=1.05)
    args = parser.parse_args()
    try:
        result = evaluate(json.loads(args.record.read_text()), args.min_speedup)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['accepted'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
