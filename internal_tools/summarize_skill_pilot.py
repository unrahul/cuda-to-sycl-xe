#!/usr/bin/env python3
"""Summarize existing trusted pilot evidence without changing the experiment."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def events(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A running CLI can be partway through its last event.
    return rows


def episode(path, task_id, arm):
    meta = read(path / 'episode.json')
    if not meta:
        return None
    final = read(path / 'final.json')
    trace = events(path / 'agent/events.jsonl')
    result_events = [x for x in trace if x.get('type') == 'result']
    terminal = result_events[-1] if result_events else None
    status = read(path / 'agent/status.json') or {}
    responses, calls = set(), set()
    for item in trace:
        if item.get('type') != 'assistant':
            continue
        message = item.get('message', {})
        if message.get('id'):
            responses.add(message['id'])
        calls.update(x['id'] for x in message.get('content', [])
                     if x.get('type') == 'tool_use' and x.get('id'))
    entries = (read(path / 'trusted/ledger.json') or {}).get('tasks', {}).get(task_id, [])
    attempts, evaluator_seconds = [], []
    for entry in entries:
        directory = path / 'trusted/tasks' / task_id / entry['label']
        result = read(directory / 'result.json')
        controller = read(directory / 'controller.json') or {}
        elapsed = controller.get('elapsed_seconds')
        if elapsed is not None:
            evaluator_seconds.append(elapsed)
        if entry['phase'] != 'assessment':
            continue
        errors = (result or {}).get('errors', [])
        build_failed = any(e.get('category') in ('candidate_load', 'worker_error')
                           and 'Error building extension ' in e.get('message', '') for e in errors)
        attempts.append({'number': len(attempts) + 1, 'label': entry['label'],
                         'status': (result or {}).get('status', 'in_progress'),
                         'correct': (result or {}).get('correct'),
                         'speedup': (result or {}).get('speedup'),
                         'build_failed': build_failed,
                         'evaluator_elapsed_seconds': elapsed})
    def first_success(threshold):
        return next((x['number'] for x in attempts
                     if x['status'] == 'success' and x['correct'] is True
                     and (x['speedup'] or 0) > threshold), None)
    wall = None
    if final:
        wall = (dt.datetime.fromisoformat(final['completed_utc']) -
                dt.datetime.fromisoformat(meta['started_utc'])).total_seconds()
    def seconds(field):
        value = terminal.get(field) if terminal else None
        return value / 1000 if isinstance(value, (int, float)) else None
    model_usage = terminal.get('modelUsage') if terminal else None
    tokens = None
    if model_usage:
        names = {'input_tokens': 'inputTokens', 'output_tokens': 'outputTokens',
                 'cache_read_input_tokens': 'cacheReadInputTokens',
                 'cache_creation_input_tokens': 'cacheCreationInputTokens',
                 'thinking_tokens_within_output': 'thinkingTokens'}
        tokens = {key: sum(model.get(source, 0) for model in model_usage.values())
                  for key, source in names.items()}
    elif len(result_events) == 1:
        tokens = terminal.get('usage')
    durations = [x.get('duration_ms') for x in result_events]
    turns = [x.get('num_turns') for x in result_events]
    result = {'task_id': task_id, 'arm': arm, 'complete': final is not None,
              'assessments_reserved': len(attempts),
              'assessments_terminal': sum(x['status'] != 'in_progress' for x in attempts),
              'assessments_correct': sum(x['correct'] is True for x in attempts),
              'policy_rejections': sum(x['status'] == 'candidate_policy_rejected' for x in attempts),
              'build_failures': sum(x['build_failed'] for x in attempts),
              'first_correct_assessment': first_success(0),
              'first_faster_assessment': first_success(1),
              'selected_assessment': (final or {}).get('selected_assessment'),
              'unique_assistant_response_ids': len(responses), 'tool_calls': len(calls),
              'episode_wall_seconds_including_final': wall,
              'evaluator_elapsed_seconds_sum': sum(evaluator_seconds) if evaluator_seconds else None,
              'cli_result_segments': len(result_events),
              'last_result_duration_seconds': seconds('duration_ms'),
              'cli_reported_segment_duration_sum_seconds': sum(durations) / 1000
                  if durations and all(isinstance(x, (int, float)) for x in durations) else None,
              'api_seconds_cumulative_reported': seconds('duration_api_ms'),
              'cli_num_turns_sum': sum(turns) if turns and all(isinstance(x, int) for x in turns) else None,
              'reported_list_price_cost_usd': terminal.get('total_cost_usd') if terminal else None,
              'tokens': tokens,
              'tokens_source': 'latest cumulative modelUsage' if model_usage else
                  'single result usage' if tokens is not None else None,
              'budget_hit': status.get('budget_hit'), 'assessments': attempts,
              'final': {k: final.get(k) for k in ('status', 'correct', 'speedup', 'baseline_median_ms',
                                                'candidate_median_ms', 'sycl_backend')} if final else None}
    return result


def summarize(run):
    protocol = read(run / 'protocol.json')
    rows = []
    for task in protocol['sampling']['selected']:
        for arm in protocol['arms']:
            row = episode(run / 'episodes' / task['id'] / arm, task['id'], arm)
            if row:
                rows.append(row)
    planned = protocol['tasks_per_arm'] * len(protocol['arms'])
    return {'run': str(run.resolve()), 'model': protocol['model'],
            'planned_episodes': planned,
            'complete': len(rows) == planned and all(x['complete'] for x in rows),
            'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'definitions': {
                'assessments': 'Controller-reserved feedback submissions; includes policy/build failures. Fresh final is separate.',
                'responses': 'Distinct assistant message IDs; not independent kernel generations or CLI num_turns.',
                'timing': 'Episode timestamps give wall time including final. CLI duration_ms and num_turns are segment-scoped; summed segment durations are not episode wall. API duration is latest cumulative value. Evaluator time overlaps agent time.',
                'tokens': 'Use latest cumulative modelUsage. Thinking tokens are part of output tokens. Per-result usage is a segment delta.',
                'cost': 'CLI-reported list-price estimate, not a billing statement; missing telemetry is null.',
                'comparability': 'Sequential feedback refinement, not pass@k independent sampling.'},
            'episodes': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--table-output', type=Path)
    parser.add_argument('--watch-service', help='Refresh every 30s until this user service stops; no model calls')
    args = parser.parse_args()
    while True:
        report = summarize(args.run_dir)
        rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix('.tmp')
            temporary.write_text(rendered)
            temporary.replace(args.output)
        else:
            print(rendered, end='')
        if args.table_output:
            args.table_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.table_output.with_suffix('.tmp')
            fields = ['task_id', 'arm', 'complete', 'assessments_reserved', 'first_correct_assessment',
                      'first_faster_assessment', 'unique_assistant_response_ids', 'cli_result_segments',
                      'cli_reported_segment_duration_sum_seconds',
                      'episode_wall_seconds_including_final', 'reported_list_price_cost_usd',
                      'status', 'correct', 'speedup', 'baseline_median_ms', 'candidate_median_ms', 'sycl_backend']
            with temporary.open('w') as stream:
                writer = csv.DictWriter(stream, fields, extrasaction='ignore')
                writer.writeheader()
                for row in report['episodes']:
                    writer.writerow({**row, **(row['final'] or {})})
            temporary.replace(args.table_output)
        if not args.watch_service or report['complete']:
            break
        active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', args.watch_service],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if active.returncode:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
