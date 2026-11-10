"""Trusted campaign regression checks; no model calls or GPU work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import campaign as c


class CampaignChecks(unittest.TestCase):
    def test_failed_and_pending_tasks_keep_fixed_denominator(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'RUN', Path(tmp)):
            for task, record in {
                'L1_005': {'status': 'success', 'correct': True, 'speedup': 2.5},
                'L1_002': {'status': 'correctness_failed', 'correct': False, 'speedup': 100},
                'L1_038': {'status': 'success', 'correct': True, 'speedup': float('inf')},
            }.items():
                c.save(c.RUN / 'episodes' / task / 'skill/final.json', record)
            sample = json.loads((c.ROOT / 'sample.json').read_text())
            c.save(c.RUN / 'episodes/UNSELECTED/skill/final.json',
                   {'status': 'success', 'correct': True, 'speedup': 100})
            score = c.score({'arms': ['control', 'skill'], 'name': 'test', 'model': 'test',
                             'sampling': sample})
            self.assertFalse(score['complete'])
            self.assertEqual(score['arms']['skill']['terminal'], 3)
            self.assertEqual(score['arms']['skill']['fast_p']['1'], 1 / 24)
            self.assertEqual(score['arms']['control']['fast_p']['1'], 0)

    def test_selection_requires_correct_finite_assessed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, record in enumerate([
                {'status': 'success', 'correct': True, 'speedup': 0.5},
                {'status': 'success', 'correct': False, 'speedup': 100},
                {'status': 'success', 'correct': True, 'speedup': float('nan')},
                {'status': 'success', 'correct': True, 'speedup': 1.5},
            ], 1):
                c.save(root / 'tasks/L1_005' / f'assessment-{i:02}/result.json', record)
            best = c.select_assessment(root, 'L1_005')
            self.assertEqual(best[1].parent.name, 'assessment-04')

    def test_freeze_is_once_and_balances_paired_order(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'RUN', Path(tmp)):
            c.freeze()
            protocol = json.loads((c.RUN / 'protocol.json').read_text())
            c.verify(protocol)
            schedule = protocol['schedule']
            self.assertEqual(len(schedule), 48)
            self.assertEqual(sum(x['arm'] == 'control' for x in schedule[::2]), 12)
            self.assertEqual(len({x['task_id'] for x in schedule}), 24)
            for left, right in zip(schedule[::2], schedule[1::2]):
                self.assertEqual(left['task_id'], right['task_id'])
                self.assertEqual({left['arm'], right['arm']}, {'skill', 'control'})
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                c.freeze()


if __name__ == '__main__':
    unittest.main()
