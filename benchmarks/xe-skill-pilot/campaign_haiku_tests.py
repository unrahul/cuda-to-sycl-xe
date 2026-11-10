"""Eight-task pilot checks; no model generation or GPU work."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import campaign_haiku as c


class HaikuPilotChecks(unittest.TestCase):
    def test_random_draw_reproduces_without_outcome_filters(self):
        child = json.loads(c.SAMPLE.read_text())
        parent = json.loads((c.ROOT / 'sample.json').read_text())
        self.assertEqual(c.sha(c.ROOT / 'sample.json'), child['parent_sample_sha256'])
        expected = []
        for level in (1, 2):
            pool = [x for x in parent['selected'] if x['level'] == level]
            pool.sort(key=lambda x: hashlib.sha256((child['seed_hex'] + '/' + x['id']).encode()).hexdigest())
            expected.extend(x['id'] for x in pool[:4])
        self.assertEqual(expected, [x['id'] for x in child['selected']])
        for row in child['selected']:
            self.assertEqual(c.sha(c.ROOT / 'tasks' / (row['id'] + '.py')), row['sha256'])

    def test_freeze_is_once_and_schedules_eight_pairs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'RUN', Path(tmp)):
            c.freeze()
            p = json.loads((c.RUN / 'protocol.json').read_text())
            c.verify(p)
            self.assertEqual((p['model'], p['effort_requested']), ('claude-haiku-5-5', 'xhigh'))
            self.assertEqual(len(p['schedule']), 16)
            self.assertEqual(sum(x['arm'] == 'control' for x in p['schedule'][::2]), 4)
            self.assertEqual(p['tasks_per_arm'], 8)
            for left, right in zip(p['schedule'][::2], p['schedule'][1::2]):
                self.assertEqual(left['task_id'], right['task_id'])
                self.assertEqual({left['arm'], right['arm']}, {'control', 'skill'})
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                c.freeze()

    def test_failures_and_pending_keep_eight_denominator(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'RUN', Path(tmp)):
            sample = json.loads(c.SAMPLE.read_text())
            task = sample['selected'][0]['id']
            c.save(c.RUN / 'episodes' / task / 'skill/final.json',
                   {'status': 'success', 'correct': True, 'speedup': 1.5})
            c.save(c.RUN / 'episodes/UNSELECTED/skill/final.json',
                   {'status': 'success', 'correct': True, 'speedup': 100})
            report = c.score({'arms': ['control', 'skill'], 'sampling': sample,
                              'tasks_per_arm': 8, 'name': 'test', 'model': c.MODEL})
            self.assertEqual(report['arms']['skill']['fast_p']['1'], 1 / 8)
            self.assertEqual(report['arms']['skill']['terminal'], 1)
            self.assertFalse(report['complete'])


if __name__ == '__main__':
    unittest.main()
