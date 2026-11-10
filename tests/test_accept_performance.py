import copy
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('gate',Path(__file__).parents[1]/'scripts/accept_performance.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def record():
    ctx={key:'recorded' for key in gate.CONTEXT}
    ctx.update(timing_scope='device_event',baseline_sha256='a'*64,candidate_sha256='b'*64)
    def phase(name):
        return dict(context=copy.deepcopy(ctx),correctness={'baseline':'PASS','candidate':'PASS'},
          interleaved=True,warmup_complete=True,uncontended=True,run_id=name,
          pairs=[dict(sample_id=f'{name}-{i}',order='AB' if i%2==0 else 'BA',baseline_us=10+i%3*.1,candidate_us=5+i%3*.1) for i in range(24)])
    return dict(discovery=phase('first'),confirmation=phase('fresh'))
class Acceptance(unittest.TestCase):
    def test_accept_repeated_gain(self):self.assertTrue(gate.evaluate(record())['accepted'])
    def test_shared_source_distinct_entrypoints(self):
        r=record()
        for phase in r.values():phase['context'].update(candidate_sha256='a'*64,baseline_variant='scalar',candidate_variant='tiled')
        self.assertTrue(gate.evaluate(r)['accepted'])
    def test_noisy_gain_rejected(self):
        r=record()
        for phase in r.values():
            for i,p in enumerate(phase['pairs']):p['candidate_us']=9 if i%2 else 12
        self.assertFalse(gate.evaluate(r)['accepted'])
    def test_reject_bad_evidence(self):
        for mutate in [lambda r:r['confirmation']['context'].update(driver='different'),
          lambda r:r['discovery']['context'].update(candidate_sha256='a'*64),
          lambda r:r['discovery']['correctness'].update(candidate='SKIP'),
          lambda r:r['discovery']['pairs'][0].update(candidate_us=float('nan')),
          lambda r:r['discovery'].update(uncontended=False),
          lambda r:r['confirmation'].update(run_id='first'),
          lambda r:r['confirmation']['pairs'][0].update(sample_id='first-0'),
          lambda r:r['discovery'].update(pairs=r['discovery']['pairs'][:2]),
          lambda r:r['discovery']['pairs'][1].update(order='AB')]:
            r=record();mutate(r)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):gate.evaluate(r)
if __name__=='__main__':unittest.main()
