import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import adapter

class RepairTests(unittest.TestCase):
    def test_private_errors_never_returned(self):
        with tempfile.TemporaryDirectory() as d:
            record={'status':'failed','correct':False,'errors':[{'category':'worker_error','message':'oracle=SECRET seed=123 /private/path Traceback'}]}
            result=json.dumps(adapter.public(record,Path(d)))
            for secret in ('SECRET','123','/private/path','Traceback'):
                self.assertNotIn(secret,result)
    def test_load_error_redacts_values(self):
        with tempfile.TemporaryDirectory() as d:
            r=adapter.public({"correct":False,"errors":[{"category":"candidate_load","message":"SECRET /private/oracle 123"}]},Path(d))
            self.assertEqual(r["errors"][0]["message"],"Candidate failed to load.")
            r=adapter.public({"correct":False,"errors":[{"category":"candidate_load","message":"libmissing.so: cannot open shared object file"}]},Path(d))
            self.assertEqual(r["errors"][0]["message"],"Missing shared library: libmissing.so")
    def test_runtime_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            r=adapter.public({'correct':False,'errors':[{'category':'worker_error','message':'private 123 UR_RESULT_ERROR_OUT_OF_RESOURCES'}]},Path(d))
            self.assertEqual(r['status'],'candidate_runtime_failed')
            self.assertEqual(r['errors'][0]['message'],'Device runtime error: UR_RESULT_ERROR_OUT_OF_RESOURCES')
    def test_mismatch_not_runtime_or_build(self):
        with tempfile.TemporaryDirectory() as d:
            r=adapter.public({'status':'failed','correct':False,'errors':[{'category':'numerical_mismatch','message':'SECRET 123'}]},Path(d))
            self.assertEqual(r['errors'][0]['category'],'numerical_mismatch')
            self.assertNotIn('SECRET',json.dumps(r))
    def test_diagnostic_redaction(self):
        s=adapter.diagnostics('/private/oracle/file.cpp:8: error: missing symbol\nTraceback secret\nseed 123\n')
        self.assertNotIn('/private',s)
        self.assertNotIn('Traceback',s)
        self.assertNotIn('123',s)
        self.assertIn('missing symbol',s)
    def test_linker_reference_and_private_payloads(self):
        output=adapter.diagnostics("ld: error: undefined reference to `missing_link_fixture`\nerror: seed 123 PRIVATE_SENTINEL\nerror: oracle=SECRET\n")
        self.assertIn("undefined reference to `missing_link_fixture`", output)
        for private in ("123", "PRIVATE_SENTINEL", "SECRET"):
            self.assertNotIn(private, output)
    def test_recovered_build_does_not_mask_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            (path/'build-diagnostics.private').write_text('error: prior build failure')
            r=adapter.public({'correct':False,'status':'failed','errors':[{'category':'worker_error','message':'UR_RESULT_ERROR_OUT_OF_RESOURCES'}]},path)
            self.assertEqual(r['status'],'candidate_runtime_failed')
            (path/'build-terminal.flag').touch()
            r=adapter.public({'correct':False,'status':'failed','errors':[{'category':'worker_error','message':'UR_RESULT_ERROR_OUT_OF_RESOURCES'}]},path)
            self.assertEqual(r['status'],'candidate_runtime_failed')
            (path/'build-terminal.flag').unlink()
            r=adapter.public({'correct':False,'status':'failed','errors':[{'category':'numerical_mismatch','message':'private'}]},path)
            self.assertEqual(r['errors'][0]['category'],'numerical_mismatch')
    def test_policy_cli(self):
        candidate=adapter.EVALUATOR/'fixtures/model_new.py'
        p=subprocess.run([sys.executable,str(adapter.HERE/'adapter.py'),'validate','--candidate',str(candidate)],text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stdout)
        with tempfile.TemporaryDirectory() as d:
            c=Path(d)/'model_new.py';c.write_text('import subprocess\nclass ModelNew: pass\n')
            p=subprocess.run([sys.executable,str(adapter.HERE/'adapter.py'),'validate','--candidate',str(c)],text=True,capture_output=True)
            self.assertEqual(p.returncode,1,p.stdout)
            self.assertIn('prohibited import',p.stdout)

if __name__=='__main__': unittest.main()
