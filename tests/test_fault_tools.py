import os
import pathlib
import subprocess
import tempfile
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
class FaultTools(unittest.TestCase):
    def test_journal_failure_is_unknown_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            sudo = pathlib.Path(tmp) / 'sudo'
            sudo.write_text('#!/bin/sh\nexit 1\n'); sudo.chmod(0o755)
            env = dict(os.environ, PATH=tmp + ':' + os.environ['PATH'])
            p = subprocess.run([str(ROOT/'scripts/fault_count.sh'), 'today'], env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 2); self.assertEqual(p.stdout.strip(), 'UNKNOWN')
            p = subprocess.run([str(ROOT/'scripts/switch_bisect.sh'), '3', 'a:', 'b:', '--', 'true'], env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 1); self.assertEqual(p.stdout.count('round '), 1)
    def test_no_faults_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            sudo = pathlib.Path(tmp) / 'sudo'
            sudo.write_text('#!/bin/sh\necho "normal kernel message"\n'); sudo.chmod(0o755)
            p = subprocess.run([str(ROOT/'scripts/fault_count.sh'), 'today'], env=dict(os.environ, PATH=tmp+':'+os.environ['PATH']), capture_output=True, text=True)
            self.assertEqual(p.returncode, 0); self.assertEqual(p.stdout.strip(), '0')
