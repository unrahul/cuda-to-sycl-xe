#!/usr/bin/env python3
import importlib.util
from build_capture import install, mark_terminal_error
install()
spec = importlib.util.spec_from_file_location('pilot_worker', '/evaluator/worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
original_load_legacy = worker.load_legacy
def load_legacy(*args, **kwargs):
    legacy = original_load_legacy(*args, **kwargs)
    original_exception_record = legacy._exception_record
    def exception_record(error, *args, **kwargs):
        mark_terminal_error(error)
        return original_exception_record(error, *args, **kwargs)
    legacy._exception_record = exception_record
    return legacy
worker.load_legacy = load_legacy
raise SystemExit(worker.main())
