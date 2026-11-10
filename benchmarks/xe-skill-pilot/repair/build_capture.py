#!/usr/bin/env python3
"""Run the frozen worker with build-scoped trusted capture."""
import os
import sys
import tempfile
from pathlib import Path

import torch.utils.cpp_extension as ext

original = ext._run_ninja_build
failures = []

def capture(*args, **kwargs):
    # Capture only the subprocess's compiler stream, not the worker's or model's
    # general stdout. PyTorch embeds non-verbose output in its exception.
    with tempfile.TemporaryFile(mode='w+b') as out:
        sys.stdout.flush()
        saved = os.dup(1)
        try:
            os.dup2(out.fileno(), 1)
            try:
                return original(*args, **kwargs)
            except Exception as error:
                out.seek(0)
                data = out.read().decode('utf-8', 'replace') + '\n' + str(error)
                failures.append((error, data))
                raise
        finally:
            os.dup2(saved, 1)
            os.close(saved)

def install():
    ext._run_ninja_build = capture

def mark_terminal_error(error):
    """Mark whether the terminal error wraps a captured failed build."""
    seen = set()
    stack = [error]
    matched = None
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        for failure, diagnostic in failures:
            if item is failure:
                matched = diagnostic
        if item.__cause__ is not None:
            stack.append(item.__cause__)
        elif not item.__suppress_context__ and item.__context__ is not None:
            stack.append(item.__context__)
    marker = Path('/results/build-terminal.flag')
    if matched is not None:
        Path('/results/build-diagnostics.private').write_text(matched, encoding='utf-8')
        marker.touch()
    else:
        marker.unlink(missing_ok=True)
