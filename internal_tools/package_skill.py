#!/usr/bin/env python3
"""Build the standalone cuda-to-sycl-xe skill, verify it, and optionally install it.

    python3 internal_tools/package_skill.py                       # dist/build-<UTC>/cuda-to-sycl-xe(.zip)
    python3 internal_tools/package_skill.py --install codex,claude,opencode [--replace]

The package holds the entrypoint, references, examples, scripts, templates and tests.
Maintainer evidence, sources, measurements and binaries stay in this workspace.
The built package must pass its own tests/test_docs.py before it is archived or installed.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cuda-to-sycl-xe'
TREES = ('references', 'scripts', 'examples', 'templates', 'tests')
SOURCE_SUFFIXES = {'.md', '.py', '.sh', '.cpp', '.hpp', '.cu', '.sycl'}
IGNORE = shutil.ignore_patterns('bin', '__pycache__', '*.pyc')
INSTALL_ROOTS = {
    'codex': Path.home() / '.codex/skills',
    'claude': Path.home() / '.claude/skills',
    'opencode': Path.home() / '.config/opencode/skills',
}
OPENAI_YAML = '''interface:
  display_name: "Intel Xe Kernel Engineering"
  short_description: "Port CUDA/HIP to SYCL and optimize kernels on Xe-HPG, Xe2 and Xe3P"
  default_prompt: "Use $cuda-to-sycl-xe to port, validate and tune this kernel on the selected Intel GPU."
'''


def build(dest):
    if dest.name != NAME:
        raise ValueError(f'output folder must be named {NAME}')
    if dest.exists():
        raise ValueError(f'{dest} exists; choose a new output')
    dest.mkdir(parents=True)
    shutil.copy2(ROOT / 'SKILL.md', dest)
    for tree in TREES:
        shutil.copytree(ROOT / tree, dest / tree, ignore=IGNORE)
    for path in (dest / 'examples').rglob('*'):
        if path.is_file() and path.suffix not in SOURCE_SUFFIXES:
            path.unlink()
    (dest / 'agents').mkdir()
    (dest / 'agents/openai.yaml').write_text(OPENAI_YAML)

    check = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_docs'], cwd=dest,
                           capture_output=True, text=True)
    if check.returncode:
        raise ValueError('package failed tests/test_docs.py:\n' + check.stderr)

    files = sorted(p for p in dest.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest = {'name': NAME, 'files': {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in files}}
    (dest / 'package-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = dest.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(p for p in dest.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
            info = zipfile.ZipInfo(str(Path(NAME) / p.relative_to(dest)), date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (p.stat().st_mode & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    return archive


def install(package, targets, replace):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for target in targets:
        dest = INSTALL_ROOTS[target] / NAME
        if dest.exists():
            if not replace:
                raise ValueError(f'{dest} exists; pass --replace to move it aside')
            dest.rename(dest.with_name(f'{NAME}.bak-{stamp}'))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(package, dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        print(f'Installed: {dest}')


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / f'dist/build-{stamp}' / NAME)
    parser.add_argument('--install', default='', help=f'comma list of {",".join(INSTALL_ROOTS)}')
    parser.add_argument('--replace', action='store_true', help='move an existing install to *.bak-<UTC>')
    args = parser.parse_args()
    targets = [t for t in args.install.split(',') if t]
    try:
        unknown = set(targets) - INSTALL_ROOTS.keys()
        if unknown:
            raise ValueError(f'unknown install target: {", ".join(sorted(unknown))}')
        archive = build(args.output)
        print(f'Package: {args.output}\nArchive: {archive}')
        install(args.output, targets, args.replace)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
