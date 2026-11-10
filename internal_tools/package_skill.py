#!/usr/bin/env python3
"""Build a standalone skill without private sources, measurements or binaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFS = ['backend-selection.md','cuda-to-sycl-mapping.md','dense-compute.md','kernel-rules.md','performance.md','playbook.md','production-patterns.md','pytorch-xpu.md','quantized-compute.md','testing.md','xe-architectures.md']

def portable_text(name, text):
    # The allowlisted source documents are themselves the shareable instructions.
    return text

def sources():
    repos=[]
    # Authoring/evaluation frameworks remain private maintainer references.
    kernel_repos = ['KernelBench', 'SyclKittens', 'TensorFold', 'ThunderKittens',
                    'Triton8', 'Xe-Forge', 'composable_kernel', 'triton',
                    'vllm-xpu-kernels', 'vllm', 'sgl-kernel-xpu', 'sglang',
                    'sycl-tla', 'oneDNN']
    for name in kernel_repos:
        folder = ROOT/'internal_references/upstream'/name
        if not (folder/'.git').exists():
            raise ValueError(f'missing pinned kernel source: {name}')
        import subprocess
        url=subprocess.check_output(['git','-C',str(folder),'remote','get-url','origin'],text=True).strip().removesuffix('.git')
        sha=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
        repos.append(f'- [{folder.name}]({url}/tree/{sha}), revision `{sha}`. Review the retained upstream license before code reuse.')
    # The source URLs are retained, not the copyrighted document bytes or local receipts.
    manifest=(ROOT/'internal_references/MANIFEST.md').read_text()
    links=[]
    for label,url in re.findall(r'\[([^]]+)\]\((https?://[^)]+)\)',manifest):
        if (label,url) not in links:links.append((label,url))
    return '# Public source catalog\n\nPinned implementation references. Inspect the matching operation contracts, build requirements, tests and licenses before reuse.\n\n'+'\n'.join(repos)+'\n\n## Specifications\n\n'+'\n'.join(f'- [{label}]({url})' for label,url in links)+'\n'

def build(dest):
    if dest.name != 'cuda-to-sycl-xe':raise ValueError('output folder must be named cuda-to-sycl-xe')
    if dest.exists():raise ValueError('output directory already exists; choose a new output')
    dest.mkdir(parents=True)
    for name in ['SKILL.md',*[f'references/{p}' for p in REFS]]:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(portable_text(Path(name).name,(ROOT/name).read_text()))
    (dest/'references/sources.md').write_text(sources())
    for name in ['scripts','examples','templates','tests']:
        shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('bin','__pycache__','*.pyc'))
    # These helpers require machine-specific infrastructure; keep them with maintainers.
    for path in (dest/'examples').rglob('*'):
        if path.is_file() and path.suffix not in ('.py','.cpp','.cu','.sycl','.h','.hpp','.sh','.md'):path.unlink()
    (dest/'agents').mkdir()
    (dest/'agents/openai.yaml').write_text('interface:\n  display_name: "Intel Xe Kernel Engineering"\n  short_description: "Port, validate and optimize SYCL kernels on Intel Xe"\n  default_prompt: "Use $cuda-to-sycl-xe to build and validate this kernel on the selected Intel GPU."\n')
    # Fail rather than silently shipping dependencies on private workspace artifacts.
    for path in dest.rglob('*.md'):
        text=path.read_text()
        if '/home/rahul' in text or 'internal_references/' in text or 'internal_tools/' in text:
            raise ValueError(f'nonportable content in {path.relative_to(dest)}')
        if re.search(r"(?:this host|our (?:tests|runs)|exercised with|(?:executed|validated|qualified|tested) on)", text, re.IGNORECASE):
            raise ValueError(f'local run-state wording in {path.relative_to(dest)}')
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if '://' not in link and not link.startswith('#') and not (path.parent/link.split('#')[0]).exists():
                raise ValueError(f'broken link in {path}: {link}')
    files={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file()}
    (dest/'package-manifest.json').write_text(json.dumps({'name':'cuda-to-sycl-xe','files':files},indent=2)+'\n')
    archive=dest.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob('*')):
            if not p.is_file():continue
            info=zipfile.ZipInfo(str(Path(dest.name)/p.relative_to(dest)),date_time=(1980,1,1,0,0,0))
            info.external_attr=(p.stat().st_mode & 0xffff)<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    return archive

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'dist/cuda-to-sycl-xe')
    ap.add_argument('--install',action='store_true')
    a=ap.parse_args()
    try:
        archive=build(a.output)
        print(f'Package: {a.output}\nArchive: {archive}')
        if a.install:
            target=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'skills/cuda-to-sycl-xe'
            if target.exists():raise ValueError(f'installed path exists; not overwriting: {target}')
            shutil.copytree(a.output,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','bin'))
            print(f'Installed: {target}')
    except (ValueError,OSError) as exc:ap.error(str(exc))
if __name__=='__main__':main()
