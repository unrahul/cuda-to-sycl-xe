# Standalone skill package

```sh
python3 internal_tools/package_skill.py                                  # dist/build-<UTC>/cuda-to-sycl-xe + .zip
python3 internal_tools/package_skill.py --install codex,claude,opencode  # also install
python3 internal_tools/package_skill.py --install claude --replace       # move an existing install to *.bak-<UTC>
```

The package contains `SKILL.md`, `references/`, `examples/` (sources only), `scripts/`, `templates/`, `tests/` and `agents/openai.yaml`. Maintainer evidence, upstream clones, measurements and binaries stay here.

The build fails unless the package passes its own `tests/test_docs.py`, which checks that:

- every documented code block is a compiled snippet;
- links resolve and every reference is reachable from `SKILL.md`;
- frontmatter and size limits hold;
- no maintainer path or benchmark-harness wording ships.

The ZIP has fixed timestamps, and `package-manifest.json` lists per-file SHA-256 hashes.

Install targets are `~/.codex/skills`, `~/.claude/skills` and `~/.config/opencode/skills`. An existing install is never overwritten. `--replace` moves it aside first.

Before installing a release, run on one GPU:

```sh
source /opt/intel/oneapi/setvars.sh
ONEAPI_DEVICE_SELECTOR=level_zero:3 scripts/smoke.sh
```
