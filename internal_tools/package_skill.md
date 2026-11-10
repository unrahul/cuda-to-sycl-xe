# Standalone skill package

Build a fresh output with `python3 internal_tools/package_skill.py --output dist/cuda-to-sycl-xe`. The output directory must not already exist. The builder creates a matching ZIP archive with fixed archive timestamps and a per-file SHA-256 manifest.

The allowlist includes the skill entrypoint, portable references, source examples, helper scripts, project template, tests and UI metadata. It excludes private source archives, hardware measurements, profiler services, binaries and Python caches. Public source URLs and pinned repository revisions remain in `references/sources.md`. The package needs a compatible SYCL compiler/runtime and an actual target device for GPU execution; it does not bundle a toolchain.

Run the skill validator and the package's CPU tests before installation. Use `--install` on a fresh build to copy the standalone package into the active Codex skills directory. Existing installations are never overwritten. A local installation becomes discoverable on the next turn. Keep maintainer measurement tools and archived references in this workspace.
