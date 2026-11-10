# Skill evaluations

Maintainer material; not shipped.

- `trigger-queries.json`: 10 prompts that should load the skill and 10 that should not. Use them with skill-creator's description-optimization loop to tune the `description` field.
- `evals.json`: task prompts in skill-creator format, aimed at the failures seen in the November 2026 pilot (build flags, kernel names, oneMKL linking, capability gating) and at generation coverage. Tasks 2 and 5 still need their fixtures under `evals/fixtures/`.

The paired GPU comparison (with skill vs without) uses `benchmarks/xe-skill-pilot/repair/`. Give both arms identical harness rules in the task prompt. The skill must not contain harness policy.
