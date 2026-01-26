# AGENTS.md

Operational rules for AI-assisted work in `data_driven_EF`.
This repository treats `ARCHITECTURE.md` as the source of truth for the stable layout. Do not propose structural changes unless explicitly requested.

## Base workflow

1) Read `ARCHITECTURE.md` before planning or editing.
2) Execute changes only according to `PLAN.md` or explicit user instructions.
3) If a change affects usage or structure, update `README.md` and/or `docs/workflow.md`.
3.1) Once code is modified or new results are generated, supplementary documentation must be added or updated to explain them.
3.2) Scripts with short runtimes that do not require parallelism can be executed directly; otherwise, perform few-shot testing in the same environment and provide the submission command or sbatch script.
3.2.1) sbatch submissions are user-only: The assistant must not directly submit sbatch jobs during a session; instead, output reusable sbatch scripts and clear submission commands for the user to execute on the cluster.
3.3) All sbatch tasks use multi-partition by default: #SBATCH -p q_fat_c,q_fat,q_fat_l (to facilitate scheduling to q_fat/q_fat_l).
4) Update `PROGRESS.md` after completing a change set.
5) Log each session in `docs/sessions/` (date-stamped file).
6) If code or documentation is modified, commit changes via git (write your own commit message).
7) If the `create-plan` skill is used, write the plan to the root `PLAN.md`.

## Scope and constraints

- Default scope is documentation-only unless the user requests engineering changes.
- Do not modify runtime artifacts under `data/` or `outputs/`.
- Do not change directory names or structure unless explicitly requested.
- Keep diffs minimal; avoid refactors not tied to the request.
- Clean up temporary smoke-test outputs under `temp/` (e.g., `temp/smoke_*`) immediately after validation; do not leave test artifacts in the repo.

## Engineering conventions (when requested)

- Use `python -m scripts.<entry>` for execution.
- Avoid ad-hoc `sys.path` hacks; if unavoidable, confine them to entry points and document why.
- All filesystem paths must come from `configs/`.

## Documentation conventions

- Use precise, scientific language.
- Write all `docs/` content in Chinese; keep filenames in English.
- Root-level Markdown files must be English only.
- Keep repository root folder names in English.
- EFNY-specific notes live in `docs/workflow.md` and `configs/paths.yaml` (dataset section).
