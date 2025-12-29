# AGENTS.md

Operational rules for AI-assisted work in `data_driven_EF`.

This repository treats `PROJECT_STRUCTURE.md` as the source of truth for the stable layout. Do not propose structural changes unless explicitly requested.

## What the AI may modify

Default scope is documentation-only. Engineering changes are permitted only when explicitly requested by the user for a specific task.

- Documentation (default): Markdown under `docs/` and this file (`AGENTS.md`).
- Engineering (only when requested): `src/**`, `scripts/**`, and `configs/**`, limited to packaging/imports, config wiring, and filesystem path handling. Do not change scientific/analysis logic unless the user explicitly requests it.

## What is forbidden (hard constraints)

The AI must not:

- Modify code/config/scripts outside an explicit user request and stated scope.
- Modify or generate runtime artifacts under `data/` or `outputs/` (including adding, deleting, renaming, or rewriting files).
- Change directory names or the directory structure defined by `PROJECT_STRUCTURE.md`.
- Invent results, numerical findings, or performance claims. Hypotheses and planned analyses must be labeled as such.

## Engineering conventions (when code changes are requested)

- Treat `src/` as importable code; prefer `python -m scripts.<entry>` for execution.
- Avoid ad-hoc `sys.path` hacks; if unavoidable, keep them in script entry points only and document why.
- All filesystem paths must come from configuration under `configs/`; do not introduce absolute paths or dataset-specific constants inside `src/**`.
- Keep diffs minimal and reviewable; avoid refactors unrelated to the stated task.

## Documentation rules

- Prefer precise, scientific language.
- Separate dataset-specific assumptions (e.g., EFNY file conventions) from general workflow statements.
- When describing implementation details, reference the authoritative script/module path; avoid paraphrasing that could diverge from code.
- Avoid embedding large tables of computed results in documentation unless they are directly produced by the pipeline and can be regenerated.
- Keep `PLAN.md` and `PROGRESS.md` current for refactor scope changes and completion status.
- When using the `create-plan` skill, update `PLAN.md` in the same change set and carry its action items into `PLAN.md`.

## How changes should be proposed

For any documentation change:

1) Identify the target audience (new contributor vs. analyst vs. reproducibility reviewer).
2) State the scope and non-goals (e.g., “docs-only; no pipeline changes”).
3) Preserve terminology and stable filenames where feasible; if a file is consolidated, keep a short “pointer stub” to avoid breaking links.
4) Prefer edits that are easy to verify against the repository (paths, script names, run order).

## How changes should be verified (docs-only)

Minimum verification before finishing a docs edit:

- Consistency check against `PROJECT_STRUCTURE.md` (paths, responsibilities, and “do not touch” areas).
- Link sanity: internal links and referenced paths should exist in the repo.
- No “results-like” claims: remove or rephrase anything that reads like an observed effect unless it is explicitly documented as a plan or as a pipeline output.
- Add/update a session note under `docs/sessions/` describing what changed and why (use `docs/notes/` for user ideas and free-form notes, not session logs).


## Project conventions (informational; do not change code)

These are operational conventions documented for orientation:

- `data/` and `outputs/` are runtime artifacts and are not version-controlled.
- Dataset-specific documentation lives under `docs/datasets/` and should not leak into general workflow statements.
- Modeling evaluations must use nested cross-validation and avoid information leakage (fit preprocessing inside training splits only). If documentation mentions modeling, it must reflect this rule.
