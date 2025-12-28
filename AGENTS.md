# AGENTS.md

Operational rules for AI-assisted work in `data_driven_EF`.

This repository treats `PROJECT_STRUCTURE.md` as the source of truth for the stable layout. Do not propose structural changes unless explicitly requested.

## What the AI may modify (default)

Unless a user request explicitly expands scope, the AI may only:

- Edit Markdown documentation under `docs/` (including `docs/report/` and `docs/datasets/`).
- Add new Markdown documentation files under `docs/` when consolidation or clarification is needed.

## What is forbidden (hard constraints)

The AI must not:

- Modify any code, configuration, or scripts under `src/`, `scripts/`, `configs/`, or `tests/`.
- Modify or generate runtime artifacts under `data/` or `outputs/` (including adding, deleting, renaming, or rewriting files).
- Change directory names or the directory structure defined by `PROJECT_STRUCTURE.md`.
- Invent results, numerical findings, or performance claims. Hypotheses and planned analyses must be labeled as such.

## Documentation rules

- Prefer precise, scientific language.
- Separate dataset-specific assumptions (e.g., EFNY file conventions) from general workflow statements.
- When describing implementation details, reference the authoritative script/module path; avoid paraphrasing that could diverge from code.
- Avoid embedding large tables of computed results in documentation unless they are directly produced by the pipeline and can be regenerated.

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
- Add/update a session note under `docs/sessions/` describing what changed and why.


## Project conventions (informational; do not change code)

These are operational conventions documented for orientation:

- `data/` and `outputs/` are runtime artifacts and are not version-controlled.
- Dataset-specific documentation lives under `docs/datasets/` and should not leak into general workflow statements.
- Modeling evaluations must use nested cross-validation and avoid information leakage (fit preprocessing inside training splits only). If documentation mentions modeling, it must reflect this rule.
