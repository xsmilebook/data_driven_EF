# Session note (2025-12-28): documentation restructure

## Scope

- Documentation-only changes.
- No code changes.
- No modifications to `data/` or `outputs/`.

## Rationale

The documentation for the EFNY dataset and the overall research plan had become fragmented across multiple files and mixed dataset-specific assumptions with general workflow statements. This session consolidates EFNY dataset documentation into a single canonical document and rewrites the research plan to match the current modular, multi-dataset repository design.

## Changes

- Rewrote `AGENTS.md` to clearly define what the AI may modify, what is forbidden, and how documentation changes should be proposed and verified.
- Consolidated EFNY dataset documentation into `docs/datasets/EFNY.md`.
  - Removed the older EFNY-specific dataset documents that are now redundant.
- Rewrote `docs/report/research_plan.md` to reflect:
  - a modular, dataset-agnostic workflow layer,
  - EFNY as the current focus,
  - ABCD as an explicitly planned (not assumed) future extension.

## Follow-ups (optional)

- If/when ABCD work begins, create `docs/datasets/ABCD.md` and document ABCD-specific ingestion/QC and alignment assumptions.
- Consider populating `docs/workflow.md` and `docs/methods.md` (currently empty) with dataset-agnostic pipeline descriptions once their intended audience is defined.
