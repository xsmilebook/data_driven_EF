# Project Structure

This document defines the stable folder structure and responsibilities of each directory in this project.

## High-level principles
- data/ and outputs/ are runtime artifacts and are NOT version-controlled
- src/ contains reusable library code only
- scripts/ are execution entry points (keep minimal; HPC + main runners)
- docs/ are the source of truth for methodology and decisions
- dataset-specific differences MUST be isolated under docs/datasets and configs/datasets
- reusable data live under data/processed; run artifacts live under outputs
- docs/README.md is the index for the documentation set

## Directory responsibilities

- src/
  Reusable Python modules. No hard-coded paths. No dataset-specific logic.

- scripts/
  Execution entry points. May parse config and dataset arguments.

- data/
  Raw and processed datasets. Never modified by git.
  - raw/: original inputs (not produced by pipeline scripts)
  - interim/: intermediate derivatives
  - processed/: reusable, cleaned outputs

- outputs/
  Run artifacts (results, figures, logs). Not tracked by git.

- docs/
  Human- and AI-readable documentation.
  - README.md: index of documentation files
  - workflow.md: reproducible pipeline
  - methods.md: paper-level methodology
  - datasets/: dataset-specific notes
  - report/: research planning and summaries
  - sessions/: AI/development session logs (one file per day, organized by yy/mm/dd.md)
  - notes/: user notes and free-form ideas
  - logs/: runtime logs (not tracked by git)

## Rules for AI-assisted modification
- Do not modify data/ or outputs/ without explicit instruction
- Do not change directory names without explicit instruction
- If src/ logic changes, update docs/workflow.md accordingly
