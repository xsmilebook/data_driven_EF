# Data dictionary (work in progress)

This document is a lightweight, dataset-agnostic index of key tables and commonly used fields. Dataset-specific schemas and assumptions belong in `docs/datasets/<DATASET>.md`.

## EFNY (current focus)

Canonical dataset documentation:

- `docs/datasets/EFNY.md`

Commonly referenced tables (paths are conventions; see `configs/paths.yaml`):

- `data/processed/EFNY/table/demo/EFNY_demo_processed.csv`: cleaned demographics.
- `data/processed/EFNY/table/demo/EFNY_demo_with_rsfmri.csv`: demographics merged with rs-fMRI QC (includes motion summary fields such as `meanFD` if available).
- `data/processed/EFNY/table/metrics/EFNY_beh_metrics.csv`: per-subject behavioral metrics wide table.
- `data/processed/EFNY/table/demo/EFNY_behavioral_data.csv`: merged demographics + behavioral metrics (if generated).
- `data/processed/EFNY/table/sublist/sublist.txt`: analysis subject list (if generated).

## Conventions

- Identifier columns vary by data source; prefer an explicit `subid` column for cross-table joins.
- Numeric behavioral metrics should be stored as numeric columns; any missing values must be represented consistently (e.g., empty/NA → parsed as NaN).
