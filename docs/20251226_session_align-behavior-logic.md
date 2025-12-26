# Session Record: align EFNY metrics with R scripts

## Summary
- Aligned EFNY behavioral metric logic with `docs/scripts/behave_function.R` and `docs/scripts/EFNY_behave_ana.R`.

## Actions Taken
- Updated trial preparation to preserve original order and restored strict min_prop gating.
- Matched n-back trial typing and RT summaries to R definitions.
- Reworked SST and Go/No-Go computations to mirror R logic.
- Aligned switch, conflict, KT, and ZYST RT/ACC handling with R outputs.
- Synced task config flags (e.g., KT/ZYST filter_rt).

## Artifacts
- `src/metric_compute/efny/preprocess.py`
- `src/metric_compute/efny/metrics.py`
- `src/metric_compute/efny/main.py`
