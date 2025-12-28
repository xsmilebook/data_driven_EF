# AGENTS.md

Project rules and conventions for data_driven_EF.

## Scope and layout
- Core workflow lives under `src/` with folders:
  - `preprocess/` (QC, subject lists, tables)
  - `functional_conn/` (FC compute, Z, vectorize, visualize)
  - `metric_compute/` (behavior metrics)
  - `models/` (model definitions + nested CV)
  - `scripts/` (entrypoints + HPC)
  - `result_summary/` (aggregation)
- Project documentation lives under `docs/` as Markdown only:
  - `report/research_plan/` (old project plans)
  - `datasets/EFNY_behavioral_metrics/` (old behavior docs)
  - `datasets/EFNY_neuroimg_prep/` (neuroimaging docs)
  - `datasets/reportEFNY_models_pipeline/` (modeling pipeline docs)
- Project execution records live under `src/docs/` as Markdown only (session notes, run logs, analysis summaries).
- Every Codex session should produce or update a session record in `src/docs/`.

## Data storage rules
- `data/EFNY/` is the canonical data root; its subfolders store all datasets and outputs.
- All results must be saved under `data/EFNY/` (not in `results/` outside the data tree).
- Result organization under `data/EFNY/results/` (create subfolders as needed, keep names stable):
  - `behavior_data/` for behavioral cleaning outputs and visualizations.
  - `real/` for real runs reaults.
  - `perm/` for permutation results.
  - `figures/` for figures not tied to one pipeline stage.

## Modeling rules (strict)
- **Only nested CV is allowed.** Model-internal CV has been removed; hyperparameters are chosen via inner CV and evaluated on the outer fold.
- All preprocessing (imputation, confound regression, standardization, optional PCA) must be fit **inside the training split only** and applied to validation/test.
- Use `src/scripts/run_single_task.py` as the single entrypoint for real (`task_id=0`) and perm (`task_id>=1`) runs.
- Supported model types: `adaptive_pls`, `adaptive_scca`, `adaptive_rcca`.

## Stepwise permutation rules
- Real runs must save per-fold loadings and train/test scores.
- Summaries are produced by `summarize_real_loadings_scores.py`.
- Permutation uses **stepwise** regression:
  - For component `k`, regress out real train scores for components `1..k-1` (both X and Y scores) as extra confounds.
  - Compute the perm score for component `k` after residualization.
- p-values are **right-tail** single-sided against the **median** real score.
- Real/perm outputs are tracked **per model type**.

## Summary outputs (per-model)
- Summaries are written to:
  - `results_root/summary/<atlas>/<model_type>/real_*`
  - `results_root/summary/<atlas>/<model_type>/perm_stepwise_pvalues.csv`
- Always pass `--atlas` and `--model_type` when running summary scripts:
  - `summarize_real_loadings_scores.py`
  - `summarize_perm_stepwise_pvalues.py`

## HPC usage
- Use `src/scripts/submit_hpc_real.sh` for repeated real runs.
- Use `src/scripts/submit_hpc_perm.sh` for permutation runs.
- Run order for stepwise:
  1) real runs
  2) real summary
  3) perm runs
  4) perm p-values summary

## Execution environment
- In CMD, activate the Conda environment before running scripts:
  - `conda activate ML`

## Outputs
- Results default to `results/real/...` and `results/perm/...` (override via `--output_dir`).
- Large arrays (e.g., `X_scores`, `Y_scores`, loadings) are stored in `artifacts/` with references in JSON/NPZ.
