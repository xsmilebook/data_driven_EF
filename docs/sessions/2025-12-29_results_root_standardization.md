# Session log (2025-12-29): results_root standardization

## Scope

- Engineering updates only (paths, CLI wiring, documentation).
- No changes to scientific or statistical logic.

## Changes

- Added `src/result_summary/cli_utils.py` to resolve `results_root` from `--dataset/--config`.
- Updated result summary scripts to accept `--dataset`/`--config` and default `results_root` to `outputs/<DATASET>/results`.
- Updated `docs/workflow.md` and `README.md` to document the standardized invocation pattern and results root.
- Updated `PROGRESS.md` to reflect completed items.

## Verification

- `python -m py_compile src/result_summary/cli_utils.py src/result_summary/summarize_real_loadings_scores.py src/result_summary/summarize_perm_stepwise_pvalues.py src/result_summary/summarize_real_perm_scores.py src/result_summary/visualize_loadings_similarity_batch.py`
- `python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --dry-run --task_id 0 --model_type adaptive_pls`

