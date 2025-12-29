# Session log (2025-12-29): SLURM path rendering and analysis defaults

## Scope

- Engineering changes only (path/config wiring, import hygiene).
- No algorithmic/statistical changes.

## Changes

- Added `scripts/render_paths.py` to render dataset roots from `configs/paths.yaml` + `configs/datasets/<DATASET>.yaml` for bash usage.
- Updated cluster-facing bash scripts to use `eval "$(python -m scripts.render_paths ...)"` instead of hard-coded `data/...` paths:
  - `src/functional_conn/submit_compute_fc.sh`
  - `src/functional_conn/submit_fisher_z.sh`
  - `src/functional_conn/cluster_convert_fc_vec.sh`
  - `src/preprocess/batch_run_xcpd.sh`
  - `src/preprocess/xcpd_36p.sh`
- Added `configs/analysis.yaml` (dataset-agnostic defaults) and wired it into `scripts/run_single_task.py` for `--max_missing_rate` and `--cv_n_splits` when those CLI flags are not provided.
- Removed remaining `sys.path` hacks under `src/` where possible and standardized imports (e.g., `src/app_data_proc/check_missing_beh_metrics.py`).

## Verification

- `python -m scripts.render_paths --dataset EFNY --config configs/paths.yaml --format bash`
- `python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --dry-run --task_id 0 --model_type adaptive_pls`

