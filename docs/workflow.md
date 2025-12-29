# Workflow (engineering)

This document describes the standardized invocation pattern for script entry points under the frozen repository structure defined in `PROJECT_STRUCTURE.md`.

## Standard CLI pattern

All runnable entry points under `scripts/` should accept:

- `--dataset <DATASET_NAME>`
- `--config <PATH_TO_CONFIGS_PATHS_YAML>`

Example:

```bash
python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --dry-run
```

Dataset-specific assumptions are configured in `configs/datasets/<DATASET>.yaml`.
Dataset-agnostic analysis defaults may be configured in `configs/analysis.yaml`.

## Data/outputs conventions

- `data/raw/<DATASET>/`: raw inputs (not produced by pipeline scripts)
- `data/interim/<DATASET>/`: intermediate products (e.g., preprocessing outputs such as MRI- and connectivity-stage derivatives)
- `data/processed/<DATASET>/`: processed, reusable data products (e.g., tables, vectorized FC features)
- `outputs/<DATASET>/`: run artifacts (results, figures, logs)

Some external inputs (e.g., fMRIPrep outputs) may live outside the repository; configure these as absolute paths under `external_inputs` in `configs/datasets/<DATASET>.yaml`.

## Results root

Summary scripts under `src/result_summary/` default to:

- `outputs/<DATASET>/results`

You can override this with `--results_root` when needed.

Example:

```bash
python -m src.result_summary.summarize_real_perm_scores --dataset EFNY --config configs/paths.yaml --analysis_type both --atlas <atlas> --model_type <model>
```

## Quick sanity check (dry-run)

From the repository root:

```bash
python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --dry-run
```

This validates imports and configuration resolution without reading `data/` or writing `outputs/`.

## Cluster execution (SLURM)

Submission scripts:

- `scripts/submit_hpc_real.sh`
- `scripts/submit_hpc_perm.sh`

They set `#SBATCH --chdir` to the cluster project root and write SLURM stdout/stderr plus per-task logs under `outputs/<DATASET>/logs/...`.
Note: SLURM `#SBATCH --output/--error` paths are static and cannot expand environment variables, so they remain dataset-specific in the script headers.

Example:

```bash
sbatch scripts/submit_hpc_real.sh
```
