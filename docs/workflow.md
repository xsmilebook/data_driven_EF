# Workflow (engineering)

This document describes the standardized invocation pattern for script entry points under the frozen repository structure defined in `PROJECT_STRUCTURE.md`.

## Standard CLI pattern

All runnable entry points under `scripts/` should accept:

- `--dataset <DATASET_NAME>`
- `--config <PATH_TO_CONFIGS_PATHS_YAML>`

Dataset-specific assumptions are configured in `configs/datasets/<DATASET>.yaml`.

## Data/outputs conventions

- `data/raw/<DATASET>/`: raw inputs (not produced by pipeline scripts)
- `data/interim/<DATASET>/`: intermediate products (e.g., preprocessing outputs such as MRI- and connectivity-stage derivatives)
- `data/processed/<DATASET>/`: processed, reusable data products (e.g., tables, vectorized FC features)
- `outputs/<DATASET>/`: run artifacts (results, figures, logs)

Some external inputs (e.g., fMRIPrep outputs) may live outside the repository; configure these as absolute paths under `external_inputs` in `configs/datasets/<DATASET>.yaml`.

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

They set `#SBATCH --chdir` to the cluster project root and write SLURM stdout/stderr plus per-task logs under `outputs/EFNY/logs/...`.
