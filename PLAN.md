# Plan

Update the EFNY decision+compute pipeline so each task is modeled with an assumption-consistent choice model (2AFC DDM only when valid), run hierarchical PyMC-based fits on the full sample via SLURM, and write back a single evidence-backed decision report under `docs/reports/`. Centralize all I/O via `configs/paths.yaml`, persist posterior traces for reproducibility, and save downloadable result tables under `data/processed/table/metrics/`.

## Scope
- In: DDM decision document (Chinese) + runnable scripts to compute the table’s model summaries + lightweight evaluation metrics (fit quality + condition effects) and report integration.
- In: True hierarchical PyMC-based models (HSSM), including random effects for `v/a/t/z` (DDM) and model comparisons (null vs effect) with LOO/WAIC.
- In: SLURM submission scripts (user submits) for multi-task parallel runs and multi-core single-task runs; posterior traces saved (netcdf).
- Out: Refactors unrelated to behavioral modeling, changes to repository structure, modifications to runtime artifacts already under `data/`/`outputs/` beyond the new results produced by these scripts.

## Action items
[ ] Audit task choice-sets on the full sample and enforce model eligibility (2AFC DDM only after recoding; multi-choice tasks use LBA), updating `docs/reports/ddm_decision.md`.
[ ] Implement 2-choice recoding datasets for `DT` (axis: `left/right` vs `down/up`, with `rule=axis`) and `EmotionSwitch` (dimension: `female/male` vs `happy/sad`, with `rule=dimension`), including logging/excluding cross-axis/dimension responses as `lapse/outlier`.
[ ] For `DT` and `EmotionSwitch`, fit two parallel hierarchical DDMs and compare them (Model A vs Model B), with condition effects on `v/a/t0`:
[ ] - Model A (Mixing): `v ~ block + rule + block:rule`, `a ~ block + rule + block:rule`, `t0 ~ block + rule + block:rule` (pure+mixed).
[ ] - Model B (Switch): `v ~ trial_type + rule + trial_type:rule`, `a ~ trial_type + rule + trial_type:rule`, `t0 ~ trial_type + rule + trial_type:rule` (mixed only).
[ ] Implement 4-choice LBA for `ColorStroop` and `EmotionStroop` (with congruency effects), and generate report-level collapsed choice rates: Target / Word / Other.
[ ] Implement `SST` go-only 2AFC hierarchical DDM (and keep SSRT/race modeling explicitly out-of-scope for the DDM table unless separately requested).
[ ] Standardize hierarchical model specification: subject random effects for `v/a/t/z` (DDM); condition effects on `v` by default, plus task-specific extensions (DT/EmotionSwitch: `a` and `t0` also vary by condition; FLANKER optional sensitivity: `a ~ congruency`); consistent priors/sampling settings in `configs/analysis.yaml`.
[ ] Add per-task model comparison (null vs effect; mixing+switch joint model where applicable) using LOO/WAIC and posterior predictive checks; write compact metrics tables for downstream plotting.
[ ] Persist outputs for cluster-to-local workflow: save `InferenceData` posterior traces (`.nc`) and lightweight summaries under `data/processed/table/metrics/` (paths resolved via `configs/paths.yaml`).
[ ] Add SLURM scripts patterned after `scripts/submit_hpc_real.sh` to run an array over `task×model` (and optionally `axis/dimension`), controlling threads/cores and writing logs to `outputs/logs/`.
[ ] Regenerate `docs/reports/ddm_decision.md` so it embeds the full-sample results tables and clearly flags any tasks still blocked by missing stimulus-to-choice mappings.
[ ] Update `docs/workflow.md`, `PROGRESS.md`, and add a dated session log under `docs/sessions/` describing how to run locally vs on SLURM and where artifacts are saved.

## Open questions
- For `EmotionStroop`, where is the trial-level mapping for “target category” vs “word-induced category” stored (extra columns in Excel vs an external stimulus dictionary)?
- For `DT/EmotionSwitch`, should cross-axis/dimension responses be excluded (recommended) or represented via a separate multi-choice model as a robustness check?
- What is the intended full-sample sampling budget on SLURM (draws/tune/chains; per-task walltime), and should we prioritize more chains or longer chains?
