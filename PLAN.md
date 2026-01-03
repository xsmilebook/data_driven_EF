# Plan

Create a DDM decision report under `docs/reports/` based on the provided task-by-task table, then implement and run the corresponding model computations and basic effectiveness checks so the report contains both decisions and evidence. Keep all I/O path resolution centralized via `configs/paths.yaml` and avoid writing to `data/` or `outputs/`.

## Scope
- In: DDM decision document (Chinese) + runnable scripts to compute the table’s model summaries + lightweight evaluation metrics (fit quality + condition effects) and report integration.
- Out: Full Bayesian hierarchical HDDM sampling pipelines (unless explicitly requested/approved after feasibility check), GPU/HPC deployment, refactoring unrelated preprocessing/metrics code.

## Action items
[ ] Translate the screenshot table into a structured “decision matrix” section in a new report `docs/reports/ddm_decision.md` (tasks, trial inclusion rules, recommended model forms, risks).
[ ] Implement a trial-level extractor for the required tasks (Flanker/Stroop family + DT/EmotionSwitch/DCCS + SST go-only option) using `configs/paths.yaml` roots and the existing EFNY normalization utilities.
[ ] Add a DDM fitting backend: default to a deterministic/fast approximation (e.g., EZ-diffusion per subject and condition) and optionally support installing/running HDDM if the environment supports it.
[ ] Add evaluation routines (per-task: N included subjects/trials, parameter stability checks, condition-effect estimates, and model-vs-baseline comparisons) and write results as tables for the report.
[ ] Run the pipeline end-to-end on the dataset (or a configurable subset), regenerate the decision report with computed tables, and sanity-check outputs against the exploratory trial counts.
[ ] Update `docs/workflow.md`, `PROGRESS.md`, and add a dated session log under `docs/sessions/` describing what was changed, how to reproduce, and known risks/limitations.

## Open questions
- Should we target true hierarchical HDDM (PyMC-based) now, or is a fast approximation acceptable for the first iteration?
- Run scope: all subjects in `data/raw/behavior_data/cibr_app_data/` or a specified subset (e.g., first N / QC-passed list)?
- Where should derived result tables live: embedded in the Markdown report only, or also as a committed CSV under `docs/reports/`?
