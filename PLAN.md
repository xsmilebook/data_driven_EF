# PLAN.md

Strategic blueprint for the ongoing refactor of `data_driven_EF`.

## Project objectives and success criteria

Objective: complete an engineering-only refactor that makes the project runnable and maintainable under the frozen directory structure defined in `PROJECT_STRUCTURE.md`, without altering scientific/analysis logic.

Success criteria:

- Entry points are standardized and runnable via `python -m scripts.<entry>`.
- All filesystem paths are centralized in `configs/paths.yaml` and `configs/datasets/<DATASET>.yaml`.
- Outputs and intermediate data follow the `data/raw` / `data/interim` / `data/processed` / `outputs` conventions.
- No dataset-specific constants remain inside reusable `src/**` modules (except dataset adapters, if explicitly defined).
- Documentation reflects the runnable workflow and configuration model.

## Technical requirements and constraints

- Do not change directory names or structure.
- Do not modify runtime artifacts under `data/` or `outputs/`.
- Do not change statistical/algorithmic logic.
- Avoid new heavy dependencies; prefer small, internal utilities.
- Maintain compatibility with Windows and Linux; bash scripts run on the cluster only.

## Architecture decisions and rationale

- Treat `src/` as importable library code and reserve `scripts/` for entry points.
- Use `configs/paths.yaml` as the single source of truth for filesystem roots.
- Use `configs/datasets/<DATASET>.yaml` for dataset-specific inputs and assumptions (including external inputs).
- Standardize `results_root` as `outputs/<DATASET>/results` and keep summaries under that tree.

## Feature roadmap and development phases

### Phase A: Path and entry-point standardization (current)
- Standardize path resolution and external inputs.
- Normalize intermediate and processed outputs to `data/interim` and `data/processed`.
- Make summary scripts accept `--dataset/--config` to infer `results_root`.

### Phase B: Documentation and workflow alignment
- Update `README.md` and `docs/workflow.md` to reflect standardized CLI patterns.
- Ensure `docs/data_dictionary.md` and `docs/datasets/EFNY.md` match the refactored paths.

### Phase C: Import hygiene and thin entry points
- Remove residual `sys.path` hacks inside `src/**`.
- Add thin `scripts/` entry points for frequently used `src/**` tools.

## Testing and deployment strategy

- Use `python -m py_compile` on modified Python modules.
- Validate configuration wiring with `python -m scripts.run_single_task --dataset <DATASET> --config configs/paths.yaml --dry-run`.
- For cluster scripts, validate syntax with `bash -n` on Linux.

