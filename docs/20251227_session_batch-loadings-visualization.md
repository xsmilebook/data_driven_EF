# Session Record: batch loadings similarity visualization

## Summary
- Added a batch script to generate loadings similarity plots for every model/seed under a given atlas.

## Decisions
- Batch scope is restricted to `results_root/real/<atlas>/**/seed_*/result.json`.

## Actions Taken
- Created `src/result_summary/visualize_loadings_similarity_batch.py` to iterate runs and call the per-seed visualization logic.
- Exposed `run_for_result_dir` helper in `src/result_summary/visualize_loadings_similarity.py`.

## Artifacts
- `src/result_summary/visualize_loadings_similarity_batch.py`: batch entrypoint.
- `src/result_summary/visualize_loadings_similarity.py`: shared helper for per-seed execution.

## Open Questions
- None.

## Next Steps
- Run the batch script for the target atlas to generate all outputs.
