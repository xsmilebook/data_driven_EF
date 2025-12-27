# Session Record: loadings similarity visualization

## Summary
- Added a script to visualize cross-fold loading similarity and compute cosine similarity tables for X/Y loadings.

## Decisions
- Save outputs under each seed run directory in a `loadings_similarity` subfolder to stay within the results tree.

## Actions Taken
- Created `src/result_summary/visualize_loadings_similarity.py` to load per-fold loadings, compute cosine similarity, and plot PCA scatter + heatmaps.

## Artifacts
- `src/result_summary/visualize_loadings_similarity.py`: new script for loadings similarity visualization and summary tables.

## Open Questions
- None.

## Next Steps
- Run the script on target result folders to generate figures and CSV summaries.
