# 2025-12-29 screen_head_motion entry point

## Summary
- Added a thin entry point `scripts/run_screen_head_motion.py` that calls the existing preprocess module.
- Updated `README.md` to document the dataset/config-based invocation.
- Recorded progress status in `PROGRESS.md`.

## Rationale
- Aligns preprocessing utilities with the standardized `python -m scripts.*` pattern.
- Keeps path resolution centralized in configs without changing scientific logic.
