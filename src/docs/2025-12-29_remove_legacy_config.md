# 2025-12-29 remove legacy config.json

## Summary
- Removed `src/models/config.json` and the legacy `--config_file` entry point.
- Updated modeling utilities to rely solely on `configs/paths.yaml` and `configs/datasets/<DATASET>.yaml`.

## Rationale
- Eliminates obsolete configuration paths and keeps all settings centralized in `configs/`.
