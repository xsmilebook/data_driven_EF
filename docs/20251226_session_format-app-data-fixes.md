# Session Record: format app data fixes

## Summary
- Added reference-based data fixes for SST and Emotion n-back sheets during app data formatting.

## Actions Taken
- Updated `src/app_data_proc/format_app_data.py` to replace SST SSRT and Emotion1Back/Emotion2Back item columns when trigger conditions are met, using the reference file in the same folder.

## Artifacts
- `src/app_data_proc/format_app_data.py`
