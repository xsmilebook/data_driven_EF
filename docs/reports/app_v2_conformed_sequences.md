# APP v2：已确认序列 JSON 的统一命名与导出（conformed）

## 目标

将 `run_corrected_v2/sequence_library` 中已确认正确的序列 JSON 统一命名，并放置到：

- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed`

以便后续流程只依赖单一、稳定命名的“确认版”文件。

## 命名规则

- `visit1_confirmed.json`
- `visit3_confirmed.json`
- `visit3_v1_confirmed.json`

## 来源映射

- `visit1_confirmed.json` <- `visit1_merged.json`
- `visit3_confirmed.json` <- `visit3_from_groups.json`
- `visit3_v1_confirmed.json` <- `visit3_v1_from_groups.json`

并在 `conformed/manifest.json` 中记录来源路径。

## 执行方式

```bash
python -m scripts.export_conformed_sequences_v2 --dataset EFNY_THU
```

脚本：

- `scripts/export_conformed_sequences_v2.py`

## 说明

- 当前实现采用“复制到 conformed”而非移动，避免影响既有依赖 `sequence_library` 旧文件名的脚本。
- 后续若要完全切换到 conformed，可在下游脚本中仅读取 `conformed/*.json` 并逐步淘汰旧命名引用。
