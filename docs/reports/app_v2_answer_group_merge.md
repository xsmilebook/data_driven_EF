# APP v2: answer group merge fixes (SST/EmotionStroop)

本次操作目的：将 `group_003/group_005/group_006` 中可通过规则修正的被试，合并回 `group_001`。

## 规则

- SST：若试次数 > 96，则截断为前 96 个 trial（删除第 96 个 trial 之后的行）。
- EmotionStroop：若第 13 个 trial 的 `正式阶段正确答案` 为空，且除第 13 个 trial 外其余答案与 `run_corrected_v2/sequence_library/visit1_merged.json` 一致，则用 `AN` 填充第 13 个 trial。
  - 备注：`visit1_merged.json` 的答案当前来自 item group_001 模板工作簿中抽取到的“观测答案模板”，仅用于定位“单个空值导致的分组差异”并进行可审计修补；不再依赖 `app_sequence` 作为答案真值。

## 执行摘要

- dry_run: False
- reference subject (group_001[0]): THU_20231203_144_LSC
- candidates: 136
- moved_to_group_001: 136

备注：本次按人工确认将 group_003/group_005/group_006 全部合并进 group_001；
下表中的 `match_key_tasks_to_group1_ref` 仅作为审计信号（SST/EmotionStroop 是否与 group_001 参考被试一致）。

## 修正明细

| subject_id | from_group | sst_truncated | sst_old_rows | sst_new_rows | emostroop_fill_trial13 | match_key_tasks_to_group1_ref | match_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| THU_20231014_131_ZXM | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231014_132_LYX | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231119_140_XJR | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231119_141_SYQ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231119_142_SRE | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231119_143_SRE | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231203_146_LHZ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231203_147_WHR | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231209_148_WRX | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231209_149_WZX | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231209_150_BYK | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20231209_153_ZZY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20240710_318_ZKM | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20240720_332_WRK | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20240720_336_XYS | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20240831_390_ZJS | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20240831_392_YQH | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20241201_438_GJJ | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250104_475_MMY | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250104_476_ZZY | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250104_477_ZMY | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250104_479_GMR | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250104_480_YJQ | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250104_481_LJS | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250106_482_ZXX | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250106_485_GYT | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250106_487_ZSL | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250107_488_YDM | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250107_489_LJN | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250107_491_ZHQ | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250108_494_FM | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250108_495_XZL | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250108_498_CSH | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250109_502_LXY | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250109_503_XZ | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250109_504_DYR | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250109_505_FHR | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250109_506_ZHL | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250109_507_LYX | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250109_508_LPW | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250110_512_LYS | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250110_514_ZWQ | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250111_515_KCY | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250111_517_ZJJ | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250111_518_SY | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250111_519_XYJ | group_005 | 1 | 99 | 96 | 0 | YES |  |
| THU_20250120_521_SRL | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250120_525_GP | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250120_526_WJR | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250121_527_LWT | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250121_528_HSQ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250121_530_WXS | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250121_531_LCY | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250122_534_HJX | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250122_535_LMZ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250122_536_LYY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250122_537_ZXL | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250123_539_LSC | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250123_540_ZL | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250124_541_LZC | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250124_543_YRQ | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250124_544_YZQ | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250125_545_HXY | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250125_546_LSC | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250125_547_WYF | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250125_548_WQX | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250125_549_WQZ | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250125_550_XZY | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250210_552_HSL | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250210_553_YKY | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250210_554_WF | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250210_555_YJR | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250211_558_YJH | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250211_559_HYL | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250211_560_QXY | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250213_557_HSY | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250213_561_SYN | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250213_563_YJY | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250213_564_WXY | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250214_566_JYN | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250214_567_JYH | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250214_568_SXH | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250214_569_ZTY | group_006 | 1 | 99 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250214_570_WQX | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250222_571_LYP | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250222_571_WSX | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250222_572_WSX | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250222_573_DQ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250301_574_LSY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250301_575_GJD | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250301_576_ZLL | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250308_577_BLY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250308_578_SY | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250308_579_MQY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250308_580_ZXY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250308_581_ZZC | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250315_583_LZY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250315_584_DSY | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250315_585_WJC | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250315_586_CYJ | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250315_587_WML | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250322_588_ZNY | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250329_589_WZN | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250329_590_CXR | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250329_591_ZWX | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250419_592_WJ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250419_593_HY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250419_594_MZZ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250419_595_SZT | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250510_596_CYH | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250517_597_CYH | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250517_598_ZYF | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250524_599_JLY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250524_600_ZPY | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250607_601_ZYY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250607_602_ZWZ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250607_603_LYH | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250607_604_LYX | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250614_605_ZSQ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250614_606_FZA | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250614_607_WL | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250621_608_LYF | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250621_609_YSM | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250621_610_LXF | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250621_611_LR | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250628_612_ZWJ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250628_613_LQY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250628_614_ZYT | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250628_615_WWX | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250628_616_MZQ | group_003 | 0 | 96 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250707_617_LXY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250707_618_ZJH | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250707_619_FYC | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250707_620_LPZ | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250707_621_LJL | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |
| THU_20250708_624_LXY | group_003 | 1 | 97 | 96 | 0 | NO | mismatch_task=EmotionStroop |

