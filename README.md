# data_driven_EF

鏈粨搴撳寘鍚?EF锛堟墽琛屽姛鑳斤級鐮旂┒鐨勭鍒扮娴佺▼锛?
- 鏁版嵁 QC / 琚瘯鍒楄〃
- 鍔熻兘杩炴帴锛團C锛夎绠椾笌鍚戦噺鍖?
- 琛屼负鎸囨爣璁＄畻
- 鑴?琛屼负鍏宠仈鍒嗘瀽锛堜粎淇濈暀鑷€傚簲妯″瀷锛歚adaptive_pls` / `adaptive_scca` / `adaptive_rcca`锛?

涓嬮潰鍙繚鐣?*鏈€甯哥敤鑴氭湰鐨勬墽琛屾柟寮忎笌鍏抽敭鍙傛暟**銆?

## 鐩綍缁撴瀯锛堟牳蹇冿級

```
src/
  imaging_preprocess/    # 影像预处理与功能连接
  behavioral_preprocess/ # 行为数据预处理与指标计算
  models/                # 脑-行为关联（建模评估/嵌套CV）
  scripts/               # 入口脚本 + HPC 提交脚本
  result_summary/        # 结果汇总脚本
```

## 杈撳嚭浣嶇疆锛堢害瀹氾級

甯哥敤杈撳嚭锛?
- QC / 涓棿缁撴灉锛歚data/interim/...`
- 琛ㄦ牸涓庡鐞嗗悗浜х墿锛歚data/processed/...`
- 鑴?琛屼负鍏宠仈缁撴灉锛歚outputs/results/...`锛堝彲鐢?`--output_dir` 鏀瑰啓锛?
- 澶у瀷鏁扮粍锛堝姣忔姌 `X_scores/Y_scores`锛変細淇濆瓨鍒板悓鐩綍鐨?`artifacts/`锛孞SON/NPZ 鍐呬粎淇濈暀绱㈠紩涓庤矾寰?





