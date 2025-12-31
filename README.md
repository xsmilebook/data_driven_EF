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
  preprocess/        # QC銆佽〃鏍间笌琚瘯鍒楄〃
  functional_conn/   # FC 璁＄畻銆乑鍙樻崲銆佸悜閲忓寲銆佸彲瑙嗗寲
  metric_compute/    # 琛屼负鎸囨爣璁＄畻涓庡彲瑙嗗寲
  models/            # 鑴?琛屼负鍏宠仈锛堝缓妯?璇勪及/宓屽CV锛?
  scripts/           # 鍏ュ彛鑴氭湰 + HPC 鎻愪氦鑴氭湰
  result_summary/    # 姹囨€昏剼鏈?
```

## 蹇€熷紑濮嬶紙甯哥敤鑴氭湰锛?

### 1) 棰勫鐞?/ QC / 琚瘯鍒楄〃锛坄src/preprocess`锛?

#### `get_mri_sublist.py`锛氫粠 fmriprep 杈撳嚭鐩綍鍒楀嚭琚瘯
```bash
python src/preprocess/get_mri_sublist.py --dir <fmriprep_rest_dir> --out <mri_sublist.txt>
```
鍏抽敭鍙傛暟锛?
- `--dir`: fmriprep 鐩綍锛堝寘鍚?`sub-*`锛?
- `--out`: 杈撳嚭 txt 璺緞

#### `screen_head_motion_efny.py`锛氱粺璁?rest FD锛屽苟鏍囨敞鏈夋晥 run
```bash
python -m src.preprocess.screen_head_motion_efny --dataset EFNY --config configs/paths.yaml
```
鍏抽敭鍙傛暟锛?
- `--dataset`: 鏁版嵁闆嗗悕绉帮紙濡?`EFNY`锛?
- `--config`: 璺緞閰嶇疆锛坄configs/paths.yaml`锛?
- `--dataset-config`: 鍙€夛紝榛樿 `configs/paths.yaml`
- `--fmriprep-dir`: 鍙€夛紝浼氳鐩?dataset config 涓殑 `external_inputs.fmriprep_dir`
- `--out`: 杈撳嚭 CSV锛堝惈 `valid_subject` 涓?`meanFD`锛?

#### `preprocess_efny_demo.py`锛氭竻娲?demo 琛ㄥ苟鍚堝苟 QC
```bash
python -m src.preprocess.preprocess_efny_demo --dataset EFNY --config configs/paths.yaml
```
鍏抽敭鍙傛暟锛?
- `--dataset`: 鏁版嵁闆嗗悕绉帮紙濡?`EFNY`锛?
- `--config`: 璺緞閰嶇疆锛坄configs/paths.yaml`锛?
- `--dataset-config`: 鍙€夛紝榛樿 `configs/paths.yaml`
- `--input/-i`: 鍙€夛紝瑕嗙洊 demo 鍘熷 CSV 璺緞
- `--output/-o`: 鍙€夛紝瑕嗙洊 demo 澶勭悊鍚?CSV 璺緞
- `--qc-file/-q`: 鍙€夛紝瑕嗙洊 QC CSV 璺緞
- `--merged-output/-m`: 鍙€夛紝瑕嗙洊鍚堝苟杈撳嚭 CSV 璺緞
- `--log/-l`: 鍙€夛紝瑕嗙洊 log 璺緞

#### `generate_valid_sublists.py`锛氫粠 QC 琛ㄧ敓鎴愭湁鏁堣璇曞垪琛?
鏃犲懡浠よ鍙傛暟锛岀洿鎺ヨ繍琛岋細
```bash
python src/preprocess/generate_valid_sublists.py
```
娉ㄦ剰锛氭暟鎹牴鐩綍鍦ㄨ剼鏈唴甯搁噺 `DATA_ROOT`銆?

#### `build_behavioral_data.py`锛氬悎骞?demo 涓?metrics锛岀敓鎴?`EFNY_behavioral_data.csv`
```bash
python -m src.app_data_proc.build_behavioral_data --dataset EFNY --config configs/paths.yaml
```
鍏抽敭鍙傛暟锛?
- `--dataset`: 鏁版嵁闆嗗悕绉帮紙濡?`EFNY`锛?
- `--config`: 璺緞閰嶇疆锛坄configs/paths.yaml`锛?
- `--dataset-config`: 鍙€夛紝榛樿 `configs/paths.yaml`
- `--metrics/-m`: 鍙€夛紝瑕嗙洊 metrics CSV 璺緞
- `--demo/-d`: 鍙€夛紝瑕嗙洊 demo CSV 璺緞
- `--output/-o`: 鍙€夛紝瑕嗙洊杈撳嚭璺緞
- `--log/-l`: 鍙€夛紝瑕嗙洊 log 璺緞

### 2) 鍔熻兘杩炴帴锛團C锛夎绠椾笌鍚戦噺鍖栵紙`src/functional_conn`锛?

#### `compute_fc_schaefer.py`: single-subject FC matrix (CSV)
```bash
python -m src.functional_conn.compute_fc_schaefer --subject <sub-xxx> --n-rois 100 --dataset EFNY --config configs/paths.yaml
```
Key args:
- `--xcpd-dir`: xcp-d output root
- `--subject`: subject ID (e.g., `sub-xxx`)
- `--n-rois`: Schaefer parcels (100/200/400)
- `--qc-file`: QC CSV (valid runs)
- `--valid-list`: valid subject list
- `--out-dir`: output dir
- `--dataset`: dataset name (e.g., `EFNY`)
- `--config`: paths config (`configs/paths.yaml`)
- `--dataset-config`: optional, default `configs/paths.yaml`

#### `fisher_z_fc.py`: Fisher-Z transform FC (CSV)
```bash
python -m src.functional_conn.fisher_z_fc --subject <sub-xxx> --n-rois 100 --dataset EFNY --config configs/paths.yaml
```
Key args:
- `--in-dir`: FC input dir
- `--out-dir`: Z output dir
- `--subject`, `--n-rois`
- `--dataset`, `--config`

#### `convert_fc_vector.py`: vectorize FC_z lower triangle (npy)
Defaults infer `input_path`, `sublist_file`, and `output_path` from dataset/config.
```bash
python -m src.functional_conn.convert_fc_vector --dataset EFNY --config configs/paths.yaml --n_rois 400
```
Key args:
- `--input_path`: override FC_z input
- `--sublist_file`: override subject list
- `--output_path`: override vector output
- `--dataset_name`: override dataset name in outputs
- `--n_rois`: 100/200/400
- `--dataset`, `--config`, `--dataset-config`

#### `compute_group_avg_fc.py`: group-average FC (optional visualization)
```bash
python -m src.functional_conn.compute_group_avg_fc --dataset EFNY --config configs/paths.yaml --visualize
```
Key args:
- `--in-dir`: input dir containing `Schaefer*`
- `--sublist`: subject list
- `--atlas` / `--n-rois`: atlas selection
- `--out-dir`: output dir
- `--fig-dir`: figure output dir
- `--dataset`, `--config`, `--dataset-config`

#### `plot_fc_matrix.py`: matrix visualization (Yeo17 ordering)
```bash
python -m src.functional_conn.plot_fc_matrix --file <matrix.csv> --title "..." --yeo17 --n-rois 100 --dataset EFNY --config configs/paths.yaml
```
Key args:
- `--file`: input CSV
- `--out`: output PNG (optional)
- `--yeo17`: Yeo17 ordering (needs `--n-rois`)
- `--dataset`, `--config` (required when `--out` is omitted)

### 3) Behavioral metrics (`src/metric_compute`)

#### `compute_efny_metrics.py`: compute behavioral metrics from app data
```bash
python -m src.metric_compute.compute_efny_metrics --dataset EFNY --config configs/paths.yaml
```
Key args:
- `--data-dir`: override app-data directory
- `--out-csv`: override metrics CSV output
- `--dataset`, `--config`, `--dataset-config`

#### `metrics_similarity_heatmap.py`: metrics similarity heatmap
```bash
python -m src.metric_compute.metrics_similarity_heatmap --dataset EFNY --config configs/paths.yaml --method pearson
```
Key args:
- `--csv`: override behavioral CSV
- `--out-png`: override PNG output
- `--method`: pearson/spearman/kendall
- `--min-valid-ratio`: per-column valid ratio threshold
- `--min-pair-ratio`: pairwise valid ratio threshold
- `--dataset`, `--config`, `--dataset-config`

#### `behavioral_metric_exploration.py`锛氳涓烘寚鏍囨帰绱㈡€у彲瑙嗗寲
```bash
python -m src.metric_compute.behavioral_metric_exploration --dataset EFNY --config configs/paths.yaml
```
鍏抽敭鍙傛暟锛?
- `--dataset`: 鏁版嵁闆嗗悕绉帮紙濡?`EFNY`锛?
- `--config`: 璺緞閰嶇疆锛坄configs/paths.yaml`锛?
- `--dataset-config`: 鍙€夛紝榛樿 `configs/paths.yaml`
- `--behavioral_csv`: 鍙€夛紝瑕嗙洊琛屼负鏁版嵁 CSV 璺緞
- `--output_dir`: 鍙€夛紝瑕嗙洊鍥惧儚杈撳嚭鐩綍
- `--summary_dir`: 鍙€夛紝瑕嗙洊 CSV 姹囨€昏緭鍑虹洰褰?
- `--log`: 鍙€夛紝瑕嗙洊 log 璺緞

## 鑴?琛屼负鍏宠仈鍒嗘瀽锛坄scripts/run_single_task.py`锛?

璇ュ叆鍙ｈ剼鏈敮鎸侊細鐪熷疄鍒嗘瀽锛坄task_id=0`锛変笌鍗曟缃崲锛坄task_id>=1`锛夈€?

### 1) 鏈€甯哥敤鍛戒护
```bash
# 鐪熷疄鏁版嵁
python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --analysis-config configs/analysis.yaml --task_id 0 --model_type adaptive_pls

# 鍗曟缃崲锛堢瀛愮敱 task_id 鍐冲畾锛屼究浜?HPC array锛?
python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --analysis-config configs/analysis.yaml --task_id 1 --model_type adaptive_pls
```

### 2) 鍏抽敭鍙傛暟閫熸煡
- `--task_id`: 0=鐪熷疄锛?..N=缃崲
- `--model_type`: `adaptive_pls` / `adaptive_scca` / `adaptive_rcca`
- `--config`: 璺緞閰嶇疆锛坄configs/paths.yaml`锛?
- `--dataset`: 鏁版嵁闆嗗悕绉帮紙濡?`EFNY`锛?
- `--analysis-config`: 鍒嗘瀽榛樿閰嶇疆锛坄configs/analysis.yaml`锛屽彲閫夛級
- `--random_state`: 闅忔満绉嶅瓙锛堢湡瀹為噸澶嶈窇鏃跺父鐢級
- `--covariates_path`: 鍙€夛紝鍗忓彉閲?CSV锛堥渶鍖呭惈 `age/sex/meanFD`锛?
- `--cv_n_splits`: 澶栧眰 CV 鎶樻暟锛堥粯璁?5锛?
- `--max_missing_rate`: 缂哄け鐜囬槇鍊?
- `--output_dir`: 杈撳嚭鏍圭洰褰曪紙榛樿鍐欏叆椤圭洰 results 鐩綍锛?
- `--output_prefix`: 杈撳嚭鍓嶇紑
- `--save_formats`: `json` / `npz`
- `--log_level`, `--log_file`

鏌ョ湅瀹屾暣鍙傛暟锛?
```bash
python -m scripts.run_single_task --help
```

### 3) 宓屽浜ゅ弶楠岃瘉锛堟帹鑽愭鏋讹級
鐩爣锛氬灞傝瘎浼版硾鍖栬兘鍔涳紝鍐呭眰閫夋嫨瓒呭弬鏁帮紝鎵€鏈夐澶勭悊涓ユ牸鍦ㄨ缁冩姌鍐呭畬鎴愪互閬垮厤淇℃伅娉勯湶銆?

鎺ㄨ崘娴佺▼锛堟瘡娆¤繍琛岄兘閬靛惊鍚屼竴閫昏緫锛夛細
1. 澶栧眰 KFold锛坣_outer锛夊垝鍒嗚缁?娴嬭瘯銆?
2. 瀵规瘡涓灞傛姌锛?
   - 鍏堣繘鍏ュ唴灞?KFold锛坣_inner锛夛紝涓嶈鎻愬墠鍦ㄢ€滃灞傝缁冮泦鏁翠綋鈥濇嫙鍚堜换浣曢澶勭悊銆?
   - 鍐呭眰姣忎釜鎶橈細
     * 鍙敤鈥滃唴灞傝缁冩姌鈥濇嫙鍚堥澶勭悊锛堢己澶卞€煎～琛ャ€佸崗鍙橀噺鍥炲綊銆佹爣鍑嗗寲銆佸彲閫?PCA锛夈€?
     * 灏嗗悓涓€棰勫鐞嗗簲鐢ㄥ埌鈥滃唴灞傞獙璇佹姌鈥濓紝璁＄畻璇勫垎銆?
     * 瀵规瘡缁勫€欓€夊弬鏁伴噸澶嶄笂杩拌繃绋嬪苟姹囨€诲緱鍒嗐€?
   - 閫夋嫨鍐呭眰骞冲潎鍒嗘渶楂樼殑鍙傛暟锛堟帹鑽愭寚鏍囷細`meancorr`=鍚勬垚鍒嗙浉鍏崇郴鏁板潎鍊硷級锛涘骞跺垪锛岀敤鏇村皬鏂瑰樊鎴栨洿灏戝弬鏁颁綔涓?tie-break銆?
   - 鐢ㄦ渶浣冲弬鏁板湪鈥滃灞傝缁冮泦鏁翠綋鈥濋噸鏂版嫙鍚堥澶勭悊涓庢ā鍨嬶紙鍥哄畾鍙傛暟銆佹棤妯″瀷鍐呴儴 CV锛夛紝鍐嶅湪鈥滃灞傛祴璇曢泦鈥濊瘎浼般€?
3. 姹囨€诲灞傜粨鏋滐細鍚勬垚鍒嗙浉鍏崇郴鏁扮殑鍧囧€?鏂瑰樊绛夋暣浣撴寚鏍囥€?
4. 缃崲妫€楠岋細姣忔缃崲浠呮墦涔?Y锛堜繚鎸?X/鍗忓彉閲忕储寮曚竴鑷达級锛屽畬鏁撮噸澶嶄笂杩板祵濂楁祦绋嬪苟璁板綍绉嶅瓙銆?

瀹炵幇绾﹀畾锛堜究浜庡悗缁慨鏀癸級锛?
- 棰勫鐞嗘楠ょ殑 fit 鍙彂鐢熷湪璁粌鎶橈紙澶栧眰涓庡唴灞傞兘閬靛畧锛夈€?
- 妯″瀷鍐呴儴涓嶅啀鍋氶€夊弬鍨?CV锛涜秴鍙傛暟浠呯敱鍐呭眰 CV 鍐冲畾銆?
- 杈撳嚭鑷冲皯鍖呭惈锛歚outer_fold_results`銆乣inner_cv_table`銆乣outer_mean_canonical_correlations`銆乣outer_all_test_canonical_correlations`銆侀殢鏈虹瀛愪笌鍙傛暟缃戞牸瑙勬ā銆?

### 4) 缃崲妫€楠岋紙stepwise锛?
鐩爣锛氭寜鎴愬垎閫愭妫€楠岀 k 涓垚鍒嗭紙鎺у埗鍓?k-1 涓垚鍒嗭級锛岄伩鍏嶅悗缁垚鍒嗚鍓嶅簭鎴愬垎鈥滃甫鍑衡€濄€?

鎺ㄨ崘娴佺▼锛?
1. 鐪熷疄鏁版嵁锛坮eal锛夎繍琛屾椂锛屼繚瀛樻瘡涓灞?fold 鐨?loadings锛圶/Y锛夛紝浠ュ強姣忔姌瀵瑰簲鐨勭浉鍏冲悜閲忎笌璇勫垎銆?
   - 鍚屾椂淇濆瓨澶栧眰璁粌闆嗕笌娴嬭瘯闆嗙殑 X/Y scores锛屼究浜?stepwise 娈嬪樊鍖栥€?
2. 鍗曠嫭鑴氭湰姹囨€绘墍鏈?real 缁撴灉锛?
   - 姹囨€昏緭鍑猴細姣忎釜鎴愬垎鐨?real score锛坢ean/median锛変笌瀵瑰簲鐨?loadings锛坢ean锛夈€?
   - 鍚屾椂杈撳嚭鍏ㄦ牱鏈殑 train/test scores锛堟寜 fold 鍐呭钩鍧囷紝鍐嶈法 real 鍙?mean锛夈€?
3. 缃崲妫€楠岋紙perm锛夊姣忎釜鎴愬垎 k 杩涜 stepwise锛?
   - 璇诲彇 real 鐨?score 涓庡叏鏍锋湰 train scores锛坢ean锛夈€?
   - 灏?real 鐨勫墠 k-1 鎴愬垎 train scores 浣滀负棰濆鍗忓彉閲忓姞鍏?confounds銆?
   - 鍦ㄦ畫宸寲鍚庣殑鏁版嵁涓婃嫙鍚堝苟鎻愬彇绗?k 涓垚鍒嗗緱鍒嗭紝璁＄畻鐩稿叧浣滀负 perm score銆?
   - 鐢?perm 鍒嗗竷瀵?real 鐨勭 k 涓垚鍒?**涓綅鏁?* score 璁＄畻鍙冲熬鍗曚晶 p 鍊硷紙real 瓒婂ぇ瓒婃樉钁楋級銆?

杈撳嚭寤鸿锛?
- real锛氭瘡鎶?loadings + 姣忔姌 train/test scores + fold 绱㈠紩 + 姹囨€诲悗鐨?mean/median锛堟姤鍛婁腑浣嶆暟锛夈€?
- perm锛氭瘡涓?k 鐨?stepwise scores + p 鍊艰〃鏍硷紙CSV锛屽熀浜?real 涓綅鏁帮級銆?

### 5) HPC锛圫LURM锛?
寤鸿浣跨敤 submit_hpc_* 鎵归噺杩愯 real 涓?perm锛坮eal 涔熼渶瑕佸娆¤繍琛屼互鑾峰緱绋冲仴涓綅鏁扮粺璁★級銆?
浠撳簱鎻愪緵浜?2 涓ず渚嬫彁浜よ剼鏈紙鍙寜闇€瑕佹敼 `MODEL_TYPE`銆乤rray 鑼冨洿銆乴og 璺緞绛夛級锛?
```bash
sbatch scripts/submit_hpc_real.sh   # 澶氭鐪熷疄杩愯锛坅rray=0-10锛?
sbatch scripts/submit_hpc_perm.sh   # 缃崲杩愯锛坅rray=1-1000锛?
```

## 缁撴灉姹囨€伙紙real/perm 鎵弿锛?

Examples call directly into `src` modules.

```bash
python -m src.result_summary.summarize_real_perm_scores --dataset EFNY --config configs/paths.yaml --analysis_type both --atlas <atlas> --model_type <model>
python -m src.result_summary.summarize_real_loadings_scores --dataset EFNY --config configs/paths.yaml --atlas <atlas> --model_type <model>
python -m src.result_summary.summarize_perm_stepwise_pvalues --dataset EFNY --config configs/paths.yaml --atlas <atlas> --model_type <model>
python -m src.result_summary.visualize_loadings_similarity --dataset EFNY --config configs/paths.yaml --atlas <atlas> --model_type <model>
python -m src.result_summary.visualize_loadings_similarity_batch --dataset EFNY --config configs/paths.yaml --atlas <atlas> --model_type <model>
```

### `src/result_summary/summarize_real_perm_scores.py`
鎵弿 `results_root/real` 涓?`results_root/perm`锛屾彁鍙栨瘡娆¤繍琛岀殑鐩稿叧鍚戦噺骞惰緭鍑?CSV銆?

```bash
python -m src.result_summary.summarize_real_perm_scores --dataset EFNY --config configs/paths.yaml --analysis_type both --atlas <atlas> --model_type <model>
```

鍏抽敭鍙傛暟锛?
- `--results_root`: 缁撴灉鏍圭洰褰曪紙鍙€夛紱榛樿 `outputs/results`锛?
- `--analysis_type`: real / perm / both
- `--atlas`: 鍙€夎繃婊?
- `--model_type`: 鍙€夎繃婊?
- `--output_csv`: 杈撳嚭 CSV
- `--score_mode`: `first_component` / `mean_all`

### 蹇呴渶椤哄簭锛坰tepwise 缃崲锛?
1. 杩愯 real锛堝缓璁敤 `submit_hpc_real.sh` 鎵归噺璺戯級銆?
2. 姹囨€?real锛歚summarize_real_loadings_scores.py`锛堢敓鎴?real mean loadings + 鍏ㄦ牱鏈?train/test scores + score median锛岄粯璁よ緭鍑哄埌 `results_root/summary/<atlas>`锛夈€?
3. 杩愯 perm锛堝缓璁敤 `submit_hpc_perm.sh` 鎵归噺璺戯紝stepwise 浼氳鍙?real 姹囨€荤粨鏋滐級銆?
4. 姹囨€?perm锛歚summarize_perm_stepwise_pvalues.py`锛堝彸灏惧崟渚?p 鍊硷紝鍩轰簬 real 涓綅鏁帮紝榛樿杈撳嚭鍒?`results_root/summary/<atlas>`锛夈€?

### `src/result_summary/summarize_real_loadings_scores.py`
姹囨€?real 鐨勫灞?fold loadings 涓庣浉鍏冲垎鏁帮紝杈撳嚭 mean/median 缁撴灉锛堥粯璁ゅ啓鍏?`results_root/summary/<atlas>`锛夈€?

```bash
python -m src.result_summary.summarize_real_loadings_scores --dataset EFNY --config configs/paths.yaml --atlas <atlas> --model_type <model>
```

鍏抽敭鍙傛暟锛?
- 鏃狅紙榛樿杈撳嚭 mean 涓?median锛?

### `src/result_summary/summarize_perm_stepwise_pvalues.py`
姹囨€?perm 鐨?stepwise 鍒嗘暟骞惰绠楀彸灏惧崟渚?p 鍊硷紙瀵?real 涓綅鏁帮紝榛樿鍐欏叆 `results_root/summary/<atlas>`锛夈€?

```bash
python -m src.result_summary.summarize_perm_stepwise_pvalues --dataset EFNY --config configs/paths.yaml --atlas <atlas> --model_type <model>
```

## 杈撳嚭浣嶇疆锛堢害瀹氾級

甯哥敤杈撳嚭锛?
- QC / 涓棿缁撴灉锛歚data/interim/...`
- 琛ㄦ牸涓庡鐞嗗悗浜х墿锛歚data/processed/...`
- 鑴?琛屼负鍏宠仈缁撴灉锛歚outputs/results/...`锛堝彲鐢?`--output_dir` 鏀瑰啓锛?
- 澶у瀷鏁扮粍锛堝姣忔姌 `X_scores/Y_scores`锛変細淇濆瓨鍒板悓鐩綍鐨?`artifacts/`锛孞SON/NPZ 鍐呬粎淇濈暀绱㈠紩涓庤矾寰?


