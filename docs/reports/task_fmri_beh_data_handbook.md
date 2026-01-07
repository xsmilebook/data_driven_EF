## 共同前提：CSV 里“哪一行算一次 trial？”

PsychoPy 的这种宽表输出里，**每一行基本对应一次 trial**，但会夹杂少数“非 trial 行”（instruction、block cue、结束 fixation 等），判断方法很稳定：

* **N-back / Switch**：用 `Trial_text.started` 是否为 NaN

  * `Trial_text.started` 非空 ⇒ 这一行是一次 trial（数字真正出现的时刻）
* **SST**：用 `Trial_image_1.started` 是否为 NaN

  * `Trial_image_1.started` 非空 ⇒ 这一行是一次 trial（猴子/香蕉那组图片开始呈现的时刻）

另外：`MRI_Signal_s.started` 在三份表里都有，说明你的程序在等待扫描触发（或模拟触发）；你后续做 fMRI 对齐时可以把它当作“扫描起点”参考（你前面写的方案是对的）。

---

# 任务 1：N-back（你这份数据里 **只有 pure blocks**，没有 mixed block）

### 1) 你图里的规则如何映射到表格？

在 CSV 里有这几个关键列：

* `Task_img`：告诉你这行 trial 属于哪种任务规则（你这份数据只有两种）

  * `material/0back_name.png`
  * `material/2back_name.png`
* `Trial`：呈现的数字
* `Ans`：正确答案编码（在你的数据里非常清楚：**1=左键/食指，2=右键/中指**）
* `key_resp.keys / key_resp.rt`：被试按键与反应时

我用表格验证了你的规则确实是：

* **0-back（“找数字 5”）**：`Trial==5` ⇒ `Ans=1`（左键）；否则 `Ans=2`（右键）
* **2-back**：`Trial == 两个 trial 之前的 Trial` ⇒ `Ans=1`；否则 `Ans=2`
  （而且每个 2-back block 的前两个 trial 都是 `Ans=2`，符合你图里“开始前两次按右键”的说明——只是这里对应的是“前两个 trial 无法形成 2-back match”）

> 注意：你图里写的是“与上一个数字是否相同”，但你这份数据的行为规则严格符合 **2-back**（不是 1-back）。如果你论文里要写任务描述，建议以代码/列表为准（或把图片文案修正为 2-back）。

### 2) 刺激出现顺序（trial 内部时间结构）

每个 trial 的程序结构在表里是稳定的（看 `Trial_fix.*`、`Trial_text.*`、`key_resp.*`）：

* `Trial_fix`（注视）≈ **0.50 s**
* `Trial_text`（数字呈现）≈ **2.00 s**
* `key_resp.started` 和 `Trial_text.started` **同一时刻**开始 ⇒ 说明反应窗与刺激呈现同步开启
* trial-to-trial 间隔基本是 **2.5 s**（只有 block 间隔会显著变长，见下）

### 3) 如何确认 block（以及你这份数据的 block 结构）

你的 N-back 有非常“硬”的 block 标记，优先级从强到弱：

**最强证据（直接给 block ID）**

* `Trial_loop_list`：每个 trial 都标着它来自哪个 excel 列表文件
  你这份数据一共 6 个 block，每个 20 trials（共 120）：

| block 顺序 | Trial_loop_list（block ID） | 任务类型   | trials |
| -------- | ------------------------- | ------ | ------ |
| 1        | `0back_block3...`         | 0-back | 20     |
| 2        | `2back_block1...`         | 2-back | 20     |
| 3        | `0back_block2...`         | 0-back | 20     |
| 4        | `2back_block2...`         | 2-back | 20     |
| 5        | `0back_block1...`         | 0-back | 20     |
| 6        | `2back_block3...`         | 2-back | 20     |

**强证据（block cue）**

* `Task_name.started / Task_name.stopped`：每个 block 开始前有一个**约 5 s**的提示屏（block 名称/提示图）
* 并且你每个 block 之间都有一个**约 15 s 的间隔**（加上 5s cue，总共 ~20s），这在时间上也能把 block 切得很干净

**备选证据（循环索引）**

* `Task_loop.thisTrialN`：在你这份数据里是 0,1,2,3,4,5（正好 6 个 block）
* `Trial_loop.thisTrialN`：在每个 block 开始会**重置为 0**（第一行 trial 的索引）

✅ 结论：你这份 nback 数据是 **纯 block 设计（0-back / 2-back 交替）**，没有 mixed block。
如果你“实验设计上确实应该有 mixed block”，那要么是**别的被试/版本**，要么是 mixed 的定义不是“同一段里交替规则”，而是你后续分析里人为合并出来的“mixed state”。

---

# 任务 2：Switch（这里同时有 pure block + mixed block，而且表里非常清楚）

### 1) 你图里的规则如何映射到表格？

关键列：

* `Cond_img`：边框颜色

  * `material/red.png`（红框：看数值大小）
  * `material/blue.png`（蓝框：看字体大小）
* `Font`：字体大小（你这里是两个水平：0.2 / 0.4）
* `Trial`：数字
* `Ans`：正确按键（同样 1=左，2=右）

表格里规则被严格编码为：

* 红框：`Trial < 5` ⇒ `Ans=1`；否则 `Ans=2`（你这份数据里似乎没有 Trial==5）
* 蓝框：`Font==0.2`（小字）⇒ `Ans=1`；`Font==0.4`（大字）⇒ `Ans=2`

### 2) 刺激出现顺序（trial 内部）

结构与 nback 非常类似：

* `Trial_fix` ≈ **0.51 s**
* `Trial_text` ≈ **2.01 s**
* `key_resp.started` 与 `Trial_text.started` 同步 ⇒ 反应窗与数字呈现同步
* trial-to-trial ≈ **2.5 s**

### 3) 如何确认 block（以及 pure/mixed 的判定）

你的 switch 的 block 标记同样很硬：

* `Trial_loop_list` / `Task_img` 直接标了 3 段（共 144 trials）：

  1. `nonswitch1_trial_loop.xlsx`（32 trials）

     * `Cond_img` **只有 red** ⇒ 这是 **pure（红规则）**
  2. `nonswitch2_trial_loop.xlsx`（32 trials）

     * `Cond_img` **只有 blue** ⇒ 这是 **pure（蓝规则）**
  3. `switch_trial_loop.xlsx`（80 trials）

     * `Cond_img` 同时有 red 和 blue ⇒ 这是 **mixed block**

同样也存在：

* 每段前 `Task_name` cue ≈ 5s
* 段间固定间隔 ≈ 15s（加 cue 总 ~20s）
* `Task_loop.thisTrialN` 在段首递增（0/1/2）
* `Trial_loop.thisTrialN` 在段首重置为 0

> 额外一层：如果你在 mixed block 内还要区分 **switch trial vs repeat trial**，用 `Cond_img` 相对上一 trial 是否变化即可（红↔蓝变化就是 switch）。

---

# 任务 3：SST（你这份数据是 **120 trials 单段**；black banana = no-go）

### 1) 图里的规则如何映射到表格？

关键列：

* `Position`：香蕉出现在左/右（你这里是两个取值：`(-0.3,0)` / `(0.3,0)`）
* `bad`：是否出现“黑香蕉”（no-go）

  * `bad` 非空（这里是 `material/Banana_1.png`） ⇒ **no-go trial**
  * `bad` 为空 ⇒ **go trial**
* `Ans`：go trial 的正确按键（1=左，2=右）；no-go trial 的 `Ans` 是 NaN（因为正确反应是不按）
* `key_resp.keys / key_resp.corr`：按了什么、是否正确

  * 你这份数据里：no-go trial 不按键时 `key_resp.corr=1`；按键时 `corr=0`（非常标准）

你这份 SST 的构成是：

* 总 trials：120
* go：90
* no-go（黑香蕉）：30
* 左/右位置总体各 60；在 go trial 内左/右各 45（平衡得很好）

### 2) 刺激出现顺序（trial 内部）

SST 的刺激是图片组件而不是 `Trial_text`，在你的表里：

* `Trial_fix` ≈ **0.50 s**
* `Trial_image_1.started`（主要图片开始）在 trial 开始后 ≈ **0.516 s**
* `key_resp.started` 与 `Trial_image_1.started` 基本同步 ⇒ 反应窗从主要图片出现时开始（约 2s）
* `Trial_image_3.started` 比 `Trial_image_1.started` 晚 **约 0.2–0.4 s**
  这非常像“黑香蕉/覆盖层”这种延迟出现的组件——也与你之前设计里“bad 用 Trial_image_3.started 做 onset”一致。

### 3) 如何确认 block（以及你这份数据是否分段）

SST 这份 CSV **没有**像前两个任务那样的 `Trial_loop_list`（只有一个 Trial_loop），所以 block/分段主要靠：

* **trial 计数 + trial 间隔**：你这里 trial-to-trial 基本一直是 2.5s，没有中途超长间隔
* `Block_fix.started/stopped`：你这份表里确实有一个 **15s 的 Block_fix**，但它出现在**最后一个 trial 结束后**（更像结束后的 rest），不是中途分段

✅ 结论：你这份 SST 是 **单段（state_part1）**，没有 mid-break 进入第二段。
如果你有 180 trials 的版本，通常会看到：

* 在 trial ~90 附近出现一次 `Block_fix`（15s），同时 `Trial.started` 会出现一个明显大间隔；据此即可把 part1/part2 切开。

---

## 给你一个“实操级”的 block 确认流程（你后面批量处理别的被试也能用）

### N-back / Switch（强标记任务）

1. 先筛 trial 行：

   * N-back/Switch：`Trial_text.started` 非空
2. 用 `Trial_loop_list` 分段（最稳）
3. 交叉核对：段首是否 `Trial_loop.thisTrialN==0`，以及 `Task_name.started` 是否只在段首非空
4. 再核对“pure vs mixed”：

   * pure：段内 `Task_img`（nback）或 `Cond_img`（switch）只有 1 个取值
   * mixed：段内出现 ≥2 个取值

### SST（弱标记任务）

1. trial 行：`Trial_image_1.started` 非空
2. 若出现中途的 `Block_fix.started`（且发生在最后一个 trial 之前），以它作为 part1/part2 边界
3. 备选：用 `Trial.started` 的相邻差值找“超长间隔”（比如 > 8–10s）作为分段点

---

如果你愿意，我也可以按你后续“任务回归建模”的需求，把**每个任务的 block/state 序列**和**event 序列（stimulus / response / banana-no-go）**直接从这三份 CSV 里导出成统一的事件表（onset/duration/trial_type/state），这样你做 HRF/FIR 或者 BGFC 的 nuisance/task regression 会更省事。
