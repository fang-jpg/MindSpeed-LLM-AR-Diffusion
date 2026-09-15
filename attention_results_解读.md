# 本次 attention 泄漏测试结果逐项解读

分析对象：仓库根目录的五份 `attention_*.json`。本文读取已有结果，不重新运行模型，也不修改测试或训练代码。

## 1. 这次结果说明了什么

**训练到 global_step_5000 的 checkpoint，在本次单 NPU、eager attention、三块短序列测试中，没有检测到当前/未来 clean target 泄漏。三个真实 checkpoint 报告各 300 项，全部通过。**

证据不只是总状态 PASS：当前/未来 clean 扰动的 logits 差异全部记录为 0；禁止位置 embedding 梯度全部记录为 0；正确前缀的扰动和梯度明确非零；每层实际 kernel mask 全部符合预期。

`attention_injected.json` 的 FAIL 是预期的检测器验证结果：我们故意让随机小模型读取当前 clean block，它确实被四种方法抓到了。它不代表训练 checkpoint 出现泄漏。

| 原始报告 | 实际实验 | 检查条数 | 状态 | 解释 |
|---|---|---:|---|---|
| [attention_tiny.json](attention_tiny.json) | 随机小模型，2 层，FP32，eager | 48 | 48 PASS | 正常结构下测试正常完成 |
| [attention_injected.json](attention_injected.json) | 同类小模型，故意放开当前 clean block | 48 | 36 PASS、12 FAIL | 人为制造的泄漏被抓住，符合预期 |
| [attention_checkpoint_bf16.json](attention_checkpoint_bf16.json) | step 5000，28 层，BF16，batch=2，全 MASK | 300 | 300 PASS | 真实权重低精度检查通过 |
| [attention_checkpoint_partial.json](attention_checkpoint_partial.json) | 同一 checkpoint，BF16，batch=2，约一半 MASK | 300 | 300 PASS | 存在未遮盖 noisy token 时也通过 |
| [attention_checkpoint_fp32_eager.json](attention_checkpoint_fp32_eager.json) | 同一 checkpoint，FP32，batch=1，全 MASK | 300 | 300 PASS | 高精度检查也通过 |

合计 996 条检查记录：984 PASS，12 个预期的注入实验 FAIL，没有 ERROR 或 INCONCLUSIVE。
这些是程序检查记录，不是 996 个独立统计样本，不能据此计算“无泄漏概率”。

真实 checkpoint 三份报告使用相同的权重目录：

```text
/share/dataset/x00840191/model_train/ckpt/phase4_cpt_ar_duffision_0910/models/global_step_5000
```

## 2. 先看懂位置，后面的数字才有意义

真实 checkpoint 的 `block_size=8`，`num_blocks=3`，所以每条原始序列 L=24。
模型实际拼成 `[noisy 24 个位置, clean 24 个位置]`，总长 48：

| 分支/块 | 拼接后的绝对位置 | 含义 |
|---|---|---|
| noisy Block0 | 0–7 | 第一块的带噪输入 |
| noisy Block1 | 8–15 | 第二块的带噪输入 |
| noisy Block2 | 16–23 | 第三块的带噪输入 |
| clean Block0 | 24–31 | 第一块的原始 token |
| clean Block1 | 32–39 | 第二块的原始 token |
| clean Block2 | 40–47 | 第三块的原始 token |

编号从 0 开始：Block1 是第二块，layer=0 是第一层。
`noisy Block1` 允许看同块 noisy 8–15、过去的 clean 24–31；不允许看当前 clean 32–39、未来 clean 40–47，也不允许看其他 noisy 块。
因此“看到同块 noisy”是算法允许的，不能把它与“偷看同块 clean 答案”混淆。

`clean_ids` 数组本身长度是 24；其中下标 8 的元素进入 clean 分支后，绝对位置是 24+8=32。
`outputs.logits` 已切出 noisy 半段，因此它的下标 8 对应 noisy 绝对位置 8。

## 3. 一个检查项的名字怎样拆开读

例如：

```text
train/seed=0/方法一/clean块1→noisy块1/当前
```

| 部分 | 意思 |
|---|---|
| train | 走模型的训练 attention 分支；没有更新权重，dropout/recompute 已关闭 |
| seed=0 | 使用随机种子 0 生成受控输入；另有种子 1、2 |
| 方法一 | 扰动输入，看输出是否改变 |
| clean块1 | 改动对象是第二块 clean token |
| →noisy块1 | 观察第二块 noisy 的所有位置、整个词表的 logits |
| 当前 | 改动块和观察块编号相同，因此应禁止依赖 |

`eval` 是本测试匹配位置编码后的评估分支，也提供 labels 并执行 block_diff 前向，不是自由生成或 KV cache 推理测试。
核对五份报告后，去掉名称中的 train/eval 前缀，各自同种子的检查字段和数值完全一致。
这说明本次两条受测路径表现一致，但脚本没有额外保存并直接逐元素比较 train/eval 原始 logits，不能把它说成一个独立的完整 logits 等价性证明。

## 4. 方法一：为什么有的差异应该为零，有的应该很大

模型在每个位置给词表中的候选 token 一个分数，称为 **logit**。它还不是概率，不限制在 0 到 1。
脚本先得到 baseline，然后只修改 clean，固定 noisy、位置编码、加噪位置和概率，再比较 noisy 输出。

实际修改方式是目标 clean 块的每个 token ID 加 1（词表范围内取模）；不是重新抽取噪声。
部分 MASK 测试同样固定整条 noisy 输入，包括原本未 MASK 的 token，因此不会混入 noisy 同时变化造成的影响。

### 4.1 差异指标

令 d 为“修改后 logit−修改前 logit”，比较范围是当前被观察块的所有 batch、token 位置、词表项。

| 字段 | 数学/直观含义 | 判定作用 |
|---|---|---|
| `max_abs` | max(abs(d))，所有元素中变化最大的那个 | 脚本实际用它与阈值比较 |
| `mean_abs` | mean(abs(d))，所有元素的平均变化量 | 帮助理解影响是否广泛，不单独判定 |
| `l2` | sqrt(sum(d²))，整体差异长度 | 元素越多可能越大，不是 loss，也不是概率 |
| `threshold` | 本项阈值，本次为 1e-6=0.000001 | 禁止依赖要求 max_abs≤阈值 |
| `expected` | `应不变` 或 `应变化（正对照）` | 决定较大的差异是正常还是异常 |
| `status` | 本项最终判断 | PASS 只表示符合本项期望，不代表输出预测正确 |

对于允许的 clean 前缀，要求 max_abs>阈值；否则是 INCONCLUSIVE，不能确认测试对输入影响足够敏感。
对于当前/未来 clean，要求 max_abs≤阈值；否则 FAIL。

### 4.2 逐个解释实际九项结果

下面完整列出 `attention_checkpoint_bf16.json` 的 `train/seed=0`：

| 修改的 clean 块 | 观察的 noisy 块 | 关系 | 实测 max_abs | 为什么 PASS |
|---:|---:|---|---:|---|
| 0 | 0 | 当前 | 0 | 第一块不能读取自己的 clean 答案 |
| 0 | 1 | 前缀 | 9.40625 | 第二块可以依赖第一块 clean，变化正常 |
| 0 | 2 | 前缀 | 4.6875 | 第三块可以依赖第一块 clean，变化正常 |
| 1 | 0 | 未来 | 0 | 第一块不能读取第二块 clean |
| 1 | 1 | 当前 | 0 | 第二块不能读取自己的 clean 答案 |
| 1 | 2 | 前缀 | 7.234375 | 第三块可以依赖第二块 clean，变化正常 |
| 2 | 0 | 未来 | 0 | 第一块不能读取第三块 clean |
| 2 | 1 | 未来 | 0 | 第二块不能读取第三块 clean |
| 2 | 2 | 当前 | 0 | 第三块不能读取自己的 clean 答案 |

例如 clean0→noisy1 的 `mean_abs=1.218668818473816`、`l2=2414.79248046875`，
表示大量 logit 元素累积产生的差异，不是训练 loss 为 2414，也不是泄漏程度 2414。
前缀本来允许影响输出，因此这些非零数值是正对照有效的证据。

### 4.3 重复性单独在检查什么

`方法一/重复性` 没改任何输入，只重复两次 forward。真实三个报告的全部重复性 max_abs 都是 0。
这说明这些实验的重复 baseline 没有记录到数值抖动，后面观察到的前缀变化可以与 baseline 随机性区分。
它本身不证明没有泄漏，只是后续比较的前提。

## 5. 方法二：逐个解释真实 kernel mask 的字段

真实 BF16 报告第一层为：

```json
{
  "layer": 0,
  "q_shape": [2, 16, 48, 128],
  "k_shape": [2, 8, 48, 128],
  "mask_shape": [2, 1, 48, 48],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": [],
  "noisy_block1_query": 8,
  "visible_key_positions": [8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31]
}
```

| 字段 | 每个值是什么意思 |
|---|---|
| `layer` | attention 层索引；0–27 共 28 层 |
| `q_shape` | [batch 条数=2，query 头数=16，query 位置数=48，每头维度=128] |
| `k_shape` | [batch 条数=2，key 头数=8，key 位置数=48，每头维度=128] |
| `mask_shape` | [batch=2，共用 mask 的头维=1，query 行数=48，key 列数=48] |
| `is_causal` | false 表示不额外要求 kernel 套普通逐 token 三角 causal 规则；可见性由自定义 4D mask 决定 |
| `dropout` | 0.0 表示 attention 不随机丢弃连接，保证扰动实验稳定 |
| `wrong_entries` | 实际 mask 与预期规则不符的元素数量，允许错屏蔽和禁止错放开都会计入；这里为 0 |
| `first_wrong_qk` | 最多列出前 10 个错误元素坐标；每项为 [batch 索引,mask 头维索引,query 位置,key 位置]；空列表表示没列出错误 |
| `noisy_block1_query` | 为便于人工查看，选 noisy 第二块的第一个 query，位置 8 |
| `visible_key_positions` | 该 query 实际可见的 key 位置；这里只展示 batch0、mask 头维0、第一层这一行 |

Q 有 16 个头、K 有 8 个头，是分组查询注意力的结构，不是维度错误。
`mask_shape` 的头维 1 表示各 query 头共用一张可见关系表，不表示模型只有一个 attention 头。

**is_causal=false 并不等于完全不屏蔽。** 当前需要同块 noisy 双向可见，以及只访问更早 clean，
所以用自定义 mask 表达规则。真实可见位置恰好是同块 noisy 8–15 和过去 clean 24–31，没有当前 clean 32–39 或未来 clean 40–47。

mask 内允许元素为 0，禁止元素为 -inf（检查器也接受足够负的 finfo.min）。
这是加在 attention score 上的 mask；禁止元素进入 softmax 后不分配正常注意力权重。
`visible_key_positions` 表示允许参与注意力，不表示这些 token 的实际 attention 权重都一样大。

还有一条独立检查：

```text
方法二/所有层 kernel 均被观察
layers=[0,1,...,27]
expected_layers=28
```

`layers` 是实际截获的层编号，`expected_layers` 是模型应有层数；两者相符才 PASS。
覆盖 PASS 只表示全部层确实被观察到，不能代替每层 mask 正确性；注入实验就同时出现覆盖 PASS、mask FAIL。

三个真实报告各有 168 条逐层检查：28 层×2 种模式×3 个种子，wrong_entries 全部为 0。
这些 mask 检查发生在每个 case 的 baseline 前向，不是在每次扰动前向都重新保存一遍。

## 6. 方法三：梯度不是“训练梯度是否爆炸”

此处求的是：改变某个**输入位置的 embedding**，目标 noisy 输出是否会发生微小变化。
不是优化器更新参数的梯度，也不是直接对包含标签的 loss 求导。

脚本对所观察块的全部 logits 乘一组随机系数并加总，得到一个标量 score，再求 score 对输入 embedding 的导数。
`投影0`、`投影1` 是两组随机系数，降低单个输出方向恰巧抵消的风险；不是第 0 层、第 1 层。
两次随机投影不是穷举完整 Jacobian，因此要结合方法一和方法二判断。

| 字段/名称 | 含义 |
|---|---|
| `noisy块1` | 当前观察第二块 noisy 的全部输出 |
| `per_position_norm` | [batch,2L] 数组；每个数字是该输入位置 embedding 梯度向量的 L2 范数 |
| `max_norm`，禁止位置项 | 在所有禁止输入位置、所有 batch 中，最大的梯度范数；不能超过阈值 |
| `max_norm`，前缀正对照项 | 在允许的 clean 前缀、所有 batch 中，最大的梯度范数；必须超过阈值才证明有可检测依赖 |
| `threshold` | 本次为 1e-8=0.00000001；这是梯度阈值，与 logits 的 1e-6 不同 |

以真实 BF16 的 `train/seed=0/方法三/noisy块1/投影0` 为例，第一条 batch 的 48 个位置可分成：

| 位置 | 来源 | 实际梯度范数 | 怎么看 |
|---|---|---|---|
| 0–7 | noisy Block0 | 全 0 | 禁止依赖，正确 |
| 8–15 | noisy Block1 | 约 1.20–2.38 | 同块 noisy 可参与预测，非零正常 |
| 16–23 | noisy Block2 | 全 0 | 禁止依赖，正确 |
| 24–31 | clean Block0 | 约 2.87–13.02 | 过去 clean 可影响预测，非零正常 |
| 32–39 | clean Block1 | 全 0 | 当前 clean 答案没有检测到梯度依赖 |
| 40–47 | clean Block2 | 全 0 | 未来 clean 没有检测到梯度依赖 |

这条禁止位置检查的 `max_norm=0`，但前缀正对照的 `max_norm=14.88833999633789`。
后者来自第二条 batch 的 clean 前缀位置 31；它并不与上表第一条 batch 的最大 13.02 矛盾。

如果你看到 `per_position_norm` 中存在 8、10、14 等数值，不要直接认为泄漏或梯度爆炸，必须先定位它在哪个输入区间。
报告要求的是禁止区间接近零，而不是所有梯度都为零。
首块 noisy 没有过去 clean，因此不生成“首块前缀正对照”条目，这不是漏测。

三个真实报告的全部禁止位置 `max_norm` 均为 0。这也检查其他 noisy block 是否通过多层计算间接影响当前块。

## 7. 方法四：秘密替换和预测 ID

当前 clean Block1 的每个 token ID 加 17（按词表取模），其余 clean 和全部 noisy 固定。
`cases/.../secret_demo` 保存演示信息：

| 字段 | 意思 |
|---|---|
| `说明` | 提醒秘密是词表 ID 序列，不是实际自然语言秘密文本 |
| `before_secret` | 替换前 clean Block1 的 token ID |
| `after_secret` | 替换后的 token ID |
| `before_prediction` | 替换前 noisy Block1 每个位置 logit 最大的候选 token ID |
| `after_prediction` | 替换后同样方式得到的预测 ID |

真实 BF16 的 seed0 第一条样本中，秘密从 `[9937,27438,...]` 改为 `[9954,27455,...]`。
预测前后都是八个 `13050`，且方法四记录全部 logits 差异为 0。
**不是仅凭预测 ID 不变判通过**：即使最高分候选不变，其他候选分数也可能改变，所以仍然要看 max_abs。

`13050`、`198`、`279`、`220` 都是词表编号，不是概率、loss 或位置。
未使用对应 tokenizer 解码，不能根据编号直接解释具体文本。
这里输入是随机 token、进行一次受控前向；预测重复并不能单独证明模型发生坍塌，也不能说明生成质量正常。

## 8. 注入实验的每个 FAIL 为什么都是预期的

`attention_injected.json` 的 `tiny=true`、`inject_leak=same`。
小模型 block_size=4、L=12、拼接总长 24；故意让 noisy Block1 的 query 4–7 读取当前 clean Block1 的 key 16–19。

| FAIL 类别 | train 的实际结果 | eval | 解释 |
|---|---|---|---|
| 方法二/层0 | wrong_entries=16 | 相同 | 4 个 query×4 个 clean key，共 16 条违规边 |
| 方法二/层1 | wrong_entries=16 | 相同 | 第二层也使用了故意改坏的 mask |
| 方法一/clean1→noisy1 | max_abs=0.23460564017295837 | 相同 | 修改被偷看的当前答案，影响了 noisy 输出 |
| 方法四/秘密替换 | max_abs=0.2866450548171997 | 相同 | 替换秘密也影响了输出 |
| 方法三/noisy1/投影0/禁止位置 | max_norm=1.791006326675415 | 相同 | 禁止区间出现梯度依赖 |
| 方法三/noisy1/投影1/禁止位置 | max_norm=2.90317964553833 | 相同 | 另一个输出方向也发现依赖 |

每种模式 6 个 FAIL，两种模式共 12 个，位置和原因都与人为注入一致。
例如错误坐标 `[0,0,4,16]` 就是第 0 条样本、mask 头维0、noisy query4 错误读取 clean key16。
每层只列前 10 个错误坐标，因此列表长度 10、wrong_entries=16 是正常的。

注入实验秘密替换的预测从 `[34,34,34,121]` 变为 `[30,121,121,121]`，可直观看到影响。
模型不必逐字抄出秘密才能算存在依赖；数值测试已经能发现。
其余 36 项通过，是因为这次只放开第二块读取自己 clean 的连接，并没有把所有边都改坏。
此次没有运行 `inject_leak=future`，所以不能说未来注入负对照也已完成。

## 9. 三个真实报告的全部数值范围

下表覆盖各自所有 train/eval 和三个种子；范围表示“各检查项指标的最小值到最大值”，不是置信区间。

| 指标 | BF16 全 MASK | BF16 部分 MASK | FP32 全 MASK |
|---|---:|---:|---:|
| 重复 baseline 最大差异 | 全 0 | 全 0 | 全 0 |
| 当前/未来 clean 扰动 max_abs | 全 0 | 全 0 | 全 0 |
| 允许前缀扰动 max_abs | 4.6875–12.46875 | 3.35168457–11.84375 | 4.76350069–10.59903908 |
| 秘密替换 max_abs | 全 0 | 全 0 | 全 0 |
| 禁止位置梯度 max_norm | 全 0 | 全 0 | 全 0 |
| 前缀梯度 max_norm | 10.30047894–115.51834869 | 8.56208229–26.46404839 | 5.63636208–24.48163223 |
| kernel mask 错误元素数 | 全 0 | 全 0 | 全 0 |

这些零是报告实际记录的数值零，不是把 1e-7 格式化成简短的 0；仍应按有限精度数值实验理解，不能推广成任意输入下的数学证明。
BF16 与 FP32 batch 不同（2 对 1），随机投影的形状也不同，不能把上述梯度范围差异全部归因于精度。
前缀梯度 115.5 不表示泄漏；此测试不是训练梯度爆炸诊断。

## 10. 为什么真实报告正好有 300 项

每个“模式+种子”组合：

| 类别 | 项数 | 组成 |
|---|---:|---|
| 每层 mask | 28 | 28 层各 1 项 |
| kernel 覆盖 | 1 | 检查层0–27都被截获 |
| baseline 重复性 | 1 | 同输入重复前向 |
| clean 扰动 | 9 | 3 个改动块×3 个观察块；3 项前缀、6 项当前/未来 |
| 秘密替换 | 1 | 修改 clean Block1 |
| 禁止位置梯度 | 6 | 3 个观察块×2 个投影 |
| 前缀梯度正对照 | 4 | 只有 Block1/2 有前缀，各 2 个投影 |
| 合计 | 50 | 一组 case 的检查数 |

2 种模式×3 个种子×50=300。小模型只有 2 层，每个 case 为 24 项；2 种模式×1 个种子=48 项。

## 11. arguments：所有运行参数逐项解释

| 字段 | 本次内容及含义 |
|---|---|
| `model_path` | 真实测试是 step 5000 目录；tiny/injected 为 null，因为没有加载训练权重 |
| `tiny` | true 表示随机小模型；false 表示真实 checkpoint |
| `diffusion_config` | 本次均 null，没有通过额外 JSON 覆盖配置；不等于缺少 diffusion 配置 |
| `device` | `npu:0`，测试进程使用的一张逻辑 NPU |
| `dtype` | bfloat16 或 float32，参数/计算使用的测试精度 |
| `backend` | 五份均 eager，实际报告也确认 eager；本次没有 SDPA/FlashAttention 对照 |
| `mode` | both，分别执行 train 和 eval 两种受控路径 |
| `block_size` | arguments 中 null 表示未从命令行覆盖；实际值查 metadata：真实为8，小模型为4 |
| `num_blocks` | 3，覆盖首块、中间块、末块 |
| `batch_size` | BF16 两份为2，FP32与小模型为1；不表示训练的全局 batch size |
| `noise_ratio` | 1.0 表示全部 MASK；partial 为0.5，是抽样概率，不强制每条正好一半 |
| `seeds` | 真实为[0,1,2]，小模型为[0]；输入及随机投影按种子复现 |
| `atol` | 1e-6，logits 比较的绝对阈值 |
| `grad_atol` | 1e-8，位置 embedding 梯度范数阈值 |
| `gradient_probes` | 2，每个观察块做两组随机输出投影 |
| `use_fused_rmsnorm` | false，未请求启用此融合补丁 |
| `use_fused_rotary_pos_emb` | false，未请求启用此融合补丁 |
| `inject_leak` | 正常报告为 null；injected 为 same，故意放开当前 clean |
| `report` | 当时程序的 JSON 输出文件名，不是模型保存路径 |

## 12. metadata：所有环境字段逐项解释

| 字段 | 意思 |
|---|---|
| `torch` | 报告为 `2.7.1+cpu`，是 PyTorch 包的版本标签，不能仅据 `+cpu` 认定实际跑在 CPU；脚本指定 npu:0 并经 torch_npu 注册设备 |
| `transformers` | `4.57.1`，加载模型及 attention 基础实现的库版本 |
| `model_class` | `Qwen3DiffusionForCausalLM`，本次受测模型类 |
| `backend` | 实际模型层选择的 attention 实现，这次均 eager |
| `block_size` | 真正生效的 block 大小，真实8，小模型4 |
| `source_root` | 服务器运行代码的根目录 `/share/f50058111/MindSpeed-LLM` |
| `weight_source` | 真实 checkpoint 目录，或“随机小模型”；用于核对有没有误测 Base 权重 |
| `scope` | 明确这是单设备完整模型、无 FSDP/TP/CP 包装，train 关闭 dropout/recompute 的测试 |
| 两个 `文件路径:sha256` | 核心模型文件和 attention 文件的内容指纹，用于确认受测代码版本 |

已核对：五份报告的两个 SHA256 一致；当前 Windows 文件将 CRLF 换行规范化为 LF 后，也与服务器报告匹配。
因此没有发现这两个核心文件存在除换行之外的文本差异。这个指纹不覆盖整个仓库、安装依赖或 checkpoint 权重，不能作更广的版本等同证明。
报告没有保存 torch_npu/CANN 版本或底层算子 profiler 信息，不能据它进一步推断具体 CANN 内核版本。

## 13. cases：输入、噪声和位置数组逐项解释

`cases` 按 `train/seed=0` 等名称组织，每个 case 的字段如下：

| 字段 | 数组形状与含义 |
|---|---|
| `clean_ids` | [batch,L]，原始随机词表 ID；不是训练集抽样文本 |
| `noisy_ids` | [batch,L]，指定位置替换为 MASK 后的输入；真实模型 MASK ID 为151669，小模型为127 |
| `masked_indices` | [batch,L] 布尔数组，true 表示此 token 被 MASK，false 表示保留 |
| `position_ids` | train 为[batch,L]，eval 为[batch,2L]；两段具有同样的逻辑位置 |
| `secret_demo` | 方法四秘密替换前后输入和预测，字段解释见方法四 |

`masked_indices` 是**哪些输入 token 被替换**，与方法二“哪些 query 能读取哪些 key”的 attention mask 是两回事。
真实 train 的 position_ids 为0–23；eval 为0–23接0–23。绝对 clean 位置24的逻辑 RoPE 位置是0。

partial 实测每个种子两条样本共48个 token：seed0 有24个 MASK，seed1 有25个，seed2 有24个；train/eval相同。
每块首 token 被强制 MASK，其余按0.5抽样，因此无需要求每条恰好12个 MASK。
它不是重新实现训练的随机 mask ratio 分布，而是固定噪声进行依赖隔离。

## 14. status 与判定范围

| 状态 | 读法 |
|---|---|
| PASS | 本项满足预设规则；总 PASS 表示该文件所有记录都通过 |
| FAIL | 某项不满足规则；注入实验预期出现它，正常实验则需要排查 |
| ERROR | 加载/设备/形状/算子等异常，测试没有正常完成；本次没有 |
| INCONCLUSIVE | baseline 不稳定或前缀正对照不足，不能可靠判定；本次没有 |

程序总状态优先级为 ERROR > FAIL > INCONCLUSIVE > PASS，不是取多数票。
相应程序退出码为2、1、3、0；JSON中记录的是文字状态，本次未读取服务器 shell 退出码日志。

能够支持的判断：step 5000、报告中的核心代码、eager、单 NPU、block_size8、三块随机短输入，
在全 MASK/部分 MASK、BF16/FP32、train/eval受控条件下没有检测到 clean-target attention 泄漏。

尚未覆盖的范围：双机 FSDP 通信与重计算、其他 attention backend、融合补丁、真实训练长序列与文档/packing/padding边界、
动态改变长度后的 mask 缓存、带 KV cache 的生成路径，以及其他 checkpoint。
如果线上训练使用 eager，这次已经对其核心 attention 依赖关系提供了较强证据；若线上是 SDPA/FlashAttention，必须另测相应路径。

报告不含训练 loss、准确率、自然语言生成质量或 loss 下降原因。
因此不能从这些 PASS 推导“训练目标全部正确”或“loss 快速下降一定是正常学习”。

## 15. 每条原始检查记录的中文索引

下列附表逐条保留原始检查顺序、名称、原始字段，并给出针对该条的中文解释，合计996条，没有只挑选好看的结果：

- [小模型48条逐项解读](attention_results_逐项/attention_tiny_逐项.md)
- [注入实验48条逐项解读](attention_results_逐项/attention_injected_逐项.md)
- [BF16全MASK 300条逐项解读](attention_results_逐项/attention_checkpoint_bf16_逐项.md)
- [BF16部分MASK 300条逐项解读](attention_results_逐项/attention_checkpoint_partial_逐项.md)
- [FP32全MASK 300条逐项解读](attention_results_逐项/attention_checkpoint_fp32_eager_逐项.md)
