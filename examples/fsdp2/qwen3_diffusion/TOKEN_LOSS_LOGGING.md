# Qwen3 逐 token loss 日志

AR 和 joint CPT 示例配置已开启逐 token 日志。其他 FSDP2 配置可在 `training` 下添加：

```yaml
log_per_token_loss: True
token_loss_logging_steps: 1
```

每个记录步骤包含所有 microbatch、所有样本、所有位置，不截断长序列。
每个全局 rank 将自己的完整记录追加写入
`<training.output_dir>/token_losses/rank_<rank>.jsonl`；全局 rank 0 同时将
`token_loss` 明细及汇总打印到现有训练日志，用 `record_type` 区分。
`token_loss_logging_steps: 10` 表示每 10 个 optimizer step 记录一次。
关闭 `log_per_token_loss` 即停止明细输出。全量输出会增加 CPU 同步、I/O 和日志体积。

每行是一个 JSON 对象。`record_type=token` 的主要字段：

| 字段 | 含义 |
| --- | --- |
| `step`, `micro_step` | 从 1 开始的训练步和步内 microbatch 编号 |
| `rank`, `batch_index`, `position` | 全局 rank、microbatch 内样本索引、当前序列位置；后两者从 0 开始 |
| `position_id` | 输入提供 position_ids 时记录其值；CP 分片的 `position` 仍是本 rank 的局部索引 |
| `component` | `ar` 或 `diffusion` |
| `input_id`, `target_id`, `target_token` | 原始输入 ID、预测目标 ID、tokenizer 的 token 字符串（有 tokenizer 时） |
| `valid`, `loss` | 是否参与该分支 loss，以及该目标的原始交叉熵 `-log p(target)` |
| `weighted_loss` | 联合训练时该 token 对 loss 分子的贡献 |
| `p_mask` | Diffusion 分支使用的 mask 采样概率 |

AR 直接使用数据加载器已经右移的 `labels[b, position]`，没有再次 shift。
因此 `input_id` 是当前位置的输入，而 `target_id` 是这一位置要预测的目标。
AR 的 `target_id=-100` 记录为 `valid=false, loss=null`。
Diffusion 的目标是当前位置的原始输入 token，仅被 mask 的位置参与 loss；
未 mask 的位置同样记录为无效和 `null`，不将其解释成零损失。

联合训练的 AR `weighted_loss = ar_loss_weight * loss`，
Diffusion `weighted_loss = dlm_loss_weight * loss / max(p_mask, 1e-3)`
（`dlm_loss_weight=None` 时权重按 1 处理）。两分支有效 token 的
`weighted_loss` 总和等于联合 loss 的分子。
联合 CPT 的归一化已对齐官方 Megatron-Bridge 默认路径：每个 rank、每个
microbatch 使用本地分母 `被 mask 的 token 数 + labels.numel()`，分母不乘
`ar_loss_weight`；随后除以当前梯度累积窗口的实际 microbatch 数，跨 rank
由 FSDP 平均梯度。不跨 rank 汇总分母，也不是整个 optimizer step 的 token 加权平均。
这里的 AR 分母包含 `-100` 标签位置，遵循官方 `numel()` 计数；这些位置仍不贡献
AR 交叉熵，日志仍记录为 `valid=false`。这不同于仅按有效标签数归一化。
`record_type=summary` 的 `mean_loss` 是该 rank、该 microbatch、该样本、该分支
的有效 token 原始 CE 均值，便于独立观察 AR；它不是跨 rank 的平均值。

日志张量已 detach，不参与反向传播。此功能支持 `qwen3_diffusion` 的
autoregressive、block_diff、bidirectional 路径，以及 Trainer 直接计算 loss 的
普通预训练路径。`chunk_loss_size` 必须关闭，因为该路径不保留 logits。
现有的 `calculate_per_token_loss` 参数与此日志开关是两个独立参数。

CPU 回归检查（不需要模型权重或 NPU）：

```bash
python scripts/test_token_loss_logging.py
python scripts/test_qwen3_token_loss.py
python scripts/test_trainer_token_loss.py
```

逐 token CE 衡量给定正确上下文时的目标预测损失；AR 生成能力还需要结合
因果 attention mask 检查与实际自回归生成结果判断。

## 按训练 step 绘制未加权的 AR / Diffusion 平均 loss

在服务器仓库根目录运行（绘图依赖 `matplotlib`）：

```bash
# 输入训练 stdout/tee 日志：通常仅包含 rank 0 的 token_loss summary。
python scripts/plot_token_loss.py /path/to/train.log -o /path/to/raw_loss.png

# 输入 token_losses 目录：汇总该目录中所有 rank_*.jsonl。
python scripts/plot_token_loss.py /path/to/output_dir/token_losses -o /path/to/raw_loss.png

# 可选：只查看某个 rank。
python scripts/plot_token_loss.py /path/to/output_dir/token_losses --rank 0 -o /path/to/rank0_raw_loss.png
```

输出 `raw_loss.png` 和同名 `raw_loss.csv`。图中每条曲线每个点是一个已记录的
optimizer step，没有乘训练权重，没有 Diffusion 的 `1/p_mask` 校正，不做平滑。
脚本读取 summary 的 **`mean_loss`**，不使用 `mean_weighted_loss`。

同一步同一分支，跨所选 rank、microbatch 和样本的计算公式为：

```text
step_mean_loss = sum(mean_loss * valid_token_count) / sum(valid_token_count)
```

这里按有效 token 数汇总，是为了还原所有有效 token 的 CE 均值，不是乘 AR 或
Diffusion 训练权重。AR 的分母是有效 next-token 标签数；Diffusion 的分母是被
mask 的有效目标数。因此与训练联合 loss 的分母和梯度累积平均口径不同。

CSV 的 `ar_loss`、`diffusion_loss` 是每一步未加权的平均 CE，同时输出
`ar_valid_tokens`、`diffusion_valid_tokens`、样本汇总数量及所含 rank。
缺失分支或有效 token 数为 0 时留空，不补成 0；NaN/Infinity 保留在 CSV 中，
在图中显示为断点。日志采样间隔大于 1 时只绘制实际记录的 step。

一个目录应只包含同一次训练的日志。脚本会去除控制台与 JSONL 中完全相同的
summary 副本，遇到同一 step/rank/microbatch/sample/component 的冲突记录时
报错，需先拆开重新运行产生的日志。正在写入或不完整的日志只代表已采集的数据；
应使用各 rank 完整日志来观察全局每步均值。

旧版只有 `ar_loss:` / `diff_loss:` 累加值的日志缺少有效 token 数和原始
Diffusion CE，无法准确恢复这里的曲线。原 `plot_training_log.py` 仍用于旧格式
的训练指标；此处的新脚本用于逐 token 日志中的原始 CE。
