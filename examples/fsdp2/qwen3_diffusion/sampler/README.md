# 扩散 Sampler 训练

本目录实现了 Nemotron-Labs-Diffusion 技术报告附录 A 中描述的学习式接受
Sampler。该实现全部由新增文件组成，不需要修改 MindSpeed-LLM 现有的模型、
生成或训练代码。

## 与论文严格对齐的部分

- 冻结扩散主干模型，不更新其参数。
- 训练数据同时包含块长度 `B=8` 和 `B=32`。
- 采集两种轨迹策略：
  - `confidence`：串行去噪，每一步提交当前所有掩码位置中置信度最高的一个位置；
  - `hybrid`：优先提交当前 top-1 预测与真实 token 一致的位置；如果没有任何位置
    匹配，则退化为提交置信度最高的一个位置。
- 对每个中间状态中的每个掩码位置，标签定义为
  `1[当前 top-1 == 同一条轨迹最终提交的 token]`。
- 每个位置使用 144 维输入特征，包括 top-3 预测 token 的 PCA 压缩语义嵌入，
  以及输出分布统计特征。
- Sampler 是一个四层、隐藏维度为 384 的双向 Transformer，并使用逐位置的
  sigmoid 分类头。
- 使用不加权的逐位置二元交叉熵进行训练，以留出验证集上的 ROC AUC 作为早停
  指标。实现采用分布式 4096 桶流式近似计算，因此验证阶段的内存占用不会随
  约 2000 万规模的轨迹数据线性增长。

论文没有公开 PCA 数据划分、完整统计特征列表、注意力头数、FFN 宽度、dropout、
优化器或学习率。为了提供可复现实现，本项目默认使用 `3 * 46` 维 PCA 特征加
6 个统计特征，得到精确的 144 维输入；模型使用 8 个注意力头、768 维 FFN 和
AdamW，总参数量约为 480 万。所有根据工程需要补充的参数都通过代码或配置显式
暴露，不会将其表述为论文未公开的原始设置。

## 输入数据格式

轨迹采集器接收 UTF-8 编码的 JSONL 文件。普通的提示词/回复记录格式如下：

```json
{"prompt":"计算 2+2。","response":"答案是 4。"}
```

对于对话数据，请传入 `--messages-field messages`。最后一条消息的角色必须是
`assistant`，此前的消息会使用模型 tokenizer 的对话模板进行渲染。

## 1. 拟合词嵌入 PCA

拟合 PCA 和采集轨迹时必须使用同一个模型检查点。

```bash
python examples/fsdp2/qwen3_diffusion/sampler/fit_sampler_pca.py \
  --model-path /path/to/qwen3-diffusion-checkpoint \
  --config-diffusion examples/fsdp2/qwen3_diffusion/config_diffusion.json \
  --output outputs/sampler/pca_46.pt \
  --pca-dim 46
```

拟合过程会从冻结模型的 token 嵌入表中采样行。随后以分块方式执行投影，避免轨迹
采集器在加速器上额外创建一份完整的 float32 词表嵌入副本。

## 2. 采集去噪轨迹

单进程运行：

```bash
python examples/fsdp2/qwen3_diffusion/sampler/collect_sampler_trajectories.py \
  --model-path /path/to/qwen3-diffusion-checkpoint \
  --config-diffusion examples/fsdp2/qwen3_diffusion/config_diffusion.json \
  --input-jsonl /path/to/post_training_mix.jsonl \
  --output-dir outputs/sampler/trajectories \
  --pca-path outputs/sampler/pca_46.pt \
  --block-lengths 8 32 \
  --policies confidence hybrid
```

数据并行采集会在每个加速器上加载一份冻结的主干模型，并为各进程分配不同的
JSONL 数据行：

```bash
torchrun --nproc-per-node 8 \
  examples/fsdp2/qwen3_diffusion/sampler/collect_sampler_trajectories.py \
  --model-path /path/to/qwen3-diffusion-checkpoint \
  --input-jsonl /path/to/post_training_mix.jsonl \
  --output-dir outputs/sampler/trajectories \
  --pca-path outputs/sampler/pca_46.pt
```

默认只保留完整的 8/32-token 块。回复中先前的 token 会通过教师强制方式加入因果
前缀，而当前块在开始时全部为掩码。训练集和验证集的划分是确定性的，并且以完整
轨迹为单位进行。

## 3. 训练 Sampler

```bash
python examples/fsdp2/qwen3_diffusion/sampler/train_sampler.py \
  --data-dir outputs/sampler/trajectories \
  --pca-path outputs/sampler/pca_46.pt \
  --output-dir outputs/sampler/checkpoints \
  --batch-size 256 \
  --dtype bfloat16
```

输出目录包含：

- `best.pt`：留出验证集 AUC 最佳时保存的检查点；
- `last.pt`：最新检查点，其中包含优化器状态；
- `metrics.jsonl`：每个 epoch 的验证损失、AUC，以及阈值为 0.5 时的准确率。

Sampler 训练也支持使用 `torchrun` 进行多设备训练。由于 Sampler 模型较小，通常
使用单个加速器就足够。分布式轨迹采集才是计算开销最大的环节，因为它需要反复
运行冻结的语言模型。

## 正式采集前先做冒烟测试

建议先使用较小的 JSONL 文件并限制工作量：

```bash
python examples/fsdp2/qwen3_diffusion/sampler/collect_sampler_trajectories.py \
  --model-path /path/to/checkpoint \
  --input-jsonl smoke.jsonl \
  --output-dir outputs/sampler/smoke \
  --pca-path outputs/sampler/pca_46.pt \
  --max-examples 4 \
  --max-blocks-per-example 1
```

检查 `train/` 和 `val/` 是否都生成了数据分片。冒烟测试的数据量很小时，可以增大
`--validation-fraction`（例如设为 `0.5`），以避免验证集为空。
