# Qwen3-1.7B d-OPD：论文对齐的双机训练

本目录与现有的 Nemotron_Diffusion 实验相互隔离。这里提供的 YAML 不再读取七个 Megatron CPT 数据前缀，而是直接使用公开的 OPDLM 指令混合数据，并启用 d-OPD 正确性门控。

## 原始训练数据

论文使用 `divelab/opdlm_train_data`，它由 61,816 条提示组成，包含以下七个数据源：

- 代码：TACO、KodCode-Light-RL、AceCode；
- 数学：DAPO、Nemotron-v2-Math；
- 科学：Nemotron-v2-STEM；
- 对话：Nemotron-v2-Chat。

每条 JSON 数据都包含 `question`、`ground_truth_answer`、`domain`、`source` 和 `tests_json` 字段。数据加载器会拒绝记录数不等于 61,816 的数据版本，并按照原项目为不同领域追加提示后缀，然后应用 Qwen 对话模板，其中 `add_generation_prompt=true`、`enable_thinking=false`。不会额外执行过滤或去重。

在共享文件系统上，从仓库根目录执行一次下载：

```bash
bash examples/fsdp2/qwen3_dopd/download_opdlm_train_data.sh
pip install -r examples/fsdp2/qwen3_dopd/requirements_dopd.txt
```

如果两个节点不共用 Python 环境，需要在两个节点上都安装依赖。如果共享数据路径不同，可通过 `DATA_DIR` 覆盖默认路径。

## 论文对齐约束

启用 `dopd.strict_paper_alignment: true` 后，只要算法超参数偏离公开论文或原项目配置，训练就会在启动时直接报错：

- 对数据随机打乱并只训练一个 epoch；共 61,816 条提示，每个外层迭代采样 128 条，共执行 483 次外层迭代；最后一个数据池不足时，从打乱后数据的开头补齐；
- 教师模型和学生模型都由 Qwen3-1.7B Base 初始化；教师保持原始 AR 注意力，学生在加载时切换为扩散注意力；不加载 Nemotron CPT checkpoint；
- block size 为 4，每个 block 执行 4 个去噪步骤；使用阈值 0.9 的动态低置信度重掩码；temperature 为 1、top-p 为 1，并禁用 top-k；
- 最大生成长度在前 100 次外层迭代中按余弦曲线从 100 增长到 4,000 token，最大提示长度为 2,000 token；
- 每个 block 只进行一次 rollout，并从去噪轨迹中独立采样一个访问状态；
- d-OPD 修正强度为 0.5；教师与学生候选 token 取 top-16 并集；教师 top-16 前向 KL；评分分块大小为 128；要求存在可见未来 token，并从损失中排除 EOS；
- 使用 AdamW：学习率 `1e-5`、betas `(0.9, 0.999)`、epsilon `1e-8`、weight decay 为 0、梯度裁剪为 1.0；使用 BF16 和激活重计算；前 5 次外层迭代进行 warmup，之后使用余弦衰减，最低学习率为峰值的 10%；
- `correctness_gate: true`：正确的数学、科学和代码 rollout 使用未来感知目标；错误样本以及无法验证的对话样本使用普通因果蒸馏目标。

论文在单机 8 张 H100 上训练，并使用带 CPU offload 的 ZeRO-3。本实现使用双机 16 张 Ascend A2 和 FSDP2，因此无法做到逐比特一致，吞吐也不会完全相同。为保持论文中的优化器全局 batch 不变，16 个 rank 全部参与 FSDP 参数分片，但每个优化器步骤只让轮换的一组 8 个 rank 产生非零局部损失，并对 FSDP 平均后的梯度重新缩放为 8 个样本的均值。这样每个包含 128 条 rollout 的外层迭代仍执行 16 次优化器更新；如果直接用全部 16 个样本进行一次更新，就属于不同的实验设置。

## 正确性验证器与安全说明

数学领域使用 `math_verify==0.9.0`；科学领域沿用原项目的最终选项判定规则；代码领域支持公开数据中的四种 `tests_json` 格式：`pytest`、`stdio`、`fn_call` 和断言列表。对话领域按设计不进行正确性验证。

代码验证会在限制运行时间、进程数、文件大小和文件描述符数量的条件下执行模型生成的 Python 代码，但这些限制不能视为完整的安全边界。请在断网的容器或虚拟机中运行训练，环境内不要放置密钥、访问令牌、SSH 密钥或敏感挂载目录。不要直接在有特权的宿主机环境中启动启用了论文对齐代码验证器的训练。

## 双机启动

两个节点必须使用相同的仓库版本、模型目录和 Python 环境。下面的数据路径和输出路径应位于共享存储上。从节点应与主节点尽量同时启动。

在主节点 `80.5.5.133` 上执行：

```bash
cd /path/to/MindSpeed-LLM
NODE_RANK=0 MASTER_ADDR=80.5.5.133 \
  bash examples/fsdp2/qwen3_dopd/train_dopd_qwen3_1.7b_fsdp2_A2.sh
```

在从节点 `80.5.5.118` 上执行：

```bash
cd /path/to/MindSpeed-LLM
NODE_RANK=1 MASTER_ADDR=80.5.5.133 \
  bash examples/fsdp2/qwen3_dopd/train_dopd_qwen3_1.7b_fsdp2_A2.sh
```

可选的环境变量包括 `MODEL_PATH`、`DOPD_DATA_PATH`、`OUTPUT_DIR`、`MASTER_PORT` 和 `NPUS_PER_NODE`，默认值记录在 YAML 和启动脚本中。每个节点都会在 `$OUTPUT_DIR/logs/node_<rank>_<timestamp>.log` 下保存持久化控制台日志；rank 0 还会写入 `metrics.jsonl`、`dopd_config.json`，并每隔 10 次外层迭代保存一次 checkpoint。

## 论文对齐推理

`inference_dopd_qwen3_1.7b_fsdp2_A2.sh` 默认严格复现论文附录 C.4 的解码设置：训练对应的 block size 4、每块 4 个去噪步骤、整块从全 MASK 开始、`low_confidence_static`、temperature 1、top-p 1、top-k 1、Qwen3 chat template、`enable_thinking=false`，最多生成 4096 个 token。脚本会在配置偏离时直接报错，防止把消融实验误标成论文对齐结果。

如果要比较其他 block size 或动态阈值，显式设置 `PAPER_ALIGNED=0`。这类结果属于解码消融，不能和论文表 1 直接比较。JSON 同时记录 `TPF_total`、`TPF_decode` 和 `TPF_denoise_only`；TPF 是本地诊断指标，并不是论文报告的 1.35–1.58 倍加速口径。论文中的该加速指达到 OPDLM 准确率所需的训练时间，而不是推理时每次前向生成的 token 数。

正式训练前，建议使用单独的输出目录进行一次双机冒烟测试：

```bash
export NODE_RANK=<0-or-1>
export OUTPUT_DIR=/share/dataset/x00840191/model_train/ckpt/dopd_smoke
torchrun --nproc_per_node 8 --nnodes 2 --node_rank "${NODE_RANK}" \
  --master_addr 80.5.5.133 --master_port 6503 train_dopd_fsdp2.py \
  examples/fsdp2/qwen3_dopd/dopd_qwen3_1.7b_fsdp2_A2.yaml \
  --training.output_dir=/share/dataset/x00840191/model_train/ckpt/dopd_smoke \
  --training.max_steps=1 --dopd.strict_paper_alignment=false \
  --dopd.prompts_per_iteration=16 --dopd.min_response_tokens=16 \
  --dopd.max_response_tokens=16 --dopd.response_schedule_steps=1 \
  --dopd.score_chunk_size=16
```

本仓库的 YAML 参数解析器要求命令行覆盖项使用 `=` 形式。重复执行冒烟测试前，请先删除或移动原来的冒烟测试输出目录；训练器会拒绝覆盖已经存在 checkpoint 的目录。
