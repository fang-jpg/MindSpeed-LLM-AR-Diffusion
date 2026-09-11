# 当前双机训练版本的离线 attention 泄漏测试

脚本：`examples/fsdp2/qwen3_diffusion/check_attention_leakage.py`。
参考文件实际位于仓库根目录 `test_attention_mask.md`，未找到 `test/_attention/_mask.md`。

## 1. 先明确当前实现与参考示例的差别

本脚本针对训练配置 `joint_cpt_qwen3_1.7b_fsdp2_A2.yaml` 的
`model_id: qwen3_diffusion`，使用当前仓库的 `Qwen3DiffusionForCausalLM`。
不适用于另一个 `Qwen3CausalBiffusion2TowerLm` 模型。

实际训练布局是 **[noisy, clean]**。当 L=12、block_size=4：

```text
noisy:  0 1 2 3 | 4 5 6 7 | 8 9 10 11
clean: 12 ...15 |16 ...19 |20 ...23
观察 noisy Block1：query=4（不是参考示例中的 16）
允许的 key：noisy 4..7，以及 clean 12..15
禁止的 key：其余 noisy，以及当前/未来 clean 16..23
clean query：只允许看自己及此前的 clean token，禁止看所有 noisy token
```

当前 `make_block_diff_mask()` 的源码表达式符合上述可见关系；这只能说明构造逻辑看起来正确，不能替代实际运行证据。

还需要适配三个实际行为：

1. 必须提供 `labels` 才进入训练的 block_diff 拼接、mask 及 logits 分割分支。
   返回的 `outputs.logits` 已是 noisy 半段，索引为 `0..L-1`。
2. `mg` 数据产生长度 L 的 `position_ids`，训练 attention 分半做 RoPE。
   脚本的 train 模式照此传 L 个位置；eval 模式传两份位置，共 2L 个。
   eval 是匹配逻辑位置的诊断路径，不能代替 train 结果。
   如果省略位置参数，模型内部会构造 2L 个位置，这与当前 train 的分半 RoPE 存在长度不匹配风险。
3. 当前子类 `from_pretrained()` 会把 embedding 权重复制到 `lm_head`。
   脚本调用父类的加载方法来实例化同一个模型类，保留 checkpoint 的输出头；
   缺失、多余、形状不匹配的参数会直接报错，避免随机补齐后误判。

## 2. 在训练服务器准备 checkpoint

进入**服务器上这份最终代码**的仓库根目录，并激活训练使用的 Python/CANN 环境。
测试需要已有 torch、torch_npu、transformers、MindSpeed 等训练依赖；不在线下载模型或安装包。
这里只需要单设备，不用 torchrun，不用同时启动两台机器，也不会执行 optimizer.step。

`--model-path` 指向一个具体训练步保存的 **HF 格式 checkpoint 目录**，而不是多个 step 的父目录，
也不要误指到原始 Qwen3 Base 权重。目录内应有 `config.json` 和完整 `.safetensors`/HF 权重分片及索引。
模型、词表维度和 diffusion 配置应与训练时一致。

若只有 DCP 分布式权重，使用仓库现有工具先合并到新目录：

```bash
python -m mindspeed_llm.fsdp2.checkpoint.merge_dcp_to_hf \
  --load-dir /你的路径/具体step的DCP权重目录 \
  --save-dir /你的路径/该step_hf_for_check \
  --model-configs /你的路径/训练使用的模型配置目录
```

`--load-dir` 应是包含 DCP 元数据的实际权重目录。合并工具可能按模型大小占用较多 CPU 内存。
如保存的 config 缺少 diffusion 字段，可在测试时显式加：

```bash
--diffusion-config examples/fsdp2/qwen3_diffusion/config_diffusion.json
```

必须确认这个 JSON 就是训练时的配置，尤其是 `mask_token_id`、`block_size`、`dlm_paradigm`。
脚本只补充内存中的配置，不修改 checkpoint。默认不改变 block_size，不自动扩词表。

## 3. 推荐执行顺序

以下命令均从仓库根目录运行。先用小模型验证测试入口，不需要 checkpoint：

```bash
python examples/fsdp2/qwen3_diffusion/check_attention_leakage.py \
  --tiny --device npu:0 --dtype float32 --backend eager \
  --mode both --seeds 0 --report attention_tiny.json
```

接着验证检测器确实能抓住人为泄漏。**这一条预期 FAIL，退出码 1**：

```bash
python examples/fsdp2/qwen3_diffusion/check_attention_leakage.py \
  --tiny --device npu:0 --dtype float32 --backend eager \
  --mode both --seeds 0 --inject-leak same --report attention_injected.json
```

也可把 `same` 改为 `future`。此参数只改测试进程中的 mask，不能用于正式通过判定。
小模型测试通过只能验证小模型和检测器，不代表真实 checkpoint 已通过。

然后测试训练后的 checkpoint，先对齐当前训练的 BF16 参数精度和自动 backend 选择：

```bash
python examples/fsdp2/qwen3_diffusion/check_attention_leakage.py \
  --model-path /你的路径/具体step_hf_checkpoint \
  --device npu:0 --dtype bfloat16 --backend auto \
  --mode both --batch-size 2 --seeds 0 1 2 --noise-ratio 1.0 \
  --report attention_checkpoint_bf16.json
```

报告会输出实际 attention backend；`auto` 不是与服务器训练 backend 一致的保证，
应结合训练环境核对。若训练明确使用 eager/SDPA，显式设置相应 `--backend`。
训练若开启了融合 RMSNorm/RoPE，测试也添加对应 `--use-fused-rmsnorm` / `--use-fused-rotary-pos-emb`；
当前参数定义中这两个开关默认关闭。

再跑部分加噪，并用 FP32 eager 做数值与实现对照：

```bash
python examples/fsdp2/qwen3_diffusion/check_attention_leakage.py \
  --model-path /你的路径/具体step_hf_checkpoint \
  --device npu:0 --dtype bfloat16 --backend auto \
  --mode both --batch-size 2 --seeds 0 1 2 --noise-ratio 0.5 \
  --report attention_checkpoint_partial.json

python examples/fsdp2/qwen3_diffusion/check_attention_leakage.py \
  --model-path /你的路径/具体step_hf_checkpoint \
  --device npu:0 --dtype float32 --backend eager \
  --mode both --seeds 0 1 2 \
  --report attention_checkpoint_fp32_eager.json
```

FP32 完整模型占用更多设备内存。可先保持 batch-size=1、默认三块短序列；
在安装了全部训练依赖的 CPU 环境也能尝试 `--device cpu --dtype float32 --backend eager`。
不在此脚本中构建 4096 训练长度的完整注意力矩阵；默认三块用于快速检查结构边界。

## 4. 四种方法分别在测什么

| 编号 | 执行内容 | 判定依据 |
|---|---|---|
| 方法一 | 同一输入重复 baseline；每次替换一个 clean block，比较每个 noisy block 的全部 logits | 前缀变化应有影响；当前/未来变化不应有影响 |
| 方法二 | 包裹真实 kernel 调用，检查每层完整 4D additive mask、Q/K 尺寸、is_causal、dropout | 所有可见边为 0，禁止边为 -inf 或 finfo.min；所有层都被观察到 |
| 方法三 | 对每块全部输出做多次随机标量投影，用 autograd.grad 求拼接后的 embedding 梯度 | 当前/未来 clean 和其他 noisy 块梯度应接近零；clean 前缀应有非零梯度 |
| 方法四 | 当前 clean Block1 的整段秘密 token 替换，展示替换前后预测 ID | 比较全部 logits，而非只看 argmax 是否改变 |

方法二在方法一 baseline 中同步执行，以确认检查的是**真实前向实际传入 kernel 的 mask**。
方法四使用词表 ID 序列作为秘密，不依赖 tokenizer；它是参考文档数字秘密实验的结构版本，
不是自然语言生成能力评测。

测试固定 noisy 内容、mask 位置、mask 概率、位置编码；关闭 KV cache、dropout、recompute。
保留带 labels 的真实模型 forward，只有 `forward_process()` 被临时替换为固定噪声。
每块首位置强制 MASK，确保低噪声时也存在 masked query；因此这是受控实验，非训练噪声分布评估。
参数权重被冻结但没有关闭输入 autograd，不保存参数梯度，不对训练 loss 求导。
梯度对象是 `[B,2L,H]` 的**位置 embedding**，不是会混合相同 token ID 位置的共享 embedding.weight。

## 5. 如何读结果

```text
[PASS] train/seed=0/方法一/clean块0→noisy块1/前缀 ...
[PASS] train/seed=0/方法一/clean块1→noisy块1/当前 ...
[PASS] train/seed=0/方法二/实际 kernel mask/层0 ...
[PASS] train/seed=0/方法三/noisy块1/投影0/禁止位置 ...
```

以上只是格式示例，不是已在你的模型上测出的结果。JSON 保存代码 SHA256、模型来源、版本、
实际 backend、全部输入、逐项差异、每位置梯度和异常栈，便于你逐条核查。

- **PASS / 退出码 0**：所有执行项通过，且前缀正对照有效。
- **FAIL / 退出码 1**：禁止依赖超阈值、mask 不符合预期或 kernel 覆盖不完整；需要排查。
- **ERROR / 退出码 2**：依赖、加载、显存、形状或算子异常，测试未完成，不能下无泄漏结论。
- **INCONCLUSIVE / 退出码 3**：重复 baseline 不稳定，或前缀影响弱到阈值以下，无法可靠判定。

先看方法一重复性，再看当前/未来扰动与方法三禁止位置梯度，最后核对所有层方法二结果。
默认 logits 阈值 1e-6、梯度阈值 1e-8；不会因 BF16 自动放宽。
如超阈值，不应直接调大阈值让它通过，应结合重复 baseline、FP32 eager 对照、正对照量级定位。
必要时可显式设置 `--atol` / `--grad-atol`，这些值会记入报告。

## 6. 能排除到什么范围

真实 checkpoint、实际训练 backend 的 train 路径、多种子/噪声比例全部通过时，
可以说明**这些受测输入和配置下未检测到 clean-target attention 泄漏**。
不能凭有限测试证明任意输入、精度和算子环境都绝无泄漏。

本测试使用完整模型层数和当前 attention 实现，但没有包裹双机 FSDP、TP/CP 或梯度重计算，
也不重放训练 dataloader 的 packing/padding/document 边界；不覆盖这些分布式和数据边界行为。
若线上异常仅在双机训练时出现，应进一步在真实训练进程导出 batch 与 kernel 信息复核。
`bd_mask` 当前只在首次构建后缓存；脚本每个实验重置缓存并保持形状固定，
因此不验证同一实例切换 batch/序列长度时的缓存问题。
loss 快速下降、模型质量或高噪声准确率需要另行评估，不能用本测试替代。
