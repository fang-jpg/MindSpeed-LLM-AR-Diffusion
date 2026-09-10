# 长期训练的 loss 日志配置

默认只保存 summary，逐 token 明细需显式开启。推荐长期训练保持：

```yaml
training:
  log_per_token_loss: True
  log_token_loss_details: False
  token_loss_logging_steps: 1
```

`log_per_token_loss` 控制是否统计各 token 的原始 CE；
`log_token_loss_details` 控制是否把每个 token 单独打印、写入文件。
将第二项设为 `False` 不会关闭 summary，也不会影响按 step 画图。
默认值也是 `False`，旧配置未填写该字段时同样只记录 summary。

每个 rank、microbatch、样本、分支只写一条 summary，保存到
`<output_dir>/token_losses/rank_<rank>.jsonl`，全局 rank 0 同时打印到训练日志。
原始 loss 的定义和训练梯度不变。summary 模式先在设备上求和、计数，
只向 CPU 传输汇总结果，不搬运完整 token 数组，也不调用 tokenizer 解码。
设备端 FP32 求和与原先 CPU 求和可能存在浮点舍入级差异。

AR 的 `mean_loss` 是有效 next-token 标签（非 `-100`）的原始 CE 均值；
Diffusion 的 `mean_loss` 是被 mask 的有效目标的原始 CE 均值。
`mean_loss` 不包含 AR/Diffusion 权重或 `1/p_mask`；
`mean_weighted_loss` 保留训练加权后的分支均值。没有有效 token 时记录 `null`。

## 日志体积估算

50B 指累计训练 500 亿输入 token 时，假设两个分支均记录每个位置、
每条 token JSON 约 300 字节，明细约为 `50e9 * 2 * 300 = 30 TB`。
原 logger 也会为无效位置写一行，因此未被 mask 的 Diffusion 位置仍占空间。
该估算已包含全局训练 token 总量，不应再次乘 rank 数；控制台副本另计。

若序列长度为 4096，summary 的条目数约为 `50e9 / 4096 * 2`，
每条约 300 字节则总量约 7.3 GB，行数约减少 4096 倍。
实际大小受字段、样本长度、填充量和训练配置影响；上述单位为十进制。
因此 summary 也有存储成本，但远小于全量明细。

`token_loss_logging_steps: 10` 可进一步降至约十分之一，但只留下每 10 步的点。
需要每一步的均值时保持 `1`。仅在短期排查具体 token 时开启
`log_token_loss_details: True`；它会恢复全量明细，适合短调试运行。

## 绘图

```bash
python scripts/plot_token_loss.py /path/to/output_dir/token_losses -o /path/to/raw_loss.png
```

也可输入包含 summary 的训练 `.log`，通常只能覆盖 rank 0。
目录模式汇总所有 rank 的已记录 summary，按
`sum(mean_loss * valid_token_count) / sum(valid_token_count)` 计算每一步的
AR 和 Diffusion 平均原始 CE，输出 PNG 和同名 CSV，无需逐 token 明细。

这些改动需要同步到服务器后生效，不会自动清理服务器上已经产生的旧日志。
