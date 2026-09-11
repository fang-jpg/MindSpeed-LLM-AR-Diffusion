# Attention Mask 泄漏验证指南

本文用于验证 **Block Diffusion LM / Nemotron-Labs-Diffusion 风格训练** 中是否存在 attention mask 泄漏，重点检查：

- noisy token 是否错误地看到了当前 block 的 clean target；
- noisy token 是否错误地看到了未来 clean block；
- noisy block 是否能够正常访问 clean prefix；
- 实际进入 Qwen3 attention kernel 的 mask 是否与预期一致。

按 NLD 的定义，对当前第 $b$ 个 noisy block $\tilde{x}_t^b$，它应该能够看到之前的 clean prefix blocks $x^{<b}$，但不能看到当前 block 的 clean target $x^b_{clean}$，也不能看到未来 clean blocks $x^{>b}_{clean}$。

---

## 1. 推荐的验证方法

建议从三层验证：

1. **扰动 / 反事实测试（最推荐）**：修改某个 clean token，观察目标 noisy query 的 logits 是否变化。
2. **直接检查最终 attention mask**：查看某个 noisy query 对所有 key 位置的 mask 行。
3. **梯度依赖测试**：检查 noisy query 输出对 forbidden clean token embedding 的梯度是否为 0。

最可靠的组合是：

$$
\boxed{\text{扰动测试} + \text{梯度测试}}
$$

因为这两种方法验证的是整个真实 forward 链路，而不仅仅是“构造出来的 mask 看起来正确”。

---

## 2. 示例布局

假设：

```text
sequence length = 12
block size      = 4
```

Clean stream：

```text
Block0       Block1       Block2
A B C D  |  E F G H  |  I J K L
```

如果 dual-stream layout 为：

```text
[ clean stream | noisy stream ]
```

则位置是：

```text
clean:
0  1  2  3 | 4  5  6  7 | 8  9 10 11

noisy:
12 13 14 15 | 16 17 18 19 | 20 21 22 23
```

假设当前测试 noisy Block1，观察：

```text
query_pos = 16
```

理论上：

```text
clean Block0   -> 可见
clean Block1   -> 不可见
clean Block2   -> 不可见
```

因此：

| 被修改位置 | noisy query logits 预期 |
|---|---|
| clean prefix block | 应明显变化 |
| 当前 clean block | 应基本不变 |
| future clean block | 应基本不变 |

---

## 3. 测试前固定条件

测试时建议：

```python
model.eval()
torch.manual_seed(0)
```

并设置：

```python
use_cache=False
```

两次 forward 之间必须保证以下内容完全相同：

- noisy input；
- timestep；
- masking ratio；
- attention mask；
- position ids；
- sampling；
- dropout 状态；
- KV cache 状态。

除了主动修改的 clean token，其他条件都不能变化。

---

## 4. 方法一：扰动测试

### 4.1 基础函数

```python
import torch


@torch.no_grad()
def get_logits(
    model,
    input_ids,
    attention_mask,
    position_ids=None,
):
    model.eval()

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }

    if position_ids is not None:
        kwargs["position_ids"] = position_ids

    outputs = model(**kwargs)
    return outputs.logits.detach()


def replace_token(input_ids, pos, new_token):
    x = input_ids.clone()
    x[:, pos] = new_token
    return x
```

### 4.2 设置测试位置

```python
L = 12
block_size = 4

# clean:
# 0~3   block0
# 4~7   block1
# 8~11  block2

# noisy:
# 12~15 block0
# 16~19 block1
# 20~23 block2

query_pos = L + 4
```

先做 baseline：

```python
logits0 = get_logits(
    model,
    input_ids,
    attention_mask,
    position_ids,
)

q0 = logits0[:, query_pos, :]
```

---

## 5. Positive Control：修改 clean prefix

选择：

```python
clean_prefix_pos = 2
```

执行：

```python
x1 = replace_token(
    input_ids,
    clean_prefix_pos,
    new_token=12345,
)

logits1 = get_logits(
    model,
    x1,
    attention_mask,
    position_ids,
)

q1 = logits1[:, query_pos, :]

diff_prefix = (q1 - q0).abs().max()

print(
    "clean prefix diff =",
    diff_prefix.item(),
)
```

正确情况下：

$$
\boxed{\text{diff\_prefix} > 0}
$$

因为 noisy Block1 本来就应该看到 clean Block0。

如果这里完全没变化，反而要检查：

- clean prefix 是否被错误屏蔽；
- attention mask 是否没有真正传到 attention kernel；
- query/key 的索引是否搞错。

---

## 6. 核心测试：修改当前 block 的 clean target

选择：

```python
clean_same_block_pos = 4
```

执行：

```python
x2 = replace_token(
    input_ids,
    clean_same_block_pos,
    new_token=23456,
)

logits2 = get_logits(
    model,
    x2,
    attention_mask,
    position_ids,
)

q2 = logits2[:, query_pos, :]

diff_same = (q2 - q0).abs().max()

print(
    "same-block clean diff =",
    diff_same.item(),
)
```

正确情况下：

$$
\boxed{\text{diff\_same} \approx 0}
$$

如果出现明显变化，例如：

```text
0.1
0.5
2.0
```

则要高度怀疑：

$$
\boxed{\text{current clean block 泄漏}}
$$

---

## 7. Future Leakage 测试

选择未来 clean block：

```python
clean_future_pos = 9
```

执行：

```python
x3 = replace_token(
    input_ids,
    clean_future_pos,
    new_token=34567,
)

logits3 = get_logits(
    model,
    x3,
    attention_mask,
    position_ids,
)

q3 = logits3[:, query_pos, :]

diff_future = (q3 - q0).abs().max()

print(
    "future clean diff =",
    diff_future.item(),
)
```

正确：

$$
\boxed{\text{diff\_future}\approx0}
$$

---

## 8. 理想输出

正常情况：

```text
clean prefix diff     = 0.72
same-block clean diff = 0.000000
future clean diff     = 0.000000
```

含义：

```text
             noisy Block1
                  ↑
clean Block0   ALLOW
clean Block1   BLOCK
clean Block2   BLOCK
```

异常情况：

```text
clean prefix diff     = 0.72
same-block clean diff = 0.83
future clean diff     = 0.00
```

则说明 current clean block 很可能存在泄漏。

---

## 9. 建议同时比较多个 logits 差异指标

```python
def compare_logits(a, b):
    d = a - b

    return {
        "max_abs_diff": d.abs().max().item(),
        "mean_abs_diff": d.abs().mean().item(),
        "l2_diff": d.float().norm().item(),
    }


print("prefix:", compare_logits(q0, q1))
print("same block:", compare_logits(q0, q2))
print("future:", compare_logits(q0, q3))
```

---

## 10. 方法二：直接检查最终 4D Attention Mask

假设：

```python
attention_mask.shape
```

为：

```text
[B, 1, Q, K]
```

或：

```text
[B, H, Q, K]
```

查看：

```python
q = query_pos
row = attention_mask[0, 0, q, :]
print(row)
```

假设：

```text
0      -> allow
-inf   -> block
```

那么 noisy Block1 对 clean stream 的预期为：

```text
clean Block0:
0 0 0 0

clean Block1:
-inf -inf -inf -inf

clean Block2:
-inf -inf -inf -inf
```

---

## 11. Attention Mask 自动断言

```python
import torch


def check_no_leak(
    attention_mask,
    q_pos,
    allowed_positions,
    forbidden_positions,
):
    row = attention_mask[0, 0, q_pos]

    for p in allowed_positions:
        assert torch.isfinite(row[p]), (
            f"Expected position {p} visible, "
            "but it is masked."
        )

    for p in forbidden_positions:
        is_blocked = (
            torch.isneginf(row[p])
            or row[p] < -1e4
        )

        assert is_blocked, (
            f"LEAK: forbidden position {p} "
            "is visible!"
        )
```

示例：

```python
allowed = [0, 1, 2, 3]
forbidden = list(range(4, 12))

check_no_leak(
    attention_mask,
    query_pos,
    allowed,
    forbidden,
)
```

注意：这只能证明当前拿到的 mask tensor 看起来正确，不能证明模型内部真正使用了它。

---

## 12. 方法三：梯度依赖测试

理论上，如果 noisy query 完全看不到 forbidden clean token：

$$
\frac{
\partial y_{noisy}
}{
\partial e_{forbidden\ clean}
}
=0
$$

### 12.1 构造 inputs_embeds

```python
model.eval()
model.zero_grad()

embedding_layer = model.get_input_embeddings()

embeds = embedding_layer(input_ids)

embeds = (
    embeds.detach()
    .requires_grad_(True)
)
```

### 12.2 Forward

```python
outputs = model(
    inputs_embeds=embeds,
    attention_mask=attention_mask,
    position_ids=position_ids,
    use_cache=False,
)
```

选一个 scalar：

```python
score = outputs.logits[
    0,
    query_pos,
].max()

score.backward()
```

读取：

```python
grad = embeds.grad[0]
```

### 12.3 检查不同区域梯度

```python
prefix_grad = (
    grad[0:4]
    .float()
    .norm()
    .item()
)

same_block_grad = (
    grad[4:8]
    .float()
    .norm()
    .item()
)

future_grad = (
    grad[8:12]
    .float()
    .norm()
    .item()
)

print("prefix:", prefix_grad)
print("same clean block:", same_block_grad)
print("future clean:", future_grad)
```

正常情况下类似：

```text
prefix:           0.034
same clean block: 0.000000
future clean:     0.000000
```

即：

$$
\nabla_{x^{<b}} \neq 0
$$

但：

$$
\nabla_{x^b_{clean}} = 0
$$

以及：

$$
\nabla_{x^{>b}_{clean}} = 0
$$

---

## 13. 更严格的逐位置梯度检查

```python
grad_norm = (
    grad.float()
    .norm(dim=-1)
)

for pos, value in enumerate(
    grad_norm.tolist()
):
    print(
        f"pos={pos:02d}, "
        f"grad_norm={value:.8e}"
    )
```

如果：

```text
pos 0~3  -> non-zero
pos 4~11 -> ~0
```

则符合预期。

---

## 14. 方法四：秘密答案泄漏测试

构造：

```text
The secret number is 928374.
```

当前 block clean target：

```text
928374
```

noisy block：

```text
[MASK] [MASK] [MASK]
```

记录第一次 noisy prediction。

然后只把当前 clean block 改成：

```text
135792
```

其他输入保持完全相同。

理论上：

```text
clean target:
928374 -> 135792
```

不应该导致 noisy prediction 跟着发生对应变化。

如果预测明显从：

```text
928374
```

变为：

```text
135792
```

则几乎可以确认存在 clean-target leakage。

---

## 15. Qwen3 eager attention 特别检查

如果使用 Qwen3 的：

```text
eager_attention_forward
```

必须确认最终 `attention_mask` 确实加到了 `attn_weights` 上。

典型逻辑：

```python
attn_weights = torch.matmul(
    query,
    key.transpose(-2, -1),
)

attn_weights = (
    attn_weights / scaling
)

if attention_mask is not None:
    attn_weights = (
        attn_weights
        + attention_mask
    )

attn_weights = torch.softmax(
    attn_weights,
    dim=-1,
)
```

重点排查：

- 自定义 block mask 是否被重新生成的 causal mask 覆盖；
- query/key 维度是否写反；
- mask 是否被错误 slice；
- broadcast 是否发生在错误维度；
- `is_causal` 是否又额外改变行为；
- eager / SDPA / FlashAttention 是否走了不同实现；
- mask dtype 是否正确；
- `-inf` 是否被错误转换。

---

## 16. 直接在 Attention Forward 中打印实际 Mask

建议临时在真正进入 attention kernel 的位置加入：

```python
if layer_idx == 0:
    print(
        "attention_mask.shape =",
        attention_mask.shape,
    )

    q = YOUR_QUERY_POS

    print(
        "mask row =",
        attention_mask[
            0,
            0,
            q,
            :,
        ],
    )
```

这样验证的是：

$$
\boxed{\text{模型真正使用的 attention mask}}
$$

而不是 dataloader 或 collator 里构造出来但可能被后续逻辑覆盖的 mask。

---

## 17. 推荐完整单元测试脚本

```python
import torch


@torch.no_grad()
def run_attention_leak_test(
    model,
    input_ids,
    attention_mask,
    query_pos,
    prefix_pos,
    same_block_pos,
    future_pos,
    position_ids=None,
):
    model.eval()

    def forward(x):
        kwargs = {
            "input_ids": x,
            "attention_mask": attention_mask,
            "use_cache": False,
        }

        if position_ids is not None:
            kwargs["position_ids"] = (
                position_ids
            )

        return model(
            **kwargs
        ).logits.detach()

    def mutate(x, pos, token_id):
        y = x.clone()
        y[:, pos] = token_id
        return y

    base = forward(input_ids)[
        :,
        query_pos,
        :,
    ]

    prefix_logits = forward(
        mutate(
            input_ids,
            prefix_pos,
            12345,
        )
    )[:, query_pos, :]

    same_logits = forward(
        mutate(
            input_ids,
            same_block_pos,
            23456,
        )
    )[:, query_pos, :]

    future_logits = forward(
        mutate(
            input_ids,
            future_pos,
            34567,
        )
    )[:, query_pos, :]

    def stats(x):
        d = x - base

        return {
            "max": (
                d.abs()
                .max()
                .item()
            ),
            "mean": (
                d.abs()
                .mean()
                .item()
            ),
            "l2": (
                d.float()
                .norm()
                .item()
            ),
        }

    result = {
        "prefix": stats(prefix_logits),
        "same_block": stats(same_logits),
        "future": stats(future_logits),
    }

    print(result)

    return result
```

调用：

```python
result = run_attention_leak_test(
    model=model,
    input_ids=input_ids,
    attention_mask=attention_mask,
    query_pos=16,
    prefix_pos=2,
    same_block_pos=4,
    future_pos=9,
    position_ids=position_ids,
)
```

---

## 18. 自动判定

浮点计算中不要机械要求：

```python
diff == 0
```

可以设置：

```python
ABS_TOL = 1e-6
```

然后：

```python
assert (
    result["prefix"]["max"]
    > ABS_TOL
), (
    "Prefix does not affect noisy query. "
    "Check whether clean-prefix attention "
    "is unexpectedly blocked."
)

assert (
    result["same_block"]["max"]
    < ABS_TOL
), (
    "Attention leakage detected: "
    "current clean block affects "
    "the noisy query."
)

assert (
    result["future"]["max"]
    < ABS_TOL
), (
    "Attention leakage detected: "
    "future clean block affects "
    "the noisy query."
)
```

实际 tolerance 需要根据：

- FP32；
- FP16；
- BF16；
- NPU kernel；
- fused attention；

进行调整。

BF16 / NPU 下如果存在微小数值扰动，可以从：

```text
1e-5 ~ 1e-4
```

开始测试。

---

## 19. 推荐验证步骤

```text
Step 1
确认 dual-stream layout
↓
明确 clean/noisy 的真实位置映射

Step 2
打印最终进入 attention kernel 的 4D mask
↓
验证 query-key 可见关系

Step 3
修改 clean prefix
↓
noisy logits 应发生变化

Step 4
修改 current clean block
↓
noisy logits 应保持不变

Step 5
修改 future clean block
↓
noisy logits 应保持不变

Step 6
执行 embedding gradient test
↓
forbidden clean positions gradient ≈ 0

Step 7
固定 timestep / noise mask
结合 diffusion loss 与推理结果做 sanity check
```

---

## 20. 最终判断标准

如果同时满足：

```text
修改 clean prefix
    -> noisy logits 明显变化

修改 current clean block
    -> noisy logits 基本不变

修改 future clean block
    -> noisy logits 基本不变

current/future clean embedding gradient
    -> 基本为 0
```

则基本可以认为：

$$
\boxed{\text{不存在 clean-target attention leakage}}
$$

如果修改 current clean block 后 noisy logits 明显变化，则优先检查：

1. `Noisy -> Clean` mask 是否错误包含当前 block；
2. 4D mask 的 query/key 维度是否写反；
3. `[clean, noisy]` 与 `[noisy, clean]` offset 是否搞反；
4. block id 边界条件是否把 `<` 写成了 `<=`；
5. Qwen3 内部是否覆盖了自定义 attention mask；
6. eager / SDPA / FlashAttention 是否走了不同实现；
7. position ids 和 KV cache 是否造成非预期路径。

---

## 21. 与 diffusion loss 快速下降的关系

如果训练中出现：

```text
diffusion loss:
12 -> 2
```

并且只用了很少的训练 step，那么 attention leakage 是需要优先排除的可能性之一。

但 loss 快速下降本身不能单独证明存在泄漏。

建议同时检查：

```text
1. attention leakage
2. high-noise / t≈1 diffusion loss
3. masked-token accuracy
4. diffusion-mode generation quality
5. 不同 noise ratio 下的 loss
```

只有这些指标一起看，才能判断 loss 快速下降究竟来自正常的 diffusion branch 快速适配，还是来自 clean target shortcut / attention leakage。
