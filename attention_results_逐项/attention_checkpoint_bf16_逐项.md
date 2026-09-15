# attention_checkpoint_bf16.json：逐条中文解读

共 300 条。原始总状态：PASS。

[全部字段定义与总体结论](../attention_results_解读.md)

每条下方折叠区完整保留原始字段，数字未改写。这里的序号从1开始，对应原始 checks 数组下标加1。

## 1. train/seed=0/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": [],
  "noisy_block1_query": 8,
  "visible_key_positions": [
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31
  ]
}
```

</details>

## 2. train/seed=0/方法二/实际 kernel mask/层1

原始状态：**PASS**。

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 3. train/seed=0/方法二/实际 kernel mask/层2

原始状态：**PASS**。

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 4. train/seed=0/方法二/实际 kernel mask/层3

原始状态：**PASS**。

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 5. train/seed=0/方法二/实际 kernel mask/层4

原始状态：**PASS**。

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 6. train/seed=0/方法二/实际 kernel mask/层5

原始状态：**PASS**。

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 7. train/seed=0/方法二/实际 kernel mask/层6

原始状态：**PASS**。

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 8. train/seed=0/方法二/实际 kernel mask/层7

原始状态：**PASS**。

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 9. train/seed=0/方法二/实际 kernel mask/层8

原始状态：**PASS**。

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 10. train/seed=0/方法二/实际 kernel mask/层9

原始状态：**PASS**。

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 11. train/seed=0/方法二/实际 kernel mask/层10

原始状态：**PASS**。

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 12. train/seed=0/方法二/实际 kernel mask/层11

原始状态：**PASS**。

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 13. train/seed=0/方法二/实际 kernel mask/层12

原始状态：**PASS**。

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 14. train/seed=0/方法二/实际 kernel mask/层13

原始状态：**PASS**。

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 15. train/seed=0/方法二/实际 kernel mask/层14

原始状态：**PASS**。

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 16. train/seed=0/方法二/实际 kernel mask/层15

原始状态：**PASS**。

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 17. train/seed=0/方法二/实际 kernel mask/层16

原始状态：**PASS**。

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 18. train/seed=0/方法二/实际 kernel mask/层17

原始状态：**PASS**。

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 19. train/seed=0/方法二/实际 kernel mask/层18

原始状态：**PASS**。

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 20. train/seed=0/方法二/实际 kernel mask/层19

原始状态：**PASS**。

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 21. train/seed=0/方法二/实际 kernel mask/层20

原始状态：**PASS**。

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 22. train/seed=0/方法二/实际 kernel mask/层21

原始状态：**PASS**。

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 23. train/seed=0/方法二/实际 kernel mask/层22

原始状态：**PASS**。

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 24. train/seed=0/方法二/实际 kernel mask/层23

原始状态：**PASS**。

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 25. train/seed=0/方法二/实际 kernel mask/层24

原始状态：**PASS**。

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 26. train/seed=0/方法二/实际 kernel mask/层25

原始状态：**PASS**。

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 27. train/seed=0/方法二/实际 kernel mask/层26

原始状态：**PASS**。

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 28. train/seed=0/方法二/实际 kernel mask/层27

原始状态：**PASS**。

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 29. train/seed=0/方法二/所有层 kernel 均被观察

原始状态：**PASS**。

观察到的层编号为[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]，模型预期共28层。覆盖检查通过表示没有漏掉层；每层mask是否正确要看前面的独立条目。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/所有层 kernel 均被观察",
  "status": "PASS",
  "layers": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27
  ],
  "expected_layers": 28
}
```

</details>

## 30. train/seed=0/方法一/重复性

原始状态：**PASS**。

输入完全不变连续做两次forward，全部noisy logits最大绝对差为0.0，阈值1e-06。这里为0，说明该次baseline可复现；此项本身不负责检测clean泄漏。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/重复性",
  "status": "PASS",
  "max_abs": 0.0,
  "threshold": 1e-06
}
```

</details>

## 31. train/seed=0/方法一/clean块0→noisy块0/当前

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块0→noisy块0/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 32. train/seed=0/方法一/clean块0→noisy块1/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=9.40625；平均绝对差mean_abs=1.218668818473816；整体L2差异l2=2414.79248046875；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 9.40625,
  "mean_abs": 1.218668818473816,
  "l2": 2414.79248046875,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 33. train/seed=0/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=4.6875；平均绝对差mean_abs=0.5723485350608826；整体L2差异l2=1117.118896484375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 4.6875,
  "mean_abs": 0.5723485350608826,
  "l2": 1117.118896484375,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 34. train/seed=0/方法一/clean块1→noisy块0/未来

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块1→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 35. train/seed=0/方法一/clean块1→noisy块1/当前

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块1→noisy块1/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 36. train/seed=0/方法一/clean块1→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=7.234375；平均绝对差mean_abs=1.0336308479309082；整体L2差异l2=2012.092041015625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 7.234375,
  "mean_abs": 1.0336308479309082,
  "l2": 2012.092041015625,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 37. train/seed=0/方法一/clean块2→noisy块0/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块2→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 38. train/seed=0/方法一/clean块2→noisy块1/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块2→noisy块1/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 39. train/seed=0/方法一/clean块2→noisy块2/当前

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块2→noisy块2/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 40. train/seed=0/方法四/秘密替换

原始状态：**PASS**。

只替换第二块clean的整段秘密token（ID加17取模），比较第二块noisy的全部logits，不能只凭最高分候选ID不变下结论。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法四/秘密替换",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 41. train/seed=0/方法三/noisy块0/投影0/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 3.352413416 | 15.45526218 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块0/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      15.455262184143066,
      12.167803764343262,
      10.307914733886719,
      6.716752052307129,
      6.720719814300537,
      6.681642532348633,
      5.192588806152344,
      3.3524134159088135,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      5.076989650726318,
      3.761712074279785,
      3.9905827045440674,
      4.295009613037109,
      6.623053550720215,
      5.718192100524902,
      4.553948402404785,
      3.5986592769622803,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 42. train/seed=0/方法三/noisy块0/投影1/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 3.697425604 | 14.25941658 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块0/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      3.9033398628234863,
      3.697425603866577,
      4.433581829071045,
      6.354450225830078,
      7.733734607696533,
      8.755304336547852,
      8.111178398132324,
      7.742833614349365,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      12.713664054870605,
      9.084609031677246,
      7.092101573944092,
      6.599458694458008,
      8.82470703125,
      11.677428245544434,
      14.259416580200195,
      10.430480003356934,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 43. train/seed=0/方法三/noisy块1/投影0/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 0.8862048388 | 3.46372962 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.874355793 | 14.88834 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.381359815597534,
      1.6780097484588623,
      1.75645911693573,
      1.4473241567611694,
      1.5031871795654297,
      1.4802683591842651,
      1.6100832223892212,
      1.2034902572631836,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      8.444995880126953,
      10.230984687805176,
      9.366537094116211,
      13.018646240234375,
      7.126495361328125,
      7.544284820556641,
      2.8743557929992676,
      8.012190818786621,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.4637296199798584,
      2.33959698677063,
      1.806106686592102,
      1.552263617515564,
      1.8679547309875488,
      1.2785395383834839,
      1.1835129261016846,
      0.8862048387527466,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.9488980770111084,
      4.467489242553711,
      4.323564052581787,
      9.340238571166992,
      6.710092067718506,
      5.306955337524414,
      10.118452072143555,
      14.88833999633789,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 44. train/seed=0/方法三/noisy块1/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为14.88833999633789，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 14.88833999633789,
  "threshold": 1e-08
}
```

</details>

## 45. train/seed=0/方法三/noisy块1/投影1/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 0.8919826746 | 2.742699146 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.425913334 | 14.41203213 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.742699146270752,
      1.0190155506134033,
      1.141419768333435,
      0.972802460193634,
      1.0713764429092407,
      1.0367375612258911,
      1.0427520275115967,
      0.8919826745986938,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.1767730712890625,
      5.9955010414123535,
      6.118861675262451,
      5.422670364379883,
      4.007998466491699,
      5.764642238616943,
      2.4259133338928223,
      5.459872722625732,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.679476737976074,
      2.1673038005828857,
      1.5342719554901123,
      1.3533720970153809,
      1.6322754621505737,
      1.8521391153335571,
      1.6422367095947266,
      1.3637981414794922,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.0705718994140625,
      5.054215431213379,
      4.151471138000488,
      5.814361572265625,
      5.18123197555542,
      4.366626739501953,
      8.001919746398926,
      14.412032127380371,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 46. train/seed=0/方法三/noisy块1/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为14.412032127380371，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 14.412032127380371,
  "threshold": 1e-08
}
```

</details>

## 47. train/seed=0/方法三/noisy块2/投影0/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 1.115670443 | 3.654742718 |
| 24–31 | clean Block0 | 允许 | 1.908205748 | 16.42089081 |
| 32–39 | clean Block1 | 允许 | 2.48019743 | 23.40812683 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块2/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.980933427810669,
      2.824221134185791,
      3.65474271774292,
      2.6059372425079346,
      2.2732694149017334,
      1.5481394529342651,
      1.9122583866119385,
      1.5453317165374756,
      4.004054546356201,
      4.143866539001465,
      3.5105113983154297,
      3.396662712097168,
      2.480971336364746,
      3.443612813949585,
      2.6574342250823975,
      1.9082057476043701,
      2.7287044525146484,
      2.4801974296569824,
      4.735294342041016,
      5.209320068359375,
      23.408126831054688,
      14.928601264953613,
      5.074024677276611,
      5.176385402679443,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.068894386291504,
      1.9464281797409058,
      1.9324079751968384,
      2.012691020965576,
      1.4461253881454468,
      1.2693965435028076,
      1.3354424238204956,
      1.1156704425811768,
      3.4475932121276855,
      3.552929401397705,
      4.126018047332764,
      5.6456217765808105,
      5.269335746765137,
      4.990004539489746,
      11.043159484863281,
      16.42089080810547,
      7.3019208908081055,
      4.26278018951416,
      8.062264442443848,
      6.732630252838135,
      3.38268780708313,
      12.110252380371094,
      21.553253173828125,
      4.169009208679199,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 48. train/seed=0/方法三/noisy块2/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为23.408126831054688，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 23.408126831054688,
  "threshold": 1e-08
}
```

</details>

## 49. train/seed=0/方法三/noisy块2/投影1/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 1.043049693 | 2.938477755 |
| 24–31 | clean Block0 | 允许 | 1.962047219 | 9.22903347 |
| 32–39 | clean Block1 | 允许 | 1.915687084 | 14.9946785 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块2/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.9384777545928955,
      1.251609444618225,
      1.075514793395996,
      1.3005907535552979,
      1.0857259035110474,
      1.3913536071777344,
      1.6542291641235352,
      2.707552194595337,
      2.777092218399048,
      4.281361103057861,
      3.4260385036468506,
      2.5202648639678955,
      2.111304521560669,
      2.7563769817352295,
      1.9620472192764282,
      2.705293655395508,
      1.915687084197998,
      2.122495651245117,
      2.785695791244507,
      5.098964214324951,
      14.994678497314453,
      9.355925559997559,
      3.959846019744873,
      8.12371826171875,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.402188777923584,
      1.9641114473342896,
      1.2810988426208496,
      1.054552435874939,
      1.1299569606781006,
      1.054717779159546,
      1.043049693107605,
      1.2750967741012573,
      3.4068286418914795,
      3.6057634353637695,
      4.1484375,
      5.4883952140808105,
      4.3887434005737305,
      3.1759395599365234,
      6.3954315185546875,
      9.229033470153809,
      7.956930637359619,
      2.5035953521728516,
      4.332805156707764,
      2.435676336288452,
      2.1911964416503906,
      4.604945182800293,
      6.770898342132568,
      3.67547345161438,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 50. train/seed=0/方法三/noisy块2/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为14.994678497314453，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 14.994678497314453,
  "threshold": 1e-08
}
```

</details>

## 51. train/seed=1/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": [],
  "noisy_block1_query": 8,
  "visible_key_positions": [
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31
  ]
}
```

</details>

## 52. train/seed=1/方法二/实际 kernel mask/层1

原始状态：**PASS**。

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 53. train/seed=1/方法二/实际 kernel mask/层2

原始状态：**PASS**。

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 54. train/seed=1/方法二/实际 kernel mask/层3

原始状态：**PASS**。

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 55. train/seed=1/方法二/实际 kernel mask/层4

原始状态：**PASS**。

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 56. train/seed=1/方法二/实际 kernel mask/层5

原始状态：**PASS**。

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 57. train/seed=1/方法二/实际 kernel mask/层6

原始状态：**PASS**。

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 58. train/seed=1/方法二/实际 kernel mask/层7

原始状态：**PASS**。

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 59. train/seed=1/方法二/实际 kernel mask/层8

原始状态：**PASS**。

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 60. train/seed=1/方法二/实际 kernel mask/层9

原始状态：**PASS**。

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 61. train/seed=1/方法二/实际 kernel mask/层10

原始状态：**PASS**。

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 62. train/seed=1/方法二/实际 kernel mask/层11

原始状态：**PASS**。

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 63. train/seed=1/方法二/实际 kernel mask/层12

原始状态：**PASS**。

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 64. train/seed=1/方法二/实际 kernel mask/层13

原始状态：**PASS**。

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 65. train/seed=1/方法二/实际 kernel mask/层14

原始状态：**PASS**。

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 66. train/seed=1/方法二/实际 kernel mask/层15

原始状态：**PASS**。

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 67. train/seed=1/方法二/实际 kernel mask/层16

原始状态：**PASS**。

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 68. train/seed=1/方法二/实际 kernel mask/层17

原始状态：**PASS**。

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 69. train/seed=1/方法二/实际 kernel mask/层18

原始状态：**PASS**。

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 70. train/seed=1/方法二/实际 kernel mask/层19

原始状态：**PASS**。

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 71. train/seed=1/方法二/实际 kernel mask/层20

原始状态：**PASS**。

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 72. train/seed=1/方法二/实际 kernel mask/层21

原始状态：**PASS**。

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 73. train/seed=1/方法二/实际 kernel mask/层22

原始状态：**PASS**。

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 74. train/seed=1/方法二/实际 kernel mask/层23

原始状态：**PASS**。

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 75. train/seed=1/方法二/实际 kernel mask/层24

原始状态：**PASS**。

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 76. train/seed=1/方法二/实际 kernel mask/层25

原始状态：**PASS**。

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 77. train/seed=1/方法二/实际 kernel mask/层26

原始状态：**PASS**。

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 78. train/seed=1/方法二/实际 kernel mask/层27

原始状态：**PASS**。

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 79. train/seed=1/方法二/所有层 kernel 均被观察

原始状态：**PASS**。

观察到的层编号为[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]，模型预期共28层。覆盖检查通过表示没有漏掉层；每层mask是否正确要看前面的独立条目。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/所有层 kernel 均被观察",
  "status": "PASS",
  "layers": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27
  ],
  "expected_layers": 28
}
```

</details>

## 80. train/seed=1/方法一/重复性

原始状态：**PASS**。

输入完全不变连续做两次forward，全部noisy logits最大绝对差为0.0，阈值1e-06。这里为0，说明该次baseline可复现；此项本身不负责检测clean泄漏。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/重复性",
  "status": "PASS",
  "max_abs": 0.0,
  "threshold": 1e-06
}
```

</details>

## 81. train/seed=1/方法一/clean块0→noisy块0/当前

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块0→noisy块0/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 82. train/seed=1/方法一/clean块0→noisy块1/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=12.46875；平均绝对差mean_abs=1.4051547050476074；整体L2差异l2=2788.575439453125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 12.46875,
  "mean_abs": 1.4051547050476074,
  "l2": 2788.575439453125,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 83. train/seed=1/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.2890625；平均绝对差mean_abs=0.8344119191169739；整体L2差异l2=1609.6400146484375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.2890625,
  "mean_abs": 0.8344119191169739,
  "l2": 1609.6400146484375,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 84. train/seed=1/方法一/clean块1→noisy块0/未来

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块1→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 85. train/seed=1/方法一/clean块1→noisy块1/当前

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块1→noisy块1/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 86. train/seed=1/方法一/clean块1→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=11.05859375；平均绝对差mean_abs=1.1529994010925293；整体L2差异l2=2256.18212890625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 11.05859375,
  "mean_abs": 1.1529994010925293,
  "l2": 2256.18212890625,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 87. train/seed=1/方法一/clean块2→noisy块0/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块2→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 88. train/seed=1/方法一/clean块2→noisy块1/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块2→noisy块1/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 89. train/seed=1/方法一/clean块2→noisy块2/当前

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块2→noisy块2/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 90. train/seed=1/方法四/秘密替换

原始状态：**PASS**。

只替换第二块clean的整段秘密token（ID加17取模），比较第二块noisy的全部logits，不能只凭最高分候选ID不变下结论。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法四/秘密替换",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 91. train/seed=1/方法三/noisy块0/投影0/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 7.341526508 | 19.04152298 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块0/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      19.041522979736328,
      14.892900466918945,
      14.343117713928223,
      14.916450500488281,
      12.508700370788574,
      13.514101028442383,
      12.680642127990723,
      9.741659164428711,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      11.90783977508545,
      13.793524742126465,
      11.305914878845215,
      7.341526508331299,
      8.600495338439941,
      12.901957511901855,
      12.318974494934082,
      9.548453330993652,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 92. train/seed=1/方法三/noisy块0/投影1/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 2.1788342 | 20.55016899 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块0/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      20.550168991088867,
      12.632847785949707,
      9.62946605682373,
      7.9582743644714355,
      10.607840538024902,
      12.520315170288086,
      14.489240646362305,
      12.120827674865723,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      8.408717155456543,
      4.439425468444824,
      3.2937726974487305,
      2.74052095413208,
      2.4651129245758057,
      2.5784027576446533,
      3.0657737255096436,
      2.1788341999053955,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 93. train/seed=1/方法三/noisy块1/投影0/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 0.9567117691 | 3.717240334 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 3.019621372 | 10.30047894 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块1/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.119480609893799,
      1.788221001625061,
      1.94688081741333,
      1.115784764289856,
      1.1060947179794312,
      0.9567117691040039,
      1.2971700429916382,
      1.4173557758331299,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.995305061340332,
      6.868326187133789,
      3.092719316482544,
      6.509781360626221,
      5.200967788696289,
      5.997320175170898,
      7.959027290344238,
      10.18474006652832,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.717240333557129,
      2.751955509185791,
      1.8618587255477905,
      1.2637096643447876,
      1.3620696067810059,
      1.2628272771835327,
      1.4205670356750488,
      1.445077896118164,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.35681676864624,
      5.02804708480835,
      5.384332180023193,
      4.9157209396362305,
      3.0196213722229004,
      4.812500953674316,
      10.3004789352417,
      9.767363548278809,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 94. train/seed=1/方法三/noisy块1/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为10.3004789352417，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 10.3004789352417,
  "threshold": 1e-08
}
```

</details>

## 95. train/seed=1/方法三/noisy块1/投影1/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 1.038024664 | 3.94515276 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 3.770582199 | 11.10655594 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块1/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.807518243789673,
      1.7905337810516357,
      1.7059438228607178,
      1.1569997072219849,
      1.0879498720169067,
      1.038024663925171,
      1.3423194885253906,
      1.576181411743164,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      5.286506652832031,
      7.495940208435059,
      3.7705821990966797,
      6.113242149353027,
      5.42629861831665,
      5.124693870544434,
      4.493069648742676,
      6.552077293395996,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.945152759552002,
      2.5489048957824707,
      2.1202385425567627,
      1.79665207862854,
      1.7023892402648926,
      1.4121524095535278,
      1.4945168495178223,
      1.6152071952819824,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      6.0630269050598145,
      6.655691623687744,
      6.726844787597656,
      5.986132621765137,
      4.712133407592773,
      5.341361999511719,
      10.63597583770752,
      11.106555938720703,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 96. train/seed=1/方法三/noisy块1/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为11.106555938720703，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 11.106555938720703,
  "threshold": 1e-08
}
```

</details>

## 97. train/seed=1/方法三/noisy块2/投影0/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 1.030108213 | 5.497277737 |
| 24–31 | clean Block0 | 允许 | 2.803390503 | 13.72489357 |
| 32–39 | clean Block1 | 允许 | 3.114816427 | 115.5183487 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块2/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.342020034790039,
      2.093918800354004,
      2.2044992446899414,
      1.5128124952316284,
      1.517828345298767,
      1.0581910610198975,
      1.3830413818359375,
      1.5179821252822876,
      9.112781524658203,
      7.992403984069824,
      3.7047929763793945,
      7.058455467224121,
      4.646365165710449,
      4.956640720367432,
      5.641040325164795,
      2.8033905029296875,
      3.114816427230835,
      7.212518215179443,
      3.9439756870269775,
      5.272112846374512,
      3.3113515377044678,
      4.2866387367248535,
      4.72344446182251,
      4.90876579284668,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      5.497277736663818,
      1.9249037504196167,
      1.8371856212615967,
      1.5418003797531128,
      1.5207383632659912,
      1.5884495973587036,
      1.6547359228134155,
      1.0301082134246826,
      10.61207103729248,
      11.434420585632324,
      13.724893569946289,
      12.445155143737793,
      6.855262279510498,
      7.705825328826904,
      12.079248428344727,
      10.610279083251953,
      13.118315696716309,
      6.821722030639648,
      4.842779159545898,
      4.897304534912109,
      6.185055732727051,
      15.5907621383667,
      20.13414192199707,
      115.51834869384766,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 98. train/seed=1/方法三/noisy块2/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为115.51834869384766，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 115.51834869384766,
  "threshold": 1e-08
}
```

</details>

## 99. train/seed=1/方法三/noisy块2/投影1/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 0.7598095536 | 2.465696096 |
| 24–31 | clean Block0 | 允许 | 1.738112926 | 5.344905853 |
| 32–39 | clean Block1 | 允许 | 1.659399152 | 10.80310822 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块2/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.465696096420288,
      1.827011227607727,
      1.5573582649230957,
      1.4950056076049805,
      1.1069612503051758,
      1.3495934009552002,
      1.2215381860733032,
      0.9492054581642151,
      3.985123634338379,
      4.758307933807373,
      2.2305312156677246,
      4.690104007720947,
      2.439833402633667,
      3.0691077709198,
      2.7392759323120117,
      1.8370078802108765,
      1.9691320657730103,
      4.176459789276123,
      2.1292006969451904,
      3.0136067867279053,
      2.9117331504821777,
      5.32433557510376,
      5.3338117599487305,
      2.5566823482513428,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.7403277158737183,
      1.3190807104110718,
      1.082687258720398,
      0.9547682404518127,
      0.9504244923591614,
      0.9707112312316895,
      0.9124192595481873,
      0.7598095536231995,
      4.420202255249023,
      4.807419776916504,
      4.341542720794678,
      3.3585333824157715,
      1.7381129264831543,
      2.5340566635131836,
      4.305287837982178,
      5.344905853271484,
      3.6824262142181396,
      2.6777191162109375,
      1.659399151802063,
      1.8881754875183105,
      2.1248724460601807,
      2.941896438598633,
      2.4788806438446045,
      10.803108215332031,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 100. train/seed=1/方法三/noisy块2/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为10.803108215332031，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 10.803108215332031,
  "threshold": 1e-08
}
```

</details>

## 101. train/seed=2/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": [],
  "noisy_block1_query": 8,
  "visible_key_positions": [
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31
  ]
}
```

</details>

## 102. train/seed=2/方法二/实际 kernel mask/层1

原始状态：**PASS**。

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 103. train/seed=2/方法二/实际 kernel mask/层2

原始状态：**PASS**。

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 104. train/seed=2/方法二/实际 kernel mask/层3

原始状态：**PASS**。

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 105. train/seed=2/方法二/实际 kernel mask/层4

原始状态：**PASS**。

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 106. train/seed=2/方法二/实际 kernel mask/层5

原始状态：**PASS**。

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 107. train/seed=2/方法二/实际 kernel mask/层6

原始状态：**PASS**。

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 108. train/seed=2/方法二/实际 kernel mask/层7

原始状态：**PASS**。

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 109. train/seed=2/方法二/实际 kernel mask/层8

原始状态：**PASS**。

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 110. train/seed=2/方法二/实际 kernel mask/层9

原始状态：**PASS**。

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 111. train/seed=2/方法二/实际 kernel mask/层10

原始状态：**PASS**。

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 112. train/seed=2/方法二/实际 kernel mask/层11

原始状态：**PASS**。

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 113. train/seed=2/方法二/实际 kernel mask/层12

原始状态：**PASS**。

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 114. train/seed=2/方法二/实际 kernel mask/层13

原始状态：**PASS**。

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 115. train/seed=2/方法二/实际 kernel mask/层14

原始状态：**PASS**。

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 116. train/seed=2/方法二/实际 kernel mask/层15

原始状态：**PASS**。

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 117. train/seed=2/方法二/实际 kernel mask/层16

原始状态：**PASS**。

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 118. train/seed=2/方法二/实际 kernel mask/层17

原始状态：**PASS**。

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 119. train/seed=2/方法二/实际 kernel mask/层18

原始状态：**PASS**。

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 120. train/seed=2/方法二/实际 kernel mask/层19

原始状态：**PASS**。

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 121. train/seed=2/方法二/实际 kernel mask/层20

原始状态：**PASS**。

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 122. train/seed=2/方法二/实际 kernel mask/层21

原始状态：**PASS**。

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 123. train/seed=2/方法二/实际 kernel mask/层22

原始状态：**PASS**。

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 124. train/seed=2/方法二/实际 kernel mask/层23

原始状态：**PASS**。

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 125. train/seed=2/方法二/实际 kernel mask/层24

原始状态：**PASS**。

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 126. train/seed=2/方法二/实际 kernel mask/层25

原始状态：**PASS**。

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 127. train/seed=2/方法二/实际 kernel mask/层26

原始状态：**PASS**。

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 128. train/seed=2/方法二/实际 kernel mask/层27

原始状态：**PASS**。

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 129. train/seed=2/方法二/所有层 kernel 均被观察

原始状态：**PASS**。

观察到的层编号为[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]，模型预期共28层。覆盖检查通过表示没有漏掉层；每层mask是否正确要看前面的独立条目。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/所有层 kernel 均被观察",
  "status": "PASS",
  "layers": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27
  ],
  "expected_layers": 28
}
```

</details>

## 130. train/seed=2/方法一/重复性

原始状态：**PASS**。

输入完全不变连续做两次forward，全部noisy logits最大绝对差为0.0，阈值1e-06。这里为0，说明该次baseline可复现；此项本身不负责检测clean泄漏。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/重复性",
  "status": "PASS",
  "max_abs": 0.0,
  "threshold": 1e-06
}
```

</details>

## 131. train/seed=2/方法一/clean块0→noisy块0/当前

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块0→noisy块0/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 132. train/seed=2/方法一/clean块0→noisy块1/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=10.53125；平均绝对差mean_abs=1.3875676393508911；整体L2差异l2=2785.00634765625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 10.53125,
  "mean_abs": 1.3875676393508911,
  "l2": 2785.00634765625,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 133. train/seed=2/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.08984375；平均绝对差mean_abs=0.7679682374000549；整体L2差异l2=1545.90673828125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.08984375,
  "mean_abs": 0.7679682374000549,
  "l2": 1545.90673828125,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 134. train/seed=2/方法一/clean块1→noisy块0/未来

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块1→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 135. train/seed=2/方法一/clean块1→noisy块1/当前

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块1→noisy块1/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 136. train/seed=2/方法一/clean块1→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=9.578125；平均绝对差mean_abs=1.010023593902588；整体L2差异l2=2059.118896484375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 9.578125,
  "mean_abs": 1.010023593902588,
  "l2": 2059.118896484375,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 137. train/seed=2/方法一/clean块2→noisy块0/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块2→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 138. train/seed=2/方法一/clean块2→noisy块1/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块2→noisy块1/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 139. train/seed=2/方法一/clean块2→noisy块2/当前

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块2→noisy块2/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 140. train/seed=2/方法四/秘密替换

原始状态：**PASS**。

只替换第二块clean的整段秘密token（ID加17取模），比较第二块noisy的全部logits，不能只凭最高分候选ID不变下结论。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法四/秘密替换",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 141. train/seed=2/方法三/noisy块0/投影0/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 2.645092249 | 6.932599068 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块0/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      6.745665073394775,
      6.350033760070801,
      5.1542134284973145,
      3.6517767906188965,
      3.528777599334717,
      4.690436363220215,
      4.639914035797119,
      3.915959119796753,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      6.932599067687988,
      5.430586338043213,
      5.0769944190979,
      6.29700231552124,
      5.945174694061279,
      4.882911682128906,
      3.869993209838867,
      2.645092248916626,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 142. train/seed=2/方法三/noisy块0/投影1/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 4.696586132 | 17.49384117 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块0/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      17.49384117126465,
      12.996794700622559,
      12.052020072937012,
      14.010272026062012,
      13.368398666381836,
      12.141448020935059,
      11.5263032913208,
      9.8638916015625,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      5.591006278991699,
      6.446035861968994,
      5.67310905456543,
      4.6965861320495605,
      4.707823276519775,
      5.544256687164307,
      9.761054039001465,
      11.434554100036621,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 143. train/seed=2/方法三/noisy块1/投影0/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 1.07258141 | 2.851112604 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.865558386 | 15.43413162 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块1/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.8876525163650513,
      2.026705503463745,
      2.8504767417907715,
      1.7312774658203125,
      2.470677375793457,
      1.9229931831359863,
      2.8511126041412354,
      2.7190017700195312,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      9.355871200561523,
      15.434131622314453,
      9.735540390014648,
      5.428525924682617,
      3.6495487689971924,
      9.082613945007324,
      9.335579872131348,
      5.368283748626709,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.3141546249389648,
      1.07258141040802,
      1.0761263370513916,
      1.282505750656128,
      1.231331467628479,
      1.334542989730835,
      1.2128642797470093,
      1.4218353033065796,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.155786991119385,
      7.966645240783691,
      2.920741319656372,
      5.052769184112549,
      3.9654195308685303,
      3.1293351650238037,
      2.865558385848999,
      3.9398601055145264,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 144. train/seed=2/方法三/noisy块1/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为15.434131622314453，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 15.434131622314453,
  "threshold": 1e-08
}
```

</details>

## 145. train/seed=2/方法三/noisy块1/投影1/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 0.9232820272 | 1.967573643 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.458497763 | 18.93818092 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块1/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.9248703718185425,
      1.5393867492675781,
      1.124711275100708,
      1.0660094022750854,
      1.4046602249145508,
      1.746181845664978,
      1.9652574062347412,
      1.0892295837402344,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      9.294981956481934,
      18.938180923461914,
      9.746371269226074,
      6.120312213897705,
      3.672649621963501,
      6.8817572593688965,
      6.783875942230225,
      6.038577556610107,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.862914800643921,
      1.5636287927627563,
      1.0669938325881958,
      0.9651774764060974,
      1.4418562650680542,
      1.967573642730713,
      0.9232820272445679,
      0.9563342928886414,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.3219075202941895,
      6.789752960205078,
      6.836311340332031,
      8.794258117675781,
      4.476724624633789,
      2.4584977626800537,
      2.5317459106445312,
      3.856125831604004,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 146. train/seed=2/方法三/noisy块1/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为18.938180923461914，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 18.938180923461914,
  "threshold": 1e-08
}
```

</details>

## 147. train/seed=2/方法三/noisy块2/投影0/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 0.9243100286 | 3.584563732 |
| 24–31 | clean Block0 | 允许 | 2.096711874 | 17.30939674 |
| 32–39 | clean Block1 | 允许 | 1.257455468 | 26.47695732 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块2/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.4716508388519287,
      3.081404685974121,
      3.584563732147217,
      2.446077823638916,
      1.4919204711914062,
      1.2420456409454346,
      1.3301860094070435,
      0.924310028553009,
      13.046133041381836,
      17.309396743774414,
      9.373955726623535,
      4.961273193359375,
      3.619123697280884,
      9.749469757080078,
      4.906312465667725,
      2.889331340789795,
      4.9458818435668945,
      2.903225898742676,
      2.313727855682373,
      9.864333152770996,
      5.593471050262451,
      6.586360931396484,
      22.231300354003906,
      26.476957321166992,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.4567862749099731,
      1.594795823097229,
      1.280011773109436,
      1.0887895822525024,
      1.344285011291504,
      1.367979645729065,
      1.1168081760406494,
      1.114450216293335,
      5.310390472412109,
      5.433557033538818,
      6.217501163482666,
      5.841769695281982,
      3.321727991104126,
      2.4263200759887695,
      2.0967118740081787,
      2.2036566734313965,
      3.009052038192749,
      2.333780527114868,
      3.7812788486480713,
      8.582927703857422,
      17.631792068481445,
      1.8616873025894165,
      5.766026020050049,
      1.2574554681777954,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 148. train/seed=2/方法三/noisy块2/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为26.476957321166992，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 26.476957321166992,
  "threshold": 1e-08
}
```

</details>

## 149. train/seed=2/方法三/noisy块2/投影1/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 0.9029234052 | 3.05314517 |
| 24–31 | clean Block0 | 允许 | 1.869611859 | 10.86892891 |
| 32–39 | clean Block1 | 允许 | 1.497926235 | 17.23698425 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块2/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.053145170211792,
      1.8786505460739136,
      1.2269399166107178,
      1.889737606048584,
      1.278321623802185,
      1.9429858922958374,
      1.9087449312210083,
      1.4393441677093506,
      8.050292015075684,
      10.868928909301758,
      6.55896520614624,
      1.9075101613998413,
      1.8696118593215942,
      3.252106189727783,
      1.9061040878295898,
      2.198599100112915,
      4.042838096618652,
      1.4979262351989746,
      1.9346370697021484,
      4.183128356933594,
      3.181840181350708,
      3.446563959121704,
      13.902247428894043,
      17.236984252929688,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.469834327697754,
      1.3216906785964966,
      1.4241493940353394,
      1.0402495861053467,
      0.9029234051704407,
      1.2608973979949951,
      1.6121505498886108,
      1.1259288787841797,
      4.944627285003662,
      4.85508918762207,
      4.00365686416626,
      8.624136924743652,
      3.534823417663574,
      3.0805740356445312,
      2.905015230178833,
      2.551527500152588,
      3.198387861251831,
      2.821225166320801,
      4.47115421295166,
      8.347105979919434,
      14.559184074401855,
      2.6465847492218018,
      7.752407073974609,
      2.3651223182678223,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 150. train/seed=2/方法三/noisy块2/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为17.236984252929688，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 17.236984252929688,
  "threshold": 1e-08
}
```

</details>

## 151. eval/seed=0/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": [],
  "noisy_block1_query": 8,
  "visible_key_positions": [
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31
  ]
}
```

</details>

## 152. eval/seed=0/方法二/实际 kernel mask/层1

原始状态：**PASS**。

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 153. eval/seed=0/方法二/实际 kernel mask/层2

原始状态：**PASS**。

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 154. eval/seed=0/方法二/实际 kernel mask/层3

原始状态：**PASS**。

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 155. eval/seed=0/方法二/实际 kernel mask/层4

原始状态：**PASS**。

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 156. eval/seed=0/方法二/实际 kernel mask/层5

原始状态：**PASS**。

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 157. eval/seed=0/方法二/实际 kernel mask/层6

原始状态：**PASS**。

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 158. eval/seed=0/方法二/实际 kernel mask/层7

原始状态：**PASS**。

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 159. eval/seed=0/方法二/实际 kernel mask/层8

原始状态：**PASS**。

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 160. eval/seed=0/方法二/实际 kernel mask/层9

原始状态：**PASS**。

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 161. eval/seed=0/方法二/实际 kernel mask/层10

原始状态：**PASS**。

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 162. eval/seed=0/方法二/实际 kernel mask/层11

原始状态：**PASS**。

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 163. eval/seed=0/方法二/实际 kernel mask/层12

原始状态：**PASS**。

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 164. eval/seed=0/方法二/实际 kernel mask/层13

原始状态：**PASS**。

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 165. eval/seed=0/方法二/实际 kernel mask/层14

原始状态：**PASS**。

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 166. eval/seed=0/方法二/实际 kernel mask/层15

原始状态：**PASS**。

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 167. eval/seed=0/方法二/实际 kernel mask/层16

原始状态：**PASS**。

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 168. eval/seed=0/方法二/实际 kernel mask/层17

原始状态：**PASS**。

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 169. eval/seed=0/方法二/实际 kernel mask/层18

原始状态：**PASS**。

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 170. eval/seed=0/方法二/实际 kernel mask/层19

原始状态：**PASS**。

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 171. eval/seed=0/方法二/实际 kernel mask/层20

原始状态：**PASS**。

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 172. eval/seed=0/方法二/实际 kernel mask/层21

原始状态：**PASS**。

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 173. eval/seed=0/方法二/实际 kernel mask/层22

原始状态：**PASS**。

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 174. eval/seed=0/方法二/实际 kernel mask/层23

原始状态：**PASS**。

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 175. eval/seed=0/方法二/实际 kernel mask/层24

原始状态：**PASS**。

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 176. eval/seed=0/方法二/实际 kernel mask/层25

原始状态：**PASS**。

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 177. eval/seed=0/方法二/实际 kernel mask/层26

原始状态：**PASS**。

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 178. eval/seed=0/方法二/实际 kernel mask/层27

原始状态：**PASS**。

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 179. eval/seed=0/方法二/所有层 kernel 均被观察

原始状态：**PASS**。

观察到的层编号为[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]，模型预期共28层。覆盖检查通过表示没有漏掉层；每层mask是否正确要看前面的独立条目。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/所有层 kernel 均被观察",
  "status": "PASS",
  "layers": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27
  ],
  "expected_layers": 28
}
```

</details>

## 180. eval/seed=0/方法一/重复性

原始状态：**PASS**。

输入完全不变连续做两次forward，全部noisy logits最大绝对差为0.0，阈值1e-06。这里为0，说明该次baseline可复现；此项本身不负责检测clean泄漏。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/重复性",
  "status": "PASS",
  "max_abs": 0.0,
  "threshold": 1e-06
}
```

</details>

## 181. eval/seed=0/方法一/clean块0→noisy块0/当前

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块0→noisy块0/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 182. eval/seed=0/方法一/clean块0→noisy块1/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=9.40625；平均绝对差mean_abs=1.218668818473816；整体L2差异l2=2414.79248046875；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 9.40625,
  "mean_abs": 1.218668818473816,
  "l2": 2414.79248046875,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 183. eval/seed=0/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=4.6875；平均绝对差mean_abs=0.5723485350608826；整体L2差异l2=1117.118896484375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 4.6875,
  "mean_abs": 0.5723485350608826,
  "l2": 1117.118896484375,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 184. eval/seed=0/方法一/clean块1→noisy块0/未来

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块1→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 185. eval/seed=0/方法一/clean块1→noisy块1/当前

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块1→noisy块1/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 186. eval/seed=0/方法一/clean块1→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=7.234375；平均绝对差mean_abs=1.0336308479309082；整体L2差异l2=2012.092041015625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 7.234375,
  "mean_abs": 1.0336308479309082,
  "l2": 2012.092041015625,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 187. eval/seed=0/方法一/clean块2→noisy块0/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块2→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 188. eval/seed=0/方法一/clean块2→noisy块1/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块2→noisy块1/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 189. eval/seed=0/方法一/clean块2→noisy块2/当前

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块2→noisy块2/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 190. eval/seed=0/方法四/秘密替换

原始状态：**PASS**。

只替换第二块clean的整段秘密token（ID加17取模），比较第二块noisy的全部logits，不能只凭最高分候选ID不变下结论。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法四/秘密替换",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 191. eval/seed=0/方法三/noisy块0/投影0/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 3.352413416 | 15.45526218 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块0/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      15.455262184143066,
      12.167803764343262,
      10.307914733886719,
      6.716752052307129,
      6.720719814300537,
      6.681642532348633,
      5.192588806152344,
      3.3524134159088135,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      5.076989650726318,
      3.761712074279785,
      3.9905827045440674,
      4.295009613037109,
      6.623053550720215,
      5.718192100524902,
      4.553948402404785,
      3.5986592769622803,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 192. eval/seed=0/方法三/noisy块0/投影1/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 3.697425604 | 14.25941658 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块0/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      3.9033398628234863,
      3.697425603866577,
      4.433581829071045,
      6.354450225830078,
      7.733734607696533,
      8.755304336547852,
      8.111178398132324,
      7.742833614349365,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      12.713664054870605,
      9.084609031677246,
      7.092101573944092,
      6.599458694458008,
      8.82470703125,
      11.677428245544434,
      14.259416580200195,
      10.430480003356934,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 193. eval/seed=0/方法三/noisy块1/投影0/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 0.8862048388 | 3.46372962 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.874355793 | 14.88834 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.381359815597534,
      1.6780097484588623,
      1.75645911693573,
      1.4473241567611694,
      1.5031871795654297,
      1.4802683591842651,
      1.6100832223892212,
      1.2034902572631836,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      8.444995880126953,
      10.230984687805176,
      9.366537094116211,
      13.018646240234375,
      7.126495361328125,
      7.544284820556641,
      2.8743557929992676,
      8.012190818786621,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.4637296199798584,
      2.33959698677063,
      1.806106686592102,
      1.552263617515564,
      1.8679547309875488,
      1.2785395383834839,
      1.1835129261016846,
      0.8862048387527466,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.9488980770111084,
      4.467489242553711,
      4.323564052581787,
      9.340238571166992,
      6.710092067718506,
      5.306955337524414,
      10.118452072143555,
      14.88833999633789,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 194. eval/seed=0/方法三/noisy块1/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为14.88833999633789，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 14.88833999633789,
  "threshold": 1e-08
}
```

</details>

## 195. eval/seed=0/方法三/noisy块1/投影1/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 0.8919826746 | 2.742699146 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.425913334 | 14.41203213 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.742699146270752,
      1.0190155506134033,
      1.141419768333435,
      0.972802460193634,
      1.0713764429092407,
      1.0367375612258911,
      1.0427520275115967,
      0.8919826745986938,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.1767730712890625,
      5.9955010414123535,
      6.118861675262451,
      5.422670364379883,
      4.007998466491699,
      5.764642238616943,
      2.4259133338928223,
      5.459872722625732,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.679476737976074,
      2.1673038005828857,
      1.5342719554901123,
      1.3533720970153809,
      1.6322754621505737,
      1.8521391153335571,
      1.6422367095947266,
      1.3637981414794922,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.0705718994140625,
      5.054215431213379,
      4.151471138000488,
      5.814361572265625,
      5.18123197555542,
      4.366626739501953,
      8.001919746398926,
      14.412032127380371,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 196. eval/seed=0/方法三/noisy块1/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为14.412032127380371，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 14.412032127380371,
  "threshold": 1e-08
}
```

</details>

## 197. eval/seed=0/方法三/noisy块2/投影0/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 1.115670443 | 3.654742718 |
| 24–31 | clean Block0 | 允许 | 1.908205748 | 16.42089081 |
| 32–39 | clean Block1 | 允许 | 2.48019743 | 23.40812683 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块2/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.980933427810669,
      2.824221134185791,
      3.65474271774292,
      2.6059372425079346,
      2.2732694149017334,
      1.5481394529342651,
      1.9122583866119385,
      1.5453317165374756,
      4.004054546356201,
      4.143866539001465,
      3.5105113983154297,
      3.396662712097168,
      2.480971336364746,
      3.443612813949585,
      2.6574342250823975,
      1.9082057476043701,
      2.7287044525146484,
      2.4801974296569824,
      4.735294342041016,
      5.209320068359375,
      23.408126831054688,
      14.928601264953613,
      5.074024677276611,
      5.176385402679443,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.068894386291504,
      1.9464281797409058,
      1.9324079751968384,
      2.012691020965576,
      1.4461253881454468,
      1.2693965435028076,
      1.3354424238204956,
      1.1156704425811768,
      3.4475932121276855,
      3.552929401397705,
      4.126018047332764,
      5.6456217765808105,
      5.269335746765137,
      4.990004539489746,
      11.043159484863281,
      16.42089080810547,
      7.3019208908081055,
      4.26278018951416,
      8.062264442443848,
      6.732630252838135,
      3.38268780708313,
      12.110252380371094,
      21.553253173828125,
      4.169009208679199,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 198. eval/seed=0/方法三/noisy块2/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为23.408126831054688，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 23.408126831054688,
  "threshold": 1e-08
}
```

</details>

## 199. eval/seed=0/方法三/noisy块2/投影1/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 1.043049693 | 2.938477755 |
| 24–31 | clean Block0 | 允许 | 1.962047219 | 9.22903347 |
| 32–39 | clean Block1 | 允许 | 1.915687084 | 14.9946785 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块2/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.9384777545928955,
      1.251609444618225,
      1.075514793395996,
      1.3005907535552979,
      1.0857259035110474,
      1.3913536071777344,
      1.6542291641235352,
      2.707552194595337,
      2.777092218399048,
      4.281361103057861,
      3.4260385036468506,
      2.5202648639678955,
      2.111304521560669,
      2.7563769817352295,
      1.9620472192764282,
      2.705293655395508,
      1.915687084197998,
      2.122495651245117,
      2.785695791244507,
      5.098964214324951,
      14.994678497314453,
      9.355925559997559,
      3.959846019744873,
      8.12371826171875,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.402188777923584,
      1.9641114473342896,
      1.2810988426208496,
      1.054552435874939,
      1.1299569606781006,
      1.054717779159546,
      1.043049693107605,
      1.2750967741012573,
      3.4068286418914795,
      3.6057634353637695,
      4.1484375,
      5.4883952140808105,
      4.3887434005737305,
      3.1759395599365234,
      6.3954315185546875,
      9.229033470153809,
      7.956930637359619,
      2.5035953521728516,
      4.332805156707764,
      2.435676336288452,
      2.1911964416503906,
      4.604945182800293,
      6.770898342132568,
      3.67547345161438,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 200. eval/seed=0/方法三/noisy块2/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为14.994678497314453，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 14.994678497314453,
  "threshold": 1e-08
}
```

</details>

## 201. eval/seed=1/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": [],
  "noisy_block1_query": 8,
  "visible_key_positions": [
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31
  ]
}
```

</details>

## 202. eval/seed=1/方法二/实际 kernel mask/层1

原始状态：**PASS**。

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 203. eval/seed=1/方法二/实际 kernel mask/层2

原始状态：**PASS**。

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 204. eval/seed=1/方法二/实际 kernel mask/层3

原始状态：**PASS**。

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 205. eval/seed=1/方法二/实际 kernel mask/层4

原始状态：**PASS**。

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 206. eval/seed=1/方法二/实际 kernel mask/层5

原始状态：**PASS**。

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 207. eval/seed=1/方法二/实际 kernel mask/层6

原始状态：**PASS**。

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 208. eval/seed=1/方法二/实际 kernel mask/层7

原始状态：**PASS**。

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 209. eval/seed=1/方法二/实际 kernel mask/层8

原始状态：**PASS**。

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 210. eval/seed=1/方法二/实际 kernel mask/层9

原始状态：**PASS**。

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 211. eval/seed=1/方法二/实际 kernel mask/层10

原始状态：**PASS**。

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 212. eval/seed=1/方法二/实际 kernel mask/层11

原始状态：**PASS**。

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 213. eval/seed=1/方法二/实际 kernel mask/层12

原始状态：**PASS**。

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 214. eval/seed=1/方法二/实际 kernel mask/层13

原始状态：**PASS**。

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 215. eval/seed=1/方法二/实际 kernel mask/层14

原始状态：**PASS**。

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 216. eval/seed=1/方法二/实际 kernel mask/层15

原始状态：**PASS**。

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 217. eval/seed=1/方法二/实际 kernel mask/层16

原始状态：**PASS**。

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 218. eval/seed=1/方法二/实际 kernel mask/层17

原始状态：**PASS**。

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 219. eval/seed=1/方法二/实际 kernel mask/层18

原始状态：**PASS**。

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 220. eval/seed=1/方法二/实际 kernel mask/层19

原始状态：**PASS**。

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 221. eval/seed=1/方法二/实际 kernel mask/层20

原始状态：**PASS**。

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 222. eval/seed=1/方法二/实际 kernel mask/层21

原始状态：**PASS**。

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 223. eval/seed=1/方法二/实际 kernel mask/层22

原始状态：**PASS**。

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 224. eval/seed=1/方法二/实际 kernel mask/层23

原始状态：**PASS**。

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 225. eval/seed=1/方法二/实际 kernel mask/层24

原始状态：**PASS**。

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 226. eval/seed=1/方法二/实际 kernel mask/层25

原始状态：**PASS**。

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 227. eval/seed=1/方法二/实际 kernel mask/层26

原始状态：**PASS**。

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 228. eval/seed=1/方法二/实际 kernel mask/层27

原始状态：**PASS**。

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 229. eval/seed=1/方法二/所有层 kernel 均被观察

原始状态：**PASS**。

观察到的层编号为[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]，模型预期共28层。覆盖检查通过表示没有漏掉层；每层mask是否正确要看前面的独立条目。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/所有层 kernel 均被观察",
  "status": "PASS",
  "layers": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27
  ],
  "expected_layers": 28
}
```

</details>

## 230. eval/seed=1/方法一/重复性

原始状态：**PASS**。

输入完全不变连续做两次forward，全部noisy logits最大绝对差为0.0，阈值1e-06。这里为0，说明该次baseline可复现；此项本身不负责检测clean泄漏。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/重复性",
  "status": "PASS",
  "max_abs": 0.0,
  "threshold": 1e-06
}
```

</details>

## 231. eval/seed=1/方法一/clean块0→noisy块0/当前

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块0→noisy块0/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 232. eval/seed=1/方法一/clean块0→noisy块1/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=12.46875；平均绝对差mean_abs=1.4051547050476074；整体L2差异l2=2788.575439453125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 12.46875,
  "mean_abs": 1.4051547050476074,
  "l2": 2788.575439453125,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 233. eval/seed=1/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.2890625；平均绝对差mean_abs=0.8344119191169739；整体L2差异l2=1609.6400146484375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.2890625,
  "mean_abs": 0.8344119191169739,
  "l2": 1609.6400146484375,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 234. eval/seed=1/方法一/clean块1→noisy块0/未来

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块1→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 235. eval/seed=1/方法一/clean块1→noisy块1/当前

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块1→noisy块1/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 236. eval/seed=1/方法一/clean块1→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=11.05859375；平均绝对差mean_abs=1.1529994010925293；整体L2差异l2=2256.18212890625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 11.05859375,
  "mean_abs": 1.1529994010925293,
  "l2": 2256.18212890625,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 237. eval/seed=1/方法一/clean块2→noisy块0/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块2→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 238. eval/seed=1/方法一/clean块2→noisy块1/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块2→noisy块1/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 239. eval/seed=1/方法一/clean块2→noisy块2/当前

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块2→noisy块2/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 240. eval/seed=1/方法四/秘密替换

原始状态：**PASS**。

只替换第二块clean的整段秘密token（ID加17取模），比较第二块noisy的全部logits，不能只凭最高分候选ID不变下结论。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法四/秘密替换",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 241. eval/seed=1/方法三/noisy块0/投影0/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 7.341526508 | 19.04152298 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块0/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      19.041522979736328,
      14.892900466918945,
      14.343117713928223,
      14.916450500488281,
      12.508700370788574,
      13.514101028442383,
      12.680642127990723,
      9.741659164428711,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      11.90783977508545,
      13.793524742126465,
      11.305914878845215,
      7.341526508331299,
      8.600495338439941,
      12.901957511901855,
      12.318974494934082,
      9.548453330993652,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 242. eval/seed=1/方法三/noisy块0/投影1/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 2.1788342 | 20.55016899 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块0/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      20.550168991088867,
      12.632847785949707,
      9.62946605682373,
      7.9582743644714355,
      10.607840538024902,
      12.520315170288086,
      14.489240646362305,
      12.120827674865723,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      8.408717155456543,
      4.439425468444824,
      3.2937726974487305,
      2.74052095413208,
      2.4651129245758057,
      2.5784027576446533,
      3.0657737255096436,
      2.1788341999053955,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 243. eval/seed=1/方法三/noisy块1/投影0/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 0.9567117691 | 3.717240334 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 3.019621372 | 10.30047894 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块1/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.119480609893799,
      1.788221001625061,
      1.94688081741333,
      1.115784764289856,
      1.1060947179794312,
      0.9567117691040039,
      1.2971700429916382,
      1.4173557758331299,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.995305061340332,
      6.868326187133789,
      3.092719316482544,
      6.509781360626221,
      5.200967788696289,
      5.997320175170898,
      7.959027290344238,
      10.18474006652832,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.717240333557129,
      2.751955509185791,
      1.8618587255477905,
      1.2637096643447876,
      1.3620696067810059,
      1.2628272771835327,
      1.4205670356750488,
      1.445077896118164,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.35681676864624,
      5.02804708480835,
      5.384332180023193,
      4.9157209396362305,
      3.0196213722229004,
      4.812500953674316,
      10.3004789352417,
      9.767363548278809,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 244. eval/seed=1/方法三/noisy块1/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为10.3004789352417，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 10.3004789352417,
  "threshold": 1e-08
}
```

</details>

## 245. eval/seed=1/方法三/noisy块1/投影1/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 1.038024664 | 3.94515276 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 3.770582199 | 11.10655594 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块1/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.807518243789673,
      1.7905337810516357,
      1.7059438228607178,
      1.1569997072219849,
      1.0879498720169067,
      1.038024663925171,
      1.3423194885253906,
      1.576181411743164,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      5.286506652832031,
      7.495940208435059,
      3.7705821990966797,
      6.113242149353027,
      5.42629861831665,
      5.124693870544434,
      4.493069648742676,
      6.552077293395996,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.945152759552002,
      2.5489048957824707,
      2.1202385425567627,
      1.79665207862854,
      1.7023892402648926,
      1.4121524095535278,
      1.4945168495178223,
      1.6152071952819824,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      6.0630269050598145,
      6.655691623687744,
      6.726844787597656,
      5.986132621765137,
      4.712133407592773,
      5.341361999511719,
      10.63597583770752,
      11.106555938720703,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 246. eval/seed=1/方法三/noisy块1/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为11.106555938720703，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 11.106555938720703,
  "threshold": 1e-08
}
```

</details>

## 247. eval/seed=1/方法三/noisy块2/投影0/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 1.030108213 | 5.497277737 |
| 24–31 | clean Block0 | 允许 | 2.803390503 | 13.72489357 |
| 32–39 | clean Block1 | 允许 | 3.114816427 | 115.5183487 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块2/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.342020034790039,
      2.093918800354004,
      2.2044992446899414,
      1.5128124952316284,
      1.517828345298767,
      1.0581910610198975,
      1.3830413818359375,
      1.5179821252822876,
      9.112781524658203,
      7.992403984069824,
      3.7047929763793945,
      7.058455467224121,
      4.646365165710449,
      4.956640720367432,
      5.641040325164795,
      2.8033905029296875,
      3.114816427230835,
      7.212518215179443,
      3.9439756870269775,
      5.272112846374512,
      3.3113515377044678,
      4.2866387367248535,
      4.72344446182251,
      4.90876579284668,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      5.497277736663818,
      1.9249037504196167,
      1.8371856212615967,
      1.5418003797531128,
      1.5207383632659912,
      1.5884495973587036,
      1.6547359228134155,
      1.0301082134246826,
      10.61207103729248,
      11.434420585632324,
      13.724893569946289,
      12.445155143737793,
      6.855262279510498,
      7.705825328826904,
      12.079248428344727,
      10.610279083251953,
      13.118315696716309,
      6.821722030639648,
      4.842779159545898,
      4.897304534912109,
      6.185055732727051,
      15.5907621383667,
      20.13414192199707,
      115.51834869384766,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 248. eval/seed=1/方法三/noisy块2/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为115.51834869384766，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 115.51834869384766,
  "threshold": 1e-08
}
```

</details>

## 249. eval/seed=1/方法三/noisy块2/投影1/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 0.7598095536 | 2.465696096 |
| 24–31 | clean Block0 | 允许 | 1.738112926 | 5.344905853 |
| 32–39 | clean Block1 | 允许 | 1.659399152 | 10.80310822 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块2/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.465696096420288,
      1.827011227607727,
      1.5573582649230957,
      1.4950056076049805,
      1.1069612503051758,
      1.3495934009552002,
      1.2215381860733032,
      0.9492054581642151,
      3.985123634338379,
      4.758307933807373,
      2.2305312156677246,
      4.690104007720947,
      2.439833402633667,
      3.0691077709198,
      2.7392759323120117,
      1.8370078802108765,
      1.9691320657730103,
      4.176459789276123,
      2.1292006969451904,
      3.0136067867279053,
      2.9117331504821777,
      5.32433557510376,
      5.3338117599487305,
      2.5566823482513428,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.7403277158737183,
      1.3190807104110718,
      1.082687258720398,
      0.9547682404518127,
      0.9504244923591614,
      0.9707112312316895,
      0.9124192595481873,
      0.7598095536231995,
      4.420202255249023,
      4.807419776916504,
      4.341542720794678,
      3.3585333824157715,
      1.7381129264831543,
      2.5340566635131836,
      4.305287837982178,
      5.344905853271484,
      3.6824262142181396,
      2.6777191162109375,
      1.659399151802063,
      1.8881754875183105,
      2.1248724460601807,
      2.941896438598633,
      2.4788806438446045,
      10.803108215332031,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 250. eval/seed=1/方法三/noisy块2/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为10.803108215332031，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 10.803108215332031,
  "threshold": 1e-08
}
```

</details>

## 251. eval/seed=2/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": [],
  "noisy_block1_query": 8,
  "visible_key_positions": [
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31
  ]
}
```

</details>

## 252. eval/seed=2/方法二/实际 kernel mask/层1

原始状态：**PASS**。

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 253. eval/seed=2/方法二/实际 kernel mask/层2

原始状态：**PASS**。

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 254. eval/seed=2/方法二/实际 kernel mask/层3

原始状态：**PASS**。

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 255. eval/seed=2/方法二/实际 kernel mask/层4

原始状态：**PASS**。

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 256. eval/seed=2/方法二/实际 kernel mask/层5

原始状态：**PASS**。

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 257. eval/seed=2/方法二/实际 kernel mask/层6

原始状态：**PASS**。

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 258. eval/seed=2/方法二/实际 kernel mask/层7

原始状态：**PASS**。

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 259. eval/seed=2/方法二/实际 kernel mask/层8

原始状态：**PASS**。

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 260. eval/seed=2/方法二/实际 kernel mask/层9

原始状态：**PASS**。

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 261. eval/seed=2/方法二/实际 kernel mask/层10

原始状态：**PASS**。

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 262. eval/seed=2/方法二/实际 kernel mask/层11

原始状态：**PASS**。

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 263. eval/seed=2/方法二/实际 kernel mask/层12

原始状态：**PASS**。

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 264. eval/seed=2/方法二/实际 kernel mask/层13

原始状态：**PASS**。

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 265. eval/seed=2/方法二/实际 kernel mask/层14

原始状态：**PASS**。

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 266. eval/seed=2/方法二/实际 kernel mask/层15

原始状态：**PASS**。

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 267. eval/seed=2/方法二/实际 kernel mask/层16

原始状态：**PASS**。

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 268. eval/seed=2/方法二/实际 kernel mask/层17

原始状态：**PASS**。

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 269. eval/seed=2/方法二/实际 kernel mask/层18

原始状态：**PASS**。

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 270. eval/seed=2/方法二/实际 kernel mask/层19

原始状态：**PASS**。

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 271. eval/seed=2/方法二/实际 kernel mask/层20

原始状态：**PASS**。

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 272. eval/seed=2/方法二/实际 kernel mask/层21

原始状态：**PASS**。

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 273. eval/seed=2/方法二/实际 kernel mask/层22

原始状态：**PASS**。

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 274. eval/seed=2/方法二/实际 kernel mask/层23

原始状态：**PASS**。

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 275. eval/seed=2/方法二/实际 kernel mask/层24

原始状态：**PASS**。

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 276. eval/seed=2/方法二/实际 kernel mask/层25

原始状态：**PASS**。

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 277. eval/seed=2/方法二/实际 kernel mask/层26

原始状态：**PASS**。

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 278. eval/seed=2/方法二/实际 kernel mask/层27

原始状态：**PASS**。

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[2, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[2, 8, 48, 128]，第二维为key头数。mask=[2, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    2,
    16,
    48,
    128
  ],
  "k_shape": [
    2,
    8,
    48,
    128
  ],
  "mask_shape": [
    2,
    1,
    48,
    48
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 0,
  "first_wrong_qk": []
}
```

</details>

## 279. eval/seed=2/方法二/所有层 kernel 均被观察

原始状态：**PASS**。

观察到的层编号为[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]，模型预期共28层。覆盖检查通过表示没有漏掉层；每层mask是否正确要看前面的独立条目。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/所有层 kernel 均被观察",
  "status": "PASS",
  "layers": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27
  ],
  "expected_layers": 28
}
```

</details>

## 280. eval/seed=2/方法一/重复性

原始状态：**PASS**。

输入完全不变连续做两次forward，全部noisy logits最大绝对差为0.0，阈值1e-06。这里为0，说明该次baseline可复现；此项本身不负责检测clean泄漏。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/重复性",
  "status": "PASS",
  "max_abs": 0.0,
  "threshold": 1e-06
}
```

</details>

## 281. eval/seed=2/方法一/clean块0→noisy块0/当前

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块0→noisy块0/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 282. eval/seed=2/方法一/clean块0→noisy块1/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=10.53125；平均绝对差mean_abs=1.3875676393508911；整体L2差异l2=2785.00634765625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 10.53125,
  "mean_abs": 1.3875676393508911,
  "l2": 2785.00634765625,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 283. eval/seed=2/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.08984375；平均绝对差mean_abs=0.7679682374000549；整体L2差异l2=1545.90673828125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.08984375,
  "mean_abs": 0.7679682374000549,
  "l2": 1545.90673828125,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 284. eval/seed=2/方法一/clean块1→noisy块0/未来

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块1→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 285. eval/seed=2/方法一/clean块1→noisy块1/当前

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块1→noisy块1/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 286. eval/seed=2/方法一/clean块1→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=9.578125；平均绝对差mean_abs=1.010023593902588；整体L2差异l2=2059.118896484375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 9.578125,
  "mean_abs": 1.010023593902588,
  "l2": 2059.118896484375,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 287. eval/seed=2/方法一/clean块2→noisy块0/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block0（位置0–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块2→noisy块0/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 288. eval/seed=2/方法一/clean块2→noisy块1/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块2→noisy块1/未来",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 289. eval/seed=2/方法一/clean块2→noisy块2/当前

原始状态：**PASS**。

只替换clean Block2（拼接位置40–47），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块2→noisy块2/当前",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 290. eval/seed=2/方法四/秘密替换

原始状态：**PASS**。

只替换第二块clean的整段秘密token（ID加17取模），比较第二块noisy的全部logits，不能只凭最高分候选ID不变下结论。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法四/秘密替换",
  "status": "PASS",
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "l2": 0.0,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 291. eval/seed=2/方法三/noisy块0/投影0/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 2.645092249 | 6.932599068 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块0/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      6.745665073394775,
      6.350033760070801,
      5.1542134284973145,
      3.6517767906188965,
      3.528777599334717,
      4.690436363220215,
      4.639914035797119,
      3.915959119796753,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      6.932599067687988,
      5.430586338043213,
      5.0769944190979,
      6.29700231552124,
      5.945174694061279,
      4.882911682128906,
      3.869993209838867,
      2.645092248916626,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 292. eval/seed=2/方法三/noisy块0/投影1/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 允许 | 4.696586132 | 17.49384117 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 禁止 | 0 | 0 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块0/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      17.49384117126465,
      12.996794700622559,
      12.052020072937012,
      14.010272026062012,
      13.368398666381836,
      12.141448020935059,
      11.5263032913208,
      9.8638916015625,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      5.591006278991699,
      6.446035861968994,
      5.67310905456543,
      4.6965861320495605,
      4.707823276519775,
      5.544256687164307,
      9.761054039001465,
      11.434554100036621,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 293. eval/seed=2/方法三/noisy块1/投影0/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 1.07258141 | 2.851112604 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.865558386 | 15.43413162 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块1/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.8876525163650513,
      2.026705503463745,
      2.8504767417907715,
      1.7312774658203125,
      2.470677375793457,
      1.9229931831359863,
      2.8511126041412354,
      2.7190017700195312,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      9.355871200561523,
      15.434131622314453,
      9.735540390014648,
      5.428525924682617,
      3.6495487689971924,
      9.082613945007324,
      9.335579872131348,
      5.368283748626709,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.3141546249389648,
      1.07258141040802,
      1.0761263370513916,
      1.282505750656128,
      1.231331467628479,
      1.334542989730835,
      1.2128642797470093,
      1.4218353033065796,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.155786991119385,
      7.966645240783691,
      2.920741319656372,
      5.052769184112549,
      3.9654195308685303,
      3.1293351650238037,
      2.865558385848999,
      3.9398601055145264,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 294. eval/seed=2/方法三/noisy块1/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为15.434131622314453，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 15.434131622314453,
  "threshold": 1e-08
}
```

</details>

## 295. eval/seed=2/方法三/noisy块1/投影1/禁止位置

原始状态：**PASS**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 允许 | 0.9232820272 | 1.967573643 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.458497763 | 18.93818092 |
| 32–39 | clean Block1 | 禁止 | 0 | 0 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块1/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.9248703718185425,
      1.5393867492675781,
      1.124711275100708,
      1.0660094022750854,
      1.4046602249145508,
      1.746181845664978,
      1.9652574062347412,
      1.0892295837402344,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      9.294981956481934,
      18.938180923461914,
      9.746371269226074,
      6.120312213897705,
      3.672649621963501,
      6.8817572593688965,
      6.783875942230225,
      6.038577556610107,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.862914800643921,
      1.5636287927627563,
      1.0669938325881958,
      0.9651774764060974,
      1.4418562650680542,
      1.967573642730713,
      0.9232820272445679,
      0.9563342928886414,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.3219075202941895,
      6.789752960205078,
      6.836311340332031,
      8.794258117675781,
      4.476724624633789,
      2.4584977626800537,
      2.5317459106445312,
      3.856125831604004,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 296. eval/seed=2/方法三/noisy块1/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为18.938180923461914，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 18.938180923461914,
  "threshold": 1e-08
}
```

</details>

## 297. eval/seed=2/方法三/noisy块2/投影0/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 0.9243100286 | 3.584563732 |
| 24–31 | clean Block0 | 允许 | 2.096711874 | 17.30939674 |
| 32–39 | clean Block1 | 允许 | 1.257455468 | 26.47695732 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块2/投影0/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.4716508388519287,
      3.081404685974121,
      3.584563732147217,
      2.446077823638916,
      1.4919204711914062,
      1.2420456409454346,
      1.3301860094070435,
      0.924310028553009,
      13.046133041381836,
      17.309396743774414,
      9.373955726623535,
      4.961273193359375,
      3.619123697280884,
      9.749469757080078,
      4.906312465667725,
      2.889331340789795,
      4.9458818435668945,
      2.903225898742676,
      2.313727855682373,
      9.864333152770996,
      5.593471050262451,
      6.586360931396484,
      22.231300354003906,
      26.476957321166992,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.4567862749099731,
      1.594795823097229,
      1.280011773109436,
      1.0887895822525024,
      1.344285011291504,
      1.367979645729065,
      1.1168081760406494,
      1.114450216293335,
      5.310390472412109,
      5.433557033538818,
      6.217501163482666,
      5.841769695281982,
      3.321727991104126,
      2.4263200759887695,
      2.0967118740081787,
      2.2036566734313965,
      3.009052038192749,
      2.333780527114868,
      3.7812788486480713,
      8.582927703857422,
      17.631792068481445,
      1.8616873025894165,
      5.766026020050049,
      1.2574554681777954,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 298. eval/seed=2/方法三/noisy块2/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为26.476957321166992，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 26.476957321166992,
  "threshold": 1e-08
}
```

</details>

## 299. eval/seed=2/方法三/noisy块2/投影1/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–7 | noisy Block0 | 禁止 | 0 | 0 |
| 8–15 | noisy Block1 | 禁止 | 0 | 0 |
| 16–23 | noisy Block2 | 允许 | 0.9029234052 | 3.05314517 |
| 24–31 | clean Block0 | 允许 | 1.869611859 | 10.86892891 |
| 32–39 | clean Block1 | 允许 | 1.497926235 | 17.23698425 |
| 40–47 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块2/投影1/禁止位置",
  "status": "PASS",
  "max_norm": 0.0,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.053145170211792,
      1.8786505460739136,
      1.2269399166107178,
      1.889737606048584,
      1.278321623802185,
      1.9429858922958374,
      1.9087449312210083,
      1.4393441677093506,
      8.050292015075684,
      10.868928909301758,
      6.55896520614624,
      1.9075101613998413,
      1.8696118593215942,
      3.252106189727783,
      1.9061040878295898,
      2.198599100112915,
      4.042838096618652,
      1.4979262351989746,
      1.9346370697021484,
      4.183128356933594,
      3.181840181350708,
      3.446563959121704,
      13.902247428894043,
      17.236984252929688,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.469834327697754,
      1.3216906785964966,
      1.4241493940353394,
      1.0402495861053467,
      0.9029234051704407,
      1.2608973979949951,
      1.6121505498886108,
      1.1259288787841797,
      4.944627285003662,
      4.85508918762207,
      4.00365686416626,
      8.624136924743652,
      3.534823417663574,
      3.0805740356445312,
      2.905015230178833,
      2.551527500152588,
      3.198387861251831,
      2.821225166320801,
      4.47115421295166,
      8.347105979919434,
      14.559184074401855,
      2.6465847492218018,
      7.752407073974609,
      2.3651223182678223,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 300. eval/seed=2/方法三/noisy块2/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为17.236984252929688，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 17.236984252929688,
  "threshold": 1e-08
}
```

</details>
