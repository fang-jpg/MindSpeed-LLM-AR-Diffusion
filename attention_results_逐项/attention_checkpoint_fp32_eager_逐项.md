# attention_checkpoint_fp32_eager.json：逐条中文解读

共 300 条。原始总状态：PASS。

[全部字段定义与总体结论](../attention_results_解读.md)

每条下方折叠区完整保留原始字段，数字未改写。这里的序号从1开始，对应原始 checks 数组下标加1。

## 1. train/seed=0/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=9.615226745605469；平均绝对差mean_abs=1.288233995437622；整体L2差异l2=1801.15185546875；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 9.615226745605469,
  "mean_abs": 1.288233995437622,
  "l2": 1801.15185546875,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 33. train/seed=0/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=4.763500690460205；平均绝对差mean_abs=0.5351479649543762；整体L2差异l2=742.7193603515625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 4.763500690460205,
  "mean_abs": 0.5351479649543762,
  "l2": 742.7193603515625,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.689870357513428；平均绝对差mean_abs=1.0978652238845825；整体L2差异l2=1502.0501708984375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.689870357513428,
  "mean_abs": 1.0978652238845825,
  "l2": 1502.0501708984375,
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
| 0–7 | noisy Block0 | 允许 | 6.335972309 | 16.28297806 |
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
      16.282978057861328,
      14.799434661865234,
      11.525925636291504,
      8.331246376037598,
      6.773989677429199,
      7.1813225746154785,
      9.708141326904297,
      6.335972309112549,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 8.558109283 | 19.26331902 |
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
      19.26331901550293,
      12.342178344726562,
      8.558109283447266,
      12.006348609924316,
      17.118667602539062,
      15.720996856689453,
      11.825777053833008,
      10.078360557556152,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 1.256666541 | 3.2825737 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 3.408213377 | 14.03880215 |
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
      3.282573699951172,
      1.9469106197357178,
      1.7449668645858765,
      1.490591049194336,
      1.7315157651901245,
      1.8077378273010254,
      1.5971623659133911,
      1.2566665410995483,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.580597877502441,
      9.374564170837402,
      10.396730422973633,
      14.038802146911621,
      5.852033615112305,
      7.867068767547607,
      3.4082133769989014,
      6.447384834289551,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为14.038802146911621，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 14.038802146911621,
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
| 8–15 | noisy Block1 | 允许 | 2.022323847 | 3.941165924 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 5.791453362 | 13.98709869 |
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
      3.9411659240722656,
      2.7399649620056152,
      2.380265712738037,
      2.0223238468170166,
      2.4695255756378174,
      2.214064598083496,
      2.028325319290161,
      2.0620906352996826,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      9.841365814208984,
      12.992112159729004,
      10.307872772216797,
      13.987098693847656,
      6.02877950668335,
      12.552664756774902,
      5.7914533615112305,
      10.888460159301758,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为13.987098693847656，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 13.987098693847656,
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
| 16–23 | noisy Block2 | 允许 | 2.024369478 | 4.154603958 |
| 24–31 | clean Block0 | 允许 | 1.719654202 | 5.579734325 |
| 32–39 | clean Block1 | 允许 | 2.129769564 | 21.94734573 |
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
      3.029308319091797,
      3.505004405975342,
      4.154603958129883,
      2.2964837551116943,
      2.3915631771087646,
      2.2833001613616943,
      2.1793248653411865,
      2.024369478225708,
      4.1893815994262695,
      5.5797343254089355,
      3.670168876647949,
      3.3469059467315674,
      2.376685619354248,
      3.1761772632598877,
      1.7196542024612427,
      2.4421205520629883,
      2.7908525466918945,
      2.1297695636749268,
      4.763181209564209,
      5.109403610229492,
      21.947345733642578,
      13.723285675048828,
      4.7110676765441895,
      6.59763765335083,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为21.947345733642578，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 21.947345733642578,
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
| 16–23 | noisy Block2 | 允许 | 1.542116642 | 5.491381645 |
| 24–31 | clean Block0 | 允许 | 1.968784213 | 6.738436222 |
| 32–39 | clean Block1 | 允许 | 2.829349756 | 20.74148178 |
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
      5.491381645202637,
      4.4206318855285645,
      1.6316111087799072,
      1.8374818563461304,
      2.497504472732544,
      1.542116641998291,
      1.7683284282684326,
      1.592655062675476,
      3.7311038970947266,
      6.738436222076416,
      4.848827838897705,
      3.472949504852295,
      3.142725944519043,
      3.489565849304199,
      1.968784213066101,
      2.8752715587615967,
      3.1357479095458984,
      2.8293497562408447,
      4.814870357513428,
      6.081122398376465,
      20.74148178100586,
      13.868244171142578,
      4.666952610015869,
      11.407054901123047,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为20.74148178100586，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 20.74148178100586,
  "threshold": 1e-08
}
```

</details>

## 51. train/seed=1/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=10.155803680419922；平均绝对差mean_abs=1.4782954454421997；整体L2差异l2=2026.05322265625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 10.155803680419922,
  "mean_abs": 1.4782954454421997,
  "l2": 2026.05322265625,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 83. train/seed=1/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.3011345863342285；平均绝对差mean_abs=0.7890196442604065；整体L2差异l2=1080.5001220703125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.3011345863342285,
  "mean_abs": 0.7890196442604065,
  "l2": 1080.5001220703125,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.463789939880371；平均绝对差mean_abs=0.9693664312362671；整体L2差异l2=1285.199462890625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.463789939880371,
  "mean_abs": 0.9693664312362671,
  "l2": 1285.199462890625,
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
| 0–7 | noisy Block0 | 允许 | 3.025094271 | 11.19643688 |
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
      7.681530475616455,
      9.27409553527832,
      10.395503997802734,
      11.196436882019043,
      7.347560405731201,
      5.257318019866943,
      4.407240390777588,
      3.0250942707061768,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 4.988809586 | 17.20315361 |
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
      17.203153610229492,
      13.494383811950684,
      10.105690956115723,
      7.622837066650391,
      7.132073402404785,
      5.977060317993164,
      6.1784539222717285,
      4.988809585571289,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 1.400677919 | 4.516222954 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 6.356587887 | 13.60091114 |
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
      4.516222953796387,
      2.8864097595214844,
      2.7005844116210938,
      2.848724365234375,
      1.5033124685287476,
      1.4006779193878174,
      1.634282112121582,
      1.6426407098770142,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.548661708831787,
      8.411681175231934,
      6.356587886810303,
      9.593992233276367,
      6.41001558303833,
      8.61397933959961,
      13.600911140441895,
      12.164996147155762,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为13.600911140441895，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 13.600911140441895,
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
| 8–15 | noisy Block1 | 允许 | 1.220423818 | 3.095654249 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 4.34126997 | 12.47044945 |
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
      3.095654249191284,
      2.688632011413574,
      3.020314931869507,
      2.8320882320404053,
      2.6328132152557373,
      1.7203038930892944,
      1.6268271207809448,
      1.2204238176345825,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.670777797698975,
      9.151833534240723,
      7.512302398681641,
      12.470449447631836,
      4.3412699699401855,
      6.202707290649414,
      7.463085651397705,
      10.619453430175781,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为12.470449447631836，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 12.470449447631836,
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
| 16–23 | noisy Block2 | 允许 | 1.287435174 | 3.012718201 |
| 24–31 | clean Block0 | 允许 | 2.087312937 | 6.879443645 |
| 32–39 | clean Block1 | 允许 | 2.498709679 | 4.410504818 |
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
      3.0127182006835938,
      2.4155218601226807,
      2.5412447452545166,
      1.576393485069275,
      1.4498794078826904,
      1.2874351739883423,
      1.8355053663253784,
      2.1706275939941406,
      4.928045749664307,
      4.90913200378418,
      3.5604889392852783,
      6.879443645477295,
      2.560215473175049,
      3.7836930751800537,
      4.277956485748291,
      2.087312936782837,
      2.4987096786499023,
      4.4105048179626465,
      4.238722801208496,
      3.3633053302764893,
      2.9714131355285645,
      4.265603065490723,
      4.393912315368652,
      4.203841686248779,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为6.879443645477295，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 6.879443645477295,
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
| 16–23 | noisy Block2 | 允许 | 1.351593375 | 2.411175728 |
| 24–31 | clean Block0 | 允许 | 2.164690733 | 5.626579285 |
| 32–39 | clean Block1 | 允许 | 2.552030325 | 5.636362076 |
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
      2.1415951251983643,
      2.4111757278442383,
      1.5082972049713135,
      1.8125334978103638,
      1.992520809173584,
      1.8566484451293945,
      1.6422595977783203,
      1.3515933752059937,
      4.784337043762207,
      5.626579284667969,
      2.4436492919921875,
      5.53748083114624,
      2.5859627723693848,
      3.6421315670013428,
      3.921311855316162,
      2.1646907329559326,
      2.552030324935913,
      4.346503257751465,
      3.0076301097869873,
      4.30131196975708,
      3.1075801849365234,
      5.636362075805664,
      5.034220218658447,
      3.50254487991333,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为5.636362075805664，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 5.636362075805664,
  "threshold": 1e-08
}
```

</details>

## 101. train/seed=2/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=10.599039077758789；平均绝对差mean_abs=1.559154748916626；整体L2差异l2=2202.206787109375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 10.599039077758789,
  "mean_abs": 1.559154748916626,
  "l2": 2202.206787109375,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 133. train/seed=2/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.064630031585693；平均绝对差mean_abs=0.8832046389579773；整体L2差异l2=1246.9151611328125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.064630031585693,
  "mean_abs": 0.8832046389579773,
  "l2": 1246.9151611328125,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=9.362726211547852；平均绝对差mean_abs=0.8762505054473877；整体L2差异l2=1265.832275390625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 9.362726211547852,
  "mean_abs": 0.8762505054473877,
  "l2": 1265.832275390625,
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
| 0–7 | noisy Block0 | 允许 | 4.173662663 | 11.4579134 |
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
      11.457913398742676,
      10.509913444519043,
      8.590126991271973,
      5.229335308074951,
      4.1736626625061035,
      7.007447719573975,
      9.450613021850586,
      7.5932440757751465,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 5.609442234 | 16.97858429 |
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
      16.97858428955078,
      14.296526908874512,
      12.679939270019531,
      14.234723091125488,
      12.988152503967285,
      11.465152740478516,
      8.596963882446289,
      5.609442234039307,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 1.517316222 | 5.50614357 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 5.075341225 | 23.54408646 |
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
      3.4688076972961426,
      2.803511142730713,
      3.841660499572754,
      5.506143569946289,
      2.9051098823547363,
      2.0435876846313477,
      2.4600231647491455,
      1.517316222190857,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      11.077143669128418,
      23.544086456298828,
      11.582091331481934,
      5.07534122467041,
      5.897240161895752,
      11.575531005859375,
      9.628326416015625,
      8.202746391296387,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为23.544086456298828，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 23.544086456298828,
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
| 8–15 | noisy Block1 | 允许 | 2.180807352 | 4.152695179 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 4.826767921 | 24.48163223 |
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
      2.981022596359253,
      2.2631287574768066,
      2.2098381519317627,
      2.18080735206604,
      2.711740016937256,
      2.9241955280303955,
      3.661702871322632,
      4.152695178985596,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      16.331336975097656,
      24.481632232666016,
      17.39482307434082,
      10.085222244262695,
      4.826767921447754,
      14.455169677734375,
      10.249527931213379,
      5.641576766967773,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为24.481632232666016，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 24.481632232666016,
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
| 16–23 | noisy Block2 | 允许 | 2.041744232 | 4.322297096 |
| 24–31 | clean Block0 | 允许 | 2.161569595 | 15.53830147 |
| 32–39 | clean Block1 | 允许 | 2.089456797 | 17.88158417 |
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
      3.270784378051758,
      2.212522268295288,
      3.2042734622955322,
      2.0417442321777344,
      2.2734124660491943,
      2.047138214111328,
      3.7941396236419678,
      4.322297096252441,
      9.351029396057129,
      15.538301467895508,
      8.983991622924805,
      5.421989917755127,
      2.161569595336914,
      5.209918022155762,
      3.656005859375,
      3.868734121322632,
      4.541912078857422,
      2.576005458831787,
      2.089456796646118,
      5.8770647048950195,
      4.36475944519043,
      5.903749942779541,
      17.88158416748047,
      11.795610427856445,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为17.88158416748047，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 17.88158416748047,
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
| 16–23 | noisy Block2 | 允许 | 1.549940348 | 3.233731747 |
| 24–31 | clean Block0 | 允许 | 1.795076251 | 17.52937889 |
| 32–39 | clean Block1 | 允许 | 1.919044137 | 15.54867458 |
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
      2.9370267391204834,
      1.9217970371246338,
      1.5499403476715088,
      1.7755937576293945,
      2.43882417678833,
      3.1263272762298584,
      2.5087571144104004,
      3.233731746673584,
      12.751720428466797,
      17.52937889099121,
      5.939986705780029,
      2.640377998352051,
      1.7950762510299683,
      3.1517751216888428,
      3.8903443813323975,
      3.3006982803344727,
      5.764036655426025,
      2.199646472930908,
      1.9190441370010376,
      5.944885730743408,
      4.284093379974365,
      5.281365871429443,
      15.548674583435059,
      12.504441261291504,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为17.52937889099121，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 17.52937889099121,
  "threshold": 1e-08
}
```

</details>

## 151. eval/seed=0/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=9.615226745605469；平均绝对差mean_abs=1.288233995437622；整体L2差异l2=1801.15185546875；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 9.615226745605469,
  "mean_abs": 1.288233995437622,
  "l2": 1801.15185546875,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 183. eval/seed=0/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=4.763500690460205；平均绝对差mean_abs=0.5351479649543762；整体L2差异l2=742.7193603515625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 4.763500690460205,
  "mean_abs": 0.5351479649543762,
  "l2": 742.7193603515625,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.689870357513428；平均绝对差mean_abs=1.0978652238845825；整体L2差异l2=1502.0501708984375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.689870357513428,
  "mean_abs": 1.0978652238845825,
  "l2": 1502.0501708984375,
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
| 0–7 | noisy Block0 | 允许 | 6.335972309 | 16.28297806 |
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
      16.282978057861328,
      14.799434661865234,
      11.525925636291504,
      8.331246376037598,
      6.773989677429199,
      7.1813225746154785,
      9.708141326904297,
      6.335972309112549,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 8.558109283 | 19.26331902 |
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
      19.26331901550293,
      12.342178344726562,
      8.558109283447266,
      12.006348609924316,
      17.118667602539062,
      15.720996856689453,
      11.825777053833008,
      10.078360557556152,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 1.256666541 | 3.2825737 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 3.408213377 | 14.03880215 |
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
      3.282573699951172,
      1.9469106197357178,
      1.7449668645858765,
      1.490591049194336,
      1.7315157651901245,
      1.8077378273010254,
      1.5971623659133911,
      1.2566665410995483,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.580597877502441,
      9.374564170837402,
      10.396730422973633,
      14.038802146911621,
      5.852033615112305,
      7.867068767547607,
      3.4082133769989014,
      6.447384834289551,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为14.038802146911621，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 14.038802146911621,
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
| 8–15 | noisy Block1 | 允许 | 2.022323847 | 3.941165924 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 5.791453362 | 13.98709869 |
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
      3.9411659240722656,
      2.7399649620056152,
      2.380265712738037,
      2.0223238468170166,
      2.4695255756378174,
      2.214064598083496,
      2.028325319290161,
      2.0620906352996826,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      9.841365814208984,
      12.992112159729004,
      10.307872772216797,
      13.987098693847656,
      6.02877950668335,
      12.552664756774902,
      5.7914533615112305,
      10.888460159301758,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为13.987098693847656，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 13.987098693847656,
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
| 16–23 | noisy Block2 | 允许 | 2.024369478 | 4.154603958 |
| 24–31 | clean Block0 | 允许 | 1.719654202 | 5.579734325 |
| 32–39 | clean Block1 | 允许 | 2.129769564 | 21.94734573 |
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
      3.029308319091797,
      3.505004405975342,
      4.154603958129883,
      2.2964837551116943,
      2.3915631771087646,
      2.2833001613616943,
      2.1793248653411865,
      2.024369478225708,
      4.1893815994262695,
      5.5797343254089355,
      3.670168876647949,
      3.3469059467315674,
      2.376685619354248,
      3.1761772632598877,
      1.7196542024612427,
      2.4421205520629883,
      2.7908525466918945,
      2.1297695636749268,
      4.763181209564209,
      5.109403610229492,
      21.947345733642578,
      13.723285675048828,
      4.7110676765441895,
      6.59763765335083,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为21.947345733642578，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 21.947345733642578,
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
| 16–23 | noisy Block2 | 允许 | 1.542116642 | 5.491381645 |
| 24–31 | clean Block0 | 允许 | 1.968784213 | 6.738436222 |
| 32–39 | clean Block1 | 允许 | 2.829349756 | 20.74148178 |
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
      5.491381645202637,
      4.4206318855285645,
      1.6316111087799072,
      1.8374818563461304,
      2.497504472732544,
      1.542116641998291,
      1.7683284282684326,
      1.592655062675476,
      3.7311038970947266,
      6.738436222076416,
      4.848827838897705,
      3.472949504852295,
      3.142725944519043,
      3.489565849304199,
      1.968784213066101,
      2.8752715587615967,
      3.1357479095458984,
      2.8293497562408447,
      4.814870357513428,
      6.081122398376465,
      20.74148178100586,
      13.868244171142578,
      4.666952610015869,
      11.407054901123047,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为20.74148178100586，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 20.74148178100586,
  "threshold": 1e-08
}
```

</details>

## 201. eval/seed=1/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=10.155803680419922；平均绝对差mean_abs=1.4782954454421997；整体L2差异l2=2026.05322265625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 10.155803680419922,
  "mean_abs": 1.4782954454421997,
  "l2": 2026.05322265625,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 233. eval/seed=1/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.3011345863342285；平均绝对差mean_abs=0.7890196442604065；整体L2差异l2=1080.5001220703125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.3011345863342285,
  "mean_abs": 0.7890196442604065,
  "l2": 1080.5001220703125,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.463789939880371；平均绝对差mean_abs=0.9693664312362671；整体L2差异l2=1285.199462890625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.463789939880371,
  "mean_abs": 0.9693664312362671,
  "l2": 1285.199462890625,
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
| 0–7 | noisy Block0 | 允许 | 3.025094271 | 11.19643688 |
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
      7.681530475616455,
      9.27409553527832,
      10.395503997802734,
      11.196436882019043,
      7.347560405731201,
      5.257318019866943,
      4.407240390777588,
      3.0250942707061768,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 4.988809586 | 17.20315361 |
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
      17.203153610229492,
      13.494383811950684,
      10.105690956115723,
      7.622837066650391,
      7.132073402404785,
      5.977060317993164,
      6.1784539222717285,
      4.988809585571289,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 1.400677919 | 4.516222954 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 6.356587887 | 13.60091114 |
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
      4.516222953796387,
      2.8864097595214844,
      2.7005844116210938,
      2.848724365234375,
      1.5033124685287476,
      1.4006779193878174,
      1.634282112121582,
      1.6426407098770142,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.548661708831787,
      8.411681175231934,
      6.356587886810303,
      9.593992233276367,
      6.41001558303833,
      8.61397933959961,
      13.600911140441895,
      12.164996147155762,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为13.600911140441895，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 13.600911140441895,
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
| 8–15 | noisy Block1 | 允许 | 1.220423818 | 3.095654249 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 4.34126997 | 12.47044945 |
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
      3.095654249191284,
      2.688632011413574,
      3.020314931869507,
      2.8320882320404053,
      2.6328132152557373,
      1.7203038930892944,
      1.6268271207809448,
      1.2204238176345825,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.670777797698975,
      9.151833534240723,
      7.512302398681641,
      12.470449447631836,
      4.3412699699401855,
      6.202707290649414,
      7.463085651397705,
      10.619453430175781,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为12.470449447631836，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 12.470449447631836,
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
| 16–23 | noisy Block2 | 允许 | 1.287435174 | 3.012718201 |
| 24–31 | clean Block0 | 允许 | 2.087312937 | 6.879443645 |
| 32–39 | clean Block1 | 允许 | 2.498709679 | 4.410504818 |
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
      3.0127182006835938,
      2.4155218601226807,
      2.5412447452545166,
      1.576393485069275,
      1.4498794078826904,
      1.2874351739883423,
      1.8355053663253784,
      2.1706275939941406,
      4.928045749664307,
      4.90913200378418,
      3.5604889392852783,
      6.879443645477295,
      2.560215473175049,
      3.7836930751800537,
      4.277956485748291,
      2.087312936782837,
      2.4987096786499023,
      4.4105048179626465,
      4.238722801208496,
      3.3633053302764893,
      2.9714131355285645,
      4.265603065490723,
      4.393912315368652,
      4.203841686248779,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为6.879443645477295，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 6.879443645477295,
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
| 16–23 | noisy Block2 | 允许 | 1.351593375 | 2.411175728 |
| 24–31 | clean Block0 | 允许 | 2.164690733 | 5.626579285 |
| 32–39 | clean Block1 | 允许 | 2.552030325 | 5.636362076 |
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
      2.1415951251983643,
      2.4111757278442383,
      1.5082972049713135,
      1.8125334978103638,
      1.992520809173584,
      1.8566484451293945,
      1.6422595977783203,
      1.3515933752059937,
      4.784337043762207,
      5.626579284667969,
      2.4436492919921875,
      5.53748083114624,
      2.5859627723693848,
      3.6421315670013428,
      3.921311855316162,
      2.1646907329559326,
      2.552030324935913,
      4.346503257751465,
      3.0076301097869873,
      4.30131196975708,
      3.1075801849365234,
      5.636362075805664,
      5.034220218658447,
      3.50254487991333,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为5.636362075805664，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 5.636362075805664,
  "threshold": 1e-08
}
```

</details>

## 251. eval/seed=2/方法二/实际 kernel mask/层0

原始状态：**PASS**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 第一层示例query=8，可见key=[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]；位置映射为noisy 0–23，clean 24–47。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层0",
  "status": "PASS",
  "layer": 0,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层1",
  "status": "PASS",
  "layer": 1,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 3 层（层索引2）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层2",
  "status": "PASS",
  "layer": 2,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 4 层（层索引3）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层3",
  "status": "PASS",
  "layer": 3,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 5 层（层索引4）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层4",
  "status": "PASS",
  "layer": 4,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 6 层（层索引5）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层5",
  "status": "PASS",
  "layer": 5,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 7 层（层索引6）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层6",
  "status": "PASS",
  "layer": 6,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 8 层（层索引7）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层7",
  "status": "PASS",
  "layer": 7,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 9 层（层索引8）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层8",
  "status": "PASS",
  "layer": 8,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 10 层（层索引9）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层9",
  "status": "PASS",
  "layer": 9,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 11 层（层索引10）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层10",
  "status": "PASS",
  "layer": 10,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 12 层（层索引11）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层11",
  "status": "PASS",
  "layer": 11,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 13 层（层索引12）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层12",
  "status": "PASS",
  "layer": 12,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 14 层（层索引13）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层13",
  "status": "PASS",
  "layer": 13,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 15 层（层索引14）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层14",
  "status": "PASS",
  "layer": 14,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 16 层（层索引15）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层15",
  "status": "PASS",
  "layer": 15,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 17 层（层索引16）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层16",
  "status": "PASS",
  "layer": 16,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 18 层（层索引17）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层17",
  "status": "PASS",
  "layer": 17,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 19 层（层索引18）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层18",
  "status": "PASS",
  "layer": 18,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 20 层（层索引19）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层19",
  "status": "PASS",
  "layer": 19,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 21 层（层索引20）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层20",
  "status": "PASS",
  "layer": 20,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 22 层（层索引21）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层21",
  "status": "PASS",
  "layer": 21,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 23 层（层索引22）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层22",
  "status": "PASS",
  "layer": 22,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 24 层（层索引23）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层23",
  "status": "PASS",
  "layer": 23,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 25 层（层索引24）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层24",
  "status": "PASS",
  "layer": 24,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 26 层（层索引25）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层25",
  "status": "PASS",
  "layer": 25,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 27 层（层索引26）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层26",
  "status": "PASS",
  "layer": 26,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

检查第 28 层（层索引27）实际传给 attention kernel 的完整 mask。发现 0 个可见关系错误。允许的边和禁止的边均符合当前 block_diff 规则。 Q形状=[1, 16, 48, 128]，分别为batch、query头数、位置数、每头维度；K形状=[1, 8, 48, 128]，第二维为key头数。mask=[1, 1, 48, 48]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法二/实际 kernel mask/层27",
  "status": "PASS",
  "layer": 27,
  "q_shape": [
    1,
    16,
    48,
    128
  ],
  "k_shape": [
    1,
    8,
    48,
    128
  ],
  "mask_shape": [
    1,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=10.599039077758789；平均绝对差mean_abs=1.559154748916626；整体L2差异l2=2202.206787109375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 10.599039077758789,
  "mean_abs": 1.559154748916626,
  "l2": 2202.206787109375,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 283. eval/seed=2/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=6.064630031585693；平均绝对差mean_abs=0.8832046389579773；整体L2差异l2=1246.9151611328125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 6.064630031585693,
  "mean_abs": 0.8832046389579773,
  "l2": 1246.9151611328125,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=9.362726211547852；平均绝对差mean_abs=0.8762505054473877；整体L2差异l2=1265.832275390625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 9.362726211547852,
  "mean_abs": 0.8762505054473877,
  "l2": 1265.832275390625,
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
| 0–7 | noisy Block0 | 允许 | 4.173662663 | 11.4579134 |
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
      11.457913398742676,
      10.509913444519043,
      8.590126991271973,
      5.229335308074951,
      4.1736626625061035,
      7.007447719573975,
      9.450613021850586,
      7.5932440757751465,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 5.609442234 | 16.97858429 |
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
      16.97858428955078,
      14.296526908874512,
      12.679939270019531,
      14.234723091125488,
      12.988152503967285,
      11.465152740478516,
      8.596963882446289,
      5.609442234039307,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 1.517316222 | 5.50614357 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 5.075341225 | 23.54408646 |
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
      3.4688076972961426,
      2.803511142730713,
      3.841660499572754,
      5.506143569946289,
      2.9051098823547363,
      2.0435876846313477,
      2.4600231647491455,
      1.517316222190857,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      11.077143669128418,
      23.544086456298828,
      11.582091331481934,
      5.07534122467041,
      5.897240161895752,
      11.575531005859375,
      9.628326416015625,
      8.202746391296387,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为23.544086456298828，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 23.544086456298828,
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
| 8–15 | noisy Block1 | 允许 | 2.180807352 | 4.152695179 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 4.826767921 | 24.48163223 |
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
      2.981022596359253,
      2.2631287574768066,
      2.2098381519317627,
      2.18080735206604,
      2.711740016937256,
      2.9241955280303955,
      3.661702871322632,
      4.152695178985596,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      16.331336975097656,
      24.481632232666016,
      17.39482307434082,
      10.085222244262695,
      4.826767921447754,
      14.455169677734375,
      10.249527931213379,
      5.641576766967773,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为24.481632232666016，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 24.481632232666016,
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
| 16–23 | noisy Block2 | 允许 | 2.041744232 | 4.322297096 |
| 24–31 | clean Block0 | 允许 | 2.161569595 | 15.53830147 |
| 32–39 | clean Block1 | 允许 | 2.089456797 | 17.88158417 |
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
      3.270784378051758,
      2.212522268295288,
      3.2042734622955322,
      2.0417442321777344,
      2.2734124660491943,
      2.047138214111328,
      3.7941396236419678,
      4.322297096252441,
      9.351029396057129,
      15.538301467895508,
      8.983991622924805,
      5.421989917755127,
      2.161569595336914,
      5.209918022155762,
      3.656005859375,
      3.868734121322632,
      4.541912078857422,
      2.576005458831787,
      2.089456796646118,
      5.8770647048950195,
      4.36475944519043,
      5.903749942779541,
      17.88158416748047,
      11.795610427856445,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为17.88158416748047，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 17.88158416748047,
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
| 16–23 | noisy Block2 | 允许 | 1.549940348 | 3.233731747 |
| 24–31 | clean Block0 | 允许 | 1.795076251 | 17.52937889 |
| 32–39 | clean Block1 | 允许 | 1.919044137 | 15.54867458 |
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
      2.9370267391204834,
      1.9217970371246338,
      1.5499403476715088,
      1.7755937576293945,
      2.43882417678833,
      3.1263272762298584,
      2.5087571144104004,
      3.233731746673584,
      12.751720428466797,
      17.52937889099121,
      5.939986705780029,
      2.640377998352051,
      1.7950762510299683,
      3.1517751216888428,
      3.8903443813323975,
      3.3006982803344727,
      5.764036655426025,
      2.199646472930908,
      1.9190441370010376,
      5.944885730743408,
      4.284093379974365,
      5.281365871429443,
      15.548674583435059,
      12.504441261291504,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为17.52937889099121，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 17.52937889099121,
  "threshold": 1e-08
}
```

</details>
