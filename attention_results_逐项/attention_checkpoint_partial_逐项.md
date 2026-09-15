# attention_checkpoint_partial.json：逐条中文解读

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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=8.890625；平均绝对差mean_abs=0.9637036323547363；整体L2差异l2=2013.7901611328125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 8.890625,
  "mean_abs": 0.9637036323547363,
  "l2": 2013.7901611328125,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 33. train/seed=0/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=3.3516845703125；平均绝对差mean_abs=0.4536869525909424；整体L2差异l2=893.5609130859375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 3.3516845703125,
  "mean_abs": 0.4536869525909424,
  "l2": 893.5609130859375,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=8.171875；平均绝对差mean_abs=0.8722968101501465；整体L2差异l2=1805.5018310546875；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 8.171875,
  "mean_abs": 0.8722968101501465,
  "l2": 1805.5018310546875,
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
| 0–7 | noisy Block0 | 允许 | 4.208150864 | 22.75146484 |
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
      19.898130416870117,
      20.61129379272461,
      22.75146484375,
      13.172547340393066,
      10.859432220458984,
      19.047103881835938,
      12.7662992477417,
      8.313169479370117,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      11.751091957092285,
      5.6960062980651855,
      5.64678955078125,
      11.725199699401855,
      10.385748863220215,
      8.179868698120117,
      4.208150863647461,
      5.390549182891846,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 6.315193176 | 22.47485542 |
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
      14.745094299316406,
      19.15774154663086,
      22.474855422973633,
      9.049482345581055,
      6.315193176269531,
      9.348450660705566,
      6.899147987365723,
      7.782634258270264,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      21.221834182739258,
      12.067121505737305,
      12.467028617858887,
      19.421281814575195,
      18.064481735229492,
      14.335693359375,
      6.464910984039307,
      11.136369705200195,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 2.636019707 | 35.41717148 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.351588726 | 26.46404839 |
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
      4.047301769256592,
      3.1756153106689453,
      4.391246318817139,
      3.791119337081909,
      20.553377151489258,
      13.866242408752441,
      24.114253997802734,
      35.417171478271484,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.9117376804351807,
      5.792185306549072,
      4.770696640014648,
      5.069343090057373,
      3.69775652885437,
      3.9007513523101807,
      2.351588726043701,
      5.205171585083008,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      11.210586547851562,
      9.503412246704102,
      14.440316200256348,
      11.712885856628418,
      10.215705871582031,
      3.5759880542755127,
      2.636019706726074,
      2.768284320831299,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      5.500870227813721,
      7.3640875816345215,
      9.838460922241211,
      13.460911750793457,
      10.82091236114502,
      10.163228988647461,
      18.27873992919922,
      26.464048385620117,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为26.464048385620117，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 26.464048385620117,
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
| 8–15 | noisy Block1 | 允许 | 2.44506669 | 38.94942856 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.373658895 | 12.96189213 |
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
      4.590728282928467,
      3.605703830718994,
      3.685353994369507,
      4.570034027099609,
      22.39337730407715,
      14.3356294631958,
      27.845455169677734,
      38.94942855834961,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.477385997772217,
      4.437796592712402,
      3.8992929458618164,
      4.428035259246826,
      3.360645055770874,
      3.764061450958252,
      2.3736588954925537,
      6.331985950469971,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      7.209214687347412,
      5.910037040710449,
      17.07869529724121,
      12.074007987976074,
      9.380870819091797,
      3.4413487911224365,
      2.4450666904449463,
      2.55080509185791,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.896336555480957,
      4.060539722442627,
      3.6834006309509277,
      6.201125621795654,
      5.403256893157959,
      4.834652423858643,
      8.250601768493652,
      12.961892127990723,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为12.961892127990723，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 12.961892127990723,
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
| 16–23 | noisy Block2 | 允许 | 1.66709733 | 20.42981148 |
| 24–31 | clean Block0 | 允许 | 1.120437264 | 8.771341324 |
| 32–39 | clean Block1 | 允许 | 1.615450144 | 11.44596863 |
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
      5.504585266113281,
      6.6565446853637695,
      3.569185256958008,
      8.34577751159668,
      10.425729751586914,
      10.270721435546875,
      4.114638328552246,
      1.8727526664733887,
      2.648564338684082,
      2.9965288639068604,
      2.2017598152160645,
      2.4554431438446045,
      1.2592347860336304,
      2.4160828590393066,
      1.1204372644424438,
      1.3714253902435303,
      1.615450143814087,
      1.7813585996627808,
      2.247431516647339,
      3.41044282913208,
      10.644376754760742,
      5.956729412078857,
      2.8452227115631104,
      6.27174711227417,
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
      2.9583444595336914,
      6.686586856842041,
      20.429811477661133,
      12.741443634033203,
      5.789741039276123,
      4.153694152832031,
      2.6263227462768555,
      1.6670973300933838,
      3.155860424041748,
      3.541957139968872,
      3.707585334777832,
      5.318604946136475,
      4.310140609741211,
      3.1167423725128174,
      6.620235919952393,
      8.771341323852539,
      4.9138689041137695,
      3.581526041030884,
      4.340480327606201,
      4.719727993011475,
      2.5233278274536133,
      8.59437370300293,
      11.445968627929688,
      3.318937301635742,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为11.445968627929688，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 11.445968627929688,
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
| 16–23 | noisy Block2 | 允许 | 1.582715631 | 20.40737534 |
| 24–31 | clean Block0 | 允许 | 1.938010097 | 6.739630222 |
| 32–39 | clean Block1 | 允许 | 1.905382514 | 19.33340645 |
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
      4.806354522705078,
      4.957788467407227,
      2.9238123893737793,
      9.858330726623535,
      15.244670867919922,
      19.861217498779297,
      8.524940490722656,
      4.538815021514893,
      3.0422961711883545,
      3.599146842956543,
      3.266359806060791,
      3.4535505771636963,
      3.07938551902771,
      3.5324244499206543,
      1.9380100965499878,
      2.357839345932007,
      2.759166717529297,
      3.1413869857788086,
      3.0660343170166016,
      6.189589023590088,
      19.333406448364258,
      9.400605201721191,
      3.2656819820404053,
      5.485589981079102,
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
      3.612872362136841,
      7.413055896759033,
      20.40737533569336,
      10.00786018371582,
      3.2689247131347656,
      3.2215664386749268,
      2.050668716430664,
      1.582715630531311,
      4.305875778198242,
      3.230903387069702,
      4.306750774383545,
      6.739630222320557,
      3.4714622497558594,
      2.6800918579101562,
      4.777240753173828,
      4.5715436935424805,
      5.065741539001465,
      2.346952199935913,
      4.105742454528809,
      3.121462821960449,
      1.905382513999939,
      3.641059160232544,
      3.635589122772217,
      2.467625141143799,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为19.333406448364258，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 19.333406448364258,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=11.84375；平均绝对差mean_abs=1.1356024742126465；整体L2差异l2=2270.12939453125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 11.84375,
  "mean_abs": 1.1356024742126465,
  "l2": 2270.12939453125,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 83. train/seed=1/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=10.21875；平均绝对差mean_abs=0.8144822120666504；整体L2差异l2=1675.4488525390625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 10.21875,
  "mean_abs": 0.8144822120666504,
  "l2": 1675.4488525390625,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=7.87109375；平均绝对差mean_abs=0.7207779884338379；整体L2差异l2=1482.54931640625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 7.87109375,
  "mean_abs": 0.7207779884338379,
  "l2": 1482.54931640625,
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
| 0–7 | noisy Block0 | 允许 | 5.519631386 | 22.47677994 |
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
      10.757540702819824,
      7.081518173217773,
      6.724259376525879,
      6.7897138595581055,
      10.61844253540039,
      9.582862854003906,
      7.827530860900879,
      11.591174125671387,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      22.47677993774414,
      15.349769592285156,
      12.69030475616455,
      8.577337265014648,
      5.519631385803223,
      8.081427574157715,
      19.849531173706055,
      11.129037857055664,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 5.093438148 | 14.27645779 |
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
      7.447962284088135,
      5.940305233001709,
      7.206332683563232,
      8.871078491210938,
      14.276457786560059,
      12.451082229614258,
      12.757172584533691,
      11.788701057434082,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      11.708489418029785,
      6.847569942474365,
      7.840662002563477,
      8.316595077514648,
      5.88059139251709,
      7.096397876739502,
      8.097024917602539,
      5.093438148498535,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 2.953763962 | 8.419481277 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 3.018456221 | 8.951904297 |
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
      3.618899345397949,
      3.8935296535491943,
      3.8194451332092285,
      5.111266136169434,
      4.356256008148193,
      6.466068744659424,
      7.377451419830322,
      3.2712981700897217,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      5.29113245010376,
      6.393915176391602,
      3.018456220626831,
      6.217255115509033,
      3.1144258975982666,
      3.9913039207458496,
      4.8097825050354,
      7.249080657958984,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      5.193175315856934,
      5.604396343231201,
      4.345341682434082,
      8.41948127746582,
      6.606832504272461,
      3.1018941402435303,
      2.953763961791992,
      7.754953861236572,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      8.498189926147461,
      8.598442077636719,
      8.094064712524414,
      4.990245819091797,
      4.218295097351074,
      5.404066562652588,
      5.574010372161865,
      8.951904296875,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为8.951904296875，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 8.951904296875,
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
| 8–15 | noisy Block1 | 允许 | 3.097472906 | 12.30475044 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 3.204726458 | 8.562082291 |
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
      3.097472906112671,
      4.999565601348877,
      3.7760508060455322,
      3.7854392528533936,
      5.821574687957764,
      5.626847267150879,
      5.971537113189697,
      3.61568021774292,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      5.341494083404541,
      5.627524375915527,
      4.1440887451171875,
      7.717036724090576,
      4.212100505828857,
      3.722778797149658,
      4.63012170791626,
      5.146763801574707,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      4.113278388977051,
      4.229953765869141,
      5.397304534912109,
      9.513001441955566,
      7.419875144958496,
      3.587629795074463,
      5.315522193908691,
      12.304750442504883,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.391645431518555,
      4.550752639770508,
      5.672774791717529,
      3.8291468620300293,
      3.4407742023468018,
      3.204726457595825,
      6.930050373077393,
      8.562082290649414,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为8.562082290649414，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 8.562082290649414,
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
| 16–23 | noisy Block2 | 允许 | 3.048775196 | 25.06311417 |
| 24–31 | clean Block0 | 允许 | 2.14785099 | 10.9323225 |
| 32–39 | clean Block1 | 允许 | 1.818301678 | 17.25789642 |
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
      3.0487751960754395,
      8.335165023803711,
      25.063114166259766,
      17.994218826293945,
      10.049216270446777,
      5.60460901260376,
      3.7040953636169434,
      3.524641990661621,
      10.93232250213623,
      10.688155174255371,
      5.086979389190674,
      5.92324686050415,
      8.305424690246582,
      4.6590142250061035,
      3.9167215824127197,
      2.14785099029541,
      2.720717191696167,
      5.599676132202148,
      4.314099311828613,
      3.6241202354431152,
      4.085791110992432,
      4.643555164337158,
      4.269227981567383,
      5.577267646789551,
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
      5.337350368499756,
      6.671751976013184,
      5.182153224945068,
      7.4951372146606445,
      4.019979953765869,
      11.607433319091797,
      12.204024314880371,
      4.581567287445068,
      5.774012088775635,
      5.613407135009766,
      5.528865337371826,
      5.233005523681641,
      2.6762661933898926,
      3.6884665489196777,
      5.159669399261475,
      4.4989776611328125,
      3.50921893119812,
      3.0119223594665527,
      2.3514626026153564,
      2.5825040340423584,
      1.8183016777038574,
      3.2111637592315674,
      4.121000289916992,
      17.257896423339844,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为17.257896423339844，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 17.257896423339844,
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
| 16–23 | noisy Block2 | 允许 | 2.61920023 | 15.15886593 |
| 24–31 | clean Block0 | 允许 | 1.729597807 | 5.220585346 |
| 32–39 | clean Block1 | 允许 | 1.784729362 | 9.34569931 |
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
      3.9900894165039062,
      4.655121326446533,
      15.158865928649902,
      12.799694061279297,
      6.454095840454102,
      4.825211524963379,
      3.4642179012298584,
      3.0746264457702637,
      3.4663166999816895,
      3.9836583137512207,
      2.834419012069702,
      4.3574090003967285,
      2.7297658920288086,
      2.4043755531311035,
      2.8882572650909424,
      1.729597806930542,
      1.7847293615341187,
      2.9294071197509766,
      2.392000436782837,
      1.7971272468566895,
      3.0676522254943848,
      4.174689292907715,
      4.208934307098389,
      2.5613248348236084,
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
      4.096056938171387,
      4.948639392852783,
      4.656919002532959,
      13.491984367370605,
      4.95646333694458,
      6.7458062171936035,
      4.407498836517334,
      2.6192002296447754,
      4.52414083480835,
      5.220585346221924,
      4.790676116943359,
      3.1323230266571045,
      2.1081383228302,
      3.1236047744750977,
      4.059847831726074,
      4.781973361968994,
      3.1117000579833984,
      2.0629074573516846,
      1.8633906841278076,
      2.0548324584960938,
      1.80654776096344,
      2.921363592147827,
      2.981276512145996,
      9.345699310302734,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为9.345699310302734，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=1/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 9.345699310302734,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=8.859375；平均绝对差mean_abs=0.8530968427658081；整体L2差异l2=1736.306396484375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 8.859375,
  "mean_abs": 0.8530968427658081,
  "l2": 1736.306396484375,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 133. train/seed=2/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=5.6171875；平均绝对差mean_abs=0.6426284313201904；整体L2差异l2=1275.8333740234375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 5.6171875,
  "mean_abs": 0.6426284313201904,
  "l2": 1275.8333740234375,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=10.046875；平均绝对差mean_abs=0.8289109468460083；整体L2差异l2=1735.479248046875；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 10.046875,
  "mean_abs": 0.8289109468460083,
  "l2": 1735.479248046875,
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
| 0–7 | noisy Block0 | 允许 | 6.681091785 | 44.57014847 |
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
      14.470599174499512,
      24.947683334350586,
      44.57014846801758,
      6.681091785430908,
      8.00078010559082,
      13.597003936767578,
      8.205004692077637,
      7.083837985992432,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      11.99417781829834,
      8.934364318847656,
      20.184263229370117,
      21.485254287719727,
      14.854667663574219,
      10.014093399047852,
      8.374771118164062,
      7.723591327667236,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 5.285215378 | 20.21134949 |
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
      10.417566299438477,
      17.230314254760742,
      15.25258731842041,
      5.285215377807617,
      5.648197650909424,
      11.130816459655762,
      7.081457614898682,
      6.23687219619751,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      10.75937271118164,
      7.999279975891113,
      16.184839248657227,
      20.211349487304688,
      13.31679630279541,
      10.61534595489502,
      8.366826057434082,
      7.9587249755859375,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 3.915030956 | 31.18437004 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.321258068 | 14.1962986 |
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
      6.075579643249512,
      3.9150309562683105,
      4.408907413482666,
      18.679296493530273,
      12.843578338623047,
      11.02767562866211,
      22.370298385620117,
      11.085244178771973,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      8.474217414855957,
      14.196298599243164,
      8.03429889678955,
      2.6740620136260986,
      2.952249050140381,
      4.546716690063477,
      3.4639954566955566,
      3.386043071746826,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      6.214752674102783,
      5.6386399269104,
      15.295442581176758,
      21.387174606323242,
      31.184370040893555,
      14.627034187316895,
      6.365627765655518,
      7.693408489227295,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.688612461090088,
      4.111546039581299,
      4.582353591918945,
      4.246678352355957,
      5.5259552001953125,
      2.321258068084717,
      3.0100128650665283,
      3.6101183891296387,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为14.196298599243164，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 14.196298599243164,
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
| 8–15 | noisy Block1 | 允许 | 3.560467005 | 68.40530396 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.472995281 | 12.51015949 |
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
      4.613965034484863,
      3.560467004776001,
      4.5349812507629395,
      15.865094184875488,
      10.9993314743042,
      13.73033618927002,
      23.37616729736328,
      17.067956924438477,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.106965065002441,
      12.510159492492676,
      6.479706287384033,
      3.747894287109375,
      2.4729952812194824,
      4.745778560638428,
      4.587330341339111,
      4.157699108123779,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      9.812017440795898,
      9.844447135925293,
      30.571670532226562,
      26.82816505432129,
      68.40530395507812,
      37.507850646972656,
      14.240970611572266,
      12.620718002319336,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      6.0435357093811035,
      4.96305513381958,
      5.593890190124512,
      6.129895210266113,
      7.180191993713379,
      3.0733304023742676,
      5.9752020835876465,
      4.723516464233398,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为12.510159492492676，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 12.510159492492676,
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
| 16–23 | noisy Block2 | 允许 | 1.644481063 | 7.555704594 |
| 24–31 | clean Block0 | 允许 | 2.157078505 | 15.65568924 |
| 32–39 | clean Block1 | 允许 | 1.365937471 | 22.52929306 |
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
      3.4879043102264404,
      3.6149916648864746,
      7.295422077178955,
      3.050565719604492,
      2.53344464302063,
      2.599881649017334,
      6.8770318031311035,
      3.9815404415130615,
      11.871499061584473,
      15.655689239501953,
      6.93649435043335,
      3.3730556964874268,
      3.909522533416748,
      9.850921630859375,
      4.595032215118408,
      3.0671441555023193,
      4.471952438354492,
      3.193218946456909,
      2.0955698490142822,
      6.457143306732178,
      5.2166523933410645,
      6.790884971618652,
      22.099945068359375,
      22.529293060302734,
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
      1.65411376953125,
      1.7124720811843872,
      1.6444810628890991,
      3.0045793056488037,
      5.318537712097168,
      7.555704593658447,
      4.1138916015625,
      2.514601707458496,
      7.839270114898682,
      10.900187492370605,
      14.461358070373535,
      6.3093061447143555,
      2.601161003112793,
      2.3753068447113037,
      2.5113399028778076,
      2.157078504562378,
      3.619844436645508,
      2.521299123764038,
      3.572225332260132,
      7.311851978302002,
      14.803731918334961,
      1.8986778259277344,
      6.205259799957275,
      1.3659374713897705,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为22.529293060302734，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 22.529293060302734,
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
| 16–23 | noisy Block2 | 允许 | 1.457379341 | 6.488688469 |
| 24–31 | clean Block0 | 允许 | 2.342390299 | 10.44825268 |
| 32–39 | clean Block1 | 允许 | 1.727907538 | 16.23640442 |
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
      3.532432794570923,
      2.685237169265747,
      4.890904426574707,
      2.6327130794525146,
      1.9732928276062012,
      2.0762572288513184,
      6.4886884689331055,
      4.258347511291504,
      6.738908290863037,
      10.44825267791748,
      5.465347766876221,
      3.701115846633911,
      3.080212116241455,
      7.910195827484131,
      3.00227952003479,
      2.4758729934692383,
      4.199572563171387,
      2.1537163257598877,
      2.6215603351593018,
      5.587625503540039,
      3.702960968017578,
      4.326131820678711,
      13.270121574401855,
      16.236404418945312,
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
      1.5319383144378662,
      1.4573793411254883,
      1.7888288497924805,
      2.3500871658325195,
      3.0439555644989014,
      2.860668182373047,
      6.476058483123779,
      2.244192361831665,
      5.095677852630615,
      5.1674485206604,
      4.03035306930542,
      7.497092247009277,
      3.257246971130371,
      2.602177619934082,
      2.7315127849578857,
      2.342390298843384,
      3.3444137573242188,
      2.7625536918640137,
      3.7844326496124268,
      8.097813606262207,
      13.759099006652832,
      2.1093688011169434,
      6.7806501388549805,
      1.7279075384140015,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为16.236404418945312，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=2/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 16.236404418945312,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=8.890625；平均绝对差mean_abs=0.9637036323547363；整体L2差异l2=2013.7901611328125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 8.890625,
  "mean_abs": 0.9637036323547363,
  "l2": 2013.7901611328125,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 183. eval/seed=0/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=3.3516845703125；平均绝对差mean_abs=0.4536869525909424；整体L2差异l2=893.5609130859375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 3.3516845703125,
  "mean_abs": 0.4536869525909424,
  "l2": 893.5609130859375,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=8.171875；平均绝对差mean_abs=0.8722968101501465；整体L2差异l2=1805.5018310546875；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 8.171875,
  "mean_abs": 0.8722968101501465,
  "l2": 1805.5018310546875,
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
| 0–7 | noisy Block0 | 允许 | 4.208150864 | 22.75146484 |
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
      19.898130416870117,
      20.61129379272461,
      22.75146484375,
      13.172547340393066,
      10.859432220458984,
      19.047103881835938,
      12.7662992477417,
      8.313169479370117,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      11.751091957092285,
      5.6960062980651855,
      5.64678955078125,
      11.725199699401855,
      10.385748863220215,
      8.179868698120117,
      4.208150863647461,
      5.390549182891846,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 6.315193176 | 22.47485542 |
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
      14.745094299316406,
      19.15774154663086,
      22.474855422973633,
      9.049482345581055,
      6.315193176269531,
      9.348450660705566,
      6.899147987365723,
      7.782634258270264,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      21.221834182739258,
      12.067121505737305,
      12.467028617858887,
      19.421281814575195,
      18.064481735229492,
      14.335693359375,
      6.464910984039307,
      11.136369705200195,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 2.636019707 | 35.41717148 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.351588726 | 26.46404839 |
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
      4.047301769256592,
      3.1756153106689453,
      4.391246318817139,
      3.791119337081909,
      20.553377151489258,
      13.866242408752441,
      24.114253997802734,
      35.417171478271484,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.9117376804351807,
      5.792185306549072,
      4.770696640014648,
      5.069343090057373,
      3.69775652885437,
      3.9007513523101807,
      2.351588726043701,
      5.205171585083008,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      11.210586547851562,
      9.503412246704102,
      14.440316200256348,
      11.712885856628418,
      10.215705871582031,
      3.5759880542755127,
      2.636019706726074,
      2.768284320831299,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      5.500870227813721,
      7.3640875816345215,
      9.838460922241211,
      13.460911750793457,
      10.82091236114502,
      10.163228988647461,
      18.27873992919922,
      26.464048385620117,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为26.464048385620117，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 26.464048385620117,
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
| 8–15 | noisy Block1 | 允许 | 2.44506669 | 38.94942856 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.373658895 | 12.96189213 |
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
      4.590728282928467,
      3.605703830718994,
      3.685353994369507,
      4.570034027099609,
      22.39337730407715,
      14.3356294631958,
      27.845455169677734,
      38.94942855834961,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.477385997772217,
      4.437796592712402,
      3.8992929458618164,
      4.428035259246826,
      3.360645055770874,
      3.764061450958252,
      2.3736588954925537,
      6.331985950469971,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      7.209214687347412,
      5.910037040710449,
      17.07869529724121,
      12.074007987976074,
      9.380870819091797,
      3.4413487911224365,
      2.4450666904449463,
      2.55080509185791,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.896336555480957,
      4.060539722442627,
      3.6834006309509277,
      6.201125621795654,
      5.403256893157959,
      4.834652423858643,
      8.250601768493652,
      12.961892127990723,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为12.961892127990723，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 12.961892127990723,
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
| 16–23 | noisy Block2 | 允许 | 1.66709733 | 20.42981148 |
| 24–31 | clean Block0 | 允许 | 1.120437264 | 8.771341324 |
| 32–39 | clean Block1 | 允许 | 1.615450144 | 11.44596863 |
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
      5.504585266113281,
      6.6565446853637695,
      3.569185256958008,
      8.34577751159668,
      10.425729751586914,
      10.270721435546875,
      4.114638328552246,
      1.8727526664733887,
      2.648564338684082,
      2.9965288639068604,
      2.2017598152160645,
      2.4554431438446045,
      1.2592347860336304,
      2.4160828590393066,
      1.1204372644424438,
      1.3714253902435303,
      1.615450143814087,
      1.7813585996627808,
      2.247431516647339,
      3.41044282913208,
      10.644376754760742,
      5.956729412078857,
      2.8452227115631104,
      6.27174711227417,
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
      2.9583444595336914,
      6.686586856842041,
      20.429811477661133,
      12.741443634033203,
      5.789741039276123,
      4.153694152832031,
      2.6263227462768555,
      1.6670973300933838,
      3.155860424041748,
      3.541957139968872,
      3.707585334777832,
      5.318604946136475,
      4.310140609741211,
      3.1167423725128174,
      6.620235919952393,
      8.771341323852539,
      4.9138689041137695,
      3.581526041030884,
      4.340480327606201,
      4.719727993011475,
      2.5233278274536133,
      8.59437370300293,
      11.445968627929688,
      3.318937301635742,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为11.445968627929688，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 11.445968627929688,
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
| 16–23 | noisy Block2 | 允许 | 1.582715631 | 20.40737534 |
| 24–31 | clean Block0 | 允许 | 1.938010097 | 6.739630222 |
| 32–39 | clean Block1 | 允许 | 1.905382514 | 19.33340645 |
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
      4.806354522705078,
      4.957788467407227,
      2.9238123893737793,
      9.858330726623535,
      15.244670867919922,
      19.861217498779297,
      8.524940490722656,
      4.538815021514893,
      3.0422961711883545,
      3.599146842956543,
      3.266359806060791,
      3.4535505771636963,
      3.07938551902771,
      3.5324244499206543,
      1.9380100965499878,
      2.357839345932007,
      2.759166717529297,
      3.1413869857788086,
      3.0660343170166016,
      6.189589023590088,
      19.333406448364258,
      9.400605201721191,
      3.2656819820404053,
      5.485589981079102,
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
      3.612872362136841,
      7.413055896759033,
      20.40737533569336,
      10.00786018371582,
      3.2689247131347656,
      3.2215664386749268,
      2.050668716430664,
      1.582715630531311,
      4.305875778198242,
      3.230903387069702,
      4.306750774383545,
      6.739630222320557,
      3.4714622497558594,
      2.6800918579101562,
      4.777240753173828,
      4.5715436935424805,
      5.065741539001465,
      2.346952199935913,
      4.105742454528809,
      3.121462821960449,
      1.905382513999939,
      3.641059160232544,
      3.635589122772217,
      2.467625141143799,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为19.333406448364258，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 19.333406448364258,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=11.84375；平均绝对差mean_abs=1.1356024742126465；整体L2差异l2=2270.12939453125；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 11.84375,
  "mean_abs": 1.1356024742126465,
  "l2": 2270.12939453125,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 233. eval/seed=1/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=10.21875；平均绝对差mean_abs=0.8144822120666504；整体L2差异l2=1675.4488525390625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 10.21875,
  "mean_abs": 0.8144822120666504,
  "l2": 1675.4488525390625,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=7.87109375；平均绝对差mean_abs=0.7207779884338379；整体L2差异l2=1482.54931640625；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 7.87109375,
  "mean_abs": 0.7207779884338379,
  "l2": 1482.54931640625,
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
| 0–7 | noisy Block0 | 允许 | 5.519631386 | 22.47677994 |
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
      10.757540702819824,
      7.081518173217773,
      6.724259376525879,
      6.7897138595581055,
      10.61844253540039,
      9.582862854003906,
      7.827530860900879,
      11.591174125671387,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      22.47677993774414,
      15.349769592285156,
      12.69030475616455,
      8.577337265014648,
      5.519631385803223,
      8.081427574157715,
      19.849531173706055,
      11.129037857055664,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 5.093438148 | 14.27645779 |
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
      7.447962284088135,
      5.940305233001709,
      7.206332683563232,
      8.871078491210938,
      14.276457786560059,
      12.451082229614258,
      12.757172584533691,
      11.788701057434082,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      11.708489418029785,
      6.847569942474365,
      7.840662002563477,
      8.316595077514648,
      5.88059139251709,
      7.096397876739502,
      8.097024917602539,
      5.093438148498535,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 2.953763962 | 8.419481277 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 3.018456221 | 8.951904297 |
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
      3.618899345397949,
      3.8935296535491943,
      3.8194451332092285,
      5.111266136169434,
      4.356256008148193,
      6.466068744659424,
      7.377451419830322,
      3.2712981700897217,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      5.29113245010376,
      6.393915176391602,
      3.018456220626831,
      6.217255115509033,
      3.1144258975982666,
      3.9913039207458496,
      4.8097825050354,
      7.249080657958984,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      5.193175315856934,
      5.604396343231201,
      4.345341682434082,
      8.41948127746582,
      6.606832504272461,
      3.1018941402435303,
      2.953763961791992,
      7.754953861236572,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      8.498189926147461,
      8.598442077636719,
      8.094064712524414,
      4.990245819091797,
      4.218295097351074,
      5.404066562652588,
      5.574010372161865,
      8.951904296875,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为8.951904296875，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 8.951904296875,
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
| 8–15 | noisy Block1 | 允许 | 3.097472906 | 12.30475044 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 3.204726458 | 8.562082291 |
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
      3.097472906112671,
      4.999565601348877,
      3.7760508060455322,
      3.7854392528533936,
      5.821574687957764,
      5.626847267150879,
      5.971537113189697,
      3.61568021774292,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      5.341494083404541,
      5.627524375915527,
      4.1440887451171875,
      7.717036724090576,
      4.212100505828857,
      3.722778797149658,
      4.63012170791626,
      5.146763801574707,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      4.113278388977051,
      4.229953765869141,
      5.397304534912109,
      9.513001441955566,
      7.419875144958496,
      3.587629795074463,
      5.315522193908691,
      12.304750442504883,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      4.391645431518555,
      4.550752639770508,
      5.672774791717529,
      3.8291468620300293,
      3.4407742023468018,
      3.204726457595825,
      6.930050373077393,
      8.562082290649414,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为8.562082290649414，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 8.562082290649414,
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
| 16–23 | noisy Block2 | 允许 | 3.048775196 | 25.06311417 |
| 24–31 | clean Block0 | 允许 | 2.14785099 | 10.9323225 |
| 32–39 | clean Block1 | 允许 | 1.818301678 | 17.25789642 |
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
      3.0487751960754395,
      8.335165023803711,
      25.063114166259766,
      17.994218826293945,
      10.049216270446777,
      5.60460901260376,
      3.7040953636169434,
      3.524641990661621,
      10.93232250213623,
      10.688155174255371,
      5.086979389190674,
      5.92324686050415,
      8.305424690246582,
      4.6590142250061035,
      3.9167215824127197,
      2.14785099029541,
      2.720717191696167,
      5.599676132202148,
      4.314099311828613,
      3.6241202354431152,
      4.085791110992432,
      4.643555164337158,
      4.269227981567383,
      5.577267646789551,
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
      5.337350368499756,
      6.671751976013184,
      5.182153224945068,
      7.4951372146606445,
      4.019979953765869,
      11.607433319091797,
      12.204024314880371,
      4.581567287445068,
      5.774012088775635,
      5.613407135009766,
      5.528865337371826,
      5.233005523681641,
      2.6762661933898926,
      3.6884665489196777,
      5.159669399261475,
      4.4989776611328125,
      3.50921893119812,
      3.0119223594665527,
      2.3514626026153564,
      2.5825040340423584,
      1.8183016777038574,
      3.2111637592315674,
      4.121000289916992,
      17.257896423339844,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为17.257896423339844，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 17.257896423339844,
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
| 16–23 | noisy Block2 | 允许 | 2.61920023 | 15.15886593 |
| 24–31 | clean Block0 | 允许 | 1.729597807 | 5.220585346 |
| 32–39 | clean Block1 | 允许 | 1.784729362 | 9.34569931 |
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
      3.9900894165039062,
      4.655121326446533,
      15.158865928649902,
      12.799694061279297,
      6.454095840454102,
      4.825211524963379,
      3.4642179012298584,
      3.0746264457702637,
      3.4663166999816895,
      3.9836583137512207,
      2.834419012069702,
      4.3574090003967285,
      2.7297658920288086,
      2.4043755531311035,
      2.8882572650909424,
      1.729597806930542,
      1.7847293615341187,
      2.9294071197509766,
      2.392000436782837,
      1.7971272468566895,
      3.0676522254943848,
      4.174689292907715,
      4.208934307098389,
      2.5613248348236084,
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
      4.096056938171387,
      4.948639392852783,
      4.656919002532959,
      13.491984367370605,
      4.95646333694458,
      6.7458062171936035,
      4.407498836517334,
      2.6192002296447754,
      4.52414083480835,
      5.220585346221924,
      4.790676116943359,
      3.1323230266571045,
      2.1081383228302,
      3.1236047744750977,
      4.059847831726074,
      4.781973361968994,
      3.1117000579833984,
      2.0629074573516846,
      1.8633906841278076,
      2.0548324584960938,
      1.80654776096344,
      2.921363592147827,
      2.981276512145996,
      9.345699310302734,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为9.345699310302734，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=1/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 9.345699310302734,
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

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block1（位置8–15）全部输出。 最大绝对差max_abs=8.859375；平均绝对差mean_abs=0.8530968427658081；整体L2差异l2=1736.306396484375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 8.859375,
  "mean_abs": 0.8530968427658081,
  "l2": 1736.306396484375,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 283. eval/seed=2/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置24–31），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=5.6171875；平均绝对差mean_abs=0.6426284313201904；整体L2差异l2=1275.8333740234375；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 5.6171875,
  "mean_abs": 0.6426284313201904,
  "l2": 1275.8333740234375,
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

只替换clean Block1（拼接位置32–39），固定全部noisy，观察noisy Block2（位置16–23）全部输出。 最大绝对差max_abs=10.046875；平均绝对差mean_abs=0.8289109468460083；整体L2差异l2=1735.479248046875；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 10.046875,
  "mean_abs": 0.8289109468460083,
  "l2": 1735.479248046875,
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
| 0–7 | noisy Block0 | 允许 | 6.681091785 | 44.57014847 |
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
      14.470599174499512,
      24.947683334350586,
      44.57014846801758,
      6.681091785430908,
      8.00078010559082,
      13.597003936767578,
      8.205004692077637,
      7.083837985992432,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      11.99417781829834,
      8.934364318847656,
      20.184263229370117,
      21.485254287719727,
      14.854667663574219,
      10.014093399047852,
      8.374771118164062,
      7.723591327667236,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 0–7 | noisy Block0 | 允许 | 5.285215378 | 20.21134949 |
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
      10.417566299438477,
      17.230314254760742,
      15.25258731842041,
      5.285215377807617,
      5.648197650909424,
      11.130816459655762,
      7.081457614898682,
      6.23687219619751,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      10.75937271118164,
      7.999279975891113,
      16.184839248657227,
      20.211349487304688,
      13.31679630279541,
      10.61534595489502,
      8.366826057434082,
      7.9587249755859375,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
| 8–15 | noisy Block1 | 允许 | 3.915030956 | 31.18437004 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.321258068 | 14.1962986 |
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
      6.075579643249512,
      3.9150309562683105,
      4.408907413482666,
      18.679296493530273,
      12.843578338623047,
      11.02767562866211,
      22.370298385620117,
      11.085244178771973,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      8.474217414855957,
      14.196298599243164,
      8.03429889678955,
      2.6740620136260986,
      2.952249050140381,
      4.546716690063477,
      3.4639954566955566,
      3.386043071746826,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      6.214752674102783,
      5.6386399269104,
      15.295442581176758,
      21.387174606323242,
      31.184370040893555,
      14.627034187316895,
      6.365627765655518,
      7.693408489227295,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      3.688612461090088,
      4.111546039581299,
      4.582353591918945,
      4.246678352355957,
      5.5259552001953125,
      2.321258068084717,
      3.0100128650665283,
      3.6101183891296387,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为14.196298599243164，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 14.196298599243164,
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
| 8–15 | noisy Block1 | 允许 | 3.560467005 | 68.40530396 |
| 16–23 | noisy Block2 | 禁止 | 0 | 0 |
| 24–31 | clean Block0 | 允许 | 2.472995281 | 12.51015949 |
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
      4.613965034484863,
      3.560467004776001,
      4.5349812507629395,
      15.865094184875488,
      10.9993314743042,
      13.73033618927002,
      23.37616729736328,
      17.067956924438477,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      7.106965065002441,
      12.510159492492676,
      6.479706287384033,
      3.747894287109375,
      2.4729952812194824,
      4.745778560638428,
      4.587330341339111,
      4.157699108123779,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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
      9.812017440795898,
      9.844447135925293,
      30.571670532226562,
      26.82816505432129,
      68.40530395507812,
      37.507850646972656,
      14.240970611572266,
      12.620718002319336,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      6.0435357093811035,
      4.96305513381958,
      5.593890190124512,
      6.129895210266113,
      7.180191993713379,
      3.0733304023742676,
      5.9752020835876465,
      4.723516464233398,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为12.510159492492676，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 12.510159492492676,
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
| 16–23 | noisy Block2 | 允许 | 1.644481063 | 7.555704594 |
| 24–31 | clean Block0 | 允许 | 2.157078505 | 15.65568924 |
| 32–39 | clean Block1 | 允许 | 1.365937471 | 22.52929306 |
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
      3.4879043102264404,
      3.6149916648864746,
      7.295422077178955,
      3.050565719604492,
      2.53344464302063,
      2.599881649017334,
      6.8770318031311035,
      3.9815404415130615,
      11.871499061584473,
      15.655689239501953,
      6.93649435043335,
      3.3730556964874268,
      3.909522533416748,
      9.850921630859375,
      4.595032215118408,
      3.0671441555023193,
      4.471952438354492,
      3.193218946456909,
      2.0955698490142822,
      6.457143306732178,
      5.2166523933410645,
      6.790884971618652,
      22.099945068359375,
      22.529293060302734,
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
      1.65411376953125,
      1.7124720811843872,
      1.6444810628890991,
      3.0045793056488037,
      5.318537712097168,
      7.555704593658447,
      4.1138916015625,
      2.514601707458496,
      7.839270114898682,
      10.900187492370605,
      14.461358070373535,
      6.3093061447143555,
      2.601161003112793,
      2.3753068447113037,
      2.5113399028778076,
      2.157078504562378,
      3.619844436645508,
      2.521299123764038,
      3.572225332260132,
      7.311851978302002,
      14.803731918334961,
      1.8986778259277344,
      6.205259799957275,
      1.3659374713897705,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为22.529293060302734，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 22.529293060302734,
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
| 16–23 | noisy Block2 | 允许 | 1.457379341 | 6.488688469 |
| 24–31 | clean Block0 | 允许 | 2.342390299 | 10.44825268 |
| 32–39 | clean Block1 | 允许 | 1.727907538 | 16.23640442 |
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
      3.532432794570923,
      2.685237169265747,
      4.890904426574707,
      2.6327130794525146,
      1.9732928276062012,
      2.0762572288513184,
      6.4886884689331055,
      4.258347511291504,
      6.738908290863037,
      10.44825267791748,
      5.465347766876221,
      3.701115846633911,
      3.080212116241455,
      7.910195827484131,
      3.00227952003479,
      2.4758729934692383,
      4.199572563171387,
      2.1537163257598877,
      2.6215603351593018,
      5.587625503540039,
      3.702960968017578,
      4.326131820678711,
      13.270121574401855,
      16.236404418945312,
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
      1.5319383144378662,
      1.4573793411254883,
      1.7888288497924805,
      2.3500871658325195,
      3.0439555644989014,
      2.860668182373047,
      6.476058483123779,
      2.244192361831665,
      5.095677852630615,
      5.1674485206604,
      4.03035306930542,
      7.497092247009277,
      3.257246971130371,
      2.602177619934082,
      2.7315127849578857,
      2.342390298843384,
      3.3444137573242188,
      2.7625536918640137,
      3.7844326496124268,
      8.097813606262207,
      13.759099006652832,
      2.1093688011169434,
      6.7806501388549805,
      1.7279075384140015,
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

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为16.236404418945312，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=2/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 16.236404418945312,
  "threshold": 1e-08
}
```

</details>
