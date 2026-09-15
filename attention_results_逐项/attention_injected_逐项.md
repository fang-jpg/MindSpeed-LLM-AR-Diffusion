# attention_injected.json：逐条中文解读

共 48 条。原始总状态：FAIL。

[全部字段定义与总体结论](../attention_results_解读.md)

每条下方折叠区完整保留原始字段，数字未改写。这里的序号从1开始，对应原始 checks 数组下标加1。

## 1. train/seed=0/方法二/实际 kernel mask/层0

原始状态：**FAIL**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 16 个可见关系错误。本文件故意放开 noisy Block1 对当前 clean Block1 的连接；4 个query×4个key=16条错误边，符合注入预期。 Q形状=[1, 4, 24, 16]，分别为batch、query头数、位置数、每头维度；K形状=[1, 2, 24, 16]，第二维为key头数。mask=[1, 1, 24, 24]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 首个错误坐标[0, 0, 4, 16]依次表示batch、mask头维、query位置、key位置；最多展示前10个错误。 第一层示例query=4，可见key=[4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19]；位置映射为noisy 0–11，clean 12–23。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层0",
  "status": "FAIL",
  "layer": 0,
  "q_shape": [
    1,
    4,
    24,
    16
  ],
  "k_shape": [
    1,
    2,
    24,
    16
  ],
  "mask_shape": [
    1,
    1,
    24,
    24
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 16,
  "first_wrong_qk": [
    [
      0,
      0,
      4,
      16
    ],
    [
      0,
      0,
      4,
      17
    ],
    [
      0,
      0,
      4,
      18
    ],
    [
      0,
      0,
      4,
      19
    ],
    [
      0,
      0,
      5,
      16
    ],
    [
      0,
      0,
      5,
      17
    ],
    [
      0,
      0,
      5,
      18
    ],
    [
      0,
      0,
      5,
      19
    ],
    [
      0,
      0,
      6,
      16
    ],
    [
      0,
      0,
      6,
      17
    ]
  ],
  "noisy_block1_query": 4,
  "visible_key_positions": [
    4,
    5,
    6,
    7,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}
```

</details>

## 2. train/seed=0/方法二/实际 kernel mask/层1

原始状态：**FAIL**。

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 16 个可见关系错误。本文件故意放开 noisy Block1 对当前 clean Block1 的连接；4 个query×4个key=16条错误边，符合注入预期。 Q形状=[1, 4, 24, 16]，分别为batch、query头数、位置数、每头维度；K形状=[1, 2, 24, 16]，第二维为key头数。mask=[1, 1, 24, 24]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 首个错误坐标[0, 0, 4, 16]依次表示batch、mask头维、query位置、key位置；最多展示前10个错误。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/实际 kernel mask/层1",
  "status": "FAIL",
  "layer": 1,
  "q_shape": [
    1,
    4,
    24,
    16
  ],
  "k_shape": [
    1,
    2,
    24,
    16
  ],
  "mask_shape": [
    1,
    1,
    24,
    24
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 16,
  "first_wrong_qk": [
    [
      0,
      0,
      4,
      16
    ],
    [
      0,
      0,
      4,
      17
    ],
    [
      0,
      0,
      4,
      18
    ],
    [
      0,
      0,
      4,
      19
    ],
    [
      0,
      0,
      5,
      16
    ],
    [
      0,
      0,
      5,
      17
    ],
    [
      0,
      0,
      5,
      18
    ],
    [
      0,
      0,
      5,
      19
    ],
    [
      0,
      0,
      6,
      16
    ],
    [
      0,
      0,
      6,
      17
    ]
  ]
}
```

</details>

## 3. train/seed=0/方法二/所有层 kernel 均被观察

原始状态：**PASS**。

观察到的层编号为[0, 1]，模型预期共2层。覆盖检查通过表示没有漏掉层；每层mask是否正确要看前面的独立条目。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法二/所有层 kernel 均被观察",
  "status": "PASS",
  "layers": [
    0,
    1
  ],
  "expected_layers": 2
}
```

</details>

## 4. train/seed=0/方法一/重复性

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

## 5. train/seed=0/方法一/clean块0→noisy块0/当前

原始状态：**PASS**。

只替换clean Block0（拼接位置12–15），固定全部noisy，观察noisy Block0（位置0–3）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

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

## 6. train/seed=0/方法一/clean块0→noisy块1/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置12–15），固定全部noisy，观察noisy Block1（位置4–7）全部输出。 最大绝对差max_abs=0.33791130781173706；平均绝对差mean_abs=0.09450638294219971；整体L2差异l2=2.7129549980163574；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 0.33791130781173706,
  "mean_abs": 0.09450638294219971,
  "l2": 2.7129549980163574,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 7. train/seed=0/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置12–15），固定全部noisy，观察noisy Block2（位置8–11）全部输出。 最大绝对差max_abs=0.3101228177547455；平均绝对差mean_abs=0.09160081297159195；整体L2差异l2=2.578565835952759；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 0.3101228177547455,
  "mean_abs": 0.09160081297159195,
  "l2": 2.578565835952759,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 8. train/seed=0/方法一/clean块1→noisy块0/未来

原始状态：**PASS**。

只替换clean Block1（拼接位置16–19），固定全部noisy，观察noisy Block0（位置0–3）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

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

## 9. train/seed=0/方法一/clean块1→noisy块1/当前

原始状态：**FAIL**。

只替换clean Block1（拼接位置16–19），固定全部noisy，观察noisy Block1（位置4–7）全部输出。 最大绝对差max_abs=0.23460564017295837；平均绝对差mean_abs=0.06482543051242828；整体L2差异l2=1.8429665565490723；阈值=1e-06。这是应禁止的依赖，但差异大于阈值；本注入实验因此成功检测到人为泄漏。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块1→noisy块1/当前",
  "status": "FAIL",
  "max_abs": 0.23460564017295837,
  "mean_abs": 0.06482543051242828,
  "l2": 1.8429665565490723,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 10. train/seed=0/方法一/clean块1→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block1（拼接位置16–19），固定全部noisy，观察noisy Block2（位置8–11）全部输出。 最大绝对差max_abs=0.2091517597436905；平均绝对差mean_abs=0.05825144052505493；整体L2差异l2=1.6277228593826294；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 0.2091517597436905,
  "mean_abs": 0.05825144052505493,
  "l2": 1.6277228593826294,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 11. train/seed=0/方法一/clean块2→noisy块0/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置20–23），固定全部noisy，观察noisy Block0（位置0–3）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

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

## 12. train/seed=0/方法一/clean块2→noisy块1/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置20–23），固定全部noisy，观察noisy Block1（位置4–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

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

## 13. train/seed=0/方法一/clean块2→noisy块2/当前

原始状态：**PASS**。

只替换clean Block2（拼接位置20–23），固定全部noisy，观察noisy Block2（位置8–11）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

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

## 14. train/seed=0/方法四/秘密替换

原始状态：**FAIL**。

只替换第二块clean的整段秘密token（ID加17取模），比较第二块noisy的全部logits，不能只凭最高分候选ID不变下结论。 最大绝对差max_abs=0.2866450548171997；平均绝对差mean_abs=0.07052814215421677；整体L2差异l2=2.0403575897216797；阈值=1e-06。这是应禁止的依赖，但差异大于阈值；本注入实验因此成功检测到人为泄漏。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法四/秘密替换",
  "status": "FAIL",
  "max_abs": 0.2866450548171997,
  "mean_abs": 0.07052814215421677,
  "l2": 2.0403575897216797,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 15. train/seed=0/方法三/noisy块0/投影0/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 允许 | 2.738930941 | 3.100445271 |
| 4–7 | noisy Block1 | 禁止 | 0 | 0 |
| 8–11 | noisy Block2 | 禁止 | 0 | 0 |
| 12–15 | clean Block0 | 禁止 | 0 | 0 |
| 16–19 | clean Block1 | 禁止 | 0 | 0 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

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
      2.7796707153320312,
      2.7389309406280518,
      2.836817741394043,
      3.10044527053833,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

## 16. train/seed=0/方法三/noisy块0/投影1/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 允许 | 2.374881506 | 2.671439886 |
| 4–7 | noisy Block1 | 禁止 | 0 | 0 |
| 8–11 | noisy Block2 | 禁止 | 0 | 0 |
| 12–15 | clean Block0 | 禁止 | 0 | 0 |
| 16–19 | clean Block1 | 禁止 | 0 | 0 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

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
      2.3748815059661865,
      2.6714398860931396,
      2.586712598800659,
      2.4166390895843506,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

## 17. train/seed=0/方法三/noisy块1/投影0/禁止位置

原始状态：**FAIL**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为1.791006326675415，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 禁止 | 0 | 0 |
| 4–7 | noisy Block1 | 允许 | 3.271456003 | 4.636708736 |
| 8–11 | noisy Block2 | 禁止 | 0 | 0 |
| 12–15 | clean Block0 | 允许 | 1.424312472 | 2.347854853 |
| 16–19 | clean Block1 | 禁止 | 1.150850654 | 1.791006327 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影0/禁止位置",
  "status": "FAIL",
  "max_norm": 1.791006326675415,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      3.271456003189087,
      3.3673853874206543,
      4.636708736419678,
      4.115452766418457,
      0.0,
      0.0,
      0.0,
      0.0,
      2.3478548526763916,
      1.68468177318573,
      1.4243124723434448,
      1.9936699867248535,
      1.3493143320083618,
      1.1508506536483765,
      1.791006326675415,
      1.2207194566726685,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 18. train/seed=0/方法三/noisy块1/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为2.3478548526763916，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 2.3478548526763916,
  "threshold": 1e-08
}
```

</details>

## 19. train/seed=0/方法三/noisy块1/投影1/禁止位置

原始状态：**FAIL**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为2.90317964553833，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 禁止 | 0 | 0 |
| 4–7 | noisy Block1 | 允许 | 2.963182449 | 4.214995384 |
| 8–11 | noisy Block2 | 禁止 | 0 | 0 |
| 12–15 | clean Block0 | 允许 | 1.703820348 | 2.813826084 |
| 16–19 | clean Block1 | 禁止 | 1.242643833 | 2.903179646 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影1/禁止位置",
  "status": "FAIL",
  "max_norm": 2.90317964553833,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      2.9631824493408203,
      3.144315242767334,
      4.214995384216309,
      3.6054434776306152,
      0.0,
      0.0,
      0.0,
      0.0,
      2.813826084136963,
      2.109025478363037,
      1.7038203477859497,
      2.3528356552124023,
      1.779486060142517,
      2.90317964553833,
      1.6269487142562866,
      1.2426438331604004,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 20. train/seed=0/方法三/noisy块1/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为2.813826084136963，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 2.813826084136963,
  "threshold": 1e-08
}
```

</details>

## 21. train/seed=0/方法三/noisy块2/投影0/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 禁止 | 0 | 0 |
| 4–7 | noisy Block1 | 禁止 | 0 | 0 |
| 8–11 | noisy Block2 | 允许 | 3.280214071 | 3.773543835 |
| 12–15 | clean Block0 | 允许 | 1.330021143 | 3.329016447 |
| 16–19 | clean Block1 | 允许 | 1.11556673 | 2.117632866 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

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
      3.508465528488159,
      3.2802140712738037,
      3.4177844524383545,
      3.7735438346862793,
      1.9574460983276367,
      1.3300211429595947,
      1.666426181793213,
      3.3290164470672607,
      2.1176328659057617,
      1.4395619630813599,
      1.1155667304992676,
      1.6222392320632935,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 22. train/seed=0/方法三/noisy块2/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为3.3290164470672607，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 3.3290164470672607,
  "threshold": 1e-08
}
```

</details>

## 23. train/seed=0/方法三/noisy块2/投影1/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 禁止 | 0 | 0 |
| 4–7 | noisy Block1 | 禁止 | 0 | 0 |
| 8–11 | noisy Block2 | 允许 | 2.923672915 | 4.421595097 |
| 12–15 | clean Block0 | 允许 | 2.055109262 | 3.733653069 |
| 16–19 | clean Block1 | 允许 | 1.565273881 | 2.546288729 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

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
      3.4121103286743164,
      3.2448923587799072,
      2.923672914505005,
      4.421595096588135,
      2.706643581390381,
      2.0551092624664307,
      2.250493049621582,
      3.7336530685424805,
      2.5462887287139893,
      1.8390003442764282,
      1.5652738809585571,
      2.4866068363189697,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 24. train/seed=0/方法三/noisy块2/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为3.7336530685424805，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "train/seed=0/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 3.7336530685424805,
  "threshold": 1e-08
}
```

</details>

## 25. eval/seed=0/方法二/实际 kernel mask/层0

原始状态：**FAIL**。

检查第 1 层（层索引0）实际传给 attention kernel 的完整 mask。发现 16 个可见关系错误。本文件故意放开 noisy Block1 对当前 clean Block1 的连接；4 个query×4个key=16条错误边，符合注入预期。 Q形状=[1, 4, 24, 16]，分别为batch、query头数、位置数、每头维度；K形状=[1, 2, 24, 16]，第二维为key头数。mask=[1, 1, 24, 24]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 首个错误坐标[0, 0, 4, 16]依次表示batch、mask头维、query位置、key位置；最多展示前10个错误。 第一层示例query=4，可见key=[4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19]；位置映射为noisy 0–11，clean 12–23。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层0",
  "status": "FAIL",
  "layer": 0,
  "q_shape": [
    1,
    4,
    24,
    16
  ],
  "k_shape": [
    1,
    2,
    24,
    16
  ],
  "mask_shape": [
    1,
    1,
    24,
    24
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 16,
  "first_wrong_qk": [
    [
      0,
      0,
      4,
      16
    ],
    [
      0,
      0,
      4,
      17
    ],
    [
      0,
      0,
      4,
      18
    ],
    [
      0,
      0,
      4,
      19
    ],
    [
      0,
      0,
      5,
      16
    ],
    [
      0,
      0,
      5,
      17
    ],
    [
      0,
      0,
      5,
      18
    ],
    [
      0,
      0,
      5,
      19
    ],
    [
      0,
      0,
      6,
      16
    ],
    [
      0,
      0,
      6,
      17
    ]
  ],
  "noisy_block1_query": 4,
  "visible_key_positions": [
    4,
    5,
    6,
    7,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}
```

</details>

## 26. eval/seed=0/方法二/实际 kernel mask/层1

原始状态：**FAIL**。

检查第 2 层（层索引1）实际传给 attention kernel 的完整 mask。发现 16 个可见关系错误。本文件故意放开 noisy Block1 对当前 clean Block1 的连接；4 个query×4个key=16条错误边，符合注入预期。 Q形状=[1, 4, 24, 16]，分别为batch、query头数、位置数、每头维度；K形状=[1, 2, 24, 16]，第二维为key头数。mask=[1, 1, 24, 24]，最后两维分别是query行和key列。 is_causal=false表示不额外套普通causal三角限制，实际可见性仍由4D mask控制；dropout=0表示关闭随机丢弃。 首个错误坐标[0, 0, 4, 16]依次表示batch、mask头维、query位置、key位置；最多展示前10个错误。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/实际 kernel mask/层1",
  "status": "FAIL",
  "layer": 1,
  "q_shape": [
    1,
    4,
    24,
    16
  ],
  "k_shape": [
    1,
    2,
    24,
    16
  ],
  "mask_shape": [
    1,
    1,
    24,
    24
  ],
  "is_causal": false,
  "dropout": 0.0,
  "wrong_entries": 16,
  "first_wrong_qk": [
    [
      0,
      0,
      4,
      16
    ],
    [
      0,
      0,
      4,
      17
    ],
    [
      0,
      0,
      4,
      18
    ],
    [
      0,
      0,
      4,
      19
    ],
    [
      0,
      0,
      5,
      16
    ],
    [
      0,
      0,
      5,
      17
    ],
    [
      0,
      0,
      5,
      18
    ],
    [
      0,
      0,
      5,
      19
    ],
    [
      0,
      0,
      6,
      16
    ],
    [
      0,
      0,
      6,
      17
    ]
  ]
}
```

</details>

## 27. eval/seed=0/方法二/所有层 kernel 均被观察

原始状态：**PASS**。

观察到的层编号为[0, 1]，模型预期共2层。覆盖检查通过表示没有漏掉层；每层mask是否正确要看前面的独立条目。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法二/所有层 kernel 均被观察",
  "status": "PASS",
  "layers": [
    0,
    1
  ],
  "expected_layers": 2
}
```

</details>

## 28. eval/seed=0/方法一/重复性

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

## 29. eval/seed=0/方法一/clean块0→noisy块0/当前

原始状态：**PASS**。

只替换clean Block0（拼接位置12–15），固定全部noisy，观察noisy Block0（位置0–3）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

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

## 30. eval/seed=0/方法一/clean块0→noisy块1/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置12–15），固定全部noisy，观察noisy Block1（位置4–7）全部输出。 最大绝对差max_abs=0.33791130781173706；平均绝对差mean_abs=0.09450638294219971；整体L2差异l2=2.7129549980163574；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块0→noisy块1/前缀",
  "status": "PASS",
  "max_abs": 0.33791130781173706,
  "mean_abs": 0.09450638294219971,
  "l2": 2.7129549980163574,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 31. eval/seed=0/方法一/clean块0→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block0（拼接位置12–15），固定全部noisy，观察noisy Block2（位置8–11）全部输出。 最大绝对差max_abs=0.3101228177547455；平均绝对差mean_abs=0.09160081297159195；整体L2差异l2=2.578565835952759；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块0→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 0.3101228177547455,
  "mean_abs": 0.09160081297159195,
  "l2": 2.578565835952759,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 32. eval/seed=0/方法一/clean块1→noisy块0/未来

原始状态：**PASS**。

只替换clean Block1（拼接位置16–19），固定全部noisy，观察noisy Block0（位置0–3）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

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

## 33. eval/seed=0/方法一/clean块1→noisy块1/当前

原始状态：**FAIL**。

只替换clean Block1（拼接位置16–19），固定全部noisy，观察noisy Block1（位置4–7）全部输出。 最大绝对差max_abs=0.23460564017295837；平均绝对差mean_abs=0.06482543051242828；整体L2差异l2=1.8429665565490723；阈值=1e-06。这是应禁止的依赖，但差异大于阈值；本注入实验因此成功检测到人为泄漏。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块1→noisy块1/当前",
  "status": "FAIL",
  "max_abs": 0.23460564017295837,
  "mean_abs": 0.06482543051242828,
  "l2": 1.8429665565490723,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 34. eval/seed=0/方法一/clean块1→noisy块2/前缀

原始状态：**PASS**。

只替换clean Block1（拼接位置16–19），固定全部noisy，观察noisy Block2（位置8–11）全部输出。 最大绝对差max_abs=0.2091517597436905；平均绝对差mean_abs=0.05825144052505493；整体L2差异l2=1.6277228593826294；阈值=1e-06。这是允许的过去clean前缀，max_abs大于阈值说明正对照有效、前缀确实影响输出；非零是期望结果。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法一/clean块1→noisy块2/前缀",
  "status": "PASS",
  "max_abs": 0.2091517597436905,
  "mean_abs": 0.05825144052505493,
  "l2": 1.6277228593826294,
  "threshold": 1e-06,
  "expected": "应变化（正对照）"
}
```

</details>

## 35. eval/seed=0/方法一/clean块2→noisy块0/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置20–23），固定全部noisy，观察noisy Block0（位置0–3）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

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

## 36. eval/seed=0/方法一/clean块2→noisy块1/未来

原始状态：**PASS**。

只替换clean Block2（拼接位置20–23），固定全部noisy，观察noisy Block1（位置4–7）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

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

## 37. eval/seed=0/方法一/clean块2→noisy块2/当前

原始状态：**PASS**。

只替换clean Block2（拼接位置20–23），固定全部noisy，观察noisy Block2（位置8–11）全部输出。 最大绝对差max_abs=0.0；平均绝对差mean_abs=0.0；整体L2差异l2=0.0；阈值=1e-06。这是应禁止的依赖；实测差异为0，符合应不变的要求。

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

## 38. eval/seed=0/方法四/秘密替换

原始状态：**FAIL**。

只替换第二块clean的整段秘密token（ID加17取模），比较第二块noisy的全部logits，不能只凭最高分候选ID不变下结论。 最大绝对差max_abs=0.2866450548171997；平均绝对差mean_abs=0.07052814215421677；整体L2差异l2=2.0403575897216797；阈值=1e-06。这是应禁止的依赖，但差异大于阈值；本注入实验因此成功检测到人为泄漏。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法四/秘密替换",
  "status": "FAIL",
  "max_abs": 0.2866450548171997,
  "mean_abs": 0.07052814215421677,
  "l2": 2.0403575897216797,
  "threshold": 1e-06,
  "expected": "应不变"
}
```

</details>

## 39. eval/seed=0/方法三/noisy块0/投影0/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 允许 | 2.738930941 | 3.100445271 |
| 4–7 | noisy Block1 | 禁止 | 0 | 0 |
| 8–11 | noisy Block2 | 禁止 | 0 | 0 |
| 12–15 | clean Block0 | 禁止 | 0 | 0 |
| 16–19 | clean Block1 | 禁止 | 0 | 0 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

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
      2.7796707153320312,
      2.7389309406280518,
      2.836817741394043,
      3.10044527053833,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

## 40. eval/seed=0/方法三/noisy块0/投影1/禁止位置

原始状态：**PASS**。

对noisy Block0的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 允许 | 2.374881506 | 2.671439886 |
| 4–7 | noisy Block1 | 禁止 | 0 | 0 |
| 8–11 | noisy Block2 | 禁止 | 0 | 0 |
| 12–15 | clean Block0 | 禁止 | 0 | 0 |
| 16–19 | clean Block1 | 禁止 | 0 | 0 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

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
      2.3748815059661865,
      2.6714398860931396,
      2.586712598800659,
      2.4166390895843506,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
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

## 41. eval/seed=0/方法三/noisy块1/投影0/禁止位置

原始状态：**FAIL**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为1.791006326675415，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 禁止 | 0 | 0 |
| 4–7 | noisy Block1 | 允许 | 3.271456003 | 4.636708736 |
| 8–11 | noisy Block2 | 禁止 | 0 | 0 |
| 12–15 | clean Block0 | 允许 | 1.424312472 | 2.347854853 |
| 16–19 | clean Block1 | 禁止 | 1.150850654 | 1.791006327 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影0/禁止位置",
  "status": "FAIL",
  "max_norm": 1.791006326675415,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      3.271456003189087,
      3.3673853874206543,
      4.636708736419678,
      4.115452766418457,
      0.0,
      0.0,
      0.0,
      0.0,
      2.3478548526763916,
      1.68468177318573,
      1.4243124723434448,
      1.9936699867248535,
      1.3493143320083618,
      1.1508506536483765,
      1.791006326675415,
      1.2207194566726685,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 42. eval/seed=0/方法三/noisy块1/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为2.3478548526763916，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 2.3478548526763916,
  "threshold": 1e-08
}
```

</details>

## 43. eval/seed=0/方法三/noisy块1/投影1/禁止位置

原始状态：**FAIL**。

对noisy Block1的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为2.90317964553833，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 禁止 | 0 | 0 |
| 4–7 | noisy Block1 | 允许 | 2.963182449 | 4.214995384 |
| 8–11 | noisy Block2 | 禁止 | 0 | 0 |
| 12–15 | clean Block0 | 允许 | 1.703820348 | 2.813826084 |
| 16–19 | clean Block1 | 禁止 | 1.242643833 | 2.903179646 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影1/禁止位置",
  "status": "FAIL",
  "max_norm": 2.90317964553833,
  "threshold": 1e-08,
  "per_position_norm": [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      2.9631824493408203,
      3.144315242767334,
      4.214995384216309,
      3.6054434776306152,
      0.0,
      0.0,
      0.0,
      0.0,
      2.813826084136963,
      2.109025478363037,
      1.7038203477859497,
      2.3528356552124023,
      1.779486060142517,
      2.90317964553833,
      1.6269487142562866,
      1.2426438331604004,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 44. eval/seed=0/方法三/noisy块1/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为2.813826084136963，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块1/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 2.813826084136963,
  "threshold": 1e-08
}
```

</details>

## 45. eval/seed=0/方法三/noisy块2/投影0/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 禁止 | 0 | 0 |
| 4–7 | noisy Block1 | 禁止 | 0 | 0 |
| 8–11 | noisy Block2 | 允许 | 3.280214071 | 3.773543835 |
| 12–15 | clean Block0 | 允许 | 1.330021143 | 3.329016447 |
| 16–19 | clean Block1 | 允许 | 1.11556673 | 2.117632866 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

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
      3.508465528488159,
      3.2802140712738037,
      3.4177844524383545,
      3.7735438346862793,
      1.9574460983276367,
      1.3300211429595947,
      1.666426181793213,
      3.3290164470672607,
      2.1176328659057617,
      1.4395619630813599,
      1.1155667304992676,
      1.6222392320632935,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 46. eval/seed=0/方法三/noisy块2/投影0/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为3.3290164470672607，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块2/投影0/前缀正对照",
  "status": "PASS",
  "max_norm": 3.3290164470672607,
  "threshold": 1e-08
}
```

</details>

## 47. eval/seed=0/方法三/noisy块2/投影1/禁止位置

原始状态：**PASS**。

对noisy Block2的logits做本次随机投影，求输入位置embedding梯度。禁止区域最大范数为0.0，阈值1e-08。 per_position_norm的每一行对应一个batch样本，每个数字对应拼接后输入位置的梯度向量长度。数值不是token ID，也不是参数训练梯度。

按区域汇总所有batch的最小/最大梯度范数：

|位置|分支|是否允许依赖|最小值|最大值|
|---|---|---|---:|---:|
| 0–3 | noisy Block0 | 禁止 | 0 | 0 |
| 4–7 | noisy Block1 | 禁止 | 0 | 0 |
| 8–11 | noisy Block2 | 允许 | 2.923672915 | 4.421595097 |
| 12–15 | clean Block0 | 允许 | 2.055109262 | 3.733653069 |
| 16–19 | clean Block1 | 允许 | 1.565273881 | 2.546288729 |
| 20–23 | clean Block2 | 禁止 | 0 | 0 |

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
      3.4121103286743164,
      3.2448923587799072,
      2.923672914505005,
      4.421595096588135,
      2.706643581390381,
      2.0551092624664307,
      2.250493049621582,
      3.7336530685424805,
      2.5462887287139893,
      1.8390003442764282,
      1.5652738809585571,
      2.4866068363189697,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}
```

</details>

## 48. eval/seed=0/方法三/noisy块2/投影1/前缀正对照

原始状态：**PASS**。

这是允许的clean前缀正对照，前缀输入位置的最大embedding梯度范数为3.7336530685424805，大于阈值1e-08。说明输出存在可检测的前缀依赖，非零是预期行为。取最大值覆盖所有batch；不要求每个前缀位置都非零。

<details>
<summary>查看这一项的全部原始字段</summary>

```json
{
  "name": "eval/seed=0/方法三/noisy块2/投影1/前缀正对照",
  "status": "PASS",
  "max_norm": 3.7336530685424805,
  "threshold": 1e-08
}
```

</details>
