# import pandas as pd
# import glob
# import json

# # 收集当前目录下所有 parquet
# parquet_files = sorted(glob.glob("/home/c00633840/dataset/dataset_addition2/*.parquet"))

# with open("/home/c00633840/dataset/dataset_addition2/merged_from_parquet.jsonl", "w", encoding="utf-8") as fout:
#     for pf in parquet_files:
#         df = pd.read_parquet(pf)
#         print("列名:", df.columns.tolist())
#         print("前几行:")
#         print(df.head())
        # 假设文本列名为 "text",按你的实际列名修改
        # for text in df["text"]:
        #     fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

# import pandas as pd
# import glob

# parquet_files = sorted(glob.glob("/home/c00633840/dataset/dataset_addition2/*.parquet"))
# df = pd.read_parquet(parquet_files[0])

# print("列名:", df.columns.tolist())
# print("前几行:")
# print(df.head())

import json

with open("/home/c00633840/dataset/dataset_addition2/eagle_chat.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        print("字段:", list(obj.keys()))
        print("内容示例:", obj)
        if i >= 1:
            break
