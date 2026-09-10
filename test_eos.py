# import numpy as np

# bin_path = "/share/dataset/x00840191/comm_math_code/megamath_qa_text_document.bin"
# eos_id = -100

# for dtype in [np.int32, np.int64, np.uint32]:
#     arr = np.memmap(bin_path, dtype=dtype, mode="r")
#     count = int((arr == eos_id).sum())
#     print(dtype, "num tokens:", len(arr), "eos count:", count)

#     if count > 0:
#         pos = np.where(arr == eos_id)[0][:20]
#         print("first eos positions:", pos.tolist())


from megatron.core.datasets.indexed_dataset import IndexedDataset

dataset = IndexedDataset(
    path_prefix="/share/dataset/x00840191/comm_math_code/megamath_qa_text_document",  # 不要加 .idx 或 .bin
    multimodal=False,
    mmap=True,
)

lengths = dataset.index.sequence_lengths

print(f"样本数: {len(lengths)}")
print(f"前 10 条长度: {lengths[:10]}")

for sample_id, length in enumerate(lengths):
    print(sample_id, int(length))
    