from mindspeed_llm.fsdp2.data.megatron_data.indexed_dataset import IndexedDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import re

dataset_list = [
    # "/share/dataset/x00840191/comm_math_code/RedStone_QA_oq_part_text_document",
    "/share/dataset/x00840191/comm_math_code/swallow_code_ablation_exp11_scor_text_document",
    # "/share/dataset/x00840191/comm_math_code/megamath_qa_llama_3.3_text_document",
    # "/share/dataset/x00840191/comm_math_code/megamath_qa_text_document",
    # "/share/dataset/x00840191/comm_math_code/MetaMathQA_text_document",
    # "/share/dataset/x00840191/comm_math_code/RedStone_QA_mcq_text_document",
    # "/share/dataset/x00840191/comm_math_code/swallow_math_text_document"
]

tokenizer = AutoTokenizer.from_pretrained("/home/c00633840/models/Qwen3-1.7B-Base", trust_remote_code=True)

# 看第 0 个文档的文本
for ds_path in dataset_list:
    ds = IndexedDataset(ds_path, multimodal=False, mmap=True)
    num_p1 = 0
    num_p2 = 0
    for i in tqdm(range(0, len(ds))):
        tokens = ds.get(i)
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        num_p1 += text.count("<|endoftext|>")
        num_p2 += text.count("<|im_end|>")

        print("<|endoftext|> count: ", num_p1)
        print("<|im_end|> count: ", num_p2)
