"""官方 OPDLM 指令混合数据的加载与提示词构造。"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


MATH_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
SCIENCE_SUFFIX = (
    'Please show your choice in the answer field with only the choice letter, '
    'e.g., "answer": "C".'
)
SUPPORTED_DOMAINS = frozenset({"math", "science", "code", "chat"})
REQUIRED_FIELDS = frozenset({"question", "ground_truth_answer", "domain", "source", "tests_json"})


def load_opdlm_records(path: str | Path, expected_records: int = 61816) -> list[dict[str, Any]]:
    """不做额外过滤，直接加载公开的 ``divelab/opdlm_train_data`` 导出文件。"""

    data_path = Path(path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Official OPDLM training file not found: {data_path}. "
            "Download divelab/opdlm_train_data/opdlm_train.json first."
        )
    with data_path.open(encoding="utf-8") as stream:
        if data_path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("OPDLM training data must be a nonempty JSON array or JSONL file")
    if expected_records > 0 and len(records) != expected_records:
        raise ValueError(
            f"Expected {expected_records} OPDLM records, found {len(records)} in {data_path}; "
            "refusing to silently train on a different dataset revision"
        )
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"OPDLM record {index} is not an object")
        missing = REQUIRED_FIELDS.difference(record)
        if missing:
            raise ValueError(f"OPDLM record {index} is missing fields: {sorted(missing)}")
        if not isinstance(record["question"], str) or not record["question"].strip():
            raise ValueError(f"OPDLM record {index} has an empty/non-string question")
        if record["domain"] not in SUPPORTED_DOMAINS:
            raise ValueError(
                f"OPDLM record {index} has unsupported domain {record['domain']!r}; "
                f"expected one of {sorted(SUPPORTED_DOMAINS)}"
            )
    return records


def shuffled_epoch(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """复现公开版本对完整数据集执行一次确定性随机打乱的行为。"""

    shuffled = list(records)
    random.Random(int(seed)).shuffle(shuffled)
    return shuffled


def records_for_iteration(
    records: list[dict[str, Any]],
    iteration: int,
    prompts_per_iteration: int,
) -> list[dict[str, Any]]:
    """返回定长的外层迭代数据池；最后一池不足时从 epoch 开头补齐。"""

    if iteration < 1:
        raise ValueError("iteration must be at least 1")
    if prompts_per_iteration <= 0:
        raise ValueError("prompts_per_iteration must be positive")
    total_iterations = math.ceil(len(records) / prompts_per_iteration)
    if iteration > total_iterations:
        raise IndexError(
            f"iteration {iteration} exceeds the one-epoch OPDLM schedule ({total_iterations})"
        )
    start = (iteration - 1) * prompts_per_iteration
    chunk = list(records[start : start + prompts_per_iteration])
    if len(chunk) < prompts_per_iteration:
        chunk.extend(records[: prompts_per_iteration - len(chunk)])
    return chunk


def format_opdlm_question(record: dict[str, Any]) -> str:
    """严格应用 OPDLM 训练配置中各领域对应的提示模板。"""

    question = record["question"]
    if record["domain"] == "math":
        return f"{question}\n{MATH_SUFFIX}"
    if record["domain"] == "science":
        return f"{question}\n{SCIENCE_SUFFIX}"
    return question


def tokenize_opdlm_prompt(record: dict[str, Any], tokenizer, max_prompt_tokens: int):
    """按照论文设置应用 Qwen 对话模板，并关闭思考模式。"""

    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": format_opdlm_question(record)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("tokenizer returned more than one prompt for one OPDLM record")
        token_ids = token_ids[0]
    if not token_ids:
        raise ValueError("Qwen chat template produced an empty OPDLM prompt")
    if len(token_ids) > max_prompt_tokens:
        raise ValueError(
            f"OPDLM prompt has {len(token_ids)} tokens, exceeding the paper limit "
            f"of {max_prompt_tokens}; no silent truncation is allowed"
        )
    return [int(token_id) for token_id in token_ids]
