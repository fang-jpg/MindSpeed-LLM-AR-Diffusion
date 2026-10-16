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
    """加载并校验公开的 ``divelab/opdlm_train_data`` 原始导出文件。"""

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
    total_iterations: int | None = None,
) -> list[dict[str, Any]]:
    """返回定长外层迭代池；不足部分按官方入口逻辑从 epoch 开头循环补齐。"""

    if iteration < 1:
        raise ValueError("iteration must be at least 1")
    if prompts_per_iteration <= 0:
        raise ValueError("prompts_per_iteration must be positive")
    if not records:
        raise ValueError("records must be nonempty")
    natural_iterations = math.ceil(len(records) / prompts_per_iteration)
    if total_iterations is None:
        total_iterations = natural_iterations
    if total_iterations < natural_iterations:
        raise ValueError(
            f"total_iterations={total_iterations} cannot cover {len(records)} records "
            f"with prompts_per_iteration={prompts_per_iteration}"
        )
    if iteration > total_iterations:
        raise IndexError(
            f"iteration {iteration} exceeds the one-epoch OPDLM schedule ({total_iterations})"
        )
    start = (iteration - 1) * prompts_per_iteration
    return [
        records[index % len(records)]
        for index in range(start, start + prompts_per_iteration)
    ]


def format_opdlm_question(record: dict[str, Any]) -> str:
    """严格应用 OPDLM 训练配置中各领域对应的提示模板。"""

    question = record["question"]
    if record["domain"] == "math":
        return f"{question}\n{MATH_SUFFIX}"
    if record["domain"] == "science":
        return f"{question}\n{SCIENCE_SUFFIX}"
    return question


def _tokenize_opdlm_prompt_batch(records: list[dict[str, Any]], tokenizer) -> list[list[int]]:
    """应用 Qwen 对话模板且明确禁用截断，与官方 OPDLM 预处理保持一致。"""

    conversations = [
        [{"role": "user", "content": format_opdlm_question(record)}]
        for record in records
    ]
    token_ids = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        truncation=False,
        return_dict=False,
    )
    if isinstance(token_ids, dict):
        token_ids = token_ids.get("input_ids")
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if len(records) == 1 and token_ids and not isinstance(token_ids[0], list):
        token_ids = [token_ids]
    if not isinstance(token_ids, list) or len(token_ids) != len(records):
        raise ValueError(
            "tokenizer returned a prompt batch whose size does not match the OPDLM records"
        )

    normalized = []
    for index, prompt_ids in enumerate(token_ids):
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(
                f"Qwen chat template produced an empty/invalid OPDLM prompt at batch index {index}"
            )
        normalized.append([int(token_id) for token_id in prompt_ids])
    return normalized


def preprocess_opdlm_records(
    records: list[dict[str, Any]],
    tokenizer,
    max_prompt_tokens: int,
    batch_size: int = 1000,
) -> tuple[list[int], dict[str, Any]]:
    """按官方逻辑在不截断的前提下过滤超长提示词。

    返回保留记录在输入列表中的索引以及可持久化的审计报告。调用方应在
    分布式切分之前应用这些索引，避免不同 rank 得到不同的 batch 形状。
    """

    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    kept_indices: list[int] = []
    dropped: list[dict[str, Any]] = []
    longest_prompt_tokens = 0
    retained_by_domain = {domain: 0 for domain in sorted(SUPPORTED_DOMAINS)}
    dropped_by_domain = {domain: 0 for domain in sorted(SUPPORTED_DOMAINS)}

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch_token_ids = _tokenize_opdlm_prompt_batch(batch_records, tokenizer)
        for offset, (record, prompt_ids) in enumerate(zip(batch_records, batch_token_ids)):
            input_index = start + offset
            prompt_tokens = len(prompt_ids)
            longest_prompt_tokens = max(longest_prompt_tokens, prompt_tokens)
            domain = record["domain"]
            if prompt_tokens <= max_prompt_tokens:
                kept_indices.append(input_index)
                retained_by_domain[domain] += 1
                continue
            dropped_by_domain[domain] += 1
            dropped.append(
                {
                    "input_index": input_index,
                    "domain": domain,
                    "source": str(record.get("source", "")),
                    "prompt_tokens": prompt_tokens,
                }
            )

    if not kept_indices:
        raise ValueError(
            f"All {len(records)} OPDLM prompts exceed max_prompt_tokens={max_prompt_tokens}"
        )
    report = {
        "policy": "official_opdlm_filter_without_truncation",
        "max_prompt_tokens": max_prompt_tokens,
        "original_records": len(records),
        "retained_records": len(kept_indices),
        "dropped_records": len(dropped),
        "longest_prompt_tokens": longest_prompt_tokens,
        "retained_by_domain": retained_by_domain,
        "dropped_by_domain": dropped_by_domain,
        "dropped": dropped,
    }
    return kept_indices, report


def tokenize_opdlm_prompt(record: dict[str, Any], tokenizer, max_prompt_tokens: int):
    """按照论文设置应用 Qwen 对话模板，并拒绝任何静默截断。"""

    token_ids = _tokenize_opdlm_prompt_batch([record], tokenizer)[0]
    if len(token_ids) > max_prompt_tokens:
        raise ValueError(
            f"OPDLM prompt has {len(token_ids)} tokens, exceeding the configured limit "
            f"of {max_prompt_tokens}; no silent truncation is allowed"
        )
    return token_ids
