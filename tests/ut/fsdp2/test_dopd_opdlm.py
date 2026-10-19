import json

import pytest
import torch

from mindspeed_llm.experimental.dopd.opdlm_data import (
    MATH_SUFFIX,
    SCIENCE_SUFFIX,
    format_opdlm_question,
    load_opdlm_records,
    preprocess_opdlm_records,
    records_for_iteration,
    tokenize_opdlm_prompt,
)
from mindspeed_llm.experimental.dopd.opdlm_verifier import (
    verify_opdlm_record,
    verify_opdlm_record_detailed,
)


def _record(domain="math", question="What is 1+1?", answer="2"):
    return {
        "question": question,
        "ground_truth_answer": answer,
        "domain": domain,
        "source": "unit-test",
        "tests_json": "N/A",
    }


class _Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "truncation": False,
            "return_dict": False,
        }
        conversations = messages if messages and isinstance(messages[0], list) else [messages]
        self.last_content = conversations[-1][0]["content"]
        return [[10, 11, 12] for _ in conversations]

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens
        return "".join(chr(item) for item in ids)


def test_official_prompt_templates_and_qwen_chat_flags():
    assert format_opdlm_question(_record("math")).endswith(MATH_SUFFIX)
    assert format_opdlm_question(_record("science")).endswith(SCIENCE_SUFFIX)
    assert format_opdlm_question(_record("code", question="write f")) == "write f"
    tokenizer = _Tokenizer()
    assert tokenize_opdlm_prompt(_record("math"), tokenizer, 3) == [10, 11, 12]
    assert tokenizer.last_content.endswith(MATH_SUFFIX)
    with pytest.raises(ValueError, match="exceeding the configured limit"):
        tokenize_opdlm_prompt(_record("math"), tokenizer, 2)


class _VariableLengthTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append(kwargs)
        assert conversations and isinstance(conversations[0], list)
        return [
            [10] * int(conversation[0]["content"])
            for conversation in conversations
        ]


def test_official_preprocessing_filters_overlong_prompts_without_truncation():
    records = [
        _record("code", question="1"),
        _record("code", question="3"),
        _record("code", question="2"),
    ]
    tokenizer = _VariableLengthTokenizer()
    kept_indices, report = preprocess_opdlm_records(
        records,
        tokenizer,
        max_prompt_tokens=2,
        batch_size=2,
    )

    assert kept_indices == [0, 2]
    assert report["policy"] == "official_opdlm_filter_without_truncation"
    assert report["original_records"] == 3
    assert report["retained_records"] == 2
    assert report["dropped_records"] == 1
    assert report["longest_prompt_tokens"] == 3
    assert report["dropped"] == [
        {
            "input_index": 1,
            "domain": "code",
            "source": "unit-test",
            "prompt_tokens": 3,
        }
    ]
    assert all(call["truncation"] is False for call in tokenizer.calls)


def test_loader_validates_schema_and_exact_record_count(tmp_path):
    path = tmp_path / "opdlm_train.json"
    path.write_text(json.dumps([_record()]), encoding="utf-8")
    assert len(load_opdlm_records(path, expected_records=1)) == 1
    with pytest.raises(ValueError, match="Expected 2 OPDLM records"):
        load_opdlm_records(path, expected_records=2)


def test_final_iteration_is_padded_from_epoch_start():
    records = [_record(question=str(index)) for index in range(5)]
    assert [row["question"] for row in records_for_iteration(records, 2, 4)] == [
        "4",
        "0",
        "1",
        "2",
    ]
    assert [
        row["question"]
        for row in records_for_iteration(records, 3, 4, total_iterations=3)
    ] == ["3", "4", "0", "1"]


def test_science_and_chat_correctness_gate():
    tokenizer = _Tokenizer()
    response = torch.tensor([ord(char) for char in '{"answer": "C"}'] + [99, ord("D")])
    assert verify_opdlm_record(
        response_ids=response,
        tokenizer=tokenizer,
        record=_record("science", answer="C"),
    )
    assert not verify_opdlm_record(
        response_ids=response,
        tokenizer=tokenizer,
        record=_record("chat", answer="N/A"),
    )


def test_detailed_verifier_explains_science_and_chat_failures():
    tokenizer = _Tokenizer()
    mismatch = torch.tensor([ord(char) for char in "The answer is B"])
    science = verify_opdlm_record_detailed(
        response_ids=mismatch,
        tokenizer=tokenizer,
        record=_record("science", answer="C"),
    )
    assert science == {
        "correct": False,
        "applicable": True,
        "reason": "science_choice_mismatch",
        "extracted_answer": "B",
        "ground_truth_answer": "C",
        "domain": "science",
    }

    chat = verify_opdlm_record_detailed(
        response_ids=mismatch,
        tokenizer=tokenizer,
        record=_record("chat", answer="N/A"),
    )
    assert chat == {
        "correct": False,
        "applicable": False,
        "reason": "no_applicable_verifier",
        "domain": "chat",
    }


def test_detailed_verifier_reports_missing_code_block():
    tokenizer = _Tokenizer()
    response = torch.tensor([ord(char) for char in "def f(): return 1"])
    result = verify_opdlm_record_detailed(
        response_ids=response,
        tokenizer=tokenizer,
        record={
            **_record("code", answer="N/A"),
            "tests_json": ["assert f() == 1"],
        },
    )
    assert result["correct"] is False
    assert result["reason"] == "code_block_missing"
