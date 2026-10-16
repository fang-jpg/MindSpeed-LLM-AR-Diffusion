import json

import pytest
import torch

from mindspeed_llm.experimental.dopd.opdlm_data import (
    MATH_SUFFIX,
    SCIENCE_SUFFIX,
    format_opdlm_question,
    load_opdlm_records,
    records_for_iteration,
    tokenize_opdlm_prompt,
)
from mindspeed_llm.experimental.dopd.opdlm_verifier import verify_opdlm_record


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
        }
        self.last_content = messages[0]["content"]
        return [10, 11, 12]

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
    with pytest.raises(ValueError, match="exceeding the paper limit"):
        tokenize_opdlm_prompt(_record("math"), tokenizer, 2)


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
