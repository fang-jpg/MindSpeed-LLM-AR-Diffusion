import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "fsdp2"
    / "qwen3_diffusion"
    / "gsm8k_experiment.py"
)
SPEC = importlib.util.spec_from_file_location("gsm8k_experiment_for_test", MODULE_PATH)
gsm8k = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gsm8k)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,234", "1234"),
        ("$001.500", "1.5"),
        ("-0", "0"),
        ("0.0010", "0.001"),
        ("not a number", None),
    ],
)
def test_normalize_numeric_answer(raw, expected):
    assert gsm8k.normalize_numeric_answer(raw) == expected


def test_extract_reference_and_prediction_answers():
    assert gsm8k.extract_reference_answer("reasoning 12\n#### 1,250") == "1250"
    assert gsm8k.extract_prediction_answer("Thus the answer is $1,250.") == "1250"
    assert gsm8k.extract_prediction_answer("work 10 then 20, so #### 30") == "30"
    assert gsm8k.extract_prediction_answer("no numeric result") is None


def test_fixed_indices_are_deterministic_and_unique():
    first = gsm8k.choose_indices(1319, 100, seed=42)
    second = gsm8k.choose_indices(1319, 100, seed=42)
    assert first == second
    assert len(first) == len(set(first)) == 100
    assert first[:5] == [1309, 228, 51, 563, 501]


def test_choose_indices_rejects_invalid_size():
    with pytest.raises(ValueError):
        gsm8k.choose_indices(10, 11, seed=42)


def test_prompt_uses_lm_eval_question_answer_shape():
    prompt = gsm8k.build_gsm8k_prompt(
        "What is 2 + 2?",
        [{"question": "What is 1 + 1?", "answer": "2 #### 2"}],
    )
    assert prompt == (
        "Question: What is 1 + 1?\nAnswer: 2 #### 2\n\n"
        "Question: What is 2 + 2?\nAnswer:"
    )
