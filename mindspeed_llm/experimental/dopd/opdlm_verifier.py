"""公开 OPDLM 数学、科学、代码和对话混合数据的任务验证器。

答案提取规则以及四种 ``tests_json`` 格式与公开 OPDLM 奖励实现保持一致。
模型生成的代码不可信，只会在受资源限制的子进程中执行；正式训练仍必须使用
断网、隔离且未挂载任何凭据的容器。

参考：https://github.com/divelab/OPDLM（采用 MIT 许可证的 ``reward`` 包）。
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from functools import lru_cache
from typing import Any


_CODE_TIMEOUT_SECONDS = int(os.environ.get("DOPD_CODE_TIMEOUT", "20"))
_OUTPUT_TAIL_CHARACTERS = 2000


def _verification_result(
    correct: bool,
    reason: str,
    *,
    applicable: bool = True,
    **details,
) -> dict[str, Any]:
    """构造可持久化的 verifier 结果，供训练审计和布尔门控共用。"""

    return {
        "correct": bool(correct),
        "applicable": bool(applicable),
        "reason": reason,
        **details,
    }


@lru_cache(maxsize=1)
def _math_verify_api():
    try:
        from math_verify import (  # type: ignore
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The paper-aligned OPDLM verifier requires math_verify==0.9.0. "
            "Install examples/fsdp2/qwen3_dopd/requirements_dopd.txt."
        ) from exc
    return parse, verify, LatexExtractionConfig, ExprExtractionConfig


def validate_runtime() -> None:
    """如果缺少验证器依赖，在执行昂贵的第一次 rollout 前立即报错。"""

    _math_verify_api()
    if os.name != "posix":
        raise RuntimeError("Paper-aligned code verification requires a POSIX training host")


def _response_text(response_ids, tokenizer) -> str:
    ids = response_ids.tolist() if hasattr(response_ids, "tolist") else list(response_ids)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and int(eos_id) in ids:
        ids = ids[: ids.index(int(eos_id))]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _verify_math_detailed(text: str, record: dict[str, Any]) -> dict[str, Any]:
    parse, verify, LatexExtractionConfig, ExprExtractionConfig = _math_verify_api()
    try:
        extracted_items = parse(
            text,
            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
        )
        if not extracted_items:
            return _verification_result(False, "math_answer_extraction_failed")
        extracted = str(extracted_items[1] if len(extracted_items) > 1 else extracted_items[0])
        gold = parse(
            "\\boxed{" + str(record["ground_truth_answer"]) + "}",
            extraction_config=[LatexExtractionConfig()],
        )
        answer = parse(
            "\\boxed{" + extracted + "}",
            extraction_config=[LatexExtractionConfig()],
        )
        correct = bool(verify(gold, answer))
        return _verification_result(
            correct,
            "correct" if correct else "math_answer_mismatch",
            extracted_answer=extracted,
            ground_truth_answer=str(record["ground_truth_answer"]),
        )
    except Exception as error:
        return _verification_result(
            False,
            "math_verification_error",
            error_type=type(error).__name__,
            error_message=str(error)[:_OUTPUT_TAIL_CHARACTERS],
        )


def _verify_math(text: str, record: dict[str, Any]) -> bool:
    return bool(_verify_math_detailed(text, record)["correct"])


def _extract_choice(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "[invalid]"
    letter = r"[A-Ja-j]"
    boxed = re.findall(
        rf"\\boxed\{{\s*(?:\\text(?:bf|rm|sf|tt|it)?\s*\{{)?\s*({letter})",
        text,
    )
    if boxed:
        return boxed[-1].upper()
    matches = re.findall(
        rf"(?i)\banswer\b[\s\*\"'\)\]]*[:=][\s\*\"'\$\(]*({letter})",
        text,
    )
    if matches:
        return matches[-1].upper()
    matches = re.findall(
        rf"(?i)\banswer\b[\s\*\"']*\bis\b[\s\*\"'\$\(\:\.]*"
        rf"(?:option|choice|letter)?[\s\*\"'\$\(\:\.]*({letter})\b",
        text,
    )
    if matches:
        return matches[-1].upper()
    bare = re.findall(r"\b([A-J])\b", text)
    return bare[-1].upper() if bare else "[invalid]"


def _verify_science_detailed(text: str, record: dict[str, Any]) -> dict[str, Any]:
    prediction = _extract_choice(text).strip().strip("()").upper()
    reference = str(record["ground_truth_answer"]).strip().strip("()").upper()
    if prediction == "[INVALID]":
        reason = "science_choice_extraction_failed"
        correct = False
    else:
        correct = prediction == reference
        reason = "correct" if correct else "science_choice_mismatch"
    return _verification_result(
        correct,
        reason,
        extracted_answer=prediction,
        ground_truth_answer=reference,
    )


def _verify_science(text: str, record: dict[str, Any]) -> bool:
    return bool(_verify_science_detailed(text, record)["correct"])


def _classify_tests(raw):
    if raw is None or raw == "N/A":
        return "unknown", None
    if isinstance(raw, (list, dict)):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                return "unknown", None
    if isinstance(parsed, list):
        return "assert_list", parsed
    if isinstance(parsed, dict):
        test_type = parsed.get("type")
        if test_type == "pytest" or "test_code" in parsed:
            return "pytest", parsed
        if test_type == "stdio":
            return "stdio", parsed
        if test_type == "fn_call":
            return "fn_call", parsed
    return "unknown", parsed


def _extract_code(text: str, record: dict[str, Any]) -> str:
    test_format, parsed = _classify_tests(record.get("tests_json"))
    function_name = record.get("fn_name")
    if not function_name and test_format in {"pytest", "fn_call"} and isinstance(parsed, dict):
        function_name = parsed.get("fn_name")
    blocks = re.findall(r"```python\s*\n?(.*?)```", text or "", re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n?(.*?)```", text or "", re.DOTALL)
    if not blocks:
        return ""
    if function_name:
        pattern = re.compile(rf"\bdef\s+{re.escape(str(function_name))}\s*\(")
        for block in blocks:
            if pattern.search(block):
                return block.strip()
    return "\n\n".join(block.strip() for block in blocks)


def _limit_child_resources() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (_CODE_TIMEOUT_SECONDS, _CODE_TIMEOUT_SECONDS + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def _run_solution_subprocess_detailed(code: str, runner_source: str) -> dict[str, Any]:
    temporary_directory = None
    process = None
    try:
        temporary_directory = tempfile.mkdtemp(prefix="dopd_code_")
        with open(os.path.join(temporary_directory, "solution.py"), "w", encoding="utf-8") as stream:
            stream.write(code)
        with open(os.path.join(temporary_directory, "test_runner.py"), "w", encoding="utf-8") as stream:
            stream.write(runner_source)
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                (
                    "import runpy,sys; sys.path.insert(0, '.'); "
                    "runpy.run_path('test_runner.py', run_name='__main__')"
                ),
            ],
            cwd=temporary_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=_limit_child_resources,
        )
        stdout, stderr = process.communicate(timeout=_CODE_TIMEOUT_SECONDS + 2)
        correct = process.returncode == 0
        return _verification_result(
            correct,
            "correct" if correct else "code_test_failed",
            return_code=process.returncode,
            stdout_tail=(stdout or "")[-_OUTPUT_TAIL_CHARACTERS:],
            stderr_tail=(stderr or "")[-_OUTPUT_TAIL_CHARACTERS:],
        )
    except subprocess.TimeoutExpired:
        if process is not None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        return _verification_result(False, "code_timeout")
    except Exception as error:
        return _verification_result(
            False,
            "code_verification_error",
            error_type=type(error).__name__,
            error_message=str(error)[:_OUTPUT_TAIL_CHARACTERS],
        )
    finally:
        if temporary_directory:
            shutil.rmtree(temporary_directory, ignore_errors=True)


def _run_solution_subprocess(code: str, runner_source: str) -> bool:
    return bool(_run_solution_subprocess_detailed(code, runner_source)["correct"])


def _check_pytest_detailed(code: str, parsed: dict[str, Any]) -> dict[str, Any]:
    test_code = parsed.get("test_code", "")
    test_functions = re.findall(r"^def (test_\w+)\s*\(", test_code, re.MULTILINE)
    if not test_code or not test_functions:
        return _verification_result(False, "invalid_pytest_spec")
    runner = (
        "from solution import *  # noqa\n"
        + test_code
        + "\n\n"
        + "\n".join(f"{name}()" for name in test_functions)
        + "\n"
    )
    return _run_solution_subprocess_detailed(code, runner)


def _check_assert_list_detailed(code: str, assertions: list[Any]) -> dict[str, Any]:
    body = "\n".join(item for item in assertions if isinstance(item, str))
    if not body.strip():
        return _verification_result(False, "invalid_assert_list")
    return _run_solution_subprocess_detailed(
        code, "from solution import *  # noqa\n" + body + "\n"
    )


def _check_function_calls_detailed(code: str, parsed: dict[str, Any]) -> dict[str, Any]:
    function_name = parsed.get("fn_name")
    cases = parsed.get("cases") or []
    if not function_name or not cases:
        return _verification_result(False, "invalid_function_call_spec")
    cases_json = json.dumps(cases)
    runner = (
        "from solution import *  # noqa\n"
        "import json\n"
        f"cases = json.loads({json.dumps(cases_json)})\n"
        "for case in cases:\n"
        "    args = case['args']\n"
        "    expected = case['expected']\n"
        f"    actual = {function_name}(*args)\n"
        "    if isinstance(expected, list) and len(expected) == 1 and not isinstance(actual, (list, tuple)):\n"
        "        expected = expected[0]\n"
        "    assert actual == expected\n"
    )
    return _run_solution_subprocess_detailed(code, runner)


def _check_stdio_detailed(code: str, parsed: dict[str, Any]) -> dict[str, Any]:
    cases = parsed.get("cases") or []
    if not cases:
        return _verification_result(False, "invalid_stdio_spec")
    cases_json = json.dumps(cases)
    runner = (
        "import json, subprocess, sys\n"
        f"cases = json.loads({json.dumps(cases_json)})\n"
        "for case in cases:\n"
        "    proc = subprocess.run([sys.executable, '-I', 'solution.py'], "
        "input=case.get('input', ''), capture_output=True, text=True, timeout=5)\n"
        "    assert proc.returncode == 0\n"
        "    assert (proc.stdout or '').strip() == (case.get('output') or '').strip()\n"
    )
    return _run_solution_subprocess_detailed(code, runner)


def _verify_code_detailed(text: str, record: dict[str, Any]) -> dict[str, Any]:
    code = _extract_code(text, record)
    if not code:
        return _verification_result(False, "code_block_missing")
    test_format, parsed = _classify_tests(record.get("tests_json"))
    if test_format == "pytest":
        result = _check_pytest_detailed(code, parsed)
    elif test_format == "stdio":
        result = _check_stdio_detailed(code, parsed)
    elif test_format == "fn_call":
        result = _check_function_calls_detailed(code, parsed)
    elif test_format == "assert_list":
        result = _check_assert_list_detailed(code, parsed)
    else:
        result = _verification_result(False, "unknown_code_test_format")
    result["test_format"] = test_format
    return result


def _verify_code(text: str, record: dict[str, Any]) -> bool:
    return bool(_verify_code_detailed(text, record)["correct"])


def verify_opdlm_record_detailed(*, response_ids, tokenizer, record, **_unused) -> dict[str, Any]:
    """返回可审计的正确性判定，包括失败阶段和提取到的答案。"""

    text = _response_text(response_ids, tokenizer)
    domain = record.get("domain")
    if domain == "math":
        result = _verify_math_detailed(text, record)
    elif domain == "science":
        result = _verify_science_detailed(text, record)
    elif domain == "code":
        result = _verify_code_detailed(text, record)
    elif domain == "chat":
        result = _verification_result(
            False,
            "no_applicable_verifier",
            applicable=False,
        )
    else:
        raise ValueError(f"Unsupported OPDLM domain: {domain!r}")
    result["domain"] = domain
    return result


def verify_opdlm_record(*, response_ids, tokenizer, record, **_unused) -> bool:
    """返回当前 rollout 是否可以使用未来感知的 d-OPD 修正。"""

    return bool(
        verify_opdlm_record_detailed(
            response_ids=response_ids,
            tokenizer=tokenizer,
            record=record,
            **_unused,
        )["correct"]
    )


# 训练器会优先使用该属性取得失败原因，同时保留原有 bool verifier 接口。
verify_opdlm_record.verify_detailed = verify_opdlm_record_detailed
