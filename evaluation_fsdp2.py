"""FSDP2 benchmark evaluation entry point.

The FSDP shard group must execute identical forwards. Rank 0 therefore reads
and formats examples, broadcasts each prompt, and is the only rank that scores
and writes results.
"""

import json
import os
import re
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist
from transformers.utils import is_torch_npu_available

from mindspeed.fsdp.utils.torch_patch import apply_hccl_premul_sum_patch
from mindspeed_llm.fsdp2.data.tokenizer import TokenizerFactory
from mindspeed_llm.fsdp2.inference.chat_model import ChatModel
from mindspeed_llm.fsdp2.models.model_factory import ModelFactory
from mindspeed_llm.fsdp2.utils.arguments import (
    InferenceArguments,
    ModelArguments,
    OptimizationArguments,
    ParallelArguments,
    fsdp2_parse_args,
)
from mindspeed_llm.fsdp2.utils.device import set_accelerator_compatible
from mindspeed_llm.fsdp2.utils.global_vars import set_args


@dataclass
class EvaluationArguments:
    task: str = field(default="mmlu", metadata={"help": "Evaluation task. Currently supports mmlu."})
    task_data_path: Optional[str] = field(default=None, metadata={"help": "MMLU test CSV directory."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Maximum examples per subject."})
    output_path: str = field(default="./outputs/fsdp2_mmlu.json", metadata={"help": "Rank-0 result JSON path."})
    use_chat_template: bool = field(
        default=True,
        metadata={
            "help": (
                "Format prompts with the HF tokenizer chat template. Keep this enabled for Qwen3 so "
                "`inference.enable_thinking=false` takes effect."
            )
        },
    )
    seed: int = field(default=42, metadata={"help": "Evaluation random seed shared by all FSDP ranks."})

    def __post_init__(self):
        if self.task != "mmlu":
            raise ValueError("FSDP2 evaluation currently supports only `mmlu`.")
        if not self.task_data_path:
            raise ValueError("`task_data_path` must be specified.")
        if self.max_eval_samples is not None and self.max_eval_samples <= 0:
            raise ValueError("`max_eval_samples` must be positive.")


@dataclass
class Arguments:
    model: ModelArguments = field(default_factory=ModelArguments)
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    inference: InferenceArguments = field(default_factory=InferenceArguments)
    evaluation: EvaluationArguments = field(default_factory=EvaluationArguments)
    optimization: OptimizationArguments = field(default_factory=OptimizationArguments)


def initialize(seed: int) -> None:
    if is_torch_npu_available():
        accelerator = torch.npu
        backend = "hccl"
        apply_hccl_premul_sum_patch()
    elif torch.cuda.is_available():
        accelerator = torch.cuda
        backend = "nccl"
    else:
        raise RuntimeError("FSDP2 evaluation requires an available NPU or CUDA device.")

    set_accelerator_compatible(accelerator)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.accelerator.set_device_index(local_rank)
    torch.accelerator.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    torch.manual_seed(seed)
    accelerator.manual_seed_all(seed)


def sync_from_rank0(value):
    objects = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def format_mmlu_prompt(row, few_shot_example: str) -> str:
    """Use the same default instruction template as mcore's MmluEval."""
    test_question = (
        f"{row['question']}\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}"
    )
    return f"{few_shot_example}\n\n{test_question}\nAnswer:"


def extract_choice(text: str) -> Optional[str]:
    """Parse mcore-style answers plus common Qwen3 answer formats."""
    answer = text.lstrip().upper()

    # Preserve mcore's "A." and single-letter behavior without allowing
    # re.match("A", "ANSWER: C") to incorrectly return A.
    dotted_match = re.match(r".*(?P<answer>[ABCD])\..*", answer)
    if dotted_match:
        return dotted_match.group("answer")
    single_match = re.fullmatch(r"(?P<answer>[ABCD])[\s.]*", answer)
    if single_match:
        return single_match.group("answer")

    # Qwen3 may emit "Answer: A", "答案：A", "(A)", or "A)" even when
    # thinking mode is disabled.
    labeled_match = re.search(
        r"(?:ANSWER|答案|选项)\s*[:：]?\s*[\(\[]?([ABCD])[\)\]]?",
        answer,
    )
    if labeled_match:
        return labeled_match.group(1)

    standalone_match = re.search(r"(?<![A-Z])[\(\[]?([ABCD])[\)\]]?(?![A-Z])", answer)
    if standalone_match:
        return standalone_match.group(1)

    return None


def evaluate_mmlu(chat_model, args: EvaluationArguments) -> dict:
    rank = dist.get_rank()
    if rank == 0:
        data_dir = Path(args.task_data_path)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"MMLU data directory does not exist: {data_dir}")
        files = sorted(str(path) for path in data_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found under: {data_dir}")
        template_path = (
            Path(__file__).parent
            / "mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/mmlu_5shot_template.json"
        )
        few_shot_templates = json.loads(template_path.read_text(encoding="utf-8"))
    else:
        files = few_shot_templates = None
    files, few_shot_templates = sync_from_rank0((files, few_shot_templates))

    results = {}
    total_correct = 0
    total_count = 0
    for file_name in files:
        if rank == 0:
            frame = pd.read_csv(file_name, names=["question", "A", "B", "C", "D", "answer"])
            if args.max_eval_samples is not None:
                frame = frame.head(args.max_eval_samples)
            rows = frame.to_dict("records")
            subject = re.sub(r"(?:_test|_val|_dev)?\.csv$", "", Path(file_name).name)
        else:
            rows = subject = None
        subject, rows = sync_from_rank0((subject, rows))

        subject_correct = 0
        samples = []
        for row in rows:
            if subject not in few_shot_templates:
                raise KeyError(f"Missing MMLU few-shot template for subject: {subject}")
            prompt = format_mmlu_prompt(row, few_shot_templates[subject])
            # Do not rely solely on tokenizer.apply_chat_template's
            # enable_thinking flag. Older Qwen3 tokenizer templates may ignore
            # it, while the explicit /no_think switch is model-visible.
            messages = [{"role": "user", "content": f"/no_think\n\n{prompt}"}]
            responses = chat_model.chat(
                messages,
                use_chat_template=args.use_chat_template,
                require_chat_template=args.use_chat_template,
                enable_thinking=False,
            )
            # print("prompt:", prompt)
            # print("responses:", responses)
            if rank == 0:
                response = responses[0].response_text
                prediction = extract_choice(response)
                expected = str(row["answer"]).strip().upper()
                correct = prediction == expected
                subject_correct += int(correct)
                samples.append({"prediction": prediction, "answer": row["answer"], "response": response})
                print(
                    f"correct: {expected}, AI: {prediction}, rank: {rank}, "
                    f"response: {response!r}",
                    flush=True,
                )

        if rank == 0:
            count = len(rows)
            results[subject] = {
                "correct": subject_correct,
                "total": count,
                "accuracy": subject_correct / count if count else 0.0,
                "samples": samples,
            }
            total_correct += subject_correct
            total_count += count
            accuracy = subject_correct / count if count else 0.0
            print(
                f"{subject} has {subject_correct} corrects in {count} questions, "
                f"with accuracy {accuracy}",
                flush=True,
            )

    if rank == 0:
        results["total"] = {
            "correct": total_correct,
            "total": total_count,
            "accuracy": total_correct / total_count if total_count else 0.0,
        }
    return results


def main() -> None:
    root_args = fsdp2_parse_args(Arguments)
    flat_args = types.SimpleNamespace(
        **{
            key: value
            for namespace in (
                root_args.model,
                root_args.parallel,
                root_args.inference,
                root_args.evaluation,
                root_args.optimization,
            )
            for key, value in namespace.__dict__.items()
        }
    )
    set_args(flat_args)
    # Keep the evaluation seed independent from InferenceArguments. Some
    # MindSpeed-LLM releases do not define `InferenceArguments.seed`.
    initialize(root_args.evaluation.seed)
    root_args.parallel.recompute = False

    tokenizer = TokenizerFactory.create(root_args.model)
    model = ModelFactory.create(root_args.model, root_args.parallel)
    chat_model = ChatModel(model, tokenizer, root_args.inference)
    results = evaluate_mmlu(chat_model, root_args.evaluation)

    if dist.get_rank() == 0:
        output_path = Path(root_args.evaluation.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        total = results["total"]
        print(
            f"mmlu acc = {total['correct']}/{total['total']}={total['accuracy']}",
            flush=True,
        )
        print(f"Results written to {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
