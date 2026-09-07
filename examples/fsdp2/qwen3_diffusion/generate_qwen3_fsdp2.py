"""Generate text with Qwen3 through the MindSpeed FSDP2 backend.

This script keeps generation semantics in Hugging Face ``generate`` and only
handles FSDP2 model construction, distributed prompt synchronization, and
tokenization.
"""

import copy
import os
import types
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from transformers.utils import is_torch_npu_available

from mindspeed.fsdp.utils.torch_patch import apply_hccl_premul_sum_patch
from mindspeed_llm.fsdp2.data.tokenizer import TokenizerFactory
from mindspeed_llm.fsdp2.models.model_factory import ModelFactory
from mindspeed_llm.fsdp2.utils.arguments import (
    ModelArguments,
    OptimizationArguments,
    ParallelArguments,
    fsdp2_parse_args,
)
from mindspeed_llm.fsdp2.utils.device import set_accelerator_compatible
from mindspeed_llm.fsdp2.utils.global_vars import set_args
from mindspeed_llm.fsdp2.utils.logging import get_logger, setup_global_logging


logger = get_logger(__name__)


@dataclass
class GenerationArguments:
    prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Generate once from this prompt. If omitted, enter interactive mode."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Optional system message used by the tokenizer chat template."},
    )
    use_chat_template: bool = field(
        default=True,
        metadata={"help": "Format input with tokenizer.apply_chat_template."},
    )
    max_new_tokens: int = field(default=256)
    min_new_tokens: int = field(default=0)
    do_sample: bool = field(
        default=True,
        metadata={"help": "Sample when true; use greedy/beam decoding when false."},
    )
    temperature: float = field(default=0.7)
    top_k: int = field(default=50)
    top_p: float = field(default=0.95)
    repetition_penalty: float = field(default=1.0)
    num_beams: int = field(default=1)
    num_return_sequences: int = field(default=1)
    seed: int = field(default=42)
    skip_special_tokens: bool = field(default=True)

    def __post_init__(self):
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.min_new_tokens < 0 or self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens must be in [0, max_new_tokens].")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative; use 0 to disable top-k filtering.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive.")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive.")
        if self.num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be positive.")
        if not self.do_sample and self.num_return_sequences > self.num_beams:
            raise ValueError(
                "For non-sampling generation, num_return_sequences cannot exceed num_beams."
            )


@dataclass
class Arguments:
    model: ModelArguments = field(default_factory=ModelArguments)
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    generation: GenerationArguments = field(default_factory=GenerationArguments)
    optimization: OptimizationArguments = field(default_factory=OptimizationArguments)


def initialize_distributed(seed: int) -> torch.device:
    if is_torch_npu_available():
        accelerator = torch.npu
        backend = "hccl"
        apply_hccl_premul_sum_patch()
    elif torch.cuda.is_available():
        accelerator = torch.cuda
        backend = "nccl"
    else:
        raise RuntimeError("Qwen3 FSDP2 generation requires an NPU or CUDA GPU.")

    set_accelerator_compatible(accelerator)
    setup_global_logging(level="INFO")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.accelerator.set_device_index(local_rank)
    torch.accelerator.set_device(local_rank)

    if not dist.is_initialized():
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = str(local_rank)
        dist.init_process_group(
            backend=backend,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )

    # Identical seeds are important: every FSDP rank must sample the same token.
    torch.manual_seed(seed)
    accelerator.manual_seed_all(seed)
    return torch.device(torch.accelerator.current_accelerator().type, local_rank)


def synchronize_prompt(prompt: Optional[str]) -> str:
    rank = dist.get_rank()
    objects = [prompt if rank == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def tokenize_prompt(tokenizer, prompt: str, system_prompt: Optional[str], use_chat_template: bool):
    if use_chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    else:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))

    return input_ids, attention_mask


@torch.inference_mode()
def generate_once(model, tokenizer, prompt: str, args: GenerationArguments, device: torch.device):
    input_ids, attention_mask = tokenize_prompt(
        tokenizer,
        prompt,
        args.system_prompt,
        args.use_chat_template,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_length = input_ids.shape[1]

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.min_new_tokens = args.min_new_tokens
    generation_config.do_sample = args.do_sample
    generation_config.temperature = args.temperature
    generation_config.top_k = args.top_k
    generation_config.top_p = args.top_p
    generation_config.repetition_penalty = args.repetition_penalty
    generation_config.num_beams = args.num_beams
    generation_config.num_return_sequences = args.num_return_sequences
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.bos_token_id = tokenizer.bos_token_id

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        # Every FSDP rank participates. This also protects uneven beam/EOS completion.
        synced_gpus=dist.get_world_size() > 1,
    )

    response_ids = outputs[:, prompt_length:]
    return tokenizer.batch_decode(
        response_ids,
        skip_special_tokens=args.skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def main():
    root_args = fsdp2_parse_args(Arguments)
    flat_args = types.SimpleNamespace(
        **{
            key: value
            for namespace in (
                root_args.model,
                root_args.parallel,
                root_args.generation,
                root_args.optimization,
            )
            for key, value in namespace.__dict__.items()
        }
    )
    set_args(flat_args)

    device = initialize_distributed(root_args.generation.seed)
    rank = dist.get_rank()

    tokenizer = TokenizerFactory.create(root_args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    root_args.parallel.recompute = False
    engine = ModelFactory.create(root_args.model, root_args.parallel)
    model = getattr(engine, "model", engine)
    model.eval()

    prompt = root_args.generation.prompt
    while True:
        if prompt is None and rank == 0:
            try:
                prompt = input("\nUser (输入 exit 退出): ").strip()
            except EOFError:
                prompt = "exit"

        prompt = synchronize_prompt(prompt)
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            prompt = None
            continue

        responses = generate_once(model, tokenizer, prompt, root_args.generation, device)
        if rank == 0:
            for index, response in enumerate(responses):
                prefix = "Assistant" if len(responses) == 1 else f"Assistant[{index}]"
                print(f"{prefix}: {response}")

        if root_args.generation.prompt is not None:
            break
        prompt = None


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
