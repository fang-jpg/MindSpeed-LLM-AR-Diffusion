#!/usr/bin/env python3
"""Inspect or paper-align decoding of a d-OPD DCP model.

The d-OPD trainer saves a PyTorch Distributed Checkpoint (DCP), not a standalone
Hugging Face directory.  This entrypoint rebuilds the Qwen3 diffusion student
with the same MindSpeed FSDP2 layout, loads the DCP weights, and runs the same
    block-diffusion decoding state machine used by the released JetEngine.

Only rank zero prints and writes results.  Every rank still executes every model
forward because the checkpoint is loaded into an FSDP2 model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


DEFAULT_CHECKPOINT = (
    "/share/dataset/x00840191/model_train/ckpt/qwen3_1.7b_dopd_1010/"
    "checkpoints/global_step_483_final"
)


@dataclass
class DecodeStats:
    """Model-forward accounting for one generation call."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_nfe: int = 0
    denoise_nfe: int = 0
    refresh_nfe: int = 0
    committed_per_step: list[int] = field(default_factory=list)
    completed_blocks: int = 0

    @property
    def total_nfe(self) -> int:
        return self.prefill_nfe + self.denoise_nfe + self.refresh_nfe

    @property
    def decode_nfe(self) -> int:
        return self.denoise_nfe + self.refresh_nfe

    @staticmethod
    def _ratio(tokens: int, forwards: int) -> float:
        return tokens / forwards if forwards else 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            total_nfe=self.total_nfe,
            decode_nfe=self.decode_nfe,
            # Primary TPF: realised output tokens per all model forwards,
            # including the one prompt-prefill forward.
            tpf_total=self._ratio(self.generated_tokens, self.total_nfe),
            tpf_decode=self._ratio(self.generated_tokens, self.decode_nfe),
            tpf_denoise_only=self._ratio(
                self.generated_tokens, self.denoise_nfe
            ),
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a MindSpeed FSDP2 d-OPD DCP checkpoint and compare block-size "
            "and confidence-threshold decoding."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Qwen3-1.7B-Base directory used to construct the model/tokenizer. "
            "By default it is read from <run_dir>/dopd_config.json."
        ),
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional explicit dopd_config.json; normally inferred from checkpoint.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Result path. Defaults to <run_dir>/dopd_decode_analysis.json.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
            "every morning and bakes muffins for her friends every day with four. "
            "She sells the remainder at the farmers' market daily for $2 per fresh "
            "duck egg. How much in dollars does she make every day?"
        ),
    )
    parser.add_argument(
        "--block-sizes",
        default="4",
        help="Comma-separated block sizes, for example 1,2,4,8,16.",
    )
    parser.add_argument(
        "--thresholds",
        default="fixed",
        help=(
            "Comma-separated modes. 'fixed' (or 0/none) commits only the static "
            "quota; a value in (0,1] additionally commits every token at or above "
            "that confidence while retaining the static quota as progress fallback."
        ),
    )
    parser.add_argument(
        "--steps-per-block",
        type=int,
        default=4,
        help="Denoising steps for every block; the paper evaluation uses 4.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-mode",
        choices=("ar", "none"),
        default="none",
        help=(
            "'none' starts each response-aligned block fully masked, as in the "
            "paper evaluation; 'ar' is the legacy hybrid decoder."
        ),
    )
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking when a compatible chat template is used.",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--paper-aligned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate the complete Appendix C.4 decoding recipe. Use "
            "--no-paper-aligned only for deliberate block/threshold ablations."
        ),
    )
    parser.add_argument(
        "--no-print-trace",
        action="store_true",
        help="Keep step traces in JSON but suppress their console rendering.",
    )
    return parser.parse_args()


def _nested(mapping: dict[str, Any], *keys: str, default=None):
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _run_dir(checkpoint: Path) -> Path:
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def load_run_config(args: argparse.Namespace) -> tuple[dict[str, Any], Optional[Path]]:
    path = Path(args.config_json) if args.config_json else _run_dir(Path(args.checkpoint)) / "dopd_config.json"
    if not path.is_file():
        return {}, None
    return json.loads(path.read_text(encoding="utf-8")), path


def parse_block_sizes(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        block_size = int(item)
        if block_size <= 0:
            raise ValueError("Every block size must be positive.")
        if block_size not in result:
            result.append(block_size)
    if not result:
        raise ValueError("At least one block size is required.")
    return result


def parse_thresholds(value: str) -> list[Optional[float]]:
    result: list[Optional[float]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"fixed", "none", "null", "0", "0.0"}:
            threshold = None
        else:
            threshold = float(item)
            if not 0.0 < threshold <= 1.0:
                raise ValueError("Dynamic thresholds must be in (0, 1].")
        if threshold not in result:
            result.append(threshold)
    if not result:
        raise ValueError("At least one threshold mode is required.")
    return result


def validate_paper_alignment(
    args: argparse.Namespace,
    block_sizes: list[int],
    thresholds: list[Optional[float]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed when a run labelled paper-aligned changes decoding semantics."""
    trained_block_size = int(_nested(run_config, "dopd", "block_size", default=4))
    expected = {
        "trained_block_size": trained_block_size,
        "block_sizes": [4],
        "steps_per_block": 4,
        "remasking": "low_confidence_static",
        "thresholds": [None],
        "response_aligned_blocks": True,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "max_new_tokens": 4096,
        "use_chat_template": True,
        "enable_thinking": False,
        "one_completion": True,
        "ignore_eos": False,
    }
    checks = [
        (
            trained_block_size == 4,
            f"checkpoint training block_size={trained_block_size}, expected 4",
        ),
        (block_sizes == [4], f"block_sizes={block_sizes}, expected [4]"),
        (args.steps_per_block == 4, f"steps_per_block={args.steps_per_block}, expected 4"),
        (thresholds == [None], f"thresholds={thresholds}, expected [fixed]"),
        (args.seed_mode == "none", f"seed_mode={args.seed_mode!r}, expected 'none'"),
        (args.temperature == 1.0, f"temperature={args.temperature}, expected 1.0"),
        (args.top_k == 1, f"top_k={args.top_k}, expected 1"),
        (args.top_p == 1.0, f"top_p={args.top_p}, expected 1.0"),
        (args.max_new_tokens == 4096, f"max_new_tokens={args.max_new_tokens}, expected 4096"),
        (args.use_chat_template, "Qwen3 chat template is disabled"),
        (not args.enable_thinking, "enable_thinking must be false"),
        (not args.ignore_eos, "ignore_eos must be false"),
        (args.warmup == 0, f"warmup={args.warmup}, expected 0"),
        (args.repeats == 1, f"repeats={args.repeats}, expected 1"),
    ]
    mismatches = [message for valid, message in checks if not valid]
    if args.paper_aligned and mismatches:
        raise ValueError(
            "--paper-aligned configuration mismatch:\n- " + "\n- ".join(mismatches)
        )
    return {
        "enabled": args.paper_aligned,
        "status": "aligned" if not mismatches else "diagnostic_override",
        "expected": expected,
        "mismatches": mismatches,
        "paper_section": "Appendix C.4 Benchmark evaluation",
    }


def resolve_paths_and_ids(
    args: argparse.Namespace, run_config: dict[str, Any]
) -> tuple[Path, Path, int]:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"DCP checkpoint directory does not exist: {checkpoint}")

    configured_model_path = _nested(
        run_config, "model", "model_name_or_path", default=None
    )
    model_path_value = args.model_path or configured_model_path
    if not model_path_value:
        raise FileNotFoundError(
            "Cannot infer the Base model/tokenizer directory because no "
            "--model-path was supplied and dopd_config.json is unavailable."
        )
    model_path = Path(model_path_value)
    if not model_path.is_dir():
        raise FileNotFoundError(
            "Cannot locate the Base model/tokenizer directory. Pass --model-path; "
            f"the path recorded in dopd_config.json was {configured_model_path!r}."
        )

    configured_mask = _nested(run_config, "dopd", "mask_token_id", default=None)
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        mask_token_id = configured_mask
    if mask_token_id is None:
        raise ValueError("Cannot infer mask token ID; pass --mask-token-id.")
    return checkpoint.resolve(), model_path.resolve(), int(mask_token_id)


def initialize_and_load_model(
    args: argparse.Namespace,
    run_config: dict[str, Any],
    checkpoint: Path,
    model_path: Path,
    mask_token_id: int,
):
    from mindspeed_llm.experimental.dopd.trainer import _initialize
    from mindspeed_llm.fsdp2.checkpoint.checkpoint_manager import CheckpointManager
    from mindspeed_llm.fsdp2.data.tokenizer import TokenizerFactory
    from mindspeed_llm.fsdp2.models.model_factory import ModelFactory
    from mindspeed_llm.fsdp2.utils.arguments import (
        ModelArguments,
        OptimizationArguments,
        ParallelArguments,
    )
    from mindspeed_llm.fsdp2.utils.global_vars import set_args

    _initialize(args.seed)
    world_size = dist.get_world_size()
    configured_parallel = _nested(run_config, "parallel", default={}) or {}
    if int(configured_parallel.get("tp_size", 1)) != 1 or int(
        configured_parallel.get("cp_size", 1)
    ) != 1:
        raise ValueError("This d-OPD inference script expects the trained TP=1, CP=1 layout.")

    model_args = ModelArguments(
        model_name_or_path=str(model_path),
        model_id="qwen3_diffusion",
        trust_remote_code=bool(
            _nested(run_config, "model", "trust_remote_code", default=False)
        ),
        train_from_scratch=False,
        init_model_with_meta_device=False,
    )
    parallel_args = ParallelArguments(
        tp_size=1,
        fsdp_size=world_size,
        ep_size=1,
        ep_fsdp_size=1,
        cp_size=1,
        fsdp_modules=list(
            configured_parallel.get(
                "fsdp_modules",
                ["model.layers.{*}", "model.embed_tokens", "lm_head"],
            )
        ),
        ep_modules=list(
            configured_parallel.get("ep_modules", ["model.layers.{*}.mlp.experts"])
        ),
        ep_fsdp_modules=list(
            configured_parallel.get(
                "ep_fsdp_modules", ["model.layers.{*}.mlp.experts"]
            )
        ),
        ep_dispatcher=configured_parallel.get("ep_dispatcher", "eager"),
        recompute=False,
        param_dtype=configured_parallel.get("param_dtype", "bf16"),
        reduce_dtype=configured_parallel.get("reduce_dtype", "bf16"),
    )
    optimization_args = OptimizationArguments(
        use_fused_rmsnorm=False,
        use_fused_rotary_pos_emb=False,
    )
    flat_args = types.SimpleNamespace(
        **{
            key: value
            for namespace in (model_args, parallel_args, optimization_args)
            for key, value in namespace.__dict__.items()
        }
    )
    set_args(flat_args)

    tokenizer = TokenizerFactory.create(model_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    engine = ModelFactory.create(model_args, parallel_args)
    CheckpointManager.load(str(checkpoint), {"model": engine})
    model = engine.model
    model.mask_token_id = mask_token_id
    model.config.mask_token_id = mask_token_id
    model.config.use_cache = True
    model.eval()

    vocab_size = int(model.config.vocab_size)
    if not 0 <= mask_token_id < vocab_size:
        raise ValueError(
            f"mask_token_id={mask_token_id} is outside model vocabulary {vocab_size}."
        )
    return model, tokenizer, torch.device(
        torch.accelerator.current_accelerator().type,
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def synchronize() -> None:
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


def tokenize_prompt(tokenizer, prompt: str, args: argparse.Namespace) -> torch.LongTensor:
    if args.use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("--use-chat-template was requested but the tokenizer has no chat template.")
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
        }
        try:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                enable_thinking=args.enable_thinking,
                **kwargs,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], **kwargs
            )
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
    else:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ]
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    return encoded


def _root_forward(model, *, lm_positions: Optional[torch.LongTensor] = None, **kwargs):
    """Run the FSDP root while projecting only selected hidden positions."""
    if lm_positions is None:
        return model(**kwargs)

    def select_hidden(_module, inputs):
        hidden_states = inputs[0].index_select(1, lm_positions)
        return (hidden_states, *inputs[1:])

    handle = model.lm_head.register_forward_pre_hook(select_hidden)
    try:
        return model(**kwargs)
    finally:
        handle.remove()


def _crop_cache(cache, length: int) -> None:
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    for layer_index in range(len(cache.key_cache)):
        cache.key_cache[layer_index] = cache.key_cache[layer_index][:, :, :length]
        cache.value_cache[layer_index] = cache.value_cache[layer_index][:, :, :length]
    cache._seen_tokens = length


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        cutoff = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        logits = torch.full_like(logits, -torch.inf).scatter(
            -1, sorted_indices, sorted_logits
        )
    return logits


def _sample(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    mask_token_id: int,
    generator: torch.Generator,
) -> tuple[torch.LongTensor, torch.Tensor]:
    logits = torch.nan_to_num(logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
    if 0 <= mask_token_id < logits.shape[-1]:
        logits[..., mask_token_id] = -torch.inf
    raw_probabilities = F.softmax(logits, dim=-1)
    if temperature == 0.0 or top_k == 1:
        proposed = logits.argmax(dim=-1)
    else:
        filtered = _filter_logits(logits / temperature, top_k, top_p)
        probabilities = F.softmax(filtered, dim=-1)
        proposed = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            num_samples=1,
            generator=generator,
        ).reshape(probabilities.shape[:-1])
    confidence = raw_probabilities.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
    return proposed, confidence


def _transfer_schedule(mask_count: int, steps: int) -> list[int]:
    base, remainder = divmod(mask_count, steps)
    return [base + int(index < remainder) for index in range(steps)]


def _partial_block_text(tokenizer, ids: list[Optional[int]]) -> str:
    parts: list[str] = []
    visible: list[int] = []

    def flush() -> None:
        if visible:
            parts.append(
                tokenizer.decode(
                    visible,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            visible.clear()

    for token_id in ids:
        if token_id is None:
            flush()
            parts.append("<MASK>")
        else:
            visible.append(token_id)
    flush()
    return "".join(parts)


@torch.inference_mode()
def generate_with_trace(
    model,
    tokenizer,
    prompt_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    block_size: int,
    steps_per_block: int,
    threshold: Optional[float],
    mask_token_id: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    seed_mode: str,
    eos_token_id: Optional[int],
    ignore_eos: bool,
    collect_trace: bool,
) -> tuple[torch.LongTensor, DecodeStats, list[dict[str, Any]]]:
    from transformers.cache_utils import DynamicCache
    from mindspeed_llm.fsdp2.models.qwen3.utils import set_diffusion_lm

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("Decode analysis currently expects one prompt with batch size 1.")
    if max_new_tokens <= 0 or block_size <= 0 or steps_per_block <= 0:
        raise ValueError("Generation length, block size, and denoising steps must be positive.")

    device = prompt_ids.device
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt_length = prompt_ids.shape[1]
    stats = DecodeStats(prompt_tokens=prompt_length)
    traces: list[dict[str, Any]] = []
    cache = DynamicCache()

    set_diffusion_lm(model, False)
    prompt_positions = torch.arange(prompt_length, device=device)
    prefill = _root_forward(
        model,
        lm_positions=torch.tensor([prompt_length - 1], device=device),
        input_ids=prompt_ids,
        position_ids=prompt_positions.unsqueeze(0),
        cache_position=prompt_positions,
        past_key_values=cache,
        use_cache=True,
    )
    cache = prefill.past_key_values
    stats.prefill_nfe = 1
    next_token = None
    if seed_mode == "ar":
        next_token, _ = _sample(
            prefill.logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            mask_token_id=mask_token_id,
            generator=generator,
        )
        next_token = next_token.unsqueeze(1)
    del prefill
    set_diffusion_lm(model, True)

    response_blocks: list[torch.LongTensor] = []
    generated = 0
    num_blocks = math.ceil(max_new_tokens / block_size)
    stop_after_block = False

    for block_index in range(num_blocks):
        current_length = min(block_size, max_new_tokens - generated)
        block_start = prompt_length + generated
        block = torch.full(
            (1, current_length),
            mask_token_id,
            dtype=prompt_ids.dtype,
            device=device,
        )
        seeded_positions: list[int] = []
        if seed_mode == "ar" and next_token is not None:
            block[:, 0] = next_token[:, 0]
            seeded_positions = [generated]

        initial_masks = int(block.eq(mask_token_id).sum().item())
        block_steps = min(steps_per_block, initial_masks) if initial_masks else 0
        schedule = (
            _transfer_schedule(initial_masks, block_steps) if block_steps else []
        )
        reveal_steps: list[Optional[int]] = []
        block_trace: dict[str, Any] = {}
        if collect_trace:
            reveal_steps = [
                0 if index == 0 and seeded_positions else None
                for index in range(current_length)
            ]
            block_trace = {
                "block_index": block_index,
                "response_positions": [generated, generated + current_length],
                "seed_mode": seed_mode,
                "seeded_response_positions": seeded_positions,
                "initial_mask_count": initial_masks,
                "denoise_step_budget": block_steps,
                "static_transfer_schedule": schedule,
                "steps": [],
            }

        for step_index in range(block_steps):
            masked = block.eq(mask_token_id)
            if not masked.any().item():
                break
            remaining_before = int(masked.sum().item())
            prefix_cache_length = cache.get_seq_length()
            positions = torch.arange(
                block_start, block_start + current_length, device=device
            )
            denoise = model(
                input_ids=block,
                position_ids=positions.unsqueeze(0),
                cache_position=torch.arange(
                    prefix_cache_length,
                    prefix_cache_length + current_length,
                    device=device,
                ),
                past_key_values=cache,
                use_cache=True,
            )
            cache = denoise.past_key_values
            _crop_cache(cache, prefix_cache_length)
            stats.denoise_nfe += 1
            proposed, confidence = _sample(
                denoise.logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mask_token_id=mask_token_id,
                generator=generator,
            )
            del denoise

            if threshold is None:
                update = torch.zeros_like(masked)
                above_threshold_count = None
            else:
                update = masked & confidence.ge(threshold)
                above_threshold_count = int(update.sum().item())

            minimum = min(schedule[step_index], remaining_before)
            selected_count = int(update.sum().item())
            if selected_count < minimum:
                needed = minimum - selected_count
                candidates = confidence.masked_fill(~masked | update, -torch.inf)
                selected = torch.topk(candidates[0], k=needed).indices
                update[0].scatter_(0, selected, True)
            # Always finish the block at the end of its denoising budget.
            if step_index + 1 == block_steps:
                update |= masked

            committed_indices = update[0].nonzero(as_tuple=True)[0]
            committed_ids = proposed[0, committed_indices]
            committed_confidence = confidence[0, committed_indices]
            block[update] = proposed[update]
            committed = int(committed_indices.numel())
            remaining_after = remaining_before - committed
            stats.committed_per_step.append(committed)

            step_number = stats.denoise_nfe
            if collect_trace:
                committed_indices_list = committed_indices.tolist()
                for local_position in committed_indices_list:
                    if reveal_steps[local_position] is None:
                        reveal_steps[local_position] = step_number
                nullable_ids = [
                    None if token_id == mask_token_id else token_id
                    for token_id in block[0].tolist()
                ]
                block_trace["steps"].append(
                    {
                        "step_in_block": step_index,
                        "global_denoise_forward": step_number,
                        "static_minimum": minimum,
                        "above_threshold_before_fallback": above_threshold_count,
                        "remaining_before": remaining_before,
                        "committed_count": committed,
                        "remaining_after": remaining_after,
                        "committed_response_positions": [
                            generated + index for index in committed_indices_list
                        ],
                        "committed_token_ids": committed_ids.tolist(),
                        "committed_token_text": [
                            tokenizer.decode(
                                [token_id],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            )
                            for token_id in committed_ids.tolist()
                        ],
                        "committed_confidence": [
                            round(float(value), 8)
                            for value in committed_confidence.tolist()
                        ],
                        "block_token_ids": nullable_ids,
                        "block_text_with_masks": _partial_block_text(
                            tokenizer, nullable_ids
                        ),
                    }
                )

        if block.eq(mask_token_id).any().item():
            raise RuntimeError(f"Block {block_index} did not fully denoise.")
        response_blocks.append(block)
        generated += current_length
        stats.completed_blocks += 1

        if collect_trace:
            block_trace["reveal_forward_per_position"] = reveal_steps
            block_trace["final_token_ids"] = block[0].tolist()
            block_trace["final_text"] = tokenizer.decode(
                block[0],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            traces.append(block_trace)

        # JetEngine schedules every completed block once in SAVING state so its
        # causal KV cache is materialized, including the final/EOS block.
        set_diffusion_lm(model, False)
        cache_length = cache.get_seq_length()
        positions = torch.arange(
            block_start, block_start + current_length, device=device
        )
        refresh = _root_forward(
            model,
            lm_positions=torch.tensor([current_length - 1], device=device),
            input_ids=block,
            position_ids=positions.unsqueeze(0),
            cache_position=torch.arange(
                cache_length, cache_length + current_length, device=device
            ),
            past_key_values=cache,
            use_cache=True,
        )
        cache = refresh.past_key_values
        stats.refresh_nfe += 1
        if seed_mode == "ar":
            next_token, _ = _sample(
                refresh.logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mask_token_id=mask_token_id,
                generator=generator,
            )
            next_token = next_token.unsqueeze(1)
        del refresh
        set_diffusion_lm(model, True)

        if not ignore_eos and eos_token_id is not None:
            stop_after_block = bool(block.eq(int(eos_token_id)).any().item())
        if stop_after_block or block_index + 1 == num_blocks:
            break

    response = torch.cat(response_blocks, dim=1)
    if not ignore_eos and eos_token_id is not None:
        eos_positions = response[0].eq(int(eos_token_id)).nonzero(as_tuple=True)[0]
        if eos_positions.numel():
            response = response[:, : int(eos_positions[0].item()) + 1]
    stats.generated_tokens = response.shape[1]
    return response, stats, traces


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def token_repetition_diagnostics(token_ids: list[int], ngram_size: int = 4) -> dict[str, Any]:
    if not token_ids:
        return {
            "unique_token_ratio": 0.0,
            "repeated_ngram_size": ngram_size,
            "repeated_ngram_ratio": 0.0,
        }
    ngrams = [
        tuple(token_ids[index : index + ngram_size])
        for index in range(max(0, len(token_ids) - ngram_size + 1))
    ]
    repeated_ratio = 0.0 if not ngrams else 1.0 - len(set(ngrams)) / len(ngrams)
    return {
        "unique_token_ratio": len(set(token_ids)) / len(token_ids),
        "repeated_ngram_size": ngram_size,
        "repeated_ngram_ratio": repeated_ratio,
    }


def print_trace(result: dict[str, Any]) -> None:
    threshold = result["threshold"]
    threshold_label = "fixed" if threshold is None else str(threshold)
    print(
        f"\n===== block_size={result['block_size']} "
        f"threshold={threshold_label} =====",
        flush=True,
    )
    for block in result["trace"]:
        print(
            f"[block {block['block_index']}] positions={block['response_positions']} "
            f"seed={block['seeded_response_positions']} "
            f"schedule={block['static_transfer_schedule']}",
            flush=True,
        )
        for step in block["steps"]:
            print(
                f"  step={step['step_in_block']} "
                f"commit={step['committed_count']} "
                f"remain={step['remaining_after']} "
                f"positions={step['committed_response_positions']} "
                f"text={step['block_text_with_masks']!r}",
                flush=True,
            )
    stats = result["generation_stats"]
    print(
        f"TPF_total={stats['tpf_total']:.4f}, "
        f"TPF_decode={stats['tpf_decode']:.4f}, "
        f"TPF_denoise_only={stats['tpf_denoise_only']:.4f} "
        f"(tokens={stats['generated_tokens']}, total_nfe={stats['total_nfe']}, "
        f"prefill={stats['prefill_nfe']}, denoise={stats['denoise_nfe']}, "
        f"refresh={stats['refresh_nfe']})",
        flush=True,
    )
    print(
        f"latency_mean={result['latency']['mean_ms']:.2f} ms, "
        f"tokens/s={result['latency']['tokens_per_second']:.3f}",
        flush=True,
    )
    print("Generated text:\n" + result["generated_text"], flush=True)


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.steps_per_block < 0:
        raise ValueError("--steps-per-block must be non-negative.")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    if args.top_k < 0 or not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-k must be non-negative and --top-p must be in (0,1].")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be non-negative and --repeats must be positive.")

    block_sizes = parse_block_sizes(args.block_sizes)
    thresholds = parse_thresholds(args.thresholds)
    run_config, run_config_path = load_run_config(args)
    paper_alignment = validate_paper_alignment(
        args, block_sizes, thresholds, run_config
    )
    checkpoint, model_path, mask_token_id = resolve_paths_and_ids(args, run_config)
    model, tokenizer, device = initialize_and_load_model(
        args, run_config, checkpoint, model_path, mask_token_id
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    prompt_values = [args.prompt if rank == 0 else None]
    dist.broadcast_object_list(prompt_values, src=0)
    prompt = prompt_values[0]
    prompt_ids = tokenize_prompt(tokenizer, prompt, args).to(device)
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and prompt_ids.shape[1] + args.max_new_tokens > int(
        max_positions
    ):
        raise ValueError(
            f"Prompt ({prompt_ids.shape[1]}) + generation ({args.max_new_tokens}) "
            f"exceeds max_position_embeddings={max_positions}."
        )

    output_json = Path(args.output_json) if args.output_json else _run_dir(checkpoint) / "dopd_decode_analysis.json"
    results: list[dict[str, Any]] = []
    generation_common = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt_ids": prompt_ids,
        "max_new_tokens": args.max_new_tokens,
        "mask_token_id": mask_token_id,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "seed_mode": args.seed_mode,
        "eos_token_id": tokenizer.eos_token_id,
        "ignore_eos": args.ignore_eos,
    }

    for block_size in block_sizes:
        steps_per_block = args.steps_per_block or block_size
        for threshold in thresholds:
            for _ in range(args.warmup):
                generate_with_trace(
                    **generation_common,
                    block_size=block_size,
                    steps_per_block=steps_per_block,
                    threshold=threshold,
                    collect_trace=False,
                )

            timings_ms: list[float] = []
            measured_stats: Optional[DecodeStats] = None
            for _ in range(args.repeats):
                dist.barrier()
                synchronize()
                started = time.perf_counter()
                _, measured_stats, _ = generate_with_trace(
                    **generation_common,
                    block_size=block_size,
                    steps_per_block=steps_per_block,
                    threshold=threshold,
                    collect_trace=False,
                )
                synchronize()
                dist.barrier()
                timings_ms.append((time.perf_counter() - started) * 1000.0)

            # Trace collection is deliberately outside latency measurement: its
            # per-step device-to-host copies and tokenizer.decode calls are costly.
            response, trace_stats, trace = generate_with_trace(
                **generation_common,
                block_size=block_size,
                steps_per_block=steps_per_block,
                threshold=threshold,
                collect_trace=(rank == 0),
            )
            if measured_stats is None:
                raise RuntimeError("No measured generation result was produced.")

            if rank == 0:
                generated_ids = response[0].tolist()
                generated_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                mean_ms = statistics.fmean(timings_ms)
                result = {
                    "block_size": block_size,
                    "steps_per_block": steps_per_block,
                    "threshold": threshold,
                    "threshold_mode": "fixed_quota" if threshold is None else "dynamic_with_quota_fallback",
                    "seed_mode": args.seed_mode,
                    "generation_stats": trace_stats.to_dict(),
                    "latency": {
                        "measurement_scope": (
                            "Includes synchronized prompt prefill and generation; "
                            "excludes model load, tokenization, and trace rendering."
                        ),
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "samples_ms": timings_ms,
                        "mean_ms": mean_ms,
                        "p50_ms": percentile(timings_ms, 0.50),
                        "p95_ms": percentile(timings_ms, 0.95),
                        "tokens_per_second": trace_stats.generated_tokens / (mean_ms / 1000.0),
                    },
                    "generated_token_ids": generated_ids,
                    "generated_text": generated_text,
                    "repetition": token_repetition_diagnostics(generated_ids),
                    "trace": trace,
                }
                results.append(result)
                if not args.no_print_trace:
                    print_trace(result)

    if rank == 0:
        payload = {
            "schema_version": 2,
            "checkpoint": str(checkpoint),
            "base_model_path": str(model_path),
            "dopd_config_json": str(run_config_path) if run_config_path else None,
            "world_size": world_size,
            "prompt": prompt,
            "prompt_token_ids": prompt_ids[0].tolist(),
            "prompt_tokens": prompt_ids.shape[1],
            "use_chat_template": args.use_chat_template,
            "enable_thinking": args.enable_thinking,
            "max_new_tokens": args.max_new_tokens,
            "mask_token_id": mask_token_id,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
            "ignore_eos": args.ignore_eos,
            "paper_alignment": paper_alignment,
            "tpf_definition": (
                "tpf_total = realised generated tokens (including EOS when emitted) / "
                "all model forwards (prefill + denoise + per-block SAVING refresh); "
                "tpf_decode excludes prefill; tpf_denoise_only excludes both prefill "
                "and SAVING refresh. TPF is a local diagnostic, not a metric reported "
                "by the d-OPD paper."
            ),
            "threshold_definition": (
                "fixed_quota commits the highest-confidence static quota each step; "
                "dynamic mode also commits every masked position with confidence >= "
                "threshold, while the static quota guarantees progress and the final "
                "step commits any remainder"
            ),
            "results": results,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_json.with_suffix(output_json.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_json)
        print("\n===== comparison summary =====", flush=True)
        print(
            "block threshold tokens nfe tpf_total tpf_decode "
            "tpf_denoise rep4 mean_ms tokens_per_second",
            flush=True,
        )
        for result in results:
            threshold_label = (
                "fixed" if result["threshold"] is None else str(result["threshold"])
            )
            result_stats = result["generation_stats"]
            latency = result["latency"]
            print(
                f"{result['block_size']:>5} {threshold_label:>9} "
                f"{result_stats['generated_tokens']:>6} "
                f"{result_stats['total_nfe']:>3} "
                f"{result_stats['tpf_total']:>9.4f} "
                f"{result_stats['tpf_decode']:>10.4f} "
                f"{result_stats['tpf_denoise_only']:>11.4f} "
                f"{result['repetition']['repeated_ngram_ratio']:>4.1%} "
                f"{latency['mean_ms']:>8.2f} "
                f"{latency['tokens_per_second']:>17.3f}",
                flush=True,
            )
        print(f"\nSaved decode analysis to: {output_json}", flush=True)
    dist.barrier()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
