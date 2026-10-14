#!/usr/bin/env python3
"""Collect paper-style serial denoising trajectories for sampler training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import torch

from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.data import (
    SamplerShardWriter,
    iter_megatron_documents,
    load_megatron_data_config,
)
from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.features import (
    EmbeddingPCA,
    SamplerFeatureExtractor,
)
from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.runtime import (
    DTYPE_MAP,
    detect_device,
    load_diffusion_model,
    load_tokenizer,
)
from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.trajectory import (
    collect_block_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-jsonl",
        help="UTF-8 JSONL containing prompt/response or chat records",
    )
    input_group.add_argument(
        "--data-config",
        help=(
            "MindSpeed FSDP2 YAML whose data.dataset.file_name contains "
            "Megatron IndexedDataset prefixes"
        ),
    )
    input_group.add_argument(
        "--input-mg-prefixes",
        nargs="+",
        help="Megatron IndexedDataset prefixes (without .bin/.idx)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pca-path", required=True)
    parser.add_argument("--config-diffusion", default=None)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--messages-field", default=None)
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[8, 32])
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["confidence", "hybrid"],
        default=["confidence", "hybrid"],
    )
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--records-per-shard", type=int, default=8192)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-blocks-per-example", type=int, default=None)
    parser.add_argument("--max-prefix-tokens", type=int, default=None)
    parser.add_argument("--include-partial-blocks", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=DTYPE_MAP, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
        return "".join(pieces)
    return str(content)


def _encode_record(record: dict[str, Any], tokenizer, args: argparse.Namespace):
    if args.messages_field and args.messages_field in record:
        messages = record[args.messages_field]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        if messages[-1].get("role") != "assistant":
            raise ValueError("the final chat message must be the assistant response")
        response = _content_to_text(messages[-1].get("content", ""))
        prompt_messages = messages[:-1]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.tensor([prompt_ids], dtype=torch.long)
    else:
        prompt = str(record[args.prompt_field])
        response = str(record[args.response_field])
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids
    response_ids = tokenizer(
        response,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    if prompt_ids.shape[1] == 0:
        fallback_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        if fallback_id is None:
            raise ValueError("empty prompt and tokenizer has no BOS/EOS token")
        prompt_ids = torch.tensor([[fallback_id]], dtype=torch.long)
    return prompt_ids.long(), response_ids.long()


def _split_for_key(key: str, validation_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / 2**64
    return "val" if value < validation_fraction else "train"


def _set_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)


def _iter_jsonl_examples(
    args: argparse.Namespace,
    tokenizer,
    rank: int,
    world_size: int,
) -> Iterator[tuple[int, str, torch.LongTensor, torch.LongTensor]]:
    input_path = Path(args.input_jsonl)
    with input_path.open(encoding="utf-8") as stream:
        for sample_index, line in enumerate(stream):
            if args.max_examples is not None and sample_index >= args.max_examples:
                break
            if sample_index % world_size != rank or not line.strip():
                continue
            record = json.loads(line)
            prompt_ids, response_ids = _encode_record(record, tokenizer, args)
            yield (
                sample_index,
                f"jsonl:{input_path}:{sample_index}",
                prompt_ids,
                response_ids,
            )


def _iter_megatron_examples(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Iterator[tuple[int, str, torch.LongTensor, torch.LongTensor]]:
    if args.data_config:
        config = load_megatron_data_config(args.data_config)
        prefixes = config.prefixes
        max_examples = (
            args.max_examples
            if args.max_examples is not None
            else config.max_samples
        )
    else:
        prefixes = tuple(args.input_mg_prefixes)
        max_examples = args.max_examples

    if rank == 0:
        limit_text = "all" if max_examples is None else str(max_examples)
        print(
            f"> Megatron input prefixes={len(prefixes)} "
            f"global_max_examples={limit_text}"
        )
    for document in iter_megatron_documents(
        prefixes,
        rank=rank,
        world_size=world_size,
        max_examples=max_examples,
        seed=args.seed,
    ):
        # CPT IndexedDataset records have no prompt/response boundary.  Keep
        # the first token as a non-empty causal prefix and collect blocks from
        # the remaining continuation without decode/re-tokenize round trips.
        if document.token_ids.numel() < 2:
            continue
        yield (
            document.sample_index,
            f"mg:{document.key}",
            document.token_ids[:1].unsqueeze(0),
            document.token_ids[1:].unsqueeze(0),
        )


def _iter_input_examples(
    args: argparse.Namespace,
    tokenizer,
    rank: int,
    world_size: int,
) -> Iterator[tuple[int, str, torch.LongTensor, torch.LongTensor]]:
    if args.input_jsonl:
        yield from _iter_jsonl_examples(args, tokenizer, rank, world_size)
    else:
        yield from _iter_megatron_examples(args, rank, world_size)


def _validate_token_ids(
    prompt_ids: torch.LongTensor,
    response_ids: torch.LongTensor,
    *,
    vocab_size: int,
    source_key: str,
) -> None:
    tokens = torch.cat([prompt_ids.reshape(-1), response_ids.reshape(-1)])
    if tokens.numel() == 0:
        raise ValueError(f"{source_key}: input contains no tokens")
    minimum = int(tokens.min())
    maximum = int(tokens.max())
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"{source_key}: token ids must be in [0, {vocab_size}), "
            f"observed [{minimum}, {maximum}]"
        )


def main() -> int:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1)")
    if any(length not in {8, 32} for length in args.block_lengths):
        raise ValueError("paper-aligned collection uses block lengths 8 and/or 32")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    device = torch.device(args.device or detect_device(local_rank))
    _set_device(device)
    torch.manual_seed(args.seed + rank)

    print(f"> rank={rank}/{world_size} loading model on {device}")
    model = load_diffusion_model(
        args.model_path,
        device=device,
        dtype=DTYPE_MAP[args.dtype],
        config_diffusion_path=args.config_diffusion,
    )
    tokenizer = load_tokenizer(args.model_path)
    pca = EmbeddingPCA.load(args.pca_path)
    pca_vocab_size = (pca.metadata or {}).get("vocab_size")
    model_vocab_size = int(model.get_input_embeddings().weight.shape[0])
    if pca_vocab_size is not None and int(pca_vocab_size) != model_vocab_size:
        raise ValueError(
            "PCA artifact vocabulary does not match the trajectory model: "
            f"{pca_vocab_size} != {model_vocab_size}"
        )
    feature_extractor = SamplerFeatureExtractor(
        pca=pca,
        embedding_weight=model.get_input_embeddings().weight.detach(),
        invalid_token_id=int(model.mask_token_id),
        table_dtype=torch.float16,
    ).to(device)

    processed_examples = 0
    trajectories = 0
    states = 0
    writer = SamplerShardWriter(
        args.output_dir,
        rank=rank,
        records_per_shard=args.records_per_shard,
    )
    examples = _iter_input_examples(args, tokenizer, rank, world_size)
    for sample_index, sample_key, prompt_ids, response_ids in examples:
        _validate_token_ids(
            prompt_ids,
            response_ids,
            vocab_size=model_vocab_size,
            source_key=sample_key,
        )
        prompt_ids = prompt_ids.to(device)
        response_ids = response_ids.to(device)

        for block_length in args.block_lengths:
            response_length = response_ids.shape[1]
            starts = list(range(0, response_length, block_length))
            if not args.include_partial_blocks:
                starts = [
                    start
                    for start in starts
                    if start + block_length <= response_length
                ]
            if args.max_blocks_per_example is not None:
                starts = starts[: args.max_blocks_per_example]
            for local_block_index, start in enumerate(starts):
                end = min(start + block_length, response_length)
                target = response_ids[:, start:end]
                if target.shape[1] != block_length:
                    # Variable-length partial blocks do not match the paper's
                    # B=8/32 training setup, so pad only when explicitly asked.
                    pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id
                    if pad_id is None:
                        raise ValueError("partial block requires EOS or PAD token")
                    padding = torch.full(
                        (1, block_length - target.shape[1]),
                        pad_id,
                        dtype=target.dtype,
                        device=device,
                    )
                    target = torch.cat([target, padding], dim=1)
                prefix = torch.cat([prompt_ids, response_ids[:, :start]], dim=1)
                if (
                    args.max_prefix_tokens is not None
                    and prefix.shape[1] > args.max_prefix_tokens
                ):
                    prefix = prefix[:, -args.max_prefix_tokens :]

                for policy in args.policies:
                    trajectory_key = (
                        f"{sample_key}:{block_length}:"
                        f"{local_block_index}:{policy}"
                    )
                    split = _split_for_key(
                        trajectory_key,
                        validation_fraction=args.validation_fraction,
                        seed=args.seed,
                    )
                    collected, _ = collect_block_trajectory(
                        model=model,
                        prefix_ids=prefix,
                        block_length=block_length,
                        feature_extractor=feature_extractor,
                        policy=policy,
                        sample_index=sample_index,
                        block_index=local_block_index,
                        target_ids=target if policy == "hybrid" else None,
                    )
                    for item in collected:
                        writer.add(split, item)
                    trajectories += 1
                    states += len(collected)
        processed_examples += 1
        if args.log_every > 0 and processed_examples % args.log_every == 0:
            print(
                f"> rank={rank} examples={processed_examples} "
                f"trajectories={trajectories} states={states}"
            )

    manifest = writer.close()
    summary = {
        "rank": rank,
        "world_size": world_size,
        "processed_examples": processed_examples,
        "trajectories": trajectories,
        "states": states,
        "manifest": manifest,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
