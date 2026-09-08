"""Independent FSDP2 inference engine for masked diffusion language models."""

from dataclasses import dataclass
import inspect
import os
from typing import Optional, Sequence

import torch
import torch.distributed as dist

from mindspeed_llm.inference.diffusion_generation import MaskDiffusionScheduler, build_diffusion_blocks


@dataclass
class DiffusionGenerationResult:
    texts: list[str]
    token_ids: list[list[int]]
    prompt_lengths: list[int]
    steps: int
    blocks: int


class DiffusionEngine:
    """Run block-autoregressive, intra-block diffusion without HF generate."""

    def __init__(self, model, tokenizer, args):
        self.model = getattr(model, "model", model)
        self.tokenizer = tokenizer
        self.args = args
        self.mask_token_id = self._resolve_mask_token_id(args.mask_token_id)
        self.model.eval()

    def _resolve_mask_token_id(self, configured_id: Optional[int]) -> int:
        mask_token_id = configured_id
        if mask_token_id is None:
            mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "No mask token is configured. Set --inference.mask_token_id or add a mask token to the tokenizer."
            )
        if mask_token_id < 0 or mask_token_id >= len(self.tokenizer):
            raise ValueError(
                f"mask_token_id={mask_token_id} is outside tokenizer vocabulary size {len(self.tokenizer)}"
            )
        return int(mask_token_id)

    def _target_device(self) -> torch.device:
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            accelerator = torch.accelerator.current_accelerator()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return torch.device(accelerator.type, local_rank)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _format_prompts(self, prompts: Sequence[str]) -> list[str]:
        if not self.args.use_chat_template or not getattr(self.tokenizer, "chat_template", None):
            return list(prompts)
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

    def _prepare_inputs(self, prompts: Sequence[str], device: torch.device):
        formatted = self._format_prompts(prompts)
        encoded = self.tokenizer(formatted, add_special_tokens=True, padding=False)
        prompt_ids = encoded["input_ids"]
        if not prompt_ids or any(len(ids) == 0 for ids in prompt_ids):
            raise ValueError("Every prompt must produce at least one token.")

        batch_size = len(prompt_ids)
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], dtype=torch.long, device=device)
        total_lengths = prompt_lengths + self.args.max_new_tokens
        max_length = int(total_lengths.max().item())
        model_config = getattr(self.model, "config", None)
        max_positions = getattr(model_config, "max_position_embeddings", None)
        if max_positions is not None and max_length > max_positions:
            raise ValueError(
                f"Prompt plus generation length {max_length} exceeds model max_position_embeddings={max_positions}."
            )
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

        tokens = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
        padding_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        mutable_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        for row, ids in enumerate(prompt_ids):
            prompt_length = len(ids)
            end = prompt_length + self.args.max_new_tokens
            tokens[row, :prompt_length] = torch.tensor(ids, dtype=torch.long, device=device)
            tokens[row, prompt_length:end] = self.mask_token_id
            padding_mask[row, :end] = 1
            mutable_mask[row, prompt_length:end] = True

        position_ids = padding_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(padding_mask == 0, 0)
        attention_mask = self._build_attention_mask(padding_mask, prompt_lengths)
        return tokens, attention_mask, position_ids, mutable_mask, prompt_lengths

    def _build_attention_mask(self, padding_mask, prompt_lengths):
        if self.args.attention_mode == "padding":
            return padding_mask

        batch_size, sequence_length = padding_mask.shape
        block_ids = torch.full(
            (batch_size, sequence_length), -2, dtype=torch.long, device=padding_mask.device
        )
        for row, prompt_length in enumerate(prompt_lengths.tolist()):
            block_ids[row, :prompt_length] = -1
            for offset in range(self.args.max_new_tokens):
                block_ids[row, prompt_length + offset] = offset // self.args.block_length

        query_blocks = block_ids.unsqueeze(-1)
        key_blocks = block_ids.unsqueeze(-2)
        query_valid = padding_mask.bool().unsqueeze(-1)
        key_valid = padding_mask.bool().unsqueeze(-2)
        prompt_to_prompt = (query_blocks == -1) & (key_blocks == -1)
        generation_to_history = (query_blocks >= 0) & (
            (key_blocks == -1) | ((key_blocks >= 0) & (key_blocks <= query_blocks))
        )
        allowed = query_valid & key_valid & (prompt_to_prompt | generation_to_history)
        # Avoid an all-masked attention row for right-padding queries. Their
        # logits are discarded, but some kernels would otherwise produce NaN.
        diagonal = torch.eye(sequence_length, dtype=torch.bool, device=padding_mask.device).unsqueeze(0)
        allowed |= (~query_valid) & diagonal

        # Additive masks are accepted by diffusion model implementations that
        # bypass the stock causal-mask builder: 0 means visible and the minimum
        # float value means blocked.
        attention_mask = torch.zeros(
            (batch_size, 1, sequence_length, sequence_length),
            dtype=torch.float32,
            device=padding_mask.device,
        )
        return attention_mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(torch.float32).min)

    @staticmethod
    def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            logits = torch.full_like(logits, -torch.inf).scatter(-1, sorted_indices, sorted_logits)
        return logits

    def _sample(self, logits: torch.Tensor, generator: torch.Generator):
        if self.args.temperature == 0:
            probabilities = torch.softmax(logits.float(), dim=-1)
            confidence, proposed = probabilities.max(dim=-1)
            return proposed, confidence

        filtered = self._filter_logits(logits.float() / self.args.temperature, self.args.top_k, self.args.top_p)
        probabilities = torch.softmax(filtered, dim=-1)
        flat = probabilities.view(-1, probabilities.size(-1))
        proposed = torch.multinomial(flat, num_samples=1, generator=generator).view(probabilities.shape[:-1])
        confidence = probabilities.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
        return proposed, confidence

    def _forward_kwargs(self, step, block, tokens, attention_mask, position_ids):
        kwargs = {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "return_dict": True,
        }
        if self.args.timestep_arg:
            kwargs[self.args.timestep_arg] = torch.full(
                (tokens.size(0),), step.timestep, dtype=torch.float32, device=tokens.device
            )
        if self.args.block_index_arg:
            kwargs[self.args.block_index_arg] = torch.full(
                (tokens.size(0),), block.index, dtype=torch.long, device=tokens.device
            )
        return kwargs

    def _validate_model_metadata_args(self):
        configured_args = [self.args.timestep_arg, self.args.block_index_arg]
        configured_args = [name for name in configured_args if name]
        if not configured_args:
            return
        signature = inspect.signature(self.model.forward)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        for name in configured_args:
            if name not in signature.parameters and not accepts_kwargs:
                raise ValueError(f"Model forward does not accept configured metadata argument {name!r}.")

    @staticmethod
    def _block_mutable_mask(all_mutable_mask, prompt_lengths, block):
        block_mask = torch.zeros_like(all_mutable_mask)
        for row, prompt_length in enumerate(prompt_lengths.tolist()):
            start = prompt_length + block.start
            end = prompt_length + block.end
            block_mask[row, start:end] = True
        return all_mutable_mask & block_mask

    @torch.inference_mode()
    def generate(self, prompts: Sequence[str]) -> DiffusionGenerationResult:
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("At least one prompt is required.")
        self._validate_model_metadata_args()

        device = self._target_device()
        if device.type != "cpu":
            torch.accelerator.set_device(device)
        tokens, attention_mask, position_ids, all_mutable_mask, prompt_lengths = self._prepare_inputs(prompts, device)
        generator = torch.Generator(device=device).manual_seed(self.args.seed)
        steps_run = 0
        blocks = build_diffusion_blocks(
            generation_length=self.args.max_new_tokens,
            block_length=self.args.block_length,
            total_steps=self.args.diffusion_steps,
        )

        # Block Diffusion LM: blocks are generated left-to-right, while tokens
        # inside the active block are denoised in parallel. Future blocks stay
        # masked and completed blocks are never modified again.
        for block in blocks:
            mutable_mask = self._block_mutable_mask(all_mutable_mask, prompt_lengths, block)
            initial_counts = mutable_mask.sum(dim=-1)
            scheduler = MaskDiffusionScheduler(
                num_steps=block.num_steps,
                schedule=self.args.schedule,
                selection=self.args.selection,
            )
            for step in scheduler.steps():
                outputs = self.model(
                    **self._forward_kwargs(step, block, tokens, attention_mask, position_ids)
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                if logits.shape[:2] != tokens.shape:
                    raise RuntimeError(
                        f"Diffusion model must return logits for every input position; got {tuple(logits.shape)} "
                        f"for tokens {tuple(tokens.shape)}."
                    )

                proposed, confidence = self._sample(logits, generator)
                confidence = confidence.masked_fill(~mutable_mask, -torch.inf)
                update_mask = scheduler.select_updates(
                    confidence, mutable_mask, initial_counts, step.index, generator=generator
                )
                tokens[update_mask] = proposed[update_mask]
                mutable_mask &= ~update_mask
                all_mutable_mask &= ~update_mask
                steps_run += 1

                if dist.is_initialized() and self.args.sync_tokens:
                    dist.broadcast(tokens, src=0)
                    dist.broadcast(mutable_mask, src=0)
                    dist.broadcast(all_mutable_mask, src=0)

                block_done = torch.tensor([not mutable_mask.any()], dtype=torch.uint8, device=device)
                if dist.is_initialized():
                    dist.all_reduce(block_done, op=dist.ReduceOp.MIN)
                if bool(block_done.item()):
                    break

            if mutable_mask.any():
                raise RuntimeError(
                    f"Block {block.index} still contains masked positions after {block.num_steps} steps."
                )

        generated_ids = []
        for row, prompt_length in enumerate(prompt_lengths.tolist()):
            ids = tokens[row, prompt_length : prompt_length + self.args.max_new_tokens].tolist()
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None and eos_token_id in ids:
                ids = ids[: ids.index(eos_token_id)]
            generated_ids.append(ids)
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return DiffusionGenerationResult(
            texts=texts,
            token_ids=generated_ids,
            prompt_lengths=prompt_lengths.tolist(),
            steps=steps_run,
            blocks=len(blocks),
        )
