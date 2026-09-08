"""CLI-facing inferencer for the independent FSDP2 diffusion engine."""

import torch.distributed as dist

from mindspeed_llm.fsdp2.utils.logging import get_logger
from .engine.diffusion_engine import DiffusionEngine


logger = get_logger(__name__)


class DiffusionInferencer:
    def __init__(self, model, tokenizer, args):
        self.engine = DiffusionEngine(model, tokenizer, args)
        self.args = args
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def _broadcast_prompt(self, prompt):
        if not dist.is_initialized():
            return prompt
        values = [prompt]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def run_once(self, prompt: str):
        prompt = self._broadcast_prompt(prompt if self.rank == 0 else None)
        result = self.engine.generate([prompt])
        if self.rank == 0:
            logger.info_plain_rank0(result.texts[0])
            logger.info_rank0(f"> Diffusion blocks: {result.blocks}, steps: {result.steps}")
        return result

    def run_interactive(self):
        logger.info_rank0(">>> Diffusion inference mode. Type 'exit' to quit.")
        while True:
            if self.rank == 0:
                try:
                    prompt = input("\nUser: ").strip()
                except EOFError:
                    prompt = "exit"
            else:
                prompt = None
            prompt = self._broadcast_prompt(prompt)
            if prompt.lower() in ("exit", "quit"):
                break
            if not prompt:
                continue
            result = self.engine.generate([prompt])
            if self.rank == 0:
                logger.info_plain_rank0(f"Assistant: {result.texts[0]}")
                logger.info_rank0(f"> Diffusion blocks: {result.blocks}, steps: {result.steps}")
