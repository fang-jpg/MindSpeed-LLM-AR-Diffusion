"""Standalone FSDP2 Block Diffusion LM inference entrypoint.

The existing autoregressive ``inference_fsdp2.py`` is intentionally not used
or modified so that AR generation and diffusion generation remain isolated.
"""

import os
import types
from dataclasses import dataclass, field

import torch
from transformers.utils import is_torch_npu_available

from mindspeed.fsdp.utils.torch_patch import apply_hccl_premul_sum_patch
from mindspeed_llm.fsdp2.data.tokenizer import TokenizerFactory
from mindspeed_llm.fsdp2.inference.diffusion_inferencer import DiffusionInferencer
from mindspeed_llm.fsdp2.models.model_factory import ModelFactory
from mindspeed_llm.fsdp2.utils.arguments import (
    ModelArguments,
    OptimizationArguments,
    ParallelArguments,
    fsdp2_parse_args,
)
from mindspeed_llm.fsdp2.utils.device import set_accelerator_compatible
from mindspeed_llm.fsdp2.utils.diffusion_arguments import DiffusionInferenceArguments
from mindspeed_llm.fsdp2.utils.global_vars import set_args
from mindspeed_llm.fsdp2.utils.logging import get_logger, setup_global_logging


logger = get_logger(__name__)


@dataclass
class DiffusionArguments:
    model: ModelArguments = field(default_factory=ModelArguments)
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    inference: DiffusionInferenceArguments = field(default_factory=DiffusionInferenceArguments)
    optimization: OptimizationArguments = field(default_factory=OptimizationArguments)


class AutoDiffusionInferencer:
    def __init__(self):
        root_args = fsdp2_parse_args(DiffusionArguments)
        self.model_args = root_args.model
        self.parallel_args = root_args.parallel
        self.inference_args = root_args.inference
        self.args = types.SimpleNamespace(
            **{
                key: value
                for namespace in (
                    root_args.model,
                    root_args.parallel,
                    root_args.inference,
                    root_args.optimization,
                )
                for key, value in namespace.__dict__.items()
            }
        )
        set_args(self.args)
        setup_global_logging()
        self._initialize_distributed()

        logger.info_rank0("> Building tokenizer for diffusion inference...")
        self.tokenizer = TokenizerFactory.create(self.model_args)
        self.parallel_args.recompute = False
        logger.info_rank0("> Building FSDP2 model for diffusion inference...")
        self.model = ModelFactory.create(self.model_args, self.parallel_args)
        self.inferencer = DiffusionInferencer(self.model, self.tokenizer, self.inference_args)

    @staticmethod
    def _initialize_distributed():
        if is_torch_npu_available():
            fallback = torch.npu
            backend = "hccl"
            apply_hccl_premul_sum_patch()
        elif torch.cuda.is_available():
            fallback = torch.cuda
            backend = "nccl"
        else:
            raise RuntimeError("FSDP2 diffusion inference requires an NPU or CUDA device.")
        set_accelerator_compatible(fallback)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.accelerator.set_device_index(local_rank)
        torch.accelerator.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)
        logger.info_rank0(f"> Distributed initialized. World size: {torch.distributed.get_world_size()}")

    def run(self):
        if self.inference_args.prompt is not None:
            return self.inferencer.run_once(self.inference_args.prompt)
        return self.inferencer.run_interactive()


def main():
    try:
        AutoDiffusionInferencer().run()
    except KeyboardInterrupt:
        logger.info_rank0("\n> Diffusion inference interrupted.")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
