"""Arguments used only by the FSDP2 diffusion inference entrypoint."""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DiffusionInferenceArguments:
    """Configuration for iterative masked-token generation.

    This class intentionally does not extend the autoregressive
    ``InferenceArguments``.  The two inference algorithms have different
    cache, streaming, stopping, and sampling semantics.
    """

    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Number of masked generation positions appended after the prompt."},
    )
    diffusion_steps: int = field(
        default=128,
        metadata={"help": "Total denoising-step budget distributed across generation blocks."},
    )
    block_length: int = field(
        default=32,
        metadata={
            "help": "Number of generated tokens in each diffusion block. The final block may be shorter."
        },
    )
    mask_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Mask token id. Defaults to tokenizer.mask_token_id."},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature. Zero selects argmax deterministically."},
    )
    top_k: int = field(default=0, metadata={"help": "Keep only the top-k logits; zero disables filtering."})
    top_p: float = field(default=1.0, metadata={"help": "Nucleus sampling threshold."})
    schedule: Literal["linear", "cosine"] = field(
        default="linear", metadata={"help": "Schedule controlling how many masked positions are committed."}
    )
    selection: Literal["low_confidence", "random"] = field(
        default="low_confidence",
        metadata={"help": "Positions retained for future denoising are low-confidence or random."},
    )
    attention_mode: Literal["block_causal", "padding"] = field(
        default="block_causal",
        metadata={
            "help": "block_causal builds a 4D additive mask with bidirectional attention inside each block; "
            "padding sends a 2D valid-token mask for models that implement block attention internally."
        },
    )
    seed: int = field(default=42, metadata={"help": "Sampling seed used identically on every rank."})
    prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Generate once for this prompt. If omitted, start interactive mode."},
    )
    use_chat_template: bool = field(
        default=True,
        metadata={"help": "Apply the tokenizer chat template when one is available."},
    )
    timestep_arg: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional model forward keyword receiving a [batch] normalized timestep tensor, "
            "for example diffusion_t."
        },
    )
    block_index_arg: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional model forward keyword receiving the current zero-based block index as [batch]."
        },
    )
    sync_tokens: bool = field(
        default=True,
        metadata={"help": "Broadcast updated tokens from global rank 0 after every denoising step."},
    )

    def __post_init__(self):
        if self.max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must be strictly positive.")
        if self.diffusion_steps <= 0:
            raise ValueError("`diffusion_steps` must be strictly positive.")
        if self.block_length <= 0:
            raise ValueError("`block_length` must be strictly positive.")
        num_blocks = (self.max_new_tokens + self.block_length - 1) // self.block_length
        if self.diffusion_steps < num_blocks:
            raise ValueError(
                f"`diffusion_steps` ({self.diffusion_steps}) must be at least the number of blocks ({num_blocks})."
            )
        if self.temperature < 0:
            raise ValueError("`temperature` must be non-negative.")
        if self.top_k < 0:
            raise ValueError("`top_k` must be non-negative.")
        if not 0 < self.top_p <= 1:
            raise ValueError("`top_p` must be in (0, 1].")
