"""Configuration owned by the isolated d-OPD reproduction."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DOPDArguments:
    """Method and rollout parameters for d-OPD.

    Training, optimizer, dataset, and FSDP settings continue to use the normal
    MindSpeed-LLM argument groups.  Keeping method-specific options here avoids
    changing the existing Nemotron/Qwen3 diffusion implementation.
    """

    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Frozen Qwen3 AR teacher. Defaults to model.model_name_or_path."},
    )
    student_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Optional MindSpeed FSDP2 checkpoint used to initialize the student."},
    )
    train_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "官方 OPDLM JSON/JSONL 训练集；启用后绕过 Megatron CPT 加载器。"},
    )
    expected_train_records: int = field(
        default=61816,
        metadata={"help": "如果官方训练集的记录数不等于该值，则立即报错。"},
    )
    strict_paper_alignment: bool = field(
        default=False,
        metadata={"help": "启动时校验公开 d-OPD 论文及原仓库中的超参数。"},
    )
    active_ranks_per_optimizer_step: int = field(
        default=0,
        metadata={
            "help": (
                "每个优化器步骤中贡献非零梯度的 rank 数；0 表示使用 WORLD_SIZE。"
                "在 16 rank 训练中设为 8，可保持论文中的全局优化 batch 为 8。"
            )
        },
    )
    mask_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Diffusion mask id. Defaults to the tokenizer/model configuration."},
    )
    prompt_tokens: int = field(
        default=2000,
        metadata={"help": "Maximum number of tokens taken from each Megatron sample as the prompt."},
    )
    min_response_tokens: int = field(
        default=100,
        metadata={"help": "Initial rollout length."},
    )
    max_response_tokens: int = field(
        default=4000,
        metadata={"help": "Maximum rollout length after curriculum warmup."},
    )
    response_schedule_steps: int = field(
        default=100,
        metadata={"help": "On-policy iterations used by the cosine response-length curriculum."},
    )
    prompts_per_iteration: int = field(
        default=128,
        metadata={"help": "Expected global number of on-policy trajectories per optimizer update."},
    )
    block_size: int = field(default=4, metadata={"help": "Student block length."})
    denoise_steps: int = field(default=4, metadata={"help": "Denoising steps per generated block."})
    temperature: float = field(default=1.0, metadata={"help": "Rollout sampling temperature."})
    top_k: int = field(default=0, metadata={"help": "Rollout top-k; zero disables filtering."})
    top_p: float = field(default=1.0, metadata={"help": "Rollout nucleus sampling threshold."})
    remasking_strategy: str = field(
        default="low_confidence_dynamic",
        metadata={"help": "low_confidence_dynamic or low_confidence_static."},
    )
    dynamic_threshold: float = field(
        default=0.9,
        metadata={"help": "Confidence threshold used by dynamic remasking."},
    )
    correction_strength: float = field(
        default=0.5,
        metadata={"help": "Multiplier on relative future log-likelihood scores."},
    )
    candidate_top_k: int = field(
        default=16,
        metadata={"help": "Top-k from teacher and student whose union is future-scored."},
    )
    loss_top_k: int = field(
        default=16,
        metadata={"help": "Teacher top-k support used by the token-level forward KL."},
    )
    require_visible_future: bool = field(
        default=True,
        metadata={"help": "Only correct a target when the sampled state reveals a later block token."},
    )
    correctness_gate: bool = field(
        default=False,
        metadata={"help": "Apply future correction only to trajectories accepted by verifier."},
    )
    rollout_audit_iterations: int = field(
        default=0,
        metadata={
            "help": (
                "保存前 N 个外层迭代的完整 rollout、verifier 判定和失败原因；"
                "0 表示关闭。"
            )
        },
    )
    verifier: Optional[str] = field(
        default=None,
        metadata={"help": "Optional module:function verifier used when correctness_gate is true."},
    )
    exclude_eos_from_loss: bool = field(
        default=True,
        metadata={"help": "Exclude EOS and positions after the first EOS from distillation."},
    )
    score_chunk_size: int = field(
        default=16,
        metadata={"help": "Maximum candidate branches scored in one teacher call."},
    )
    teacher_cache_max_bytes: int = field(
        default=17179869184,
        metadata={"help": "Upper bound used to shrink candidate branch batches (16 GiB by default)."},
    )

    def __post_init__(self):
        positive = (
            "expected_train_records",
            "prompt_tokens",
            "min_response_tokens",
            "max_response_tokens",
            "response_schedule_steps",
            "prompts_per_iteration",
            "block_size",
            "denoise_steps",
            "candidate_top_k",
            "loss_top_k",
            "score_chunk_size",
            "teacher_cache_max_bytes",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"dopd.{name} must be a positive integer")
        if self.block_size < 2:
            raise ValueError("dopd.block_size must be at least 2 for future-aware correction")
        if self.min_response_tokens > self.max_response_tokens:
            raise ValueError("dopd.min_response_tokens cannot exceed dopd.max_response_tokens")
        if self.temperature < 0:
            raise ValueError("dopd.temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("dopd.top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("dopd.top_p must lie in (0, 1]")
        if not 0 <= self.dynamic_threshold <= 1:
            raise ValueError("dopd.dynamic_threshold must lie in [0, 1]")
        if self.remasking_strategy not in {"low_confidence_dynamic", "low_confidence_static"}:
            raise ValueError(
                "dopd.remasking_strategy must be low_confidence_dynamic or low_confidence_static"
            )
        if self.correction_strength < 0:
            raise ValueError("dopd.correction_strength must be non-negative")
        if self.correctness_gate and not self.verifier:
            raise ValueError("dopd.correctness_gate=true requires dopd.verifier=module:function")
        if self.active_ranks_per_optimizer_step < 0:
            raise ValueError("dopd.active_ranks_per_optimizer_step cannot be negative")
        if self.rollout_audit_iterations < 0:
            raise ValueError("dopd.rollout_audit_iterations cannot be negative")

