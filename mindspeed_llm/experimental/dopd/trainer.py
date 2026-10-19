"""MindSpeed FSDP2 training loop for the isolated d-OPD reproduction."""

from __future__ import annotations

import contextlib
import copy
import importlib
import inspect
import json
import math
import os
import random
import time
import types
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config

from mindspeed_llm.experimental.dopd.config import DOPDArguments
from mindspeed_llm.experimental.dopd.core import (
    active_rank_training_schedule,
    cosine_response_length,
    response_valid_mask,
    sample_intermediate_state,
)
from mindspeed_llm.experimental.dopd.opdlm_data import (
    load_opdlm_records,
    preprocess_opdlm_records,
    records_for_iteration,
    shuffled_epoch,
    tokenize_opdlm_prompt,
)
from mindspeed_llm.experimental.dopd.rollout import block_diffusion_rollout
from mindspeed_llm.experimental.dopd.scoring import compute_dopd_loss
from mindspeed_llm.fsdp2.checkpoint.checkpoint_manager import CheckpointManager
from mindspeed_llm.fsdp2.checkpoint.utils import cleanup_old_checkpoints
from mindspeed_llm.fsdp2.data.data_factory import DataFactory
from mindspeed_llm.fsdp2.data.template import get_template_and_fix_tokenizer
from mindspeed_llm.fsdp2.data.tokenizer import TokenizerFactory
from mindspeed_llm.fsdp2.distributed.clip_grad_norm import clip_grad_norm
from mindspeed_llm.fsdp2.models.model_factory import ModelFactory
from mindspeed_llm.fsdp2.optim.optimizer import OptimizerFactory
from mindspeed_llm.fsdp2.optim.scheduler import SchedulerFactory
from mindspeed_llm.fsdp2.utils.arguments import (
    DataArguments,
    ModelArguments,
    OptimizationArguments,
    ParallelArguments,
    TrainingArguments,
    fsdp2_parse_args,
)
from mindspeed_llm.fsdp2.utils.global_vars import set_args
from mindspeed_llm.fsdp2.utils.logging import get_logger


logger = get_logger(__name__)


class DOPDA2DQwen3Config(Qwen3Config):
    """与官方 ``dopd.modeling.A2DQwen3Config`` 等价的本地配置注册。"""

    model_type = "a2d-qwen3"


def _register_a2d_qwen3_config() -> None:
    """让 MindSpeed 的 ModelFactory 能读取官方 a2d-init 的 config.json。"""

    try:
        AutoConfig.register("a2d-qwen3", DOPDA2DQwen3Config, exist_ok=True)
    except TypeError:
        # 兼容没有 exist_ok 参数的较旧 transformers。
        try:
            AutoConfig.register("a2d-qwen3", DOPDA2DQwen3Config)
        except ValueError:
            pass


_register_a2d_qwen3_config()


def _read_local_model_config(path: str, label: str) -> dict:
    model_path = Path(path)
    config_path = model_path / "config.json"
    if not model_path.is_dir() or not config_path.is_file():
        raise ValueError(f"{label} must be a local model directory containing config.json: {path}")
    try:
        with config_path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} config: {config_path}: {error}") from error


def _validate_paper_model_pair(student_path: str, teacher_path: str) -> None:
    """复现官方 train.py 对 A2D 学生和 AR 教师的启动前校验。"""

    student_config = _read_local_model_config(student_path, "d-OPD student")
    teacher_config = _read_local_model_config(teacher_path, "d-OPD teacher")
    if student_config.get("model_type") != "a2d-qwen3":
        raise ValueError(
            "strict paper alignment requires an a2d-qwen3 student "
            f"(got {student_config.get('model_type')!r} from {student_path})"
        )
    if teacher_config.get("model_type") != "qwen3":
        raise ValueError(
            "strict paper alignment requires a qwen3 autoregressive teacher "
            f"(got {teacher_config.get('model_type')!r} from {teacher_path})"
        )
    shape_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    )
    mismatches = [
        f"{name}: student={student_config.get(name)!r}, teacher={teacher_config.get(name)!r}"
        for name in shape_fields
        if student_config.get(name) != teacher_config.get(name)
    ]
    if mismatches:
        raise ValueError(
            "d-OPD student and teacher architectures do not match:\n- "
            + "\n- ".join(mismatches)
        )


@dataclass
class Arguments:
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)
    optimization: OptimizationArguments = field(default_factory=OptimizationArguments)
    dopd: DOPDArguments = field(default_factory=DOPDArguments)


def _device() -> torch.device:
    accelerator = torch.accelerator.current_accelerator()
    return torch.device(accelerator.type, int(os.environ.get("LOCAL_RANK", "0")))


def _empty_cache() -> None:
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()


def _load_verifier(spec):
    if not spec:
        return None
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise ValueError("dopd.verifier must identify a callable")
    validate_runtime = getattr(module, "validate_runtime", None)
    if callable(validate_runtime):
        validate_runtime()
    return function


def _scalar(value) -> float:
    if isinstance(value, DTensor):
        value = value.full_tensor()
    return float(value.item() if hasattr(value, "item") else value)


class DOPDTrainer:
    def __init__(self, args: Arguments):
        self.root_args = args
        self.model_args = args.model
        self.data_args = args.data
        self.parallel_args = args.parallel
        self.training_args = args.training
        self.optimization_args = args.optimization
        self.dopd_args = args.dopd
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = _device()
        self._validate_runtime_config()
        self.output_dir = Path(self.training_args.output_dir)
        if (
            self.training_args.resume_from_checkpoint is None
            and (self.output_dir / "checkpoints").exists()
        ):
            raise FileExistsError(
                f"{self.output_dir} already contains checkpoints; use a new output_dir or resume explicitly"
            )
        self.verifier = _load_verifier(self.dopd_args.verifier)
        self.tokenizer = TokenizerFactory.create(self.model_args)
        self.opdlm_records = None
        self.opdlm_total_iterations = None
        self.opdlm_preprocessing_report = None
        if self.dopd_args.train_data_path:
            records = load_opdlm_records(
                self.dopd_args.train_data_path,
                expected_records=self.dopd_args.expected_train_records,
            )
            records = shuffled_epoch(records, self.training_args.seed)
            original_record_count = len(records)
            self.opdlm_total_iterations = math.ceil(
                original_record_count / self.dopd_args.prompts_per_iteration
            )
            preprocessing_payload = [None]
            if self.rank == 0:
                try:
                    kept_indices, report = preprocess_opdlm_records(
                        records,
                        self.tokenizer,
                        self.dopd_args.prompt_tokens,
                        batch_size=self.data_args.preprocessing_batch_size or 1000,
                    )
                    preprocessing_payload[0] = {
                        "ok": True,
                        "kept_indices": kept_indices,
                        "report": report,
                    }
                except Exception as error:
                    preprocessing_payload[0] = {
                        "ok": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
            dist.broadcast_object_list(preprocessing_payload, src=0)
            preprocessing_result = preprocessing_payload[0]
            if not preprocessing_result["ok"]:
                raise RuntimeError(
                    "OPDLM preprocessing failed on rank 0: "
                    f"{preprocessing_result['error_type']}: {preprocessing_result['error']}"
                )
            self.opdlm_preprocessing_report = preprocessing_result["report"]
            self.opdlm_records = [
                records[index] for index in preprocessing_result["kept_indices"]
            ]
            expected_iterations = self.opdlm_total_iterations
            scheduled_prompts = (
                expected_iterations * self.dopd_args.prompts_per_iteration
            )
            self.opdlm_preprocessing_report.update(
                {
                    "scheduling_policy": "fixed_global_batch_cycle_padding",
                    "scheduled_iterations": expected_iterations,
                    "scheduled_prompts": scheduled_prompts,
                    "repeated_prompts": scheduled_prompts - len(self.opdlm_records),
                }
            )
            if (
                self.dopd_args.strict_paper_alignment
                and self.training_args.max_steps != expected_iterations
            ):
                raise ValueError(
                    "A one-epoch OPDLM run requires "
                    f"training.max_steps={expected_iterations}, got {self.training_args.max_steps}"
                )
            report = self.opdlm_preprocessing_report
            logger.info_rank0(
                f"> OPDLM preprocessing (truncation disabled): "
                f"original={report['original_records']}, "
                f"retained={report['retained_records']}, "
                f"dropped_over_{report['max_prompt_tokens']}={report['dropped_records']}, "
                f"longest={report['longest_prompt_tokens']}, "
                f"cycle_padding={report['repeated_prompts']}, "
                f"iterations={expected_iterations}"
            )

        logger.info_rank0("> Building d-OPD diffusion student with MindSpeed FSDP2")
        self.student_engine = ModelFactory.create(self.model_args, self.parallel_args)
        self.student = self.student_engine.model
        self.template = None
        if not self.dopd_args.train_data_path:
            self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.mask_token_id = self._resolve_mask_token_id()
        self.student.mask_token_id = self.mask_token_id
        self.student.config.mask_token_id = self.mask_token_id
        max_positions = getattr(self.student.config, "max_position_embeddings", None)
        # 训练输入按 [提示词, 干净回答, 加噪回答] 拼接。
        requested_context = (
            self.dopd_args.prompt_tokens + 2 * self.dopd_args.max_response_tokens
        )
        if max_positions is not None and requested_context > int(max_positions):
            raise ValueError(
                f"d-OPD requests {requested_context} tokens but the model supports {max_positions}"
            )

        if self.training_args.resume_from_checkpoint is None and self.dopd_args.student_checkpoint:
            logger.info_rank0("> Loading Nemotron_Diffusion student checkpoint")
            CheckpointManager.load(
                self.dopd_args.student_checkpoint,
                {"model": self.student_engine},
            )

        self.optimizer = OptimizerFactory.create(
            model=self.student_engine,
            ep_size=self.parallel_args.ep_size,
            lr=self.training_args.lr,
            optimizer_type=self.training_args.optimizer,
            weight_decay=self.training_args.weight_decay,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            adam_epsilon=self.training_args.adam_epsilon,
        )
        self.scheduler = SchedulerFactory.create(
            optimizer=self.optimizer,
            train_steps=self.training_args.max_steps * self.inner_steps_per_iteration,
            lr=self.training_args.lr,
            lr_decay_style=self.training_args.lr_scheduler_type,
            lr_warmup_ratio=self.training_args.warmup_ratio,
            lr_min=self.training_args.min_lr,
        )
        self.data_manager = None
        self.dataloader = None
        self.data_iterator = None
        if self.opdlm_records is None:
            data_training_args = copy.copy(self.training_args)
            # 旧版 CPT 数据路径仅保留给已有的消融实验使用。
            data_training_args.max_steps *= self.inner_steps_per_iteration
            self.data_manager = DataFactory.create(
                data_manager_type=self.data_args.data_manager_type,
                model_args=self.model_args,
                data_args=self.data_args,
                training_args=data_training_args,
                parallel_args=self.parallel_args,
                stage=self.training_args.stage,
                tokenizer=self.tokenizer,
                template=self.template,
            )
            self.dataloader = self.data_manager.create_train_dataloader()
            self.data_iterator = iter(self.dataloader)
        self.data_epoch = 0
        self.data_batches_consumed = 0
        self.teacher = self._load_teacher()
        self.global_step = 0
        self.optimizer_step = 0
        self._load_resume_state()

        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "dopd_config.json").open("w", encoding="utf-8") as stream:
                json.dump(asdict(self.root_args), stream, indent=2, ensure_ascii=False)
            if self.opdlm_preprocessing_report is not None:
                with (self.output_dir / "opdlm_preprocessing_report.json").open(
                    "w", encoding="utf-8"
                ) as stream:
                    json.dump(
                        self.opdlm_preprocessing_report,
                        stream,
                        indent=2,
                        ensure_ascii=False,
                    )
        dist.barrier()

    def _validate_runtime_config(self):
        if self.model_args.model_id != "qwen3_diffusion":
            raise ValueError("model.model_id must be qwen3_diffusion")
        if self.training_args.stage != "pt":
            raise ValueError("d-OPD requires training.stage=pt")
        if not self.dopd_args.train_data_path and self.data_args.data_manager_type != "mg":
            raise ValueError("the legacy Nemotron data path requires data_manager_type=mg")
        if self.training_args.per_device_train_batch_size != 1:
            raise ValueError("d-OPD currently requires per_device_train_batch_size=1")
        self.active_ranks_per_step = (
            self.dopd_args.active_ranks_per_optimizer_step or self.world_size
        )
        if self.active_ranks_per_step > self.world_size:
            raise ValueError("dopd.active_ranks_per_optimizer_step cannot exceed WORLD_SIZE")
        if self.world_size % self.active_ranks_per_step:
            raise ValueError(
                "WORLD_SIZE must be divisible by dopd.active_ranks_per_optimizer_step"
            )
        if (
            self.active_ranks_per_step < self.world_size
            and self.training_args.gradient_accumulation_steps != 1
        ):
            raise ValueError(
                "gradient accumulation must be 1 when rotating active-rank cohorts"
            )
        effective_training_batch = (
            self.active_ranks_per_step
            * self.training_args.per_device_train_batch_size
            * self.training_args.gradient_accumulation_steps
        )
        if self.dopd_args.prompts_per_iteration % effective_training_batch:
            raise ValueError(
                "dopd.prompts_per_iteration must be divisible by the global effective "
                f"training batch: {self.dopd_args.prompts_per_iteration} % "
                f"{effective_training_batch} != 0"
            )
        if self.dopd_args.prompts_per_iteration % self.world_size:
            raise ValueError("dopd.prompts_per_iteration must be divisible by WORLD_SIZE")
        self.inner_steps_per_iteration = (
            self.dopd_args.prompts_per_iteration // effective_training_batch
        )
        self.local_rollouts_per_iteration = (
            self.dopd_args.prompts_per_iteration // self.world_size
        )
        if self.training_args.max_steps <= 0:
            raise ValueError("training.max_steps must be positive")
        if self.parallel_args.fsdp_size != self.world_size:
            raise ValueError("d-OPD requires parallel.fsdp_size=WORLD_SIZE")
        if self.dopd_args.strict_paper_alignment:
            self._validate_paper_alignment()

    def _validate_paper_alignment(self):
        if not self.dopd_args.train_data_path:
            raise ValueError("strict paper alignment requires dopd.train_data_path")
        if self.dopd_args.student_checkpoint is not None:
            raise ValueError(
                "strict paper alignment starts from the official Qwen3-1.7B-a2d-init "
                "weights; dopd.student_checkpoint must be null"
            )
        teacher_path = self.dopd_args.teacher_model_name_or_path
        if not teacher_path:
            raise ValueError("strict paper alignment requires an explicit AR teacher path")
        _validate_paper_model_pair(self.model_args.model_name_or_path, teacher_path)
        if self.model_args.train_from_scratch:
            raise ValueError("strict paper alignment requires model.train_from_scratch=false")
        if self.active_ranks_per_step != 8:
            raise ValueError(
                "strict paper alignment requires 8 active ranks per optimizer step "
                "(the physical WORLD_SIZE may be 8 or 16)"
            )
        if self.dopd_args.expected_train_records != 61816:
            raise ValueError("strict paper alignment requires 61,816 training records")
        expected_dopd = {
            "prompt_tokens": 2000,
            "min_response_tokens": 100,
            "max_response_tokens": 4000,
            "response_schedule_steps": 100,
            "prompts_per_iteration": 128,
            "block_size": 4,
            "denoise_steps": 4,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "remasking_strategy": "low_confidence_dynamic",
            "dynamic_threshold": 0.9,
            "correction_strength": 0.5,
            "candidate_top_k": 16,
            "loss_top_k": 16,
            "require_visible_future": True,
            "correctness_gate": True,
            "exclude_eos_from_loss": True,
            "score_chunk_size": 128,
            "teacher_cache_max_bytes": 17179869184,
        }
        mismatches = []
        for name, expected in expected_dopd.items():
            actual = getattr(self.dopd_args, name)
            equal = (
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                if isinstance(expected, float)
                else actual == expected
            )
            if not equal:
                mismatches.append(f"dopd.{name}={actual!r} (expected {expected!r})")
        expected_training = {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "seed": 10086,
            "optimizer": "adamw",
            "lr": 1.0e-5,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1.0e-8,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
            "min_lr": 1.0e-6,
            "max_steps": 483,
            "save_steps": 50,
            "save_total_limit": 10,
            "save_only_model": False,
        }
        for name, expected in expected_training.items():
            actual = getattr(self.training_args, name)
            equal = (
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                if isinstance(expected, float)
                else actual == expected
            )
            if not equal:
                mismatches.append(f"training.{name}={actual!r} (expected {expected!r})")
        expected_warmup = 5.0 / 483.0
        if not math.isclose(
            self.training_args.warmup_ratio, expected_warmup, rel_tol=0.0, abs_tol=1e-10
        ):
            mismatches.append(
                f"training.warmup_ratio={self.training_args.warmup_ratio!r} "
                f"(expected 5/483={expected_warmup!r})"
            )
        warmup_inner_steps = int(
            self.training_args.max_steps
            * self.inner_steps_per_iteration
            * self.training_args.warmup_ratio
        )
        if warmup_inner_steps != 5 * self.inner_steps_per_iteration:
            mismatches.append(
                f"warmup resolves to {warmup_inner_steps} inner steps "
                f"(expected {5 * self.inner_steps_per_iteration})"
            )
        if not self.parallel_args.recompute:
            mismatches.append("parallel.recompute=false (expected true)")
        if self.parallel_args.param_dtype != "bf16" or self.parallel_args.reduce_dtype != "bf16":
            mismatches.append("parallel param/reduce dtype must both be bf16")
        if mismatches:
            raise ValueError("Paper-alignment check failed:\n- " + "\n- ".join(mismatches))

    def _resolve_mask_token_id(self) -> int:
        value = self.dopd_args.mask_token_id
        if value is None:
            value = getattr(self.tokenizer, "mask_token_id", None)
        if value is None:
            value = getattr(self.student.config, "mask_token_id", None)
        if value is None:
            raise ValueError("set dopd.mask_token_id or configure a tokenizer mask token")
        value = int(value)
        if not 0 <= value < int(self.student.config.vocab_size):
            raise ValueError("dopd.mask_token_id is outside the model vocabulary")
        return value

    def _load_teacher(self):
        path = self.dopd_args.teacher_model_name_or_path or self.model_args.model_name_or_path
        logger.info_rank0(f"> Loading frozen AR teacher from {path}")
        teacher = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=self.model_args.trust_remote_code,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        teacher.requires_grad_(False)
        teacher.eval()
        teacher.config.use_cache = True
        teacher.to(self.device)
        if int(teacher.config.vocab_size) != int(self.student.config.vocab_size):
            raise ValueError("teacher and student must use the same vocabulary")
        return teacher

    def _load_resume_state(self):
        path = self.training_args.resume_from_checkpoint
        if not path:
            return
        state = {"model": self.student_engine, "optimizer": self.optimizer, "extra_state": {}}
        CheckpointManager.load(path, state)
        extra = state.get("extra_state", {})
        self.global_step = int(extra.get("global_step", 0))
        self.optimizer_step = int(
            extra.get("optimizer_step", self.global_step * self.inner_steps_per_iteration)
        )
        if "lr_scheduler" in extra:
            self.scheduler.load_state_dict(extra["lr_scheduler"])
        if "torch_rng_state" in extra:
            torch.set_rng_state(extra["torch_rng_state"])
        target_batches = int(
            extra.get(
                "data_batches_consumed",
                self.global_step * self.local_rollouts_per_iteration,
            )
        )
        if self.opdlm_records is None:
            for _ in range(target_batches):
                self._next_batch()
        logger.info_rank0(f"> Resumed d-OPD at iteration {self.global_step}")

    def _next_batch(self):
        if self.data_iterator is None:
            raise RuntimeError("Megatron data iterator is unavailable in official OPDLM mode")
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_epoch += 1
            sampler = getattr(self.dataloader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.data_epoch)
            self.data_iterator = iter(self.dataloader)
            batch = next(self.data_iterator)
        self.data_batches_consumed += 1
        return batch

    def _prompt_and_reference(self, batch):
        input_ids = batch["input_ids"][0].to(self.device, non_blocking=True)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            non_pad = input_ids.ne(int(pad_id)).nonzero(as_tuple=True)[0]
            if non_pad.numel():
                input_ids = input_ids[: int(non_pad[-1].item()) + 1]
        if input_ids.numel() < 2:
            raise ValueError("dataset sample is too short to form a prompt")
        prompt_length = min(self.dopd_args.prompt_tokens, input_ids.numel())
        prompt_length -= prompt_length % self.dopd_args.block_size
        if prompt_length <= 0:
            raise ValueError("dataset sample is shorter than one diffusion block")
        prompt = input_ids[:prompt_length].unsqueeze(0)
        reference = input_ids[prompt_length:]
        return prompt, reference

    def _trajectory_verification(self, prompt, response, reference, record=None) -> dict:
        if self.verifier is None:
            return {
                "correct": False,
                "applicable": False,
                "reason": "verifier_not_configured",
            }
        kwargs = {
            "prompt_ids": prompt[0].detach().cpu(),
            "response_ids": response[0].detach().cpu(),
            "reference_ids": reference.detach().cpu(),
            "tokenizer": self.tokenizer,
        }
        signature = inspect.signature(self.verifier)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if record is not None and ("record" in signature.parameters or accepts_kwargs):
            kwargs["record"] = record
        detailed_verifier = getattr(self.verifier, "verify_detailed", None)
        if callable(detailed_verifier):
            result = detailed_verifier(**kwargs)
            if not isinstance(result, dict) or "correct" not in result:
                raise TypeError("verifier.verify_detailed must return a dict containing 'correct'")
            normalized = dict(result)
            normalized["correct"] = bool(normalized["correct"])
            normalized.setdefault("applicable", True)
            normalized.setdefault(
                "reason",
                "accepted" if normalized["correct"] else "rejected_by_verifier",
            )
            return normalized
        correct = bool(self.verifier(**kwargs))
        return {
            "correct": correct,
            "applicable": True,
            "reason": "accepted" if correct else "rejected_by_verifier",
        }

    def _trajectory_is_correct(self, prompt, response, reference, record=None) -> bool:
        return bool(
            self._trajectory_verification(prompt, response, reference, record)["correct"]
        )

    def _rollout_audit_entry(
        self,
        *,
        iteration: int,
        trajectory_index: int,
        prompt,
        response,
        record,
        seed: int,
        response_token_cap: int,
        verification: dict,
    ) -> dict:
        response_ids = response[0].detach().cpu().tolist()
        eos_id = self.tokenizer.eos_token_id
        eos_found = eos_id is not None and int(eos_id) in response_ids
        if eos_found:
            response_ids = response_ids[: response_ids.index(int(eos_id))]
        prompt_ids = prompt[0].detach().cpu().tolist()
        return {
            "iteration": iteration,
            "rank": self.rank,
            "local_trajectory_index": trajectory_index,
            "seed": int(seed),
            "response_token_cap": int(response_token_cap),
            "response_token_count": len(response_ids),
            "response_token_ids": response_ids,
            "eos_generated": eos_found,
            "hit_response_token_cap": not eos_found and len(response_ids) >= response_token_cap,
            "domain": record.get("domain") if record else None,
            "source": record.get("source") if record else None,
            "question": record.get("question") if record else None,
            "ground_truth_answer": record.get("ground_truth_answer") if record else None,
            "tests_json": record.get("tests_json") if record else None,
            "formatted_prompt": self.tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
            ),
            "response": self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            ),
            "verifier": verification,
        }

    def _write_rollout_audit(self, iteration: int, local_entries: list[dict]) -> None:
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_entries)
        if self.rank != 0:
            return
        entries = [entry for rank_entries in gathered for entry in rank_entries]
        entries.sort(key=lambda item: (item["rank"], item["local_trajectory_index"]))
        if len(entries) != self.dopd_args.prompts_per_iteration:
            raise RuntimeError(
                "rollout audit is incomplete: "
                f"collected {len(entries)}, expected {self.dopd_args.prompts_per_iteration}"
            )
        audit_dir = self.output_dir / "rollout_audits"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"iteration_{iteration:04d}.jsonl"
        with audit_path.open("w", encoding="utf-8") as stream:
            for entry in entries:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

        summary = {
            "iteration": iteration,
            "audit_file": str(audit_path),
            "total": len(entries),
            "applicable": sum(
                bool(entry["verifier"].get("applicable", True)) for entry in entries
            ),
            "correct": sum(bool(entry["verifier"]["correct"]) for entry in entries),
            "failure_reasons": {},
            "by_domain": {},
        }
        for entry in entries:
            domain = entry.get("domain") or "unknown"
            domain_summary = summary["by_domain"].setdefault(
                domain,
                {"total": 0, "applicable": 0, "correct": 0},
            )
            domain_summary["total"] += 1
            domain_summary["applicable"] += int(
                bool(entry["verifier"].get("applicable", True))
            )
            domain_summary["correct"] += int(bool(entry["verifier"]["correct"]))
            if not entry["verifier"]["correct"]:
                reason = entry["verifier"].get("reason", "unknown")
                summary["failure_reasons"][reason] = (
                    summary["failure_reasons"].get(reason, 0) + 1
                )
        with (audit_dir / "summary.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        logger.info_rank0(f"> Saved rollout/verifier audit to {audit_path}")

    def _official_iteration_inputs(self, iteration: int):
        pool = records_for_iteration(
            self.opdlm_records,
            iteration,
            self.dopd_args.prompts_per_iteration,
            total_iterations=self.opdlm_total_iterations,
        )
        step_seed = int(self.training_args.seed) + iteration
        global_indices = list(range(len(pool)))
        random.Random(step_seed).shuffle(global_indices)
        local_indices = global_indices[self.rank :: self.world_size]
        records = [pool[index] for index in local_indices]
        prompts = [
            torch.tensor(
                tokenize_opdlm_prompt(
                    record, self.tokenizer, self.dopd_args.prompt_tokens
                ),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            for record in records
        ]
        references = [torch.empty(0, dtype=torch.long) for _ in records]
        sample_seeds = [
            (step_seed * 1_000_003 + global_index) % (2**31 - 1)
            for global_index in local_indices
        ]
        return prompts, references, records, sample_seeds, step_seed

    def _pad_prompts(self, prompts):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("the tokenizer must define pad_token_id or eos_token_id")
        maximum = max(prompt.shape[1] for prompt in prompts)
        batch = torch.full(
            (len(prompts), maximum),
            int(pad_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(batch, dtype=torch.bool)
        for row, prompt in enumerate(prompts):
            length = prompt.shape[1]
            batch[row, maximum - length :] = prompt[0]
            attention_mask[row, maximum - length :] = True
        return batch, attention_mask

    def _collect_rollouts(self, iteration: int, response_tokens: int):
        """Generate every rank-local trajectory in one cache-efficient batch."""

        local_batch_size = self.local_rollouts_per_iteration
        if self.opdlm_records is not None:
            prompts, references, records, sample_seeds, seed = (
                self._official_iteration_inputs(iteration)
            )
        else:
            prompts = []
            references = []
            for _ in range(local_batch_size):
                prompt, reference = self._prompt_and_reference(self._next_batch())
                prompts.append(prompt)
                references.append(reference)
            records = [None] * len(prompts)
            seed = (
                int(self.training_args.seed)
                + iteration * 1_000_003
                + self.rank
            ) % (2**31 - 1)
            sample_seeds = None
        if len(prompts) != local_batch_size:
            raise RuntimeError(
                f"rank {self.rank} received {len(prompts)} rollouts; expected {local_batch_size}"
            )
        prompt_batch, prompt_attention_mask = self._pad_prompts(prompts)
        responses, first_unmask_steps = block_diffusion_rollout(
            self.student,
            prompt_batch,
            response_tokens=response_tokens,
            block_size=self.dopd_args.block_size,
            denoise_steps=self.dopd_args.denoise_steps,
            mask_token_id=self.mask_token_id,
            temperature=self.dopd_args.temperature,
            top_k=self.dopd_args.top_k,
            top_p=self.dopd_args.top_p,
            remasking_strategy=self.dopd_args.remasking_strategy,
            dynamic_threshold=self.dopd_args.dynamic_threshold,
            seed=seed,
            eos_token_id=self.tokenizer.eos_token_id,
            prompt_attention_mask=prompt_attention_mask,
            sample_seeds=sample_seeds,
        )
        trajectory_seeds = sample_seeds or [
            (seed + index * self.world_size) % (2**31 - 1)
            for index in range(len(prompts))
        ]
        return (
            prompts,
            references,
            records,
            responses,
            first_unmask_steps,
            trajectory_seeds,
            seed,
        )

    def _trajectory_loss(
        self,
        prompt,
        reference,
        response,
        first_unmask_steps,
        *,
        seed: int,
        record=None,
        correction_enabled: bool = True,
    ):
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            eos_positions = response[0].eq(int(eos_id)).nonzero(as_tuple=True)[0]
            if eos_positions.numel():
                response_end = int(eos_positions[0].item()) + 1
                response = response[:, :response_end]
                first_unmask_steps = first_unmask_steps[:, :response_end]
        state = sample_intermediate_state(
            response[0],
            first_unmask_steps[0],
            self.dopd_args.block_size,
            self.mask_token_id,
            random.Random(seed + 17),
        )
        valid = response_valid_mask(
            response[0],
            self.tokenizer.eos_token_id,
            self.dopd_args.exclude_eos_from_loss,
        )
        future_valid = response_valid_mask(
            response[0],
            self.tokenizer.eos_token_id,
            exclude_eos=False,
        )
        loss_mask = (state.loss_mask & valid).unsqueeze(0)
        verification = (
            self._trajectory_verification(prompt, response, reference, record)
            if correction_enabled
            else {
                "correct": False,
                "applicable": False,
                "reason": "inactive_rank_copy",
            }
        )
        correct = bool(verification["correct"])
        loss, stats = compute_dopd_loss(
            self.student,
            self.teacher,
            prompt,
            response,
            state.noised_response.unsqueeze(0),
            loss_mask,
            block_size=self.dopd_args.block_size,
            candidate_top_k=self.dopd_args.candidate_top_k,
            loss_top_k=self.dopd_args.loss_top_k,
            correction_strength=self.dopd_args.correction_strength,
            require_visible_future=self.dopd_args.require_visible_future,
            correction_allowed=(
                correction_enabled
                and (correct or not self.dopd_args.correctness_gate)
            ),
            score_chunk_size=self.dopd_args.score_chunk_size,
            cache_max_bytes=self.dopd_args.teacher_cache_max_bytes,
            future_valid_length=int(future_valid.sum().item()),
        )
        return loss, stats, verification

    def _save(self, final: bool = False):
        name = f"global_step_{self.global_step}"
        if final:
            name = f"global_step_{self.global_step}_final"
        path = self.output_dir / "checkpoints" / name
        state = {
            "model": self.student_engine,
            "optimizer": self.optimizer,
            "extra_state": {
                "global_step": self.global_step,
                "optimizer_step": self.optimizer_step,
                "data_batches_consumed": self.data_batches_consumed,
                "lr_scheduler": self.scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
        }
        CheckpointManager.save(
            path=str(path),
            state=state,
            save_async=self.training_args.save_async,
            save_only_model=self.training_args.save_only_model,
            global_steps=self.global_step,
        )
        dist.barrier()
        cleanup_old_checkpoints(
            self.training_args,
            self.data_args.data_shared_file_system,
        )
        logger.info_rank0(f"> Saved d-OPD checkpoint to {path}")

    def train(self):
        self.optimizer.zero_grad()
        start_time = time.time()
        while self.global_step < self.training_args.max_steps:
            iteration = self.global_step + 1
            response_tokens = cosine_response_length(
                iteration,
                self.dopd_args.min_response_tokens,
                self.dopd_args.max_response_tokens,
                self.dopd_args.response_schedule_steps,
                self.dopd_args.block_size,
            )
            local_loss = 0.0
            local_audit_entries = []
            # loss/correction counters + per-domain total/correct counters.
            local_stats = torch.zeros(14, dtype=torch.float32, device=self.device)
            accumulation = self.training_args.gradient_accumulation_steps
            (
                prompts,
                references,
                records,
                responses,
                reveal_steps,
                trajectory_seeds,
                rollout_seed,
            ) = self._collect_rollouts(iteration, response_tokens)
            local_trajectories = responses.shape[0]
            order_generator = torch.Generator(device="cpu").manual_seed(rollout_seed + 23)
            training_order = torch.randperm(
                local_trajectories, generator=order_generator
            ).tolist()
            training_schedule = active_rank_training_schedule(
                training_order,
                rank=self.rank,
                world_size=self.world_size,
                active_ranks=self.active_ranks_per_step,
                accumulation=accumulation,
            )
            last_grad_norm = torch.zeros((), device=self.device)
            for trajectory_index, active, optimizer_boundary in training_schedule:
                trajectory_seed = trajectory_seeds[trajectory_index]
                sync_context = (
                    self.student.no_sync()
                    if not optimizer_boundary and hasattr(self.student, "no_sync")
                    else contextlib.nullcontext()
                )
                with sync_context:
                    loss, stats, verification = self._trajectory_loss(
                        prompts[trajectory_index],
                        references[trajectory_index],
                        responses[trajectory_index : trajectory_index + 1],
                        reveal_steps[trajectory_index : trajectory_index + 1],
                        seed=trajectory_seed,
                        record=records[trajectory_index],
                        correction_enabled=active,
                    )
                    correct = int(verification["correct"])
                    if self.active_ranks_per_step < self.world_size:
                        # FSDP averages gradients across all physical ranks. Scale
                        # the active cohort so the result is the paper's 8-sample
                        # 目标是得到论文中的 8 样本全局均值；未激活的 rank 仍执行
                        # 相同计算图，以保持 FSDP 集合通信对齐。
                        loss_scale = (
                            self.world_size / self.active_ranks_per_step if active else 0.0
                        )
                    else:
                        loss_scale = 1.0 / accumulation
                    (loss * loss_scale).backward()
                if active:
                    local_loss += float(loss.detach().item())
                    if iteration <= self.dopd_args.rollout_audit_iterations:
                        local_audit_entries.append(
                            self._rollout_audit_entry(
                                iteration=iteration,
                                trajectory_index=trajectory_index,
                                prompt=prompts[trajectory_index],
                                response=responses[trajectory_index : trajectory_index + 1],
                                record=records[trajectory_index],
                                seed=trajectory_seed,
                                response_token_cap=response_tokens,
                                verification=verification,
                            )
                        )
                    trajectory_stats = torch.zeros(
                        14, dtype=torch.float32, device=self.device
                    )
                    trajectory_stats[:6] = torch.tensor(
                        [
                            stats.loss_tokens,
                            stats.eligible_positions,
                            stats.corrected_positions,
                            stats.scored_branches,
                            stats.skipped_no_visible_future,
                            correct,
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    record = records[trajectory_index]
                    if record is not None:
                        domain_offset = {
                            "math": 0,
                            "science": 1,
                            "code": 2,
                            "chat": 3,
                        }[record["domain"]]
                        trajectory_stats[6 + domain_offset] = 1
                        trajectory_stats[10 + domain_offset] = correct
                    local_stats += trajectory_stats
                if optimizer_boundary:
                    last_grad_norm = clip_grad_norm(
                        self.student_engine, self.training_args.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.optimizer_step += 1
            self.global_step = iteration

            if iteration <= self.dopd_args.rollout_audit_iterations:
                self._write_rollout_audit(iteration, local_audit_entries)

            loss_tensor = torch.tensor(local_loss, dtype=torch.float32, device=self.device)
            # 检查通信张量
            assert loss_tensor.dtype == torch.float32
            assert local_stats.dtype == torch.float32

            # 只汇总激活组的统计量，总计恰好为 128 条轨迹。
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

            mean_loss = loss_tensor.item() / self.dopd_args.prompts_per_iteration
            if self.global_step % self.training_args.logging_steps == 0:
                elapsed = time.time() - start_time
                metrics = {
                    "iteration": self.global_step,
                    "optimizer_step": self.optimizer_step,
                    "loss": mean_loss,
                    "lr": float(self.scheduler.get_last_lr()[0]),
                    "grad_norm": _scalar(last_grad_norm),
                    "response_token_cap": response_tokens,
                    "response_tokens": int(responses.shape[1]),
                    "loss_tokens": int(local_stats[0].item()),
                    "eligible_positions": int(local_stats[1].item()),
                    "corrected_positions": int(local_stats[2].item()),
                    "scored_branches": int(local_stats[3].item()),
                    "skipped_no_visible_future": int(local_stats[4].item()),
                    "correct_trajectories": int(local_stats[5].item()),
                    "active_ranks_per_optimizer_step": self.active_ranks_per_step,
                    "elapsed_seconds": elapsed,
                }
                for domain_offset, domain in enumerate(
                    ("math", "science", "code", "chat")
                ):
                    total = int(local_stats[6 + domain_offset].item())
                    correct = int(local_stats[10 + domain_offset].item())
                    metrics[f"{domain}_trajectories"] = total
                    metrics[f"{domain}_correct"] = correct
                    metrics[f"{domain}_accuracy"] = (
                        correct / total if total else None
                    )
                logger.info_rank0("d-OPD " + " | ".join(f"{k}={v}" for k, v in metrics.items()))
                if self.rank == 0:
                    with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            if self.training_args.save_steps > 0 and self.global_step % self.training_args.save_steps == 0:
                self._save()
            _empty_cache()

        if self.training_args.save_steps <= 0 or self.global_step % self.training_args.save_steps:
            self._save(final=True)
        logger.info_rank0(f"> d-OPD finished in {time.time() - start_time:.1f}s")


def parse_args() -> Arguments:
    args = fsdp2_parse_args(Arguments)
    flat = types.SimpleNamespace(
        **{
            key: value
            for namespace in (
                args.model,
                args.data,
                args.parallel,
                args.training,
                args.optimization,
                args.dopd,
            )
            for key, value in namespace.__dict__.items()
        }
    )
    set_args(flat)
    return args


def _initialize(seed: int) -> None:
    # Reuse the repository's NPU/NCCL selection and deterministic setup without
    # changing train_fsdp2.py.
    from train_fsdp2 import MindSpeedAutoTrainer

    MindSpeedAutoTrainer._initialize(seed)


def main():
    args = parse_args()
    _initialize(args.training.seed)
    if dist.get_rank() == 0:
        logger.info_plain_rank0("***** d-OPD configuration *****")
        for group_name in ("model", "data", "parallel", "training", "dopd"):
            group = getattr(args, group_name)
            logger.info_plain_rank0(f"[{group_name}]")
            for item in fields(group):
                logger.info_plain_rank0(f"  {item.name}: {getattr(group, item.name)}")
    trainer = DOPDTrainer(args)
    trainer.train()

