# Qwen3-1.7B d-OPD on 8x Ascend A2

This directory is an isolated reproduction of **d-OPD: Future-Aware On-Policy
Distillation for Block Diffusion Language Models**.  It reuses the existing
MindSpeed FSDP2 Qwen3 diffusion student and the seven Megatron-format datasets,
without modifying any Nemotron_Diffusion source or experiment file.

## What is reproduced

- frozen Qwen3 autoregressive teacher and Qwen3 block-diffusion student;
- student on-policy block rollouts with every token's first reveal step;
- one uniformly sampled denoising state per block;
- the reference `[prompt, clean response, noised response]` training layout, so
  every noised block reads completed clean blocks on its left;
- distributed EOS termination with a synchronized curriculum length cap;
- teacher/student top-16 candidate union;
- causal future-likelihood scoring with shared teacher KV cache;
- relative future-score teacher reweighting and teacher-top-16 forward KL;
- block size 4, four denoising steps, 483 on-policy iterations;
- 128 global prompts per iteration on 8 ranks;
- one rank-local batch of 16 rollouts, followed by the official 16 optimizer
  steps at global training batch 8;
- cosine rollout curriculum from 100 to 4000 tokens in the first 100 iterations;
- BF16 FSDP2 student, AdamW at `1e-5`, five rollout-iteration warmup (80 inner
  optimizer steps), cosine decay, and checkpoints every 10 rollout iterations.

## Deliberate dataset adaptation

The paper uses a 61,816-prompt instruction mixture with task verifiers.  The
existing Nemotron_Diffusion inputs are continuous Megatron `.bin/.idx` CPT data.
This implementation takes the first `dopd.prompt_tokens` of each sampled
sequence as the rollout prompt.  Because those records have no structured
answer or verifier, the supplied configuration sets `correctness_gate: false`.
It therefore reproduces the paper's **d-OPD gate-off** method on the requested
local data, not the paper's verifier-gated benchmark numbers.

To enable the full gate, set:

```yaml
dopd:
  correctness_gate: true
  verifier: your_python_module:check
```

The callable receives `prompt_ids`, `response_ids`, `reference_ids`, and
`tokenizer`, and must return a boolean.

## Start training

Confirm these paths in `dopd_qwen3_1.7b_fsdp2_A2.yaml` first:

1. `model.model_name_or_path` and `dopd.teacher_model_name_or_path` point to the
   original Qwen3-1.7B AR checkpoint.
2. `dopd.student_checkpoint` points to the desired existing
   Nemotron_Diffusion FSDP2 checkpoint.
3. All seven dataset prefixes exist as `.bin/.idx` pairs.
4. The Qwen3 config contains the diffusion fields and mask id used by the
   existing experiment.

Then run from the MindSpeed-LLM repository root:

```bash
bash examples/fsdp2/qwen3_dopd/train_dopd_qwen3_1.7b_fsdp2_A2.sh
```

For a short integration run, override the expensive paper-scale curriculum:

```bash
torchrun --standalone --nproc_per_node 8 train_dopd_fsdp2.py \
  examples/fsdp2/qwen3_dopd/dopd_qwen3_1.7b_fsdp2_A2.yaml \
  --training.max_steps=1 \
  --training.gradient_accumulation_steps=1 \
  --dopd.prompts_per_iteration=8 \
  --dopd.min_response_tokens=16 \
  --dopd.max_response_tokens=16 \
  --training.output_dir=/data2/model_train/ckpt/qwen3_1.7b_dopd_smoke/
```

The `=` form is required for command-line values to override this repository's
YAML parser. For the OPDLM control, additionally set
`--dopd.correction_strength=0`. Use a
new `training.output_dir` for every control so checkpoints cannot overwrite one
another.

## Outputs and resume

The run writes distributed checkpoints under
`<output_dir>/checkpoints/global_step_N`, resolved config to
`<output_dir>/dopd_config.json`, and rank-zero metrics to
`<output_dir>/metrics.jsonl`.  Resume a d-OPD checkpoint with
`training.resume_from_checkpoint`; `dopd.student_checkpoint` is used only for a
fresh d-OPD run. `training.max_steps` and checkpoint names count outer
on-policy rollout iterations; `optimizer_step` in metrics/checkpoint state
counts the 16 inner updates per iteration.

