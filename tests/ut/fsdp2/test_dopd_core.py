import random

import torch

from mindspeed_llm.experimental.dopd.core import (
    active_rank_training_schedule,
    apply_future_reweighting,
    build_dopd_attention_mask,
    cosine_response_length,
    sample_intermediate_state,
    topk_forward_kl,
)
from mindspeed_llm.experimental.dopd.rollout import _prompt_block_attention_mask


def test_two_node_rotation_retains_global_optimizer_batch_eight():
    schedules = [
        active_rank_training_schedule(
            list(range(8)), rank=rank, world_size=16, active_ranks=8, accumulation=1
        )
        for rank in range(16)
    ]
    assert all(len(schedule) == 16 for schedule in schedules)
    for update in range(16):
        assert sum(int(schedule[update][1]) for schedule in schedules) == 8
    for schedule in schedules:
        active_trajectories = [trajectory for trajectory, active, _ in schedule if active]
        assert active_trajectories == list(range(8))


def test_cosine_response_curriculum_hits_endpoints_and_block_alignment():
    values = [cosine_response_length(i, 100, 4000, 100, 4) for i in range(1, 101)]
    assert values[0] == 100
    assert values[-1] == 4000
    assert values == sorted(values)
    assert all(value % 4 == 0 for value in values)


def test_dopd_mask_uses_clean_history_and_current_noised_block():
    mask = build_dopd_attention_mask(3, 4, 2, dtype=torch.float32, device="cpu")[0, 0]
    assert mask[0, 0] == 0 and mask[0, 1] < 0
    assert mask[1, :3].eq(0).all()
    assert mask[3, :5].eq(0).all() and mask[3, 5:].lt(0).all()
    assert mask[7, :3].eq(0).all() and mask[7, 3:7].lt(0).all()
    assert mask[7, 7:9].eq(0).all() and mask[7, 9:].lt(0).all()
    assert mask[9, :5].eq(0).all() and mask[9, 5:9].lt(0).all()
    assert mask[9, 9:].eq(0).all()


def test_rollout_prefill_mask_is_start_aligned_block_causal_with_left_padding():
    valid = torch.tensor(
        [[False, True, True, True, True], [False, False, True, True, True]]
    )
    mask = _prompt_block_attention_mask(valid, block_size=2, dtype=torch.float32)

    # Four-token prompt: [1,2] and [3,4] are bidirectional blocks.
    assert mask[0, 0, 1:3, 1:3].eq(0).all()
    assert mask[0, 0, 1:3, 3:].lt(0).all()
    assert mask[0, 0, 3:5, 1:5].eq(0).all()
    # Three-token prompt is start-aligned: [2,3] then [4].
    assert mask[1, 0, 2:4, 2:4].eq(0).all()
    assert mask[1, 0, 2:4, 4].lt(0).all()
    assert mask[1, 0, 4, 2:5].eq(0).all()
    # Left-padding keys never leak into valid queries.
    assert mask[0, 0, 1:, 0].lt(0).all()
    assert mask[1, 0, 2:, :2].lt(0).all()


def test_sample_intermediate_state_masks_current_and_later_reveal_groups():
    response = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23])
    steps = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    state = sample_intermediate_state(response, steps, 4, 99, random.Random(7))
    for start in (0, 4):
        selected_step = steps[start : start + 4][state.loss_mask[start : start + 4]][0]
        expected_mask = steps[start : start + 4] >= selected_step
        assert torch.equal(state.noised_response[start : start + 4].eq(99), expected_mask)


def test_future_reweighting_normalizes_and_topk_kl_backpropagates():
    teacher = torch.tensor([0.5, 0.3, 0.2]).log()
    corrected = apply_future_reweighting(
        teacher, torch.tensor([0, 2]), torch.tensor([-1.0, 1.0]), strength=0.5
    )
    torch.testing.assert_close(corrected.exp().sum(), torch.tensor(1.0))
    assert corrected[2] > teacher[2]

    logits = torch.randn(1, 3, requires_grad=True)
    student = logits.log_softmax(-1)
    loss = topk_forward_kl(corrected.unsqueeze(0), student, top_k=2).mean()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0

