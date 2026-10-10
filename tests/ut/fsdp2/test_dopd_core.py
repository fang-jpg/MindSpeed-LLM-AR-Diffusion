import random

import torch

from mindspeed_llm.experimental.dopd.core import (
    apply_future_reweighting,
    build_dopd_attention_mask,
    cosine_response_length,
    sample_intermediate_state,
    topk_forward_kl,
)


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

