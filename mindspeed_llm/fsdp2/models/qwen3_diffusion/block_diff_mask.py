# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Phase 3: Dense block-diagonal attention mask for block_diff paradigm.
#
# Replaces Nemotron's flex_attention + create_block_mask with a standard 4D
# float mask compatible with SDPA/eager attention on Ascend NPU.
#
# Algorithm source: Nemotron modeling_nemotron_labs_diffusion.py:56-92

import torch


def make_block_diff_mask(
    seq_len: int,
    block_size: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """Construct the block_diff 4D attention mask.

    For a concatenated input [noisy(0..L), original(L..2L)]:
      - M_BD: noisy→noisy, same block → bidirectional (attend)
      - M_OBC: noisy→original, earlier blocks → attend (conditioning)
      - M_BC: original→original → standard causal

    Args:
        seq_len: length of one half (L). Total sequence is 2L.
        block_size: block size for block-diagonal partitioning.
        batch_size: batch dimension (expanded after construction).
        dtype: mask dtype (should match model dtype).
        device: device for the mask tensor.

    Returns:
        mask: (batch_size, 1, 2L, 2L) float tensor.
              0.0 = attend, -inf = blocked
    """
    L = seq_len
    n = L  # boundary: first half is noisy, second half is original
    total_len = 2 * L

    # Index grids
    q_idx = torch.arange(total_len, device=device).unsqueeze(1)  # (2L, 1)
    kv_idx = torch.arange(total_len, device=device).unsqueeze(0)  # (1, 2L)

    x0_flag_q = q_idx >= n    # True for original half
    x0_flag_kv = kv_idx >= n  # True for original half

    block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

    # M_BD: noisy→noisy, same block
    m_bd = (block_q == block_kv) & (~x0_flag_kv) & (~x0_flag_q)

    # M_OBC: noisy→original, block_q > block_kv (earlier blocks)
    m_obc = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)

    # M_BC: original→original, causal (q_idx >= kv_idx)
    m_bc = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q

    attend = m_bd | m_obc | m_bc

    # Build float mask: 0 where attend, -inf where blocked
    mask = torch.full((1, 1, total_len, total_len), float("-inf"), dtype=dtype, device=device)
    mask[0, 0][attend] = 0.0

    # Expand for batch
    if batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1)
    return mask

def make_causal_train_mask(
    seq_len: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """Construct the standard causal 4D attention mask for AR training.

    For a sequence input_ids[0..L):
      - token i can attend to token j only if j <= i
      - future tokens j > i are masked

    Args:
        seq_len: sequence length L.
        batch_size: batch dimension.
        dtype: mask dtype, should match model dtype.
        device: device for the mask tensor.

    Returns:
        mask: (batch_size, 1, L, L) float tensor.
              0.0 = attend
              -inf = blocked
    """
    L = seq_len

    q_idx = torch.arange(L, device=device).unsqueeze(1)   # (L, 1)
    kv_idx = torch.arange(L, device=device).unsqueeze(0)  # (1, L)

    # causal attend rule:
    # query position q can attend key/value position kv if kv <= q
    attend = kv_idx <= q_idx

    mask = torch.full(
        (1, 1, L, L),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    mask[0, 0][attend] = 0.0

    if batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1)

    return mask