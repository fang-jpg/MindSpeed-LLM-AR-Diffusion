#!/usr/bin/env python3
"""Fit the semantic-embedding PCA used by the diffusion acceptance sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.features import (
    fit_embedding_pca,
)
from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.runtime import (
    DTYPE_MAP,
    detect_device,
    load_diffusion_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-diffusion", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=DTYPE_MAP, default="bfloat16")
    parser.add_argument("--pca-dim", type=int, default=46)
    parser.add_argument("--sample-size", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device or detect_device()
    print(f"> Loading frozen diffusion model on {device}")
    model = load_diffusion_model(
        args.model_path,
        device=device,
        dtype=DTYPE_MAP[args.dtype],
        config_diffusion_path=args.config_diffusion,
    )
    embedding_weight = model.get_input_embeddings().weight.detach()
    mask_token_id = int(model.mask_token_id)
    pca_source = (
        embedding_weight[:mask_token_id]
        if 0 < mask_token_id < embedding_weight.shape[0]
        else embedding_weight
    )
    print(
        f"> Fitting PCA: rows={min(args.sample_size, pca_source.shape[0])}, "
        f"input_dim={pca_source.shape[1]}, output_dim={args.pca_dim}"
    )
    pca = fit_embedding_pca(
        pca_source,
        output_dim=args.pca_dim,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    model_path = Path(args.model_path)
    model_identifier = (
        str(model_path.resolve()) if model_path.exists() else args.model_path
    )
    pca.metadata = {
        "model_path": model_identifier,
        "vocab_size": int(embedding_weight.shape[0]),
        "embedding_dim": int(embedding_weight.shape[1]),
        "mask_token_id": mask_token_id,
    }
    output_path = Path(args.output)
    pca.save(output_path)
    summary = {
        "output": str(output_path.resolve()),
        "source_rows": pca.source_rows,
        "input_dim": pca.input_dim,
        "output_dim": pca.output_dim,
        "retained_explained_variance": pca.explained_variance.sum().item(),
        "seed": pca.seed,
        "model": pca.metadata,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
