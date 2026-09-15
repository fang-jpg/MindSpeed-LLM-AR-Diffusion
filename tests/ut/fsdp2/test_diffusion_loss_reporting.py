# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""CPU tests for Nemotron-style diffusion loss reporting."""

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "diffusion_loss_reporting_under_test",
    ROOT / "mindspeed_llm/fsdp2/models/common/diffusion_loss_reporting.py",
)
reporting_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting_module)

build_diffusion_loss_report = reporting_module.build_diffusion_loss_report
reduce_diffusion_loss_reports = reporting_module.reduce_diffusion_loss_reports


def _report(weighted, total_tokens, ar, ar_tokens, dlm, dlm_tokens):
    return build_diffusion_loss_report(
        weighted_loss_sum=torch.tensor(weighted),
        total_token_count=torch.tensor(total_tokens),
        ar_loss_sum=torch.tensor(ar),
        ar_token_count=torch.tensor(ar_tokens),
        dlm_loss_sum=torch.tensor(dlm),
        dlm_token_count=torch.tensor(dlm_tokens),
    )


def test_build_report_matches_official_fields():
    report = _report(weighted=7.0, total_tokens=8, ar=3.0, ar_tokens=5, dlm=8.0, dlm_tokens=3)

    assert list(report) == ["lm loss", "ar loss", "dlm loss", "num_tokens_dlm"]
    torch.testing.assert_close(report["lm loss"], torch.tensor([7.0, 8.0]))
    torch.testing.assert_close(report["ar loss"], torch.tensor([3.0, 5.0]))
    torch.testing.assert_close(report["dlm loss"], torch.tensor([8.0, 3.0]))
    torch.testing.assert_close(report["num_tokens_dlm"], torch.tensor([3.0]))


def test_reduce_report_uses_component_token_denominators():
    reports = [
        _report(weighted=7.0, total_tokens=8, ar=3.0, ar_tokens=5, dlm=8.0, dlm_tokens=3),
        _report(weighted=13.0, total_tokens=12, ar=8.0, ar_tokens=7, dlm=10.0, dlm_tokens=5),
    ]

    reduced = reduce_diffusion_loss_reports(reports)

    torch.testing.assert_close(reduced["lm loss"], torch.tensor(1.0))
    torch.testing.assert_close(reduced["ar loss"], torch.tensor(11.0 / 12.0))
    torch.testing.assert_close(reduced["dlm loss"], torch.tensor(18.0 / 8.0))
    torch.testing.assert_close(reduced["num_tokens_dlm"], torch.tensor(4.0))


def test_empty_report_list_is_supported():
    assert reduce_diffusion_loss_reports([]) == {}
