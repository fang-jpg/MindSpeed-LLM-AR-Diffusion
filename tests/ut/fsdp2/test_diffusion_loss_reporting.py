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
finalize_global_token_gradients = reporting_module.finalize_global_token_gradients
reduce_diffusion_loss_reports = reporting_module.reduce_diffusion_loss_reports


def _report(
    weighted,
    total_tokens,
    ar,
    ar_tokens,
    dlm,
    dlm_tokens,
    calculate_per_token_loss=True,
):
    return build_diffusion_loss_report(
        weighted_loss_sum=torch.tensor(weighted),
        total_token_count=torch.tensor(total_tokens),
        ar_loss_sum=torch.tensor(ar),
        ar_token_count=torch.tensor(ar_tokens),
        dlm_loss_sum=torch.tensor(dlm),
        dlm_token_count=torch.tensor(dlm_tokens),
        calculate_per_token_loss=calculate_per_token_loss,
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


def test_reduce_report_matches_local_average_training_when_per_token_loss_is_disabled():
    reports = [
        _report(
            weighted=7.0,
            total_tokens=8,
            ar=3.0,
            ar_tokens=5,
            dlm=8.0,
            dlm_tokens=3,
            calculate_per_token_loss=False,
        ),
        _report(
            weighted=13.0,
            total_tokens=12,
            ar=8.0,
            ar_tokens=7,
            dlm=10.0,
            dlm_tokens=5,
            calculate_per_token_loss=False,
        ),
    ]

    reduced = reduce_diffusion_loss_reports(reports)

    expected_lm_loss = ((7.0 / 8.0) + (13.0 / 12.0)) / 2.0
    torch.testing.assert_close(reduced["lm loss"], torch.tensor(expected_lm_loss))
    # Component metrics retain their official token-weighted reporting rules.
    torch.testing.assert_close(reduced["ar loss"], torch.tensor(11.0 / 12.0))
    torch.testing.assert_close(reduced["dlm loss"], torch.tensor(18.0 / 8.0))


def test_empty_report_list_is_supported():
    assert reduce_diffusion_loss_reports([]) == {}


def test_finalize_global_token_gradients_uses_one_global_denominator():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.tensor([[20.0]])

    global_tokens, scale = finalize_global_token_gradients(
        model,
        [torch.tensor(10), torch.tensor(100)],
    )

    torch.testing.assert_close(global_tokens, torch.tensor(110.0))
    torch.testing.assert_close(scale, torch.tensor(1.0 / 110.0))
    torch.testing.assert_close(model.weight.grad, torch.tensor([[20.0 / 110.0]]))


def test_finalize_global_token_gradients_compensates_for_fsdp_average(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    # This represents the gradient after FSDP averaged two identical ranks.
    model.weight.grad = torch.tensor([[20.0]])

    monkeypatch.setattr(reporting_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(reporting_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(reporting_module.dist, "get_world_size", lambda group=None: 2)

    def fake_all_reduce(value, op=None, group=None):
        value.mul_(2)

    monkeypatch.setattr(reporting_module.dist, "all_reduce", fake_all_reduce)

    global_tokens, scale = finalize_global_token_gradients(
        model,
        [torch.tensor(10), torch.tensor(100)],
    )

    torch.testing.assert_close(global_tokens, torch.tensor(220.0))
    torch.testing.assert_close(scale, torch.tensor(2.0 / 220.0))
    torch.testing.assert_close(model.weight.grad, torch.tensor([[20.0 / 110.0]]))


def test_finalize_global_token_gradients_rejects_empty_counts():
    model = torch.nn.Linear(1, 1, bias=False)

    try:
        finalize_global_token_gradients(model, [])
    except ValueError as error:
        assert "at least one local token count" in str(error)
    else:
        raise AssertionError("Expected an empty token-count list to fail.")
