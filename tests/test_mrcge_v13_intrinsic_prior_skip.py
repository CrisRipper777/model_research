from __future__ import annotations

import io
import math

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from src.models.mrcge_v13 import Model as FrozenV13Model
from src.models.mrcge_v13_intrinsic_prior_skip import Model as IntrinsicPriorModel
from src.tasks.nc import _compose_full_graph_loss


DATA_INFO = {"input_dim": 8, "text_dim": 4, "visual_dim": 4, "num_nodes": 12, "num_classes": 3}
X = torch.arange(96, dtype=torch.float32).reshape(12, 8) / 31.0
PHYSICAL = torch.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]], dtype=torch.long,
)
SEMANTIC = torch.tensor([[0, 2, 4, 6, 8, 10], [2, 4, 6, 8, 10, 0]], dtype=torch.long)
LABELS = torch.tensor([0, 1, 2, 1, 0, 2, 0, 1, 2, 1, 0, 2])
TRAIN_IDX = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])


def attach_semantic(model):
    for modality, edges in (("text", SEMANTIC), ("visual", SEMANTIC.flip(0))):
        setattr(model, f"semantic_{modality}_edge_index", edges.clone())
        setattr(model, f"semantic_{modality}_weight", torch.ones(edges.size(1)))
    model.semantic_candidate_fingerprint = "a" * 64
    model.semantic_feature_fingerprint = "b" * 64
    model.semantic_v7b_fingerprint = "c" * 64


def _config(model_file: str):
    cfg = OmegaConf.load(model_file)
    return OmegaConf.create({"model": cfg})


def make_candidate(seed=42):
    torch.manual_seed(seed)
    model = IntrinsicPriorModel(
        _config("configs/model/mrcge_v13_intrinsic_prior_skip.yaml"), DATA_INFO
    )
    attach_semantic(model)
    head = nn.Linear(model.out_dim, DATA_INFO["num_classes"])
    return model, head


def make_frozen(seed=42):
    torch.manual_seed(seed)
    cfg = _config("configs/model/mrcge_v13.yaml")
    cfg.model.condition = "M_repaired"
    cfg.model.v13_variant = "joint_view"
    cfg.model.sender_private_ratio = 0.0
    model = FrozenV13Model(cfg, DATA_INFO)
    attach_semantic(model)
    head = nn.Linear(model.out_dim, DATA_INFO["num_classes"])
    return model, head


def task_loss(model, head, z, aux, info):
    main = F.cross_entropy(head(z[TRAIN_IDX]), LABELS[TRAIN_IDX])
    total, private_ce = _compose_full_graph_loss(
        model=model,
        info=info,
        main_ce=main,
        aux_loss=aux,
        aux_weight=1.0,
        unimodal_weight=0.3,
        labels=LABELS,
        train_idx=TRAIN_IDX,
    )
    return total, main, private_ce


def _assert_nested_equal(left, right, path="root"):
    assert type(left) is type(right), path
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0, msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            _assert_nested_equal(a, b, f"{path}[{i}]")
    elif isinstance(left, float):
        assert left == pytest.approx(right, abs=0.0, rel=0.0), path
    else:
        assert left == right, path


@pytest.mark.parametrize("device", ["cpu"])
def test_zero_prior_is_exact_v13a_initialization_eval_and_external_head(device):
    frozen, frozen_head = make_frozen(seed=123)
    candidate, candidate_head = make_candidate(seed=123)
    frozen, frozen_head = frozen.to(device), frozen_head.to(device)
    candidate, candidate_head = candidate.to(device), candidate_head.to(device)

    reference_state = frozen.state_dict()
    candidate_state = candidate.state_dict()
    assert set(candidate_state) == set(reference_state) | {"prior_skip_raw"}
    assert all(torch.equal(value, candidate_state[name]) for name, value in reference_state.items())
    assert candidate.prior_skip_raw.shape == torch.Size([])
    assert candidate.prior_skip_raw.item() == 0.0
    assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in frozen.parameters()) + 1
    assert torch.equal(frozen_head.weight, candidate_head.weight)
    assert torch.equal(frozen_head.bias, candidate_head.bias)

    frozen.eval()
    candidate.eval()
    frozen._capture_state_audit = True
    candidate._capture_state_audit = True
    with torch.no_grad():
        z0, lp0, aux0, balance0, info0 = frozen(X.to(device), PHYSICAL.to(device), return_details=True)
        z1, lp1, aux1, balance1, info1 = candidate(X.to(device), PHYSICAL.to(device), return_details=True)
    assert lp0 is lp1 is None and aux0 is aux1 is None
    torch.testing.assert_close(z0, z1, atol=0.0, rtol=0.0)
    torch.testing.assert_close(balance0, balance1, atol=0.0, rtol=0.0)
    torch.testing.assert_close(info0["balance_loss"], info1["balance_loss"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(info0["aux_loss"], info1["aux_loss"], atol=0.0, rtol=0.0)
    for modality in ("text", "visual"):
        torch.testing.assert_close(info0["unimodal_logits"][modality], info1["unimodal_logits"][modality], atol=0.0, rtol=0.0)
    for key in ("routes", "collaborative_routes", "late", "private_collaborative_route_changes"):
        _assert_nested_equal(info0["details"][key], info1["details"][key], key)
    assert frozen._last_state_audit.keys() == (
        candidate._last_state_audit.keys() - {"intrinsic_prior_skip"}
    )
    for key in frozen._last_state_audit:
        _assert_nested_equal(frozen._last_state_audit[key], candidate._last_state_audit[key], key)
    torch.testing.assert_close(frozen_head(z0), candidate_head(z1), atol=0.0, rtol=0.0)
    d = info1["details"]["intrinsic_prior_skip"]
    assert d["lambda_prior"] == 0.0
    assert d["prior_skip_linear_rms"] == 0.0

    # The main, private auxiliary, and balance loss components remain identical.
    main0 = F.cross_entropy(frozen_head(z0[TRAIN_IDX]), LABELS[TRAIN_IDX].to(device))
    main1 = F.cross_entropy(candidate_head(z1[TRAIN_IDX]), LABELS[TRAIN_IDX].to(device))
    torch.testing.assert_close(main0, main1, atol=0.0, rtol=0.0)
    torch.testing.assert_close(info0["balance_loss"], info1["balance_loss"], atol=0.0, rtol=0.0)


def test_zero_prior_train_loss_shared_gradients_and_rng_match_and_scalar_is_trainable():
    frozen, frozen_head = make_frozen(seed=211)
    candidate, candidate_head = make_candidate(seed=211)
    assert torch.equal(frozen_head.weight, candidate_head.weight)
    assert torch.equal(frozen_head.bias, candidate_head.bias)
    frozen.train()
    candidate.train()

    torch.manual_seed(20261027)
    z0, _, _, aux0, info0 = frozen(X, PHYSICAL)
    rng0 = torch.random.get_rng_state().clone()
    loss0, main0, private0 = task_loss(frozen, frozen_head, z0, aux0, info0)
    loss0.backward()

    torch.manual_seed(20261027)
    z1, _, _, aux1, info1 = candidate(X, PHYSICAL)
    rng1 = torch.random.get_rng_state().clone()
    loss1, main1, private1 = task_loss(candidate, candidate_head, z1, aux1, info1)
    loss1.backward()

    torch.testing.assert_close(z0, z1, atol=0.0, rtol=0.0)
    torch.testing.assert_close(loss0, loss1, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(main0, main1, atol=0.0, rtol=0.0)
    torch.testing.assert_close(private0, private1, atol=0.0, rtol=0.0)
    torch.testing.assert_close(aux0, aux1, atol=0.0, rtol=0.0)
    assert torch.equal(rng0, rng1)

    ref_params = dict(frozen.named_parameters())
    new_params = dict(candidate.named_parameters())
    assert set(new_params) == set(ref_params) | {"prior_skip_raw"}
    for name, param in ref_params.items():
        other = new_params[name]
        assert (param.grad is None) == (other.grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(other.grad, param.grad, atol=2e-7, rtol=2e-6, msg=name)
    torch.testing.assert_close(frozen_head.weight.grad, candidate_head.weight.grad, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(frozen_head.bias.grad, candidate_head.bias.grad, atol=2e-7, rtol=2e-6)
    grad = candidate.prior_skip_raw.grad
    assert grad is not None and torch.isfinite(grad) and grad.abs().item() > 0.0
    candidate.capture_gradient_diagnostics()

    optimizer = torch.optim.AdamW(
        list(candidate.parameters()) + list(candidate_head.parameters()), lr=1e-3, weight_decay=1e-4
    )
    assert any(p is candidate.prior_skip_raw for group in optimizer.param_groups for p in group["params"])
    grad_diag = candidate._last_gradient_diagnostics
    assert grad_diag["prior_skip_raw"]["gradient_present"]
    assert grad_diag["prior_skip_raw"]["gradient_finite"]
    assert grad_diag["prior_skip_raw"]["gradient_nonzero"]


def test_nonzero_prior_changes_only_final_fusion_and_matches_frozen_equation():
    model, _ = make_candidate(seed=307)
    model.eval()
    model._capture_state_audit = True
    z0, _, _, aux0, info0 = model(X, PHYSICAL, return_details=True)
    audit0 = model._last_state_audit
    with torch.no_grad():
        model.prior_skip_raw.fill_(0.45)
    z1, _, _, aux1, info1 = model(X, PHYSICAL, return_details=True)
    audit1 = model._last_state_audit

    assert not torch.equal(z0, z1)
    torch.testing.assert_close(aux0, aux1, atol=0.0, rtol=0.0)
    for key in ("routes", "collaborative_routes", "late", "private_collaborative_route_changes"):
        _assert_nested_equal(info0["details"][key], info1["details"][key], key)
    for key in audit0:
        if key != "intrinsic_prior_skip":
            _assert_nested_equal(audit0[key], audit1[key], f"state_audit.{key}")

    tensors = audit1["intrinsic_prior_skip"]
    h0, fused = tensors["h0_fused"], tensors["fused"]
    lam = model.lambda_prior
    expected_skip = model.fusion_skip(fused + lam * h0)
    expected_mlp = model.fusion_linear2(model.fusion_dropout(F.gelu(model.fusion_linear1(fused))))
    expected_z = model.fusion_norm(expected_skip + expected_mlp)
    torch.testing.assert_close(z1, expected_z, atol=0.0, rtol=0.0)
    assert h0.requires_grad
    grad_h0 = torch.autograd.grad(z1.square().sum(), h0, retain_graph=False)[0]
    assert torch.isfinite(grad_h0).all() and torch.count_nonzero(grad_h0) > 0

    d = info1["details"]["intrinsic_prior_skip"]
    w = model.fusion_skip.weight.detach()
    no_bias_old = F.linear(fused.detach(), w, bias=None)
    no_bias_prior = F.linear((lam.detach() * h0.detach()), w, bias=None)
    expected_old_rms = model._rms(no_bias_old)
    expected_prior_rms = model._rms(no_bias_prior)
    expected_cosine = F.cosine_similarity(
        no_bias_old.reshape(1, -1), no_bias_prior.reshape(1, -1), dim=-1, eps=model.eps
    ).item()
    assert d["fusion_skip_weight_fused_rms"] == pytest.approx(expected_old_rms.item(), rel=1e-6)
    assert d["prior_skip_linear_rms"] == pytest.approx(expected_prior_rms.item(), rel=1e-6)
    assert d["prior_to_old_skip_cosine"] == pytest.approx(expected_cosine, rel=1e-6)
    assert d["lambda_prior"] == pytest.approx(torch.tanh(torch.tensor(0.45)).item())


def test_prior_rng_checkpoint_and_finite_empty_zero_cross_and_isolated_graphs():
    model, head = make_candidate(seed=401)
    model.eval()
    with torch.no_grad():
        model.prior_skip_raw.fill_(-0.3)
        z, _, _, aux, _ = model(X, PHYSICAL)
        stream = io.BytesIO()
        torch.save({"model": model.state_dict(), "head": head.state_dict()}, stream)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        restored, restored_head = make_candidate(seed=999)
        restored.load_state_dict(payload["model"], strict=True)
        restored_head.load_state_dict(payload["head"], strict=True)
        restored.eval()
        z2, _, _, aux2, _ = restored(X, PHYSICAL)
    torch.testing.assert_close(z, z2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(aux, aux2, atol=0.0, rtol=0.0)

    empty_edges = torch.empty((2, 0), dtype=torch.long)
    isolated_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    with torch.no_grad():
        for edges in (empty_edges, isolated_edges):
            output, _, _, aux_out, info = restored(X, edges, return_details=True)
            assert torch.isfinite(output).all() and torch.isfinite(aux_out)
            assert torch.isfinite(info["balance_loss"])
            assert math.isfinite(info["details"]["intrinsic_prior_skip"]["prior_skip_linear_rms"])
        # With a finite, strongly negative cross logit the sigmoid rounds to zero;
        # this exercises the parent's zero-cross/fallback path without non-finite state.
        restored.cross_lambda_raw.fill_(-1000.0)
        output, _, _, aux_out, _ = restored(X, PHYSICAL, return_details=True)
        assert torch.isfinite(output).all() and torch.isfinite(aux_out)


def test_epoch_history_contains_prior_scalar_trajectory_fields():
    model, _ = make_candidate(seed=509)
    row = model.collect_epoch_diagnostics()
    assert row["intrinsic_prior_raw"] == 0.0
    assert row["intrinsic_prior_lambda"] == 0.0
    assert row["intrinsic_prior_gradient_norm"] == 0.0
    assert row["intrinsic_prior_gradient_nonzero"] == 0
    assert model.prior_skip_raw.numel() == 1
