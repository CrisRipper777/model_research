from __future__ import annotations

import io

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from src.models import build_model
from src.models.mrcge_v13 import Model as FrozenV13Model


DATA_INFO = {"input_dim": 7, "text_dim": 3, "visual_dim": 4, "num_nodes": 6, "num_classes": 3}
X = torch.arange(42, dtype=torch.float32).reshape(6, 7) / 29.0
PHYSICAL = torch.tensor(
    [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
)
SEMANTIC_TEXT = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 1]], dtype=torch.long)
SEMANTIC_VISUAL = torch.tensor([[0, 2, 4, 1, 3, 5], [2, 4, 1, 3, 5, 0]], dtype=torch.long)


def attach_semantic(model):
    for modality, edges in (("text", SEMANTIC_TEXT), ("visual", SEMANTIC_VISUAL)):
        setattr(model, f"semantic_{modality}_edge_index", edges)
        setattr(model, f"semantic_{modality}_weight", torch.ones(edges.size(1)))
    model.semantic_candidate_fingerprint = "a" * 64
    model.semantic_feature_fingerprint = "b" * 64
    model.semantic_v7b_fingerprint = "c" * 64


def make_cfg(*, static: bool, seed: int = 42):
    cfg = OmegaConf.create({"model": OmegaConf.load("configs/model/mrcge_v13.yaml"), "seed": seed})
    cfg.model.name = "mrcge_v13_static_strength" if static else "mrcge_v13"
    cfg.model.condition = "M_repaired"
    cfg.model.v13_variant = "joint_view"
    cfg.model.sender_private_ratio = 0.0
    if static:
        cfg.model = OmegaConf.load("configs/model/mrcge_v13_static_strength.yaml")
        cfg.model.static_strength_variant = "static_strength"
        cfg.model.static_strength_enabled = True
        cfg.model.static_strength_init = 0.0
    return cfg


def make_model(*, static: bool, seed: int = 42):
    torch.manual_seed(seed)
    cfg = make_cfg(static=static, seed=seed)
    model = build_model(cfg, DATA_INFO) if static else FrozenV13Model(cfg, DATA_INFO)
    attach_semantic(model)
    return model


def make_model_and_head(*, static: bool, seed: int):
    torch.manual_seed(seed)
    cfg = make_cfg(static=static, seed=seed)
    model = build_model(cfg, DATA_INFO) if static else FrozenV13Model(cfg, DATA_INFO)
    attach_semantic(model)
    return model, nn.Linear(model.out_dim, DATA_INFO["num_classes"])


def test_factory_builds_only_four_explicit_static_parameters():
    model = make_model(static=True)
    assert type(model).__module__ == "src.models.mrcge_v13_static_strength"
    assert model.static_strength_variant == "static_strength"
    assert model.supports_link_prediction is False
    assert model.condition == "M_repaired" and model.v13_variant == "joint_view"
    assert not hasattr(model, "joint_view")
    assert not any(name.startswith("joint_view.") for name, _ in model.named_parameters())
    assert tuple(model.static_logits.shape) == (2, 2)
    assert torch.equal(model.static_logits, torch.zeros((2, 2)))
    assert model.static_logits.numel() == 4
    assert torch.equal(model.static_strengths_all(), torch.ones((2, 2)))


def test_public_initialization_and_classifier_are_bitwise_equal():
    torch.manual_seed(61)
    reference = FrozenV13Model(make_cfg(static=False, seed=61), DATA_INFO)
    reference_head = nn.Linear(reference.out_dim, DATA_INFO["num_classes"])
    torch.manual_seed(61)
    candidate = build_model(make_cfg(static=True, seed=61), DATA_INFO)
    candidate_head = nn.Linear(candidate.out_dim, DATA_INFO["num_classes"])
    attach_semantic(reference)
    attach_semantic(candidate)

    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    public = [name for name in reference_state if not name.startswith("joint_view.")]
    assert set(public) == set(candidate_state) - {"static_logits"}
    for name in public:
        assert torch.equal(reference_state[name], candidate_state[name]), name
    assert torch.equal(reference.filter_bank.theta, candidate.filter_bank.theta)
    assert torch.equal(reference_head.weight, candidate_head.weight)
    assert torch.equal(reference_head.bias, candidate_head.bias)
    assert torch.equal(candidate.static_logits, torch.zeros_like(candidate.static_logits))


def test_initial_eval_private_collaborative_embedding_logits_and_aux_are_equal():
    reference, ref_head = make_model_and_head(static=False, seed=67)
    candidate, new_head = make_model_and_head(static=True, seed=67)
    assert torch.equal(ref_head.weight, new_head.weight)
    assert torch.equal(ref_head.bias, new_head.bias)
    reference._capture_state_audit = True
    candidate._capture_state_audit = True
    reference.eval(); candidate.eval()
    with torch.no_grad():
        z_ref, _, _, aux_ref, info_ref = reference(X, PHYSICAL, return_details=True)
        z_new, _, _, aux_new, info_new = candidate(X, PHYSICAL, return_details=True)
    torch.testing.assert_close(z_new, z_ref, atol=0, rtol=0)
    torch.testing.assert_close(new_head(z_new), ref_head(z_ref), atol=0, rtol=0)
    torch.testing.assert_close(aux_new, aux_ref, atol=0, rtol=0)
    for modality in ("text", "visual"):
        for branch in ("private", "collaborative"):
            torch.testing.assert_close(candidate._last_state_audit[branch][modality],
                                       reference._last_state_audit[branch][modality], atol=0, rtol=0)
    assert info_new["details"]["late"] == info_ref["details"]["late"]
    assert info_new["details"]["v13_mechanism"]["static_sP_text"] == 1.0
    assert info_new["details"]["v13_mechanism"]["static_sS_visual"] == 1.0


def test_initial_train_rng_and_all_common_parameter_gradients_match():
    reference, ref_head = make_model_and_head(static=False, seed=71)
    candidate, new_head = make_model_and_head(static=True, seed=71)
    reference.train(); candidate.train(); ref_head.train(); new_head.train()
    torch.manual_seed(20261026)
    z_ref, _, _, aux_ref, _ = reference(X, PHYSICAL)
    loss_ref = F.cross_entropy(ref_head(z_ref), torch.tensor([0, 1, 2, 1, 0, 2])) + aux_ref
    loss_ref.backward()
    torch.manual_seed(20261026)
    z_new, _, _, aux_new, _ = candidate(X, PHYSICAL)
    loss_new = F.cross_entropy(new_head(z_new), torch.tensor([0, 1, 2, 1, 0, 2])) + aux_new
    loss_new.backward()
    torch.testing.assert_close(z_new, z_ref, atol=0, rtol=0)
    torch.testing.assert_close(aux_new, aux_ref, atol=0, rtol=0)
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    for name in candidate_parameters:
        if name == "static_logits":
            continue
        assert name in reference_parameters, name
        left, right = reference_parameters[name].grad, candidate_parameters[name].grad
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6, msg=name)
    torch.testing.assert_close(ref_head.weight.grad, new_head.weight.grad, atol=0, rtol=0)
    torch.testing.assert_close(ref_head.bias.grad, new_head.bias.grad, atol=0, rtol=0)
    grad = candidate.static_logits.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) == 4


def test_modality_and_relation_strengths_are_independent_and_node_constant():
    model = make_model(static=True, seed=73)
    with torch.no_grad():
        model.static_logits.copy_(torch.tensor([[0.2, -0.3], [0.5, 0.1]]))
    values = [model.view_strength(i, torch.zeros_like(X[:, :1]).expand(-1, model.hidden_dim),
                                  torch.zeros(6, model.hidden_dim), torch.zeros(6, model.hidden_dim))
              for i in range(2)]
    assert values[0].shape == (6, 2) and values[1].shape == (6, 2)
    assert torch.equal(values[0], values[0][:1].expand_as(values[0]))
    assert torch.equal(values[1], values[1][:1].expand_as(values[1]))
    assert not torch.equal(values[0][0], values[1][0])
    assert values[0][0, 0] != values[0][0, 1]
    assert values[1][0, 0] != values[1][0, 1]
    grads = torch.autograd.grad(sum(v[:, i].sum() for v in values for i in range(2)),
                                model.static_logits)[0]
    assert torch.isfinite(grads).all() and torch.count_nonzero(grads) == 4


def test_same_strength_is_used_for_private_and_collaborative_and_zero_cross_is_exact():
    model = make_model(static=True, seed=79)
    model._capture_state_audit = True
    with torch.no_grad():
        model.static_logits.copy_(torch.tensor([[0.25, -0.2], [-0.3, 0.4]]))
        for parameter in model.cross_projections.parameters():
            parameter.zero_()
    model.train()
    torch.manual_seed(8801)
    _, _, _, _, info = model(X, PHYSICAL, return_details=True)
    for modality in ("text", "visual"):
        assert torch.equal(model._last_state_audit["private"][modality],
                           model._last_state_audit["collaborative"][modality])
        assert torch.equal(model._last_state_audit["private_strengths"][modality],
                           model._last_state_audit["collaborative_strengths"][modality])
        assert info["details"]["v13_mechanism"][f"collaborative_vs_private_sP_abs_mean_{modality}"] == 0.0


def test_static_parameter_task_gradient_and_both_early_cross_directions():
    model = make_model(static=True, seed=83)
    model.train()
    logits, _, _, aux, _ = model(X, PHYSICAL)
    loss = F.cross_entropy(nn.Linear(model.out_dim, DATA_INFO["num_classes"])(logits),
                           torch.tensor([0, 1, 2, 1, 0, 2])) + aux
    loss.backward()
    gradient = model.static_logits.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 4
    model.eval()
    with torch.no_grad():
        _, _, _, _, info = model(X, PHYSICAL, return_details=True)
    mechanism = info["details"]["v13_mechanism"]
    for modality in ("text", "visual"):
        assert mechanism[f"early_cross_calibrated_rms_{modality}"] > 0
        assert mechanism[f"early_cross_response_delta_rms_{modality}"] > 0
        assert mechanism[f"effective_rho_sP_{modality}"] > 0
        assert mechanism[f"effective_gamma_sS_{modality}"] > 0


def test_no_graph_support_has_no_graph_message_and_static_remains_finite():
    model = make_model(static=True, seed=89)
    empty = torch.empty((2, 0), dtype=torch.long)
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index", empty)
        setattr(model, f"semantic_{modality}_weight", torch.empty(0))
    model.train()
    z, _, _, aux, info = model(X, empty, return_details=True)
    assert torch.isfinite(z).all() and torch.isfinite(aux)
    assert info["details"]["physical_support_nodes"] == 0
    for modality in ("text", "visual"):
        assert info["details"]["semantic_support_nodes"][modality] == 0
        assert model._last_diagnostics["physical_response_rms"][modality] == 0.0


def test_low_memory_training_path_matches_eval_path_and_static_gradients():
    full = make_model(static=True, seed=97)
    low = make_model(static=True, seed=97)
    low.load_state_dict(full.state_dict(), strict=True)
    for module in list(full.modules()) + list(low.modules()):
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    full.eval(); low.train(); low.late_cross_dropout = 0.0
    z_full, _, _, aux_full, _ = full(X, PHYSICAL)
    (z_full.square().mean() + aux_full).backward()
    z_low, _, _, aux_low, _ = low(X, PHYSICAL)
    (z_low.square().mean() + aux_low).backward()
    torch.testing.assert_close(z_low, z_full, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(aux_low, aux_full, atol=2e-6, rtol=2e-5)
    full_params, low_params = dict(full.named_parameters()), dict(low.named_parameters())
    assert full_params.keys() == low_params.keys()
    for name in full_params:
        g0, g1 = full_params[name].grad, low_params[name].grad
        assert (g0 is None) == (g1 is None), name
        if g0 is not None:
            torch.testing.assert_close(g1, g0, atol=3e-5, rtol=3e-4, msg=name)
    for name in ("static_logits", "filter_bank.theta", "router.net.3.weight"):
        assert full_params[name].grad is not None and torch.isfinite(full_params[name].grad).all()


def test_strict_checkpoint_reload_reproduces_eval_output_and_validation_metrics():
    model = make_model(static=True, seed=101)
    head = nn.Linear(model.out_dim, DATA_INFO["num_classes"])
    model.eval(); head.eval()
    with torch.no_grad():
        z, _, _, aux, _ = model(X, PHYSICAL)
        logits = head(z)
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "head": head.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=False)
    reloaded = make_model(static=True, seed=997)
    reloaded_head = nn.Linear(reloaded.out_dim, DATA_INFO["num_classes"])
    reloaded.load_state_dict(payload["model"], strict=True)
    reloaded_head.load_state_dict(payload["head"], strict=True)
    reloaded.eval(); reloaded_head.eval()
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        z2, _, _, aux2, _ = reloaded(X, PHYSICAL)
        logits2 = reloaded_head(z2)
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(z2, z, atol=0, rtol=0)
    torch.testing.assert_close(aux2, aux, atol=0, rtol=0)
    torch.testing.assert_close(logits2, logits, atol=0, rtol=0)
    labels = torch.tensor([0, 1, 2, 1, 0, 2])
    val_idx = torch.tensor([1, 2, 5])
    torch.testing.assert_close(F.cross_entropy(logits2[val_idx], labels[val_idx]),
                               F.cross_entropy(logits[val_idx], labels[val_idx]), atol=0, rtol=0)
