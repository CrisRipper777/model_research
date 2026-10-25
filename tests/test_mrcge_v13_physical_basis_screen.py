from __future__ import annotations

import io

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from omegaconf import OmegaConf
from torch import nn

from src.models import build_model
from src.models.hybrid_cge_paper_v1 import Model as CorrectedHybridModel
from src.models.mrcge_v13 import Model as FrozenV13Model
from src.models.rmcge_v2 import build_modality_knn_operators


DATA_INFO = {"input_dim": 7, "text_dim": 3, "visual_dim": 4, "num_nodes": 6, "num_classes": 3}
X = torch.arange(42, dtype=torch.float32).reshape(6, 7) / 29.0
PHYSICAL = torch.tensor(
    [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
)
SEMANTIC_UNIQUE = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 1]], dtype=torch.long)
SEMANTIC_FLAGS = torch.tensor(
    [[1, 0], [0, 1], [1, 1], [0, 1], [1, 0], [1, 1]], dtype=torch.uint8
)


def make_basis_model(*, no_p0: bool, seed: int = 42):
    cfg = OmegaConf.create({
        "model": OmegaConf.load("configs/model/mrcge_v13.yaml"),
        "seed": seed,
    })
    cfg.model.name = "mrcge_v13_basis_screen"
    cfg.model.condition = "M_repaired"
    cfg.model.v13_variant = "joint_view"
    cfg.model.sender_private_ratio = 0.0
    cfg.model.basis_screen_variant = (
        "joint_view_physical_no_p0" if no_p0 else "joint_view_original"
    )
    cfg.model.physical_no_p0 = no_p0
    torch.manual_seed(seed)
    model = build_model(cfg, DATA_INFO)
    attach_v13_semantic(model)
    return model


def make_frozen_v13(seed: int = 42):
    cfg = OmegaConf.create({"model": OmegaConf.load("configs/model/mrcge_v13.yaml"), "seed": seed})
    cfg.model.name = "mrcge_v13"
    cfg.model.condition = "M_repaired"
    cfg.model.v13_variant = "joint_view"
    cfg.model.sender_private_ratio = 0.0
    torch.manual_seed(seed)
    model = FrozenV13Model(cfg, DATA_INFO)
    attach_v13_semantic(model)
    return model


def attach_v13_semantic(model):
    text = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    visual = torch.tensor([[0, 2, 4, 1, 3, 5], [2, 4, 1, 3, 5, 0]], dtype=torch.long)
    for modality, edges in (("text", text), ("visual", visual)):
        setattr(model, f"semantic_{modality}_edge_index", edges)
        setattr(model, f"semantic_{modality}_weight", torch.ones(edges.size(1)))
    model.semantic_candidate_fingerprint = "a" * 64
    model.semantic_feature_fingerprint = "b" * 64
    model.semantic_v7b_fingerprint = "c" * 64


def make_corrected_hybrid(seed: int = 42):
    cfg = OmegaConf.create({
        "model": OmegaConf.load("configs/model/hybrid_cge_paper_v1.yaml"), "seed": seed
    })
    cfg.model.name = "hybrid_cge_paper_v1"
    cfg.model.paper_condition = "H2_physical_3hop"
    cfg.model.condition = "H2_hybrid_plus"
    cfg.model.physical_hops = 3
    torch.manual_seed(seed)
    model = CorrectedHybridModel(cfg, DATA_INFO)
    operators = build_modality_knn_operators(SEMANTIC_UNIQUE, SEMANTIC_FLAGS, DATA_INFO["num_nodes"])
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index_v2", operators[modality]["edge_index"])
        setattr(model, f"semantic_{modality}_weight_v2", operators[modality]["weight"])
    model.semantic_candidate_fingerprint = "d" * 64
    model.semantic_v7b_fingerprint = "e" * 64
    return model


def test_factory_loads_explicit_no_p0_model_and_frozen_files_stay_separate():
    model = make_basis_model(no_p0=True)
    assert type(model).__module__ == "src.models.mrcge_v13_basis_screen"
    assert model.basis_screen_variant == "joint_view_physical_no_p0"
    assert model.physical_no_p0 is True
    assert model.supports_link_prediction is False
    assert model.condition == "M_repaired"
    assert model.v13_variant == "joint_view"


def test_original_switch_is_bitwise_initial_and_forward_compatible_with_v13_a():
    reference = make_frozen_v13(seed=61)
    reference_head = nn.Linear(reference.out_dim, DATA_INFO["num_classes"])
    candidate = make_basis_model(no_p0=False, seed=61)
    candidate_head = nn.Linear(candidate.out_dim, DATA_INFO["num_classes"])
    assert list(reference.state_dict()) == list(candidate.state_dict())
    for name, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    assert torch.equal(reference_head.weight, candidate_head.weight)
    assert torch.equal(reference_head.bias, candidate_head.bias)
    assert torch.equal(reference.filter_bank.theta, candidate.filter_bank.theta)
    assert torch.equal(reference.filter_bank.coefficients(3, torch.float32), candidate.physical_alpha())
    assert torch.equal(reference.filter_bank.coefficients(1, torch.float32), candidate.semantic_alpha())

    reference.eval(); candidate.eval()
    with torch.no_grad():
        z_ref, _, _, aux_ref, _ = reference(X, PHYSICAL)
        z_new, _, _, aux_new, _ = candidate(X, PHYSICAL)
        logits_ref, logits_new = reference_head(z_ref), candidate_head(z_new)
    torch.testing.assert_close(z_new, z_ref, atol=0, rtol=0)
    torch.testing.assert_close(aux_new, aux_ref, atol=0, rtol=0)
    torch.testing.assert_close(logits_new, logits_ref, atol=0, rtol=0)


def test_original_switch_matches_training_rng_and_initial_parameter_gradients():
    reference = make_frozen_v13(seed=67)
    ref_head = nn.Linear(reference.out_dim, DATA_INFO["num_classes"])
    candidate = make_basis_model(no_p0=False, seed=67)
    new_head = nn.Linear(candidate.out_dim, DATA_INFO["num_classes"])
    assert torch.equal(ref_head.weight, new_head.weight)
    assert torch.equal(ref_head.bias, new_head.bias)
    reference.train(); candidate.train()
    torch.manual_seed(20261026)
    z_ref, _, _, aux_ref, _ = reference(X, PHYSICAL)
    loss_ref = z_ref.square().mean() + aux_ref
    loss_ref.backward()
    torch.manual_seed(20261026)
    z_new, _, _, aux_new, _ = candidate(X, PHYSICAL)
    loss_new = z_new.square().mean() + aux_new
    loss_new.backward()
    torch.testing.assert_close(z_new, z_ref, atol=0, rtol=0)
    torch.testing.assert_close(aux_new, aux_ref, atol=0, rtol=0)
    ref_params, new_params = dict(reference.named_parameters()), dict(candidate.named_parameters())
    assert ref_params.keys() == new_params.keys()
    for name in ref_params:
        a, b = ref_params[name].grad, new_params[name].grad
        assert (a is None) == (b is None), name
        if a is not None:
            torch.testing.assert_close(a, b, atol=0, rtol=0, msg=name)


def test_physical_no_p0_masks_only_physical_coefficients_and_keeps_theta_trainable():
    model = make_basis_model(no_p0=True, seed=71)
    alpha_p = model.physical_alpha()
    alpha_s = model.semantic_alpha()
    assert torch.equal(alpha_p[:, 0], torch.zeros_like(alpha_p[:, 0]))
    expected_raw = model.filter_bank.theta.detach().clone()
    expected_raw[:, 0] = 0
    expected = expected_raw / (expected_raw.abs().sum(dim=-1, keepdim=True) + model.eps)
    torch.testing.assert_close(alpha_p, expected, atol=0, rtol=0)
    torch.testing.assert_close(alpha_p[:, 1:].abs().sum(-1), torch.ones(4), atol=1e-7, rtol=1e-7)
    expected_semantic = model.filter_bank.theta.detach().clone()
    expected_semantic[:, 2:] = 0
    expected_semantic /= expected_semantic.abs().sum(dim=-1, keepdim=True) + model.eps
    torch.testing.assert_close(alpha_s, expected_semantic, atol=0, rtol=0)
    assert model.filter_bank.theta.requires_grad
    assert "filter_bank.theta" in dict(model.named_parameters())

    model.eval()
    with torch.no_grad():
        _, _, _, _, info = model(X, PHYSICAL, return_details=True)
    filters = info["details"]["filters"]
    for name in ("private_text_physical", "private_visual_physical",
                 "collaborative_text_physical", "collaborative_visual_physical"):
        assert torch.equal(torch.tensor(filters[name]["coefficients"])[:, 0], torch.zeros(4)), name
    for name in ("private_text_semantic", "private_visual_semantic"):
        coefficients = torch.tensor(filters[name]["coefficients"])
        assert torch.equal(coefficients[:, 0], alpha_s[:, 0])
        assert torch.equal(coefficients[:, 1], alpha_s[:, 1])


def test_p0_remains_in_propagation_basis_and_all_three_higher_orders_are_built():
    model = make_basis_model(no_p0=True, seed=73)
    edges, weights, support, _ = model._physical_operator(PHYSICAL, 6, torch.float32)
    initial = torch.randn(6, model.hidden_dim)
    basis = model._basis(initial, edges, weights, support, 3, model.physical_chunk_size)
    assert len(basis) == 4
    assert torch.equal(basis[0], F.layer_norm(initial, (model.hidden_dim,)))
    assert all(torch.isfinite(value).all() for value in basis)
    assert all(torch.count_nonzero(value) > 0 for value in basis[1:])
    assert torch.equal(model.physical_alpha()[:, 0], torch.zeros(4))


def test_eval_stack_and_low_memory_checkpointed_paths_match_outputs_aux_and_gradients():
    full = make_basis_model(no_p0=True, seed=79)
    low = make_basis_model(no_p0=True, seed=79)
    low.load_state_dict(full.state_dict(), strict=True)
    for module in list(full.modules()) + list(low.modules()):
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    low.late_cross_dropout = 0.0
    full.eval()
    low.train()
    z_full, _, _, aux_full, _ = full(X, PHYSICAL)
    loss_full = z_full.square().mean() + aux_full
    loss_full.backward()
    torch.manual_seed(10101)
    z_low, _, _, aux_low, _ = low(X, PHYSICAL)
    loss_low = z_low.square().mean() + aux_low
    loss_low.backward()
    torch.testing.assert_close(z_low, z_full, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(aux_low, aux_full, atol=2e-6, rtol=2e-5)
    full_params, low_params = dict(full.named_parameters()), dict(low.named_parameters())
    assert full_params.keys() == low_params.keys()
    for name in full_params:
        g0, g1 = full_params[name].grad, low_params[name].grad
        assert (g0 is None) == (g1 is None), name
        if g0 is not None:
            torch.testing.assert_close(g1, g0, atol=2e-6, rtol=2e-5, msg=name)
    for prefix in ("filter_bank.theta", "joint_view.", "router."):
        names = [name for name in full_params if name == prefix or name.startswith(prefix)]
        assert names
        assert all(full_params[name].grad is not None for name in names), prefix


def test_no_p0_cross_zero_paired_dropout_and_gradient_through_semantic_theta_zero():
    model = make_basis_model(no_p0=True, seed=83)
    model._capture_state_audit = True
    model.train()
    with torch.no_grad():
        for parameter in model.cross_projections.parameters():
            parameter.zero_()
    torch.manual_seed(4401)
    z, _, _, aux, _ = model(X, PHYSICAL)
    for modality in ("text", "visual"):
        assert torch.equal(model._last_state_audit["private"][modality],
                           model._last_state_audit["collaborative"][modality])
    (z.square().mean() + aux).backward()
    theta_grad = model.filter_bank.theta.grad
    assert theta_grad is not None and torch.isfinite(theta_grad).all()
    assert bool((theta_grad[:, 0].abs() > 0).any()), "theta[:,0] remains active through Semantic order 0"
    assert bool((theta_grad[:, 1:].abs() > 0).any()), "Physical orders 1-3 remain trainable"


def test_no_support_keeps_graph_messages_zero_with_no_p0():
    model = make_basis_model(no_p0=True, seed=89)
    empty = torch.empty((2, 0), dtype=torch.long)
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index", empty)
        setattr(model, f"semantic_{modality}_weight", torch.empty(0))
    model._capture_state_audit = True
    model.train()
    z, _, _, aux, info = model(X, empty, return_details=True)
    assert torch.isfinite(z).all() and torch.isfinite(aux)
    assert info["details"]["physical_support_nodes"] == 0
    for modality in ("text", "visual"):
        assert info["details"]["semantic_support_nodes"][modality] == 0
        assert torch.count_nonzero(model._last_state_audit["responses"][modality]["physical"]) == 0
        assert torch.count_nonzero(model._last_state_audit["responses"][modality]["semantic"]) == 0
    (z.square().mean() + aux).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_no_p0_strict_state_reload_reproduces_eval_logits_and_validation_metrics():
    model = make_basis_model(no_p0=True, seed=97)
    head = nn.Linear(model.out_dim, DATA_INFO["num_classes"])
    model.eval(); head.eval()
    with torch.no_grad():
        z, _, _, aux, _ = model(X, PHYSICAL)
        logits = head(z)
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "head": head.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=False)
    reloaded = make_basis_model(no_p0=True, seed=997)
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
    metric0 = F.cross_entropy(logits[val_idx], labels[val_idx])
    metric1 = F.cross_entropy(logits2[val_idx], labels[val_idx])
    torch.testing.assert_close(metric1, metric0, atol=0, rtol=0)
    pred0, pred1 = logits[val_idx].argmax(-1), logits2[val_idx].argmax(-1)
    assert torch.equal(pred0, pred1)
    assert float((pred0 == labels[val_idx]).float().mean()) == float((pred1 == labels[val_idx]).float().mean())
    assert f1_score(labels[val_idx].numpy(), pred0.numpy(), labels=[0, 1, 2],
                    average="macro", zero_division=0) == f1_score(
        labels[val_idx].numpy(), pred1.numpy(), labels=[0, 1, 2],
        average="macro", zero_division=0)


def test_no_p0_early_physical_cross_runs_in_both_modality_directions():
    model = make_basis_model(no_p0=True, seed=101)
    model.eval()
    with torch.no_grad():
        _, _, _, _, info = model(X, PHYSICAL, return_details=True)
    mechanism = info["details"]["v13_mechanism"]
    for modality in ("text", "visual"):
        assert mechanism[f"early_cross_raw_rms_{modality}"] > 0
        assert mechanism[f"early_cross_calibrated_rms_{modality}"] > 0
        assert mechanism[f"early_cross_to_neighbor_rms_{modality}"] > 0


def test_corrected_hybrid_h2_three_hop_same_node_candidate_is_live():
    model = make_corrected_hybrid(seed=103)
    model.eval()
    with torch.no_grad():
        _, _, _, _, info = model(torch.randn(6, 7), PHYSICAL, return_details=True)
    assert model.paper_condition == "H2_physical_3hop"
    assert model.physical_hops == 3
    for modality in ("text", "visual"):
        late = info["details"]["modalities"][modality]["late_completion"]
        assert late["same_node_support_nodes"] == 6
        assert late["same_node_candidate_rms"] > 0
        assert late["same_node_source_weight"] > 0
