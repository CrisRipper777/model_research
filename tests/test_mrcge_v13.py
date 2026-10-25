from __future__ import annotations

import io

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from scripts import run_mrcge_v13_screen as screen
from src.models.mrcge_v12 import Model as ModelV12
from src.models.mrcge_v13 import Model as ModelV13


DATA_INFO = {
    "input_dim": 8,
    "text_dim": 4,
    "visual_dim": 4,
    "num_nodes": 12,
    "num_classes": 3,
}
X = torch.arange(96, dtype=torch.float32).reshape(12, 8) / 31.0
PHYSICAL = torch.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]],
    dtype=torch.long,
)
SEMANTIC = torch.tensor(
    [[0, 2, 4, 6, 8, 10], [2, 4, 6, 8, 10, 0]], dtype=torch.long
)


def make_model(variant: str = "base", ratio: float = 0.0, seed: int = 42):
    torch.manual_seed(seed)
    name = "mrcge_v12" if variant == "v12" else "mrcge_v13"
    config_name = "mrcge_v12" if variant == "v12" else "mrcge_v13"
    model_cfg = OmegaConf.load(f"configs/model/{config_name}.yaml")
    model_cfg.condition = "M_repaired"
    if variant != "v12":
        model_cfg.v13_variant = variant
        model_cfg.sender_private_ratio = ratio
    model = (ModelV12 if variant == "v12" else ModelV13)(
        OmegaConf.create({"model": model_cfg}), DATA_INFO
    )
    for modality, edges in (("text", SEMANTIC), ("visual", SEMANTIC.flip(0))):
        setattr(model, f"semantic_{modality}_edge_index", edges.clone())
        setattr(model, f"semantic_{modality}_weight", torch.ones(edges.size(1)))
    model.semantic_candidate_fingerprint = "synthetic-candidate-fingerprint"
    model.semantic_feature_fingerprint = "synthetic-feature-fingerprint"
    model.semantic_v7b_fingerprint = "synthetic-v7b-fingerprint"
    return model


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max())


@pytest.mark.parametrize("variant,ratio", [("base", 0.0), ("joint_view", 0.0), ("sender_private", 0.0)])
def test_v13_compatibility_variants_match_v12_and_classifier_initialization(variant, ratio):
    baseline = make_model("v12", seed=123)
    head_baseline = nn.Linear(baseline.out_dim, DATA_INFO["num_classes"])
    candidate = make_model(variant, ratio=ratio, seed=123)
    for name, tensor in baseline.state_dict().items():
        assert name in candidate.state_dict()
        assert torch.equal(tensor, candidate.state_dict()[name]), name
    head_candidate = nn.Linear(candidate.out_dim, DATA_INFO["num_classes"])
    for name, tensor in head_baseline.state_dict().items():
        assert torch.equal(tensor, head_candidate.state_dict()[name]), name

    baseline.eval()
    candidate.eval()
    old = baseline(X, PHYSICAL)
    new = candidate(X, PHYSICAL)
    assert torch.allclose(old[0], new[0], atol=2e-6, rtol=2e-5)
    assert max_abs(old[3], new[3]) <= 1e-8
    assert not candidate.supports_link_prediction
    assert candidate.condition == "M_repaired"
    assert candidate.v13_variant == (variant if variant != "v12" else "base")


def test_joint_view_starts_at_unit_strength_and_preserves_forward():
    baseline = make_model("v12", seed=19)
    candidate = make_model("joint_view", seed=19)
    assert torch.equal(candidate.joint_view.output.weight, torch.zeros_like(candidate.joint_view.output.weight))
    assert torch.equal(candidate.joint_view.output.bias, torch.zeros_like(candidate.joint_view.output.bias))
    parts = candidate.joint_view(X[:, :4].repeat(1, 64), X[:, :4].repeat(1, 64), X[:, :4].repeat(1, 64))
    assert torch.equal(parts, torch.ones_like(parts))
    baseline.eval()
    candidate.eval()
    assert torch.allclose(baseline(X, PHYSICAL)[0], candidate(X, PHYSICAL)[0], atol=2e-6, rtol=2e-5)


def test_joint_view_has_gradients_for_both_strength_outputs():
    model = make_model("joint_view", seed=71)
    model.train()
    torch.manual_seed(9)
    output, _, _, aux, _ = model(X, PHYSICAL)
    weights = torch.linspace(0.2, 1.7, output.numel()).reshape_as(output)
    loss = (output * weights).sum() + aux
    loss.backward()
    grad = model.joint_view.output.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert bool((grad.norm(dim=1) > 0).all())
    assert model.supports_link_prediction is False


@pytest.mark.parametrize("variant", ["base", "joint_view", "sender_private"])
def test_gradient_audit_supports_each_m_repaired_variant(variant):
    model = make_model(variant, ratio=0.3 if variant == "sender_private" else 0.0, seed=75)
    model.train()
    output, _, _, aux, _ = model(X, PHYSICAL)
    (output.square().mean() + aux).backward()
    diagnostics = model.capture_gradient_diagnostics()
    assert diagnostics
    assert all(row["gradient_finite"] for row in diagnostics.values())
    if variant == "joint_view":
        assert diagnostics["joint_view.output.weight"]["gradient_nonzero"]
    else:
        assert not any(name.startswith("single_filter") for name in diagnostics)


@pytest.mark.parametrize("variant", ["joint_view", "sender_private"])
def test_zero_cross_invariant_holds_with_paired_train_dropout(variant):
    model = make_model(variant, ratio=0.3 if variant == "sender_private" else 0.0, seed=89)
    model._capture_state_audit = True
    model.train()
    with torch.no_grad():
        for parameter in model.cross_projections.parameters():
            parameter.zero_()
    torch.manual_seed(2025)
    model(X, PHYSICAL)
    for modality in ("text", "visual"):
        assert torch.equal(
            model._last_state_audit["private"][modality],
            model._last_state_audit["collaborative"][modality],
        )
        assert torch.count_nonzero(model._last_state_audit["early_cross"][modality]) == 0


def test_sender_private_gradient_flows_to_source_private_encoder():
    model = make_model("sender_private", ratio=0.3, seed=101)
    model._capture_state_audit = True
    model.train()
    model(X, PHYSICAL)
    text_receiver_cross = model._last_state_audit["early_cross"]["text"]
    grad = torch.autograd.grad(
        text_receiver_cross.square().sum(),
        model.projectors.visual.linear1.weight,
        allow_unused=True,
    )[0]
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.norm()) > 0


def test_view_support_masks_and_isolated_nodes_do_not_create_graph_messages():
    model = make_model("joint_view", seed=111)
    model._capture_state_audit = True
    model.train()
    empty = torch.empty((2, 0), dtype=torch.long)
    model.semantic_text_edge_index = empty
    model.semantic_text_weight = torch.empty((0,))
    model.semantic_visual_edge_index = empty
    model.semantic_visual_weight = torch.empty((0,))
    output, _, _, aux, info = model(X, empty, return_details=True)
    assert torch.isfinite(output).all() and torch.isfinite(aux)
    assert info["details"]["physical_support_nodes"] == 0
    assert info["details"]["semantic_support_nodes"] == {"text": 0, "visual": 0}
    for modality in ("text", "visual"):
        assert torch.count_nonzero(model._last_state_audit["responses"][modality]["physical"]) == 0
        assert torch.count_nonzero(model._last_state_audit["responses"][modality]["semantic"]) == 0
        assert torch.equal(
            model._last_state_audit["private"][modality],
            model._last_state_audit["collaborative"][modality],
        )
    (output.square().mean() + aux).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpoint_strict_reload_reproduces_output():
    model = make_model("joint_view", seed=141)
    model.eval()
    expected = model(X, PHYSICAL)[0]
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = make_model("joint_view", seed=333)
    restored.load_state_dict(state, strict=True)
    restored.eval()
    actual = restored(X, PHYSICAL)[0]
    assert torch.equal(expected, actual)


def test_screen_matrix_is_exact_and_rejects_unapproved_cells():
    cells = screen.target_cells()
    assert len(cells) == 12
    assert set(cells) == {
        (dataset, variant, seed)
        for dataset in ("Movies", "Grocery", "ele-fashion")
        for variant in ("joint_view", "sender_private")
        for seed in (42, 43)
    }
    with pytest.raises(ValueError, match="unsupported V1.3 variant"):
        screen.target_cells(variants=("joint_view+sender_private",))
    with pytest.raises(ValueError, match="unsupported V1.3 seeds"):
        screen.target_cells(seeds=(44,))


def test_eval_diagnostics_preserve_rng_and_epoch_collection_returns_scalars():
    model = make_model("joint_view", seed=151)
    model.train()
    train_output = model(X, PHYSICAL)[0]
    train_output.square().mean().backward()
    model.eval()
    cpu_rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        model(X, PHYSICAL)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    diagnostics = model.collect_epoch_diagnostics()
    assert "train_v13_private_sP_mean_text" in diagnostics
    assert "val_v13_early_cross_to_neighbor_rms_visual" in diagnostics
    assert all(isinstance(value, (float, int)) for value in diagnostics.values())


@pytest.mark.parametrize("variant", ["base", "joint_view", "sender_private"])
def test_weighted_contribution_rms_matches_captured_validation_tensors(variant):
    model = make_model(variant, ratio=0.3 if variant == "sender_private" else 0.0, seed=157)
    model.eval()
    model._capture_state_audit = True
    with torch.no_grad():
        _, _, _, _, info = model(X, PHYSICAL, return_details=True)
    mechanism = info["details"]["v13_mechanism"]
    physical_support = model._support(PHYSICAL, X.size(0), X.device)
    audit = model._last_state_audit
    for modality in ("text", "visual"):
        semantic_edges = getattr(model, f"semantic_{modality}_edge_index")
        semantic_support = model._support(semantic_edges, X.size(0), X.device)
        expected = {
            f"weighted_physical_private_contribution_rms_{modality}": model._rms(
                audit["private_contributions"][modality]["physical"], physical_support),
            f"weighted_semantic_private_contribution_rms_{modality}": model._rms(
                audit["private_contributions"][modality]["semantic"], semantic_support),
            f"weighted_physical_collaborative_contribution_rms_{modality}": model._rms(
                audit["collaborative_contributions"][modality]["physical"], physical_support),
        }
        for name, value in expected.items():
            assert abs(float(mechanism[name]) - float(value)) <= 1e-7
