from __future__ import annotations

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
LABELS = torch.tensor([0, 1, 2, 1, 0, 2])


def attach_semantic(model):
    for modality, edges in (("text", SEMANTIC_TEXT), ("visual", SEMANTIC_VISUAL)):
        setattr(model, f"semantic_{modality}_edge_index", edges)
        setattr(model, f"semantic_{modality}_weight", torch.ones(edges.size(1)))
    model.semantic_candidate_fingerprint = "a" * 64
    model.semantic_feature_fingerprint = "b" * 64
    model.semantic_v7b_fingerprint = "c" * 64


def make_cfg(*, candidate: bool, seed: int = 42):
    if candidate:
        model_cfg = OmegaConf.load("configs/model/mrcge_v13_early_cross_decoupled.yaml")
    else:
        model_cfg = OmegaConf.load("configs/model/mrcge_v13.yaml")
        model_cfg.name = "mrcge_v13"
        model_cfg.condition = "M_repaired"
        model_cfg.v13_variant = "joint_view"
        model_cfg.sender_private_ratio = 0.0
    return OmegaConf.create({"model": model_cfg, "seed": seed})


def make_model(*, candidate: bool, seed: int = 42):
    torch.manual_seed(seed)
    cfg = make_cfg(candidate=candidate, seed=seed)
    model = build_model(cfg, DATA_INFO) if candidate else FrozenV13Model(cfg, DATA_INFO)
    attach_semantic(model)
    return model


def test_explicit_variant_adds_only_four_deterministically_copied_parameters():
    torch.manual_seed(610)
    reference = FrozenV13Model(make_cfg(candidate=False, seed=610), DATA_INFO)
    ref_head = nn.Linear(reference.out_dim, DATA_INFO["num_classes"])
    torch.manual_seed(610)
    candidate = build_model(make_cfg(candidate=True, seed=610), DATA_INFO)
    new_head = nn.Linear(candidate.out_dim, DATA_INFO["num_classes"])
    attach_semantic(reference); attach_semantic(candidate)

    ref_state, new_state = reference.state_dict(), candidate.state_dict()
    assert set(new_state) == set(ref_state) | {"cross_coefficients"}
    for name, value in ref_state.items():
        assert torch.equal(value, new_state[name]), name
    alpha = candidate.filter_bank.coefficients(3, candidate.filter_bank.theta.dtype)
    assert candidate.cross_coefficients.shape == (4,)
    assert candidate.cross_coefficients.numel() == 4
    assert torch.equal(candidate.cross_coefficients, alpha[:, 1])
    assert torch.equal(ref_head.weight, new_head.weight)
    assert torch.equal(ref_head.bias, new_head.bias)
    assert candidate.supports_link_prediction is False
    assert candidate.v13_variant == "joint_view" and candidate.condition == "M_repaired"
    assert candidate.early_cross_decoupling_variant == "expert_decoupled_early_cross"


def test_initial_eval_and_train_paths_are_bitwise_equal_with_paired_dropout():
    reference, candidate = make_model(candidate=False, seed=670), make_model(candidate=True, seed=670)
    ref_head, new_head = nn.Linear(reference.out_dim, 3), nn.Linear(candidate.out_dim, 3)
    torch.manual_seed(670)
    ref_head = nn.Linear(reference.out_dim, 3)
    torch.manual_seed(670)
    new_head = nn.Linear(candidate.out_dim, 3)
    reference._capture_state_audit = True; candidate._capture_state_audit = True
    reference.eval(); candidate.eval()
    with torch.no_grad():
        z0, _, _, aux0, info0 = reference(X, PHYSICAL, return_details=True)
        z1, _, _, aux1, info1 = candidate(X, PHYSICAL, return_details=True)
    assert torch.equal(z0, z1) and torch.equal(aux0, aux1)
    assert torch.equal(ref_head(z0), new_head(z1))
    assert info0["details"]["late"] == info1["details"]["late"]
    for modality in ("text", "visual"):
        for branch in ("private", "collaborative"):
            assert torch.equal(reference._last_state_audit[branch][modality],
                               candidate._last_state_audit[branch][modality])
        for name in (f"private_{modality}_physical", f"collaborative_{modality}_physical"):
            assert info0["details"]["filters"][name]["G_expert_rms"] == \
                info1["details"]["filters"][name]["G_expert_rms"]
            assert info0["details"]["filters"][name]["G_pairwise_cosine"] == \
                info1["details"]["filters"][name]["G_pairwise_cosine"]

    reference.train(); candidate.train(); ref_head.train(); new_head.train()
    torch.manual_seed(20261026)
    z0, _, _, aux0, _ = reference(X, PHYSICAL)
    (F.cross_entropy(ref_head(z0), LABELS) + aux0).backward()
    torch.manual_seed(20261026)
    z1, _, _, aux1, _ = candidate(X, PHYSICAL)
    (F.cross_entropy(new_head(z1), LABELS) + aux1).backward()
    assert torch.equal(z0, z1) and torch.equal(aux0, aux1)
    ref_params, new_params = dict(reference.named_parameters()), dict(candidate.named_parameters())
    assert set(new_params) == set(ref_params) | {"cross_coefficients"}
    for name, p_new in new_params.items():
        if name == "cross_coefficients":
            continue
        p_ref = ref_params[name]
        assert (p_ref.grad is None) == (p_new.grad is None), name
        if p_ref.grad is not None and name != "filter_bank.theta":
            torch.testing.assert_close(p_ref.grad, p_new.grad, atol=2e-7, rtol=2e-6, msg=name)
    assert ref_params["filter_bank.theta"].grad is not None
    assert new_params["filter_bank.theta"].grad is not None
    assert torch.isfinite(new_params["filter_bank.theta"].grad).all()
    assert not torch.equal(ref_params["filter_bank.theta"].grad, new_params["filter_bank.theta"].grad)
    torch.testing.assert_close(ref_head.weight.grad, new_head.weight.grad, atol=0, rtol=0)
    torch.testing.assert_close(ref_head.bias.grad, new_head.bias.grad, atol=0, rtol=0)
    b_grad = new_params["cross_coefficients"].grad
    assert b_grad is not None and torch.isfinite(b_grad).all() and torch.count_nonzero(b_grad) == 4


def test_analytic_theta_gradient_cancels_only_old_cross_coefficient_path():
    model = make_model(candidate=True, seed=710)
    theta = model.filter_bank.theta
    alpha = model.filter_bank.coefficients(3, torch.float32)
    beta = model.beta[0]
    generator = torch.Generator().manual_seed(711)
    bases = [torch.randn((3, 5), generator=generator) for _ in range(4)]
    cross = torch.randn((3, 5), generator=generator)
    private_g = torch.einsum("kl,lnd->knd", alpha, torch.stack(bases))
    old_cross = alpha[:, 1, None, None] * beta * cross[None]
    old_g = private_g + old_cross
    new_g = old_g + beta * (model.cross_coefficients - alpha[:, 1])[:, None, None] * cross[None]

    old_theta_grad = torch.autograd.grad(old_cross.square().sum(), theta, retain_graph=True)[0]
    new_theta_grad, b_grad = torch.autograd.grad(new_g.square().sum(), (theta, model.cross_coefficients),
                                                  retain_graph=True)
    # Isolate the path derivative by using a scalar linear probe so the expected
    # theta derivative is precisely that of the Private polynomial input.
    probe = torch.arange(new_g.numel(), dtype=new_g.dtype).reshape_as(new_g) / new_g.numel()
    expected_theta_grad = torch.autograd.grad((private_g * probe).sum(), theta, retain_graph=True)[0]
    actual_theta_grad = torch.autograd.grad((new_g * probe).sum(), theta, retain_graph=True)[0]
    actual_b_grad = torch.autograd.grad((new_g * probe).sum(), model.cross_coefficients)[0]
    assert torch.isfinite(old_theta_grad).all() and torch.count_nonzero(old_theta_grad) > 0
    torch.testing.assert_close(actual_theta_grad, expected_theta_grad, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(actual_b_grad).all() and torch.count_nonzero(actual_b_grad) == 4
    assert torch.isfinite(b_grad).all()


def test_perturbed_current_alpha_can_be_recopied_for_exact_equivalence():
    reference, candidate = make_model(candidate=False, seed=730), make_model(candidate=True, seed=730)
    theta = torch.tensor([[0.8, -0.2, 0.3, -0.1], [-0.4, 0.9, -0.2, 0.1],
                          [0.3, 0.2, -0.7, 0.4], [-0.2, -0.5, 0.6, 0.3]])
    with torch.no_grad():
        reference.filter_bank.theta.copy_(theta)
        candidate.filter_bank.theta.copy_(theta)
        alpha = candidate.filter_bank.coefficients(3, torch.float32)
        candidate.cross_coefficients.copy_(alpha[:, 1])
    reference.eval(); candidate.eval()
    with torch.no_grad():
        z0, _, _, aux0, _ = reference(X, PHYSICAL, return_details=True)
        z1, _, _, aux1, _ = candidate(X, PHYSICAL, return_details=True)
    assert torch.equal(z0, z1) and torch.equal(aux0, aux1)


def test_b_is_not_reset_and_only_collaborative_physical_input_router_output_changes():
    model = make_model(candidate=True, seed=750)
    model.eval()
    with torch.no_grad():
        _, _, _, _, before = model(X, PHYSICAL, return_details=True)
        state_b = model.cross_coefficients.clone()
        model.cross_coefficients.add_(torch.tensor([0.1, -0.08, 0.06, -0.04]))
        changed_b = model.cross_coefficients.clone()
        _, _, _, _, after = model(X, PHYSICAL, return_details=True)
    assert torch.equal(changed_b, model.cross_coefficients)
    for modality in ("text", "visual"):
        old, new = before["details"]["filters"], after["details"]["filters"]
        assert old[f"private_{modality}_physical"]["G_expert_rms"] == \
            new[f"private_{modality}_physical"]["G_expert_rms"]
        assert old[f"private_{modality}_semantic"]["G_expert_rms"] == \
            new[f"private_{modality}_semantic"]["G_expert_rms"]
        assert before["details"]["collaborative_routes"][modality] == \
            after["details"]["collaborative_routes"][modality]
        for key in ("raw_rms", "calibrated_rms", "scale", "beta", "nonzero_nodes",
                    "private_response_rms", "sender_state_rms", "sender_private_ratio",
                    "early_cross_to_neighbor_rms"):
            assert before["details"]["early_physical_cross"][modality][key] == \
                after["details"]["early_physical_cross"][modality][key]
        assert old[f"collaborative_{modality}_physical"]["G_expert_rms"] != \
            new[f"collaborative_{modality}_physical"]["G_expert_rms"]
        old_mech = old[f"collaborative_{modality}_physical"]["expert_decoupled_cross"]
        new_mech = new[f"collaborative_{modality}_physical"]["expert_decoupled_cross"]
        assert old_mech["cross_addition_rms_by_expert"] != new_mech["cross_addition_rms_by_expert"]
        assert old_mech["cross_calibrated_rms"] == new_mech["cross_calibrated_rms"]
    assert not torch.equal(state_b, changed_b)


def test_low_memory_checkpoint_and_stacked_eval_paths_agree_with_nonzero_delta():
    stacked, low = make_model(candidate=True, seed=770), make_model(candidate=True, seed=770)
    low.load_state_dict(stacked.state_dict(), strict=True)
    with torch.no_grad():
        stacked.cross_coefficients.add_(torch.tensor([0.08, -0.06, 0.04, -0.03]))
        low.cross_coefficients.copy_(stacked.cross_coefficients)
    for model in (stacked, low):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        model.late_cross_dropout = 0.0
    stacked.train(); low.train()
    z_full, _, _, aux_full, _ = stacked(X, PHYSICAL, return_details=True)
    (F.cross_entropy(nn.Linear(stacked.out_dim, 3)(z_full), LABELS) + aux_full).backward()
    torch.manual_seed(771)
    z_low, _, _, aux_low, _ = low(X, PHYSICAL)
    (z_low.square().mean() + aux_low).backward()
    # Use equal scalar losses for a strict path/gradient comparison.
    stacked.zero_grad(set_to_none=True); low.zero_grad(set_to_none=True)
    stacked_head, low_head = nn.Linear(stacked.out_dim, 3), nn.Linear(low.out_dim, 3)
    low_head.load_state_dict(stacked_head.state_dict())
    z_full, _, _, aux_full, _ = stacked(X, PHYSICAL, return_details=True)
    (F.cross_entropy(stacked_head(z_full), LABELS) + aux_full).backward()
    torch.manual_seed(771)
    z_low, _, _, aux_low, _ = low(X, PHYSICAL)
    (F.cross_entropy(low_head(z_low), LABELS) + aux_low).backward()
    torch.testing.assert_close(z_low, z_full, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(aux_low, aux_full, atol=2e-6, rtol=2e-5)
    full_params, low_params = dict(stacked.named_parameters()), dict(low.named_parameters())
    for name in full_params:
        a, b = full_params[name].grad, low_params[name].grad
        assert (a is None) == (b is None), name
        if a is not None:
            torch.testing.assert_close(b, a, atol=4e-5, rtol=4e-4, msg=name)
    assert low_params["cross_coefficients"].grad is not None
    assert torch.isfinite(low_params["cross_coefficients"].grad).all()


def test_zero_cross_fallback_and_empty_support_remain_finite_and_unchanged():
    model = make_model(candidate=True, seed=790)
    model._capture_state_audit = True
    with torch.no_grad():
        model.cross_coefficients.add_(0.2)
        for parameter in model.cross_projections.parameters():
            parameter.zero_()
    model.train(); torch.manual_seed(791)
    _, _, _, _, info = model(X, PHYSICAL, return_details=True)
    for modality in ("text", "visual"):
        assert torch.equal(model._last_state_audit["private"][modality],
                           model._last_state_audit["collaborative"][modality])
        item = info["details"]["filters"][f"collaborative_{modality}_physical"]["expert_decoupled_cross"]
        assert item["cross_calibrated_rms"] == 0.0
        assert item["difference_correction_rms_by_expert"] == [0.0] * 4

    empty = torch.empty((2, 0), dtype=torch.long)
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index", empty)
        setattr(model, f"semantic_{modality}_weight", torch.empty(0))
    z, _, _, aux, empty_info = model(X, empty, return_details=True)
    assert torch.isfinite(z).all() and torch.isfinite(aux)
    assert empty_info["details"]["physical_support_nodes"] == 0

