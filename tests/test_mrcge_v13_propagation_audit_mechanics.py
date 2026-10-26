import torch
from omegaconf import OmegaConf

from scripts.mrcge_v13_propagation_audit_ops import early_cross, physical_operator, sparse_propagate, v13_basis
from src.models.mrcge_v13 import Model as FrozenV13
from src.models.psce_mag_v7a import chunked_weighted_spmm


def _toy_model():
    cfg = OmegaConf.create({"model": OmegaConf.load("configs/model/mrcge_v13.yaml")})
    cfg.model.v13_variant = "joint_view"
    cfg.model.condition = "M_repaired"
    cfg.model.hidden_dim = 256
    cfg.model.physical_chunk_size = 8
    model = FrozenV13(cfg, {"input_dim": 8, "text_dim": 4, "visual_dim": 4,
                            "num_nodes": 6, "num_classes": 2})
    # Empty V7B candidate edges are sufficient for the local physical-path test.
    model.semantic_candidate_fingerprint = "toy-cache-present"
    return model.eval()


def test_physical_operator_matches_frozen_code_and_graph_orientation():
    edge = torch.tensor([[0, 1, 1, 2, 2, 4, 3], [1, 0, 2, 1, 2, 2, 4]], dtype=torch.long)
    audit_edge, audit_weight, audit_support, audit_loops = physical_operator(edge, 6, torch.float32)
    frozen_edge, frozen_weight, frozen_support, frozen_loops = FrozenV13._physical_operator(edge, 6, torch.float32)
    assert torch.equal(audit_edge, frozen_edge)
    assert torch.equal(audit_support, frozen_support)
    assert torch.equal(audit_weight, frozen_weight)
    assert audit_loops == frozen_loops == 1
    assert audit_support.tolist() == [True, True, True, False, True, False]
    source = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    result = sparse_propagate(source, audit_edge, audit_weight)
    expected_node1 = source[0] / 2.0**0.5 + source[2] * 0.5
    assert torch.allclose(result[1], expected_node1)
    assert torch.equal(result[5], torch.zeros(2))


def test_basis_and_cross_reconstructions_match_frozen_helpers():
    torch.manual_seed(29)
    model = _toy_model()
    edge = torch.tensor([[0, 1, 1, 2, 2, 4, 3, 5], [1, 0, 2, 1, 2, 2, 4, 5]])
    frozen_edge, weight, support, _ = model._physical_operator(edge, 6, torch.float32)
    initial = torch.randn(6, 256)
    reconstructed = v13_basis(initial, frozen_edge, weight, support)
    frozen = model._basis(initial, frozen_edge, weight, support, 3, model.physical_chunk_size)
    assert all(torch.allclose(a, b, rtol=1.0e-6, atol=1.0e-7) for a, b in zip(reconstructed, frozen))
    projection = torch.nn.Linear(256, 256, bias=False)
    source = torch.randn(6, 256)
    raw_a, x1_a, scale_a = early_cross(source, projection, frozen_edge, weight, support, frozen[1])
    raw_b, x1_b, scale_b = model._early_cross("text", 0, source, projection, frozen_edge,
                                              weight, support, frozen[1])
    assert torch.allclose(raw_a, raw_b, rtol=1.0e-6, atol=1.0e-7)
    assert torch.allclose(x1_a, x1_b, rtol=1.0e-6, atol=1.0e-7)
    assert abs(scale_a - float(scale_b)) < 1.0e-6
    x2 = sparse_propagate(x1_a, frozen_edge, weight) * support[:, None]
    x3 = sparse_propagate(x2, frozen_edge, weight) * support[:, None]
    assert torch.allclose(x2, chunked_weighted_spmm(x1_a, frozen_edge, weight, 8) * support[:, None])
    assert torch.allclose(x3, chunked_weighted_spmm(x2, frozen_edge, weight, 8) * support[:, None])


def test_full_eval_forward_captures_real_early_cross_without_mutation():
    torch.manual_seed(31)
    model = _toy_model()
    model._capture_state_audit = True
    edge = torch.tensor([[0, 1, 1, 2, 2, 4, 3, 5, 5], [1, 0, 2, 1, 2, 2, 4, 5, 5]])
    x = torch.randn(6, 8)
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        _, _, _, _, info = model(x, edge, return_details=True)
    rng_after = torch.random.get_rng_state()
    assert info["test_evaluation"] is False
    assert torch.equal(rng_before, rng_after)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state_before.items())
    audit = model._last_state_audit
    assert set(audit["early_cross"]) == {"text", "visual"}
    assert set(audit["private"]) == {"text", "visual"}

    physical_edge, weight, support, _ = model._physical_operator(edge, 6, x.dtype)
    h = {"text": model.projectors["text"](x[:, :4]),
         "visual": model.projectors["visual"](x[:, 4:])}
    bases = {m: model._basis(h[m], physical_edge, weight, support, 3, model.physical_chunk_size)
             for m in ("text", "visual")}
    for receiver, sender, projection_name in (("text", "visual", "visual_to_text"),
                                               ("visual", "text", "text_to_visual")):
        _, reconstructed, _ = model._early_cross(
            receiver, int(receiver == "visual"), h[sender], model.cross_projections[projection_name],
            physical_edge, weight, support, bases[receiver][1]
        )
        assert torch.allclose(reconstructed, audit["early_cross"][receiver], rtol=1.0e-6, atol=1.0e-7)


def test_empty_graph_and_zero_cross_are_explicit_zero_states():
    model = _toy_model()
    empty = torch.empty((2, 0), dtype=torch.long)
    edge, weight, support, loops = model._physical_operator(empty, 4, torch.float32)
    assert edge.shape == (2, 0) and not bool(support.any()) and loops == 0
    initial = torch.randn(4, 256)
    basis = model._basis(initial, edge, weight, support, 3, 8)
    assert all(torch.equal(value, torch.zeros_like(value)) for value in basis[1:])
    zero_projection = torch.nn.Linear(256, 256, bias=False)
    torch.nn.init.zeros_(zero_projection.weight)
    raw, calibrated, scale = model._early_cross("text", 0, initial, zero_projection,
                                                edge, weight, support, basis[1])
    assert torch.equal(raw, torch.zeros_like(raw))
    assert torch.equal(calibrated, torch.zeros_like(calibrated))
    assert float(scale) == 0.0
