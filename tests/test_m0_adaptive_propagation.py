from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.run_m0_adaptive_propagation import restrict_labels_to_train_val
from src.data.types import MAGData
from src.models.adaptive_prop_m0 import (
    Model,
    common_parameter_names,
    compose_message,
    fixed_degree_mean,
    graph_standardized_log_degree,
    incoming_degree,
    leave_one_out_context,
    neighborhood_shuffle_indices,
    parameter_counts,
    remove_self_messages,
    tie_modality_controls,
)


def make_cfg(variant: str, chunk: int = 100000, dropout: float = 0.0):
    return OmegaConf.create({"model": {
        "variant": variant, "hidden_dim": 128, "dropout": dropout,
        "relation_dim": 32, "relation_state_dim": 64,
        "modality_embed_dim": 8, "edge_chunk_size": chunk,
    }})


def make_graph():
    x = torch.arange(8 * 12, dtype=torch.float32).view(8, 12) / 50
    # Includes a self-loop and directed edges with in-degree 3 at node 1.
    edges = torch.tensor([
        [0, 2, 3, 1, 4, 1, 5, 0, 7],
        [1, 1, 1, 1, 2, 2, 3, 4, 6],
    ], dtype=torch.long)
    return x, edges, {"input_dim": 12, "num_nodes": 8, "num_classes": 3, "text_dim": 5, "visual_dim": 7}


def make_model(variant: str, chunk: int = 100000):
    torch.manual_seed(13)
    x, edges, info = make_graph()
    return Model(make_cfg(variant, chunk), info), x, edges


def test_modality_slicing_is_text_then_visual():
    model, x, _ = make_model("semantic")
    text, visual = model.split_modalities(x)
    assert torch.equal(text, x[:, :5])
    assert torch.equal(visual, x[:, 5:])
    assert text.shape[1] == 5 and visual.shape[1] == 7


def test_physical_support_preserves_directed_nonself_edges():
    _, edges, _ = make_graph()
    kept, src, dst = remove_self_messages(edges)
    expected = edges[:, edges[0] != edges[1]]
    assert torch.equal(kept, expected)
    assert torch.equal(src, expected[0])
    assert torch.equal(dst, expected[1])
    assert torch.equal(kept.flip(0).flip(0), expected)  # orientation/order unchanged


def test_incoming_degree_excludes_self_loops_and_standardizes_log_degree():
    _, edges, info = make_graph()
    _, _, dst = remove_self_messages(edges)
    degree = incoming_degree(dst, info["num_nodes"])
    assert degree.tolist() == [0, 3, 2, 1, 1, 0, 1, 0]
    z = graph_standardized_log_degree(degree)
    assert torch.isfinite(z).all()
    assert abs(float(z.mean())) < 1e-6


def test_directed_pair_evidence_places_target_before_source():
    model, _, _ = make_model("extent")
    p = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    context = torch.tensor([[0.25]])
    deg_z = torch.tensor([-1.0, 0.0, 1.0])
    evidence = model._pair_evidence(p, context, deg_z, torch.tensor([0]), torch.tensor([2]))
    assert torch.equal(evidence[0, :2], p[2])
    assert torch.equal(evidence[0, 2:4], p[0])
    assert torch.equal(evidence[0, 4:6], (p[2] - p[0]).abs())
    assert torch.equal(evidence[0, 6:8], p[2] * p[0])
    assert evidence.shape[-1] == 4 * 2 + 4


def test_leave_one_out_context_bruteforce_excludes_current_source_and_handles_degree_one():
    projected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], requires_grad=True)
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([3, 3, 2])
    degree = incoming_degree(dst, 4)
    context = leave_one_out_context(projected, src, dst, degree)
    # 0->3 sees only source 1; 1->3 sees only source 0. Both orthogonal.
    assert torch.allclose(context[:2, 0], torch.zeros(2), atol=1e-7)
    # Degree-one target 2 has no leave-one-out neighborhood.
    assert context[2, 0].item() == 0.0
    assert not context.requires_grad


def test_fixed_denominator_is_original_in_degree_even_after_zero_gate():
    degree = torch.tensor([2])
    messages = torch.tensor([[2.0], [0.0]])  # gate=(1, 0), source base=(2, 9)
    aggregate = messages.sum(dim=0, keepdim=True)
    assert fixed_degree_mean(aggregate, degree).item() == 1.0
    assert fixed_degree_mean(aggregate, torch.tensor([1])).item() == 2.0


def test_relation_state_shapes_and_finite_forward():
    model, x, edges = make_model("extent")
    model.eval()
    h0, _, src, dst, degree, deg_z, p, context = model._prepare(x, edges)
    u = model._functional_states(p, context, deg_z, src, dst)
    assert [tuple(v.shape) for v in p] == [(8, 32), (8, 32)]
    assert tuple(u[0].shape) == (8, 64)
    z, _, _, aux, _ = model(x, edges)
    assert tuple(z.shape) == (8, 128)
    assert aux.item() == 0
    assert all(torch.isfinite(v).all() for v in h0 + p + u)
    assert int(degree.sum()) == src.numel()


def test_common_initialization_is_exact_for_abc_and_basis_params_match():
    models = {}
    for variant in ("extent", "single_basis", "multi_basis"):
        torch.manual_seed(77)
        _, _, info = make_graph()
        models[variant] = Model(make_cfg(variant), info)
    names = common_parameter_names(models["extent"])
    for variant in ("single_basis", "multi_basis"):
        assert names == common_parameter_names(models[variant])
        for name in names:
            assert torch.equal(models["extent"].state_dict()[name], models[variant].state_dict()[name]), name
    assert parameter_counts(models["single_basis"])["correction_basis"] == parameter_counts(models["multi_basis"])["correction_basis"]


def test_gate_correction_and_basis_heads_have_frozen_initial_values():
    expected_gate = torch.sigmoid(torch.tensor(2.0)).item()
    expected_correction = torch.sigmoid(torch.tensor(-2.0)).item()
    extent, _, _ = make_model("extent")
    single, _, _ = make_model("single_basis")
    multi, _, _ = make_model("multi_basis")
    u = torch.randn(20, 64)
    for gate in extent.gate_heads:
        assert abs(float(torch.sigmoid(gate(u)).mean()) - expected_gate) < 0.02
    for model in (single, multi):
        for gate in model.gate_heads:
            assert abs(float(torch.sigmoid(gate(u)).mean()) - expected_gate) < 0.02
        for head in model.correction_heads:
            assert abs(float(torch.sigmoid(head(u)).mean()) - expected_correction) < 0.02
    for head in multi.basis_heads:
        pi = torch.softmax(head(u), dim=-1)
        assert torch.allclose(pi.mean(0), torch.full((4,), 0.25), atol=0.01)
        assert torch.allclose(pi.sum(-1), torch.ones(20), atol=1e-6)


def test_degenerate_message_identities_and_uniform_basis_composition():
    base = torch.randn(5, 128)
    gate = torch.rand(5, 1)
    correction = torch.randn(5, 128)
    assert torch.equal(compose_message(base, torch.ones_like(gate)), base)
    assert torch.equal(compose_message(base, gate, correction, torch.zeros_like(gate)), gate * base)
    model, _, _ = make_model("multi_basis")
    source = torch.randn(6, 128)
    outputs = torch.stack([
        torch.nn.functional.linear(torch.nn.functional.linear(source, model.basis_v[0, k]), model.basis_u[0, k])
        for k in range(4)
    ], dim=1)
    uniform = model._basis_correction(source, 0, torch.full((6, 4), 0.25))
    assert torch.allclose(uniform, outputs.mean(dim=1), atol=1e-7)


def test_chunked_forward_matches_single_chunk_forward():
    small, x, edges = make_model("multi_basis", chunk=2)
    large = copy.deepcopy(small)
    small.edge_chunk_size = 2
    large.edge_chunk_size = 1000
    small.eval()
    large.eval()
    out_small = small(x, edges)[0]
    out_large = large(x, edges)[0]
    assert torch.allclose(out_small, out_large, atol=1e-6, rtol=1e-6)


def test_shuffle_preserves_target_control_multisets_and_changes_assignments():
    dst = torch.tensor([2, 1, 1, 2, 2, 3, 1])
    controls = {
        "g": torch.arange(dst.numel())[:, None].float(),
        "c": (torch.arange(dst.numel())[:, None] + 20).float(),
        "pi": torch.stack([torch.arange(dst.numel()).float(), torch.ones(dst.numel())], dim=-1),
    }
    permutation = neighborhood_shuffle_indices(dst, 1001)
    shuffled = {name: value[permutation] for name, value in controls.items()}
    original_tuples = torch.cat([controls["g"], controls["c"], controls["pi"]], dim=-1)
    shuffled_tuples = torch.cat([shuffled["g"], shuffled["c"], shuffled["pi"]], dim=-1)
    changed = False
    for target in torch.unique(dst):
        mask = dst == target
        assert sorted(map(tuple, original_tuples[mask].tolist())) == sorted(map(tuple, shuffled_tuples[mask].tolist()))
        changed |= not torch.equal(original_tuples[mask], shuffled_tuples[mask])
    assert changed


def test_modality_tied_controls_are_equal_and_pi_renormalized():
    controls = {
        "g": (torch.rand(7, 1), torch.rand(7, 1)),
        "c": (torch.rand(7, 1), torch.rand(7, 1)),
        "pi": (torch.rand(7, 4), torch.rand(7, 4)),
    }
    tied = tie_modality_controls(controls)
    for name in controls:
        assert torch.equal(tied[name][0], tied[name][1])
    assert torch.allclose(tied["pi"][0].sum(-1), torch.ones(7), atol=1e-6)


@pytest.mark.parametrize("variant", ["extent", "single_basis", "multi_basis"])
def test_adaptive_variant_gradients_are_finite_and_active(variant):
    model, x, edges = make_model(variant, chunk=3)
    model.train()
    head = nn.Linear(128, 3)
    z = model(x, edges)[0]
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 1, 0])
    nn.functional.cross_entropy(head(z), labels).backward()
    audit = {
        prefix: [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        for prefix in ("gate_heads.", "correction_heads.", "basis_heads.", "basis_u", "basis_v")
    }
    required = ["gate_heads."]
    if variant in {"single_basis", "multi_basis"}:
        required += ["correction_heads.", "basis_u", "basis_v"]
    if variant == "multi_basis":
        required += ["basis_heads."]
    for prefix in required:
        grads = audit[prefix]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.norm()) for g in grads) > 0


def test_function_off_matches_extent_only_common_path():
    basis, x, edges = make_model("single_basis")
    extent, _, _ = make_model("extent")
    for name in common_parameter_names(extent):
        extent.state_dict()[name].copy_(basis.state_dict()[name])
    for key in ("weight", "bias"):
        for m in range(2):
            extent.gate_heads[m].state_dict()[key].copy_(basis.gate_heads[m].state_dict()[key])
    basis.eval()
    extent.eval()
    controls = basis(x, edges, return_diagnostics=True)[4]["controls"]
    overrides = {
        "g": controls["g"],
        "c": tuple(torch.zeros_like(v) for v in controls["c"]),
    }
    output_b = basis(x, edges, control_overrides=overrides)[0]
    output_a = extent(x, edges)[0]
    assert torch.allclose(output_b, output_a, atol=1e-6, rtol=1e-6)


def test_inference_matches_eval_forward():
    model, x, edges = make_model("single_basis")
    model.eval()
    expected = model(x, edges)[0].detach().cpu()
    actual = model.inference(x, edges, device="cpu")
    assert torch.allclose(expected, actual, atol=1e-7, rtol=1e-7)


def test_no_test_indices_or_labels_survive_m0_label_restriction():
    data = MAGData(
        name="synthetic", source="test", task="nc", x=torch.zeros(6, 2),
        edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=6, y=torch.arange(6),
        train_idx=torch.tensor([0, 1]), val_idx=torch.tensor([2]), test_idx=torch.tensor([3, 4, 5]),
        num_classes=6,
    )
    result = restrict_labels_to_train_val(data)
    assert result.keys() == {"train", "validation"}
    assert data.test_idx is None
    assert data.y.tolist() == [0, 1, 2, -100, -100, -100]
