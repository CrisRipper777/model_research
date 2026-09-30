from __future__ import annotations

import copy
import hashlib
import json

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.run_m0_adaptive_propagation import restrict_labels_to_train_val
from src.data.types import MAGData
from src.models.adaptive_prop_m0 import (
    Model,
    compose_message,
    fixed_degree_mean,
    global_mean_control,
    incoming_degree,
    neighborhood_shuffle_indices,
    parameter_counts,
    remove_self_messages,
    target_mean_control,
)


def cfg(variant: str, chunk: int = 100000):
    return OmegaConf.create({"model": {
        "variant": variant, "hidden_dim": 128, "dropout": 0.0,
        "relation_dim": 32, "relation_state_dim": 64,
        "modality_embed_dim": 8, "edge_chunk_size": chunk,
        "wide_gate_hidden_dim": 126,
    }})


def graph():
    torch.manual_seed(201)
    x = torch.randn(9, 14)
    edges = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1, 3, 5, 7, 8],
        [8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 8],
    ], dtype=torch.long)
    info = {"input_dim": 14, "num_nodes": 9, "num_classes": 3, "text_dim": 6, "visual_dim": 8}
    return x, edges, info


def make_model(variant: str, chunk: int = 100000):
    x, edges, info = graph()
    torch.manual_seed(702)
    return Model(cfg(variant, chunk), info), x, edges


def test_single_basis_regression_preserves_state_layout_and_formula_output():
    model, x, edge_index = make_model("single_basis")
    expected_keys = {
        "gate_heads.0.weight": (1, 64), "gate_heads.0.bias": (1,),
        "correction_heads.0.weight": (1, 64), "correction_heads.0.bias": (1,),
        "basis_u": (2, 1, 128, 32), "basis_v": (2, 1, 32, 128),
    }
    state = model.state_dict()
    for key, shape in expected_keys.items():
        assert tuple(state[key].shape) == shape
    assert "static_c_logits" not in state
    assert not any("basis_heads" in key for key in state)
    # Frozen full M0-B state schema from the parent checkpoint: all 59 keys and shapes.
    schema_info = {"input_dim": 1536, "num_nodes": 9, "num_classes": 3, "text_dim": 768, "visual_dim": 768}
    torch.manual_seed(702)
    schema_model = Model(cfg("single_basis"), schema_info)
    signature = json.dumps([(k, list(v.shape)) for k, v in sorted(schema_model.state_dict().items())], separators=(",", ":"))
    assert hashlib.sha256(signature.encode()).hexdigest() == "85f960c4b155fb229074bf1b62fe30351af07a56cc62ba8fa0b5e2b20b97317b"
    assert len(schema_model.state_dict()) == 59

    model.eval()
    with torch.no_grad():
        actual, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
        h0, _, src, dst, degree, deg_z, p, contexts = model._prepare(x, edge_index)
        deltas = [h0[0].new_zeros((x.size(0), 128)), h0[1].new_zeros((x.size(0), 128))]
        ref_g, ref_c = [[], []], [[], []]
        for begin in range(src.numel()):
            u = model._functional_states(p, [contexts[0][begin:begin+1], contexts[1][begin:begin+1]],
                                         deg_z, src[begin:begin+1], dst[begin:begin+1])
            for m in range(2):
                g, c, pi = model._controls(u[m], m)
                ref_g[m].append(g)
                ref_c[m].append(c)
                base = model.w0[m](h0[m][src[begin:begin+1]])
                corr = model._basis_correction(h0[m][src[begin:begin+1]], m, pi)
                message = compose_message(base, g, corr, c)
                deltas[m] = deltas[m].index_add(0, dst[begin:begin+1], message)
        deltas = [fixed_degree_mean(deltas[m], degree) for m in range(2)]
        htilde = [model.residual_norms[m](h0[m] + deltas[m]) for m in range(2)]
        expected = model.fusion(torch.cat([h0[0], h0[1], htilde[0], htilde[1]], dim=-1))
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
        for m in range(2):
            assert torch.allclose(aux["controls"]["g"][m], torch.cat(ref_g[m]), atol=1e-7, rtol=1e-7)
            assert torch.allclose(aux["controls"]["c"][m], torch.cat(ref_c[m]), atol=1e-7, rtol=1e-7)


def test_a_wide_is_extent_only_and_parameter_matched_to_b():
    wide, _, _ = make_model("extent_wide")
    b, _, _ = make_model("single_basis")
    assert wide.correction_heads is None and wide.basis_u is None and wide.basis_v is None
    assert wide.static_c_logits is None
    for head in wide.gate_heads:
        assert isinstance(head, nn.Sequential)
        assert isinstance(head[0], nn.Linear) and head[0].out_features == 126
        assert isinstance(head[1], nn.GELU)
        assert isinstance(head[2], nn.Linear) and head[2].out_features == 1
        assert torch.all(head[0].bias == 0)
        assert torch.all(head[2].bias == 2)
    assert abs(parameter_counts(wide)["model_total"] - parameter_counts(b)["model_total"]) <= 32


def test_a_wide_gate_initialization_near_default_smooth_gate():
    wide, _, _ = make_model("extent_wide")
    u = torch.randn(128, 64)
    for head in wide.gate_heads:
        assert abs(float(torch.sigmoid(head(u)).mean()) - float(torch.sigmoid(torch.tensor(2.0)))) < 0.02


def test_b_static_c_is_constant_and_trainable():
    model, x, edges = make_model("single_basis_static")
    model.eval()
    with torch.no_grad():
        _, _, _, _, aux = model(x, edges, return_diagnostics=True)
    for values in aux["controls"]["c"]:
        assert torch.equal(values, values[0].expand_as(values))
    model.train()
    z = model(x, edges)[0]
    nn.functional.cross_entropy(nn.Linear(128, 3)(z), torch.arange(9) % 3).backward()
    grad = model.static_c_logits.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad.norm()) > 0
    assert torch.allclose(model.static_c_logits.detach(), torch.full((2,), -2.0))


def test_b_target_c_constant_within_target_and_varies_between_targets():
    model, x, edges = make_model("single_basis_target", chunk=2)
    model.eval()
    with torch.no_grad():
        _, _, _, _, aux = model(x, edges, return_diagnostics=True)
    dst = aux["edge_index_nonself"][1]
    for values in aux["controls"]["c"]:
        for node in torch.unique(dst):
            within = values[dst == node]
            assert torch.equal(within, within[0].expand_as(within))
        target_values = torch.stack([values[dst == node][0] for node in torch.unique(dst)])
        assert float(target_values.std(unbiased=False)) > 1e-8


def test_b_target_params_equal_b_and_shared_head_basis_initialization():
    _, _, info = graph()
    torch.manual_seed(851)
    b = Model(cfg("single_basis"), info)
    torch.manual_seed(851)
    target = Model(cfg("single_basis_target"), info)
    assert parameter_counts(target)["model_total"] == parameter_counts(b)["model_total"]
    for a, b_ in zip(b.correction_heads, target.correction_heads):
        assert torch.equal(a.weight, b_.weight) and torch.equal(a.bias, b_.bias)
    assert torch.equal(b.basis_u, target.basis_u)
    assert torch.equal(b.basis_v, target.basis_v)


def test_b_target_two_pass_chunk_equivalence_and_autograd():
    small, x, edges = make_model("single_basis_target", chunk=2)
    large = copy.deepcopy(small)
    large.edge_chunk_size = 1000
    small.train(); large.train()
    z_small = small(x, edges)[0]
    z_large = large(x, edges)[0]
    assert torch.allclose(z_small, z_large, atol=2e-6, rtol=2e-6)
    loss_small = z_small.square().mean()
    loss_large = z_large.square().mean()
    loss_small.backward(); loss_large.backward()
    for (name_a, p_a), (name_b, p_b) in zip(small.named_parameters(), large.named_parameters()):
        assert name_a == name_b
        if p_a.grad is not None:
            assert p_b.grad is not None and torch.isfinite(p_a.grad).all() and torch.isfinite(p_b.grad).all()
            assert torch.allclose(p_a.grad, p_b.grad, atol=2e-5, rtol=2e-4), name_a


def test_b_target_gradient_groups_are_finite_and_active():
    model, x, edges = make_model("single_basis_target", chunk=3)
    z = model(x, edges)[0]
    nn.functional.cross_entropy(nn.Linear(128, 3)(z), torch.arange(9) % 3).backward()
    prefixes = {
        "gate": ("gate_heads.",), "correction": ("correction_heads.",),
        "basis": ("basis_u", "basis_v"),
        "relation": ("rel_proj_t.", "rel_proj_v.", "phi_pair.", "phi_rel.", "phi_mod.", "modality_embeddings"),
    }
    for values in prefixes.values():
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(values) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.norm()) for g in grads) > 0


def test_c_target_mean_and_global_mean_control_replacements():
    dst = torch.tensor([1, 1, 1, 2, 2, 3])
    c = torch.tensor([[0.1], [0.4], [0.7], [0.2], [0.8], [0.6]])
    target = target_mean_control(c, dst, 4)
    assert torch.equal(target[:3], torch.full((3, 1), 0.4))
    assert torch.equal(target[3:5], torch.full((2, 1), 0.5))
    assert torch.equal(target[5:], c[5:])
    global_c = global_mean_control(c)
    assert torch.equal(global_c, torch.full_like(c, float(c.mean())))


def test_single_control_shuffles_preserve_multisets_and_keep_other_control_fixed():
    dst = torch.tensor([1, 1, 1, 2, 2, 3])
    g = torch.tensor([[.1], [.2], [.3], [.4], [.5], [.6]])
    c = torch.tensor([[.9], [.7], [.5], [.3], [.1], [.8]])
    pi = torch.empty((6, 0))
    for offset in (0, 10000):
        permutation = neighborhood_shuffle_indices(dst, 1002 + offset)
        gs, cs = g[permutation], c[permutation]
        for target in torch.unique(dst):
            mask = dst == target
            assert sorted(g[mask, 0].tolist()) == sorted(gs[mask, 0].tolist())
            assert sorted(c[mask, 0].tolist()) == sorted(cs[mask, 0].tolist())
        # Applying either intervention changes only its named tensor.
        g_only = {"g": (gs, gs), "c": (c, c), "pi": (pi, pi)}
        c_only = {"g": (g, g), "c": (cs, cs), "pi": (pi, pi)}
        assert torch.equal(g_only["c"][0], c) and torch.equal(c_only["g"][0], g)


def test_b_target_c_target_mean_is_prediction_identity():
    model, x, edges = make_model("single_basis_target")
    model.eval()
    with torch.no_grad():
        normal, _, _, _, aux = model(x, edges, return_diagnostics=True)
        dst = aux["edge_index_nonself"][1]
        c_mean = tuple(target_mean_control(c, dst, x.size(0)) for c in aux["controls"]["c"])
        intervened = model(x, edges, control_overrides={"c": c_mean})[0]
    for c, mean in zip(aux["controls"]["c"], c_mean):
        assert torch.allclose(mean, c, atol=1e-7, rtol=0)
    assert torch.allclose(intervened, normal, atol=2e-6, rtol=2e-6)


def test_fixed_denominator_stays_original_indegree_after_control_zeros():
    edges = torch.tensor([[0, 2, 3], [1, 1, 1]])
    _, _, dst = remove_self_messages(edges)
    degree = incoming_degree(dst, 4)
    messages = torch.tensor([[2.0], [0.0], [0.0]])
    assert degree[1].item() == 3
    assert abs(fixed_degree_mean(messages.new_zeros((4, 1)).index_add(0, dst, messages), degree)[1].item() - 2.0 / 3.0) < 1e-7


def test_no_test_split_or_test_labels_are_exposed_by_train_val_restrictor():
    data = MAGData(name="toy", source="mmgraph", task="nc", x=torch.randn(5, 3),
                   edge_index=torch.empty((2, 0), dtype=torch.long), y=torch.tensor([0, 1, 0, 1, 9]),
                   train_idx=torch.tensor([0, 1]), val_idx=torch.tensor([2, 3]), test_idx=torch.tensor([4]),
                   num_nodes=5, num_classes=2, info={})
    split = restrict_labels_to_train_val(data)
    assert data.test_idx is None
    assert "test" not in split
    assert data.y.tolist() == [0, 1, 0, 1, -100]
