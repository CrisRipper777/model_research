from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.run_m0_adaptive_propagation import restrict_labels_to_train_val
from src.data.types import MAGData
from src.models.adaptive_prop_m0 import Model as M0Model, neighborhood_shuffle_indices
from src.models.adaptive_prop_m0 import fixed_degree_mean, incoming_degree, remove_self_messages
from src.models.structured_executor_e0 import (
    FUNCTION_NAMES,
    Model,
    _rms,
    pi_global_mean,
    pi_modality_tied,
    pi_target_mean,
    renormalize_without,
)


def cfg(variant: str, chunk: int = 100000):
    return OmegaConf.create({"model": {
        "variant": variant, "hidden_dim": 128, "dropout": 0.0,
        "relation_dim": 32, "relation_state_dim": 64, "modality_embed_dim": 8,
        "edge_chunk_size": chunk, "eps": 1e-8,
    }, "task": {"evaluate_test": False}})


def graph():
    torch.manual_seed(201)
    x = torch.randn(9, 14)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1, 3, 5, 7, 8],
        [8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 8],
    ], dtype=torch.long)
    info = {"input_dim": 14, "num_nodes": 9, "num_classes": 3, "text_dim": 6, "visual_dim": 8}
    return x, edge_index, info


def make_model(variant: str, chunk: int = 100000):
    x, edge_index, info = graph()
    torch.manual_seed(702)
    return Model(cfg(variant, chunk), info), x, edge_index


def _copy_common(m0: nn.Module, e0: nn.Module):
    e0_state = e0.state_dict()
    common = {key: value for key, value in m0.state_dict().items() if key in e0_state}
    e0.load_state_dict(common, strict=False)


def test_modal_slicing_physical_support_no_self_and_fixed_degree():
    model, x, edge_index = make_model("edge_mix")
    text, visual = model.split_modalities(x)
    assert torch.equal(text, x[:, :6]) and torch.equal(visual, x[:, 6:])
    nonself, src, dst = remove_self_messages(edge_index)
    assert torch.equal(nonself, edge_index[:, edge_index[0] != edge_index[1]])
    assert not bool((src == dst).any())
    degree = incoming_degree(dst, x.size(0))
    assert degree.tolist() == [0, 0, 0, 0, 0, 0, 4, 4, 8]
    msg = torch.ones(src.numel(), 128)
    assert torch.equal(fixed_degree_mean(torch.zeros(x.size(0), 128).index_add(0, dst, msg), degree)[6],
                       torch.ones(128))


def test_m0_relation_state_regression_p_q_r_u():
    x, edge_index, info = graph()
    torch.manual_seed(921)
    m0 = M0Model(OmegaConf.create({"model": {
        "variant": "single_basis", "hidden_dim": 128, "dropout": 0.0,
        "relation_dim": 32, "relation_state_dim": 64, "modality_embed_dim": 8,
        "edge_chunk_size": 3, "eps": 1e-8,
    }}), info).eval()
    torch.manual_seed(98)
    e0 = Model(cfg("edge_mix", chunk=3), info).eval()
    _copy_common(m0, e0)
    m0_data = m0._prepare(x, edge_index)
    h0, _, src, dst, degree, deg_z, p, contexts = m0_data
    evidence_t = m0._pair_evidence(p[0], contexts[0], deg_z, src, dst)
    evidence_v = m0._pair_evidence(p[1], contexts[1], deg_z, src, dst)
    q_t, q_v = m0.phi_pair(evidence_t), m0.phi_pair(evidence_v)
    relation = m0.phi_rel(torch.cat([q_t, q_v, (q_t - q_v).abs(), q_t * q_v], dim=-1))
    u = m0._functional_states(p, contexts, deg_z, src, dst)
    e0_state = e0.relation_state(x, edge_index)
    assert torch.equal(src, e0_state["src"]) and torch.equal(dst, e0_state["dst"])
    for left, right in zip(h0, e0_state["h0"]):
        assert torch.allclose(left, right, rtol=1e-6, atol=1e-7)
    for left, right in zip(p, e0_state["p"]):
        assert torch.allclose(left, right, rtol=1e-6, atol=1e-7)
    for left, right in zip(contexts, e0_state["contexts"]):
        assert torch.allclose(left, right, rtol=1e-6, atol=1e-7)
    for left, right in zip((q_t, q_v), e0_state["q"]):
        assert torch.allclose(left, right, rtol=1e-6, atol=1e-7)
    assert torch.allclose(relation, e0_state["r"], rtol=1e-6, atol=1e-7)
    for left, right in zip(u, e0_state["u"]):
        assert torch.allclose(left, right, rtol=1e-6, atol=1e-7)
    assert torch.equal(degree, e0_state["degree"])


def test_smooth_only_is_exact_ordinary_uniform_propagation():
    x, edge_index, info = graph()
    torch.manual_seed(80)
    uniform = M0Model(OmegaConf.create({"model": {
        "variant": "uniform", "hidden_dim": 128, "dropout": 0.0,
        "relation_dim": 32, "relation_state_dim": 64, "modality_embed_dim": 8,
        "edge_chunk_size": 100000, "eps": 1e-8,
    }}), info).eval()
    torch.manual_seed(1)
    e0 = Model(cfg("edge_mix"), info).eval()
    _copy_common(uniform, e0)
    with torch.no_grad():
        for m in range(2):
            e0.smooth_transforms[m].weight.copy_(uniform.w0[m].weight)
        e0_state = e0.relation_state(x, edge_index)
        _, src, dst = remove_self_messages(edge_index)
        h0 = e0_state["h0"]
        for m in range(2):
            fs = e0.smooth_transforms[m](h0[m][src])
            manual = h0[m].new_zeros(h0[m].shape).index_add(0, dst, fs)
            manual = fixed_degree_mean(manual, e0_state["degree"])
            m0_h0, _, _, _, _, _, _, _ = uniform._prepare(x, edge_index)
            m0_msg = uniform.w0[m](m0_h0[m][src])
            m0_manual = m0_h0[m].new_zeros(m0_h0[m].shape).index_add(0, dst, m0_msg)
            m0_manual = fixed_degree_mean(m0_manual, incoming_degree(dst, x.size(0)))
            assert torch.allclose(manual, m0_manual, rtol=1e-7, atol=1e-7)
        forced_pi = torch.zeros(src.numel(), len(FUNCTION_NAMES))
        forced_pi[:, 1] = 1
        e0_z = e0(x, edge_index, control_overrides={"pi": (forced_pi, forced_pi)})[0]
        m0_z = uniform(x, edge_index)[0]
    assert torch.allclose(e0_z, m0_z, rtol=1e-6, atol=1e-7)


def test_null_is_exact_zero_message_and_router_prior_is_smooth_default():
    model, x, edge_index = make_model("edge_mix")
    model.eval()
    with torch.no_grad():
        state = model.relation_state(x, edge_index)
        src, dst, h0 = state["src"], state["dst"], state["h0"]
        null_pi = torch.zeros(src.numel(), 4)
        null_pi[:, 0] = 1
        z = model(x, edge_index, control_overrides={"pi": (null_pi, null_pi)})[0]
        h_tilde = [model.residual_norms[m](h0[m]) for m in range(2)]
        expected = model.fusion(torch.cat([h0[0], h0[1], h_tilde[0], h_tilde[1]], dim=-1))
        assert torch.allclose(z, expected, rtol=1e-6, atol=1e-7)
        prior = torch.softmax(model.router(torch.randn(2048, 64)), dim=-1).mean(0)
    expected_prior = torch.tensor([0.096255, 0.711235, 0.096255, 0.096255])
    assert torch.allclose(prior, expected_prior, atol=0.004, rtol=0)


def test_relational_and_cross_modal_read_the_required_sources():
    model, x, edge_index = make_model("edge_mix")
    state = model.relation_state(x, edge_index)
    src, dst = state["src"], state["dst"]
    h0 = [value.detach().clone() for value in state["h0"]]
    outputs_a = model._expert_outputs(h0, src, dst)
    changed_visual = [h0[0], h0[1].clone()]
    changed_visual[1][src[0]] += 2.0
    outputs_b = model._expert_outputs(changed_visual, src, dst)
    assert torch.equal(outputs_a[0][0], outputs_b[0][0])
    assert torch.equal(outputs_a[1][0], outputs_b[1][0])  # Text R does not read Visual source.
    assert not torch.allclose(outputs_a[2][0][0], outputs_b[2][0][0])  # Text X reads Visual source.

    changed_text = [h0[0].clone(), h0[1]]
    changed_text[0][src[0]] += 2.0
    outputs_c = model._expert_outputs(changed_text, src, dst)
    assert not torch.allclose(outputs_a[1][0][0], outputs_c[1][0][0])
    assert torch.equal(outputs_a[1][1], outputs_c[1][1])  # Visual R does not read Text source.
    assert not torch.allclose(outputs_a[2][1][0], outputs_c[2][1][0])  # Visual X reads Text source.
    # Raw Text-X direction is unchanged; its calibrated magnitude can change
    # because the specification references Text-Smooth RMS.
    assert torch.equal(outputs_a[3]["raw_cross_modal"][0][0],
                       outputs_c[3]["raw_cross_modal"][0][0])


def test_calibration_rms_matches_smooth_and_retains_expert_gradients():
    model, x, edge_index = make_model("edge_mix", chunk=3)
    model.train()
    z = model(x, edge_index)[0]
    head = nn.Linear(model.out_dim, 3)
    labels = torch.arange(x.size(0)) % 3
    nn.functional.cross_entropy(head(z), labels).backward()
    groups = {
        "router": ("router.",),
        "smooth": ("smooth_transforms.",),
        "relational_source": ("relational_source.",),
        "relational_mlp": ("relational_mlps.",),
        "cross_source": ("cross_modal_source.",),
        "cross_mlp": ("cross_modal_mlps.",),
        "relation": ("rel_proj_t.", "rel_proj_v.", "phi_pair.", "phi_rel.",
                     "phi_mod.", "modality_embeddings"),
    }
    for prefixes in groups.values():
        grads = [p.grad for name, p in model.named_parameters()
                 if name.startswith(prefixes) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.norm()) for g in grads) > 0

    state = model.relation_state(x, edge_index)
    outputs = model._expert_outputs(state["h0"], state["src"], state["dst"])
    smooth, relational, cross_modal = outputs[:3]
    for m in range(2):
        assert torch.allclose(_rms(relational[m], model.eps), _rms(smooth[m], model.eps), atol=2e-6, rtol=2e-6)
        assert torch.allclose(_rms(cross_modal[m], model.eps), _rms(smooth[m], model.eps), atol=2e-6, rtol=2e-6)
    direction_probe = outputs[1][0].square().sum() + outputs[2][1].square().sum()
    grads = torch.autograd.grad(direction_probe, [model.relational_mlps[0][2].weight,
                                                  model.cross_modal_mlps[1][2].weight])
    assert all(torch.isfinite(g).all() and float(g.norm()) > 0 for g in grads)


def test_all_variants_have_exactly_equal_parameter_counts_and_bitwise_init():
    x, edge_index, info = graph()
    models = {}
    for variant in ("static_mix", "target_mix", "edge_mix"):
        torch.manual_seed(114)
        models[variant] = Model(cfg(variant), info)
    names = set(models["static_mix"].state_dict())
    base_state = models["static_mix"].state_dict()
    for variant, model in models.items():
        assert set(model.state_dict()) == names
        assert sum(p.numel() for p in model.parameters()) == sum(
            p.numel() for p in models["static_mix"].parameters())
        for name, value in model.state_dict().items():
            assert torch.equal(value, base_state[name]), (variant, name)


@pytest.mark.parametrize("variant", ["static_mix", "target_mix", "edge_mix"])
def test_router_granularity_and_chunk_equivalence(variant):
    small, x, edge_index = make_model(variant, chunk=2)
    large = copy.deepcopy(small)
    large.edge_chunk_size = 1000
    small.eval(); large.eval()
    with torch.no_grad():
        z_small, _, _, _, aux = small(x, edge_index, return_diagnostics=True)
        z_large = large(x, edge_index)[0]
    assert torch.allclose(z_small, z_large, rtol=2e-5, atol=2e-6)
    for m in range(2):
        pi = aux["controls"]["pi"][m]
        assert torch.isfinite(pi).all()
        assert torch.allclose(pi.sum(-1), torch.ones(pi.size(0)), atol=1e-6)
        if variant == "static_mix":
            assert torch.equal(pi, pi[:1].expand_as(pi))
        elif variant == "target_mix":
            dst = aux["edge_index_nonself"][1]
            for node in torch.unique(dst):
                values = pi[dst == node]
                assert torch.equal(values, values[:1].expand_as(values))
        elif variant == "edge_mix":
            dst = aux["edge_index_nonself"][1]
            within = pi[dst == 8]
            assert (within - within[:1]).abs().max() > 1e-8


def test_target_global_tied_and_expert_off_intervention_identities():
    dst = torch.tensor([2, 1, 1, 2, 2, 3, 1])
    logits = torch.randn(dst.numel(), 4)
    pi = torch.softmax(logits, dim=-1)
    target = pi_target_mean(pi, dst, 4)
    global_value = pi_global_mean(pi)
    tied = pi_modality_tied(pi, torch.softmax(torch.randn_like(logits), dim=-1))
    assert torch.allclose(global_value.sum(-1), torch.ones(dst.numel()), atol=1e-6)
    assert torch.allclose(tied[0].sum(-1), torch.ones(dst.numel()), atol=1e-6)
    assert torch.equal(tied[0], tied[1])
    for node in torch.unique(dst):
        assert torch.equal(target[dst == node], target[dst == node][:1].expand_as(target[dst == node]))
    for index in range(4):
        off = renormalize_without(pi, index)
        assert torch.equal(off[:, index], torch.zeros_like(off[:, index]))
        assert torch.allclose(off.sum(-1), torch.ones(dst.numel()), atol=1e-6)

    static, x, edge_index = make_model("static_mix")
    target_model, _, _ = make_model("target_mix")
    static.eval(); target_model.eval()
    with torch.no_grad():
        _, _, _, _, static_aux = static(x, edge_index, return_diagnostics=True)
        _, _, _, _, target_aux = target_model(x, edge_index, return_diagnostics=True)
        s_pi = static_aux["controls"]["pi"]
        t_pi = target_aux["controls"]["pi"]
        dst_all = static_aux["edge_index_nonself"][1]
        s_mean = tuple(pi_global_mean(v) for v in s_pi)
        t_mean = tuple(pi_target_mean(v, dst_all, x.size(0)) for v in t_pi)
        for m in range(2):
            assert torch.allclose(s_pi[m], s_mean[m], atol=1e-6, rtol=0)  # Static/global identity.
            assert torch.allclose(t_pi[m], t_mean[m], atol=1e-6, rtol=0)  # Target/target-mean identity.


def test_within_target_shuffle_preserves_pi_vector_multisets():
    dst = torch.tensor([2, 1, 1, 2, 2, 3, 1])
    pi = torch.softmax(torch.arange(dst.numel() * 4).view(-1, 4).float(), dim=-1)
    permuted = neighborhood_shuffle_indices(dst, 1001)
    moved = pi[permuted]
    for node in torch.unique(dst):
        original = sorted(tuple(row.tolist()) for row in pi[dst == node])
        shuffled = sorted(tuple(row.tolist()) for row in moved[dst == node])
        assert original == shuffled


def test_no_test_labels_or_indices_are_available_to_e0_protocol():
    data = MAGData(
        name="synthetic", source="unit", task="nc", x=torch.zeros(6, 2),
        edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=6,
        y=torch.arange(6), train_idx=torch.tensor([0, 1]), val_idx=torch.tensor([2]),
        test_idx=torch.tensor([3, 4, 5]), num_classes=6,
    )
    splits = restrict_labels_to_train_val(data)
    assert set(splits) == {"train", "validation"}
    assert data.test_idx is None
    assert data.y.tolist() == [0, 1, 2, -100, -100, -100]
    assert cfg("edge_mix").task.evaluate_test is False
