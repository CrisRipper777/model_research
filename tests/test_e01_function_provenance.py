from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.run_m0_adaptive_propagation import restrict_labels_to_train_val
from src.data.types import MAGData
from src.models.adaptive_prop_m0 import Model as M0Model
from src.models.adaptive_prop_m0 import fixed_degree_mean, incoming_degree, neighborhood_shuffle_indices, remove_self_messages
from src.models.provenance_executor_e01 import E01_VARIANTS, Model
from src.models.structured_executor_e0 import (
    Model as E0Model,
    pi_global_mean,
    pi_modality_tied,
    pi_target_mean,
)


def cfg(variant="keep_edge", chunk=100000):
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


def make_model(variant="keep_edge", chunk=100000):
    x, edge_index, info = graph()
    torch.manual_seed(702)
    return Model(cfg(variant, chunk), info), x, edge_index


def _load_common(source, target):
    target.load_state_dict({key: value for key, value in source.state_dict().items()
                            if key in target.state_dict()}, strict=False)


def _contexts(model, x, edge_index, pi_override=None):
    return model(x, edge_index, control_overrides=pi_override,
                 return_diagnostics=True)[4]


def test_modality_slicing_physical_support_no_self_and_fixed_degree():
    model, x, edge_index = make_model()
    text, visual = model.split_modalities(x)
    assert torch.equal(text, x[:, :6]) and torch.equal(visual, x[:, 6:])
    nonself, src, dst = remove_self_messages(edge_index)
    assert torch.equal(nonself, edge_index[:, edge_index[0] != edge_index[1]])
    assert not bool((src == dst).any())
    degree = incoming_degree(dst, x.size(0))
    assert degree.tolist() == [0, 0, 0, 0, 0, 0, 4, 4, 8]
    msg = torch.ones(src.numel(), 128)
    agg = torch.zeros(x.size(0), 128).index_add(0, dst, msg)
    assert torch.equal(fixed_degree_mean(agg, degree)[6], torch.ones(128))


def test_m0_e0_relation_state_and_function_bank_regression():
    x, edge_index, info = graph()
    torch.manual_seed(921)
    m0 = M0Model(OmegaConf.create({"model": {
        "variant": "single_basis", "hidden_dim": 128, "dropout": 0.0,
        "relation_dim": 32, "relation_state_dim": 64, "modality_embed_dim": 8,
        "edge_chunk_size": 3, "eps": 1e-8,
    }}), info).eval()
    torch.manual_seed(98)
    e0 = E0Model(cfg("edge_mix", 3), info).eval()
    torch.manual_seed(98)
    e01 = Model(cfg("keep_edge", 3), info).eval()
    _load_common(m0, e0)
    _load_common(e0, e01)
    s0, s1 = e0.relation_state(x, edge_index), e01.relation_state(x, edge_index)
    for key in ("h0", "p", "q", "contexts", "u"):
        assert all(torch.allclose(a, b, rtol=1e-6, atol=1e-7)
                   for a, b in zip(s0[key], s1[key]))
    assert torch.allclose(s0["r"], s1["r"], rtol=1e-6, atol=1e-7)
    assert torch.equal(s0["src"], s1["src"]) and torch.equal(s0["dst"], s1["dst"])
    assert torch.equal(s0["degree"], s1["degree"])
    m0_state = m0._prepare(x, edge_index)
    _, _, m0_src, m0_dst, m0_degree, m0_deg_z, m0_p, m0_contexts = m0_state
    m0_evidence_t = m0._pair_evidence(m0_p[0], m0_contexts[0], m0_deg_z, m0_src, m0_dst)
    m0_evidence_v = m0._pair_evidence(m0_p[1], m0_contexts[1], m0_deg_z, m0_src, m0_dst)
    m0_q = (m0.phi_pair(m0_evidence_t), m0.phi_pair(m0_evidence_v))
    m0_r = m0.phi_rel(torch.cat([m0_q[0], m0_q[1], (m0_q[0]-m0_q[1]).abs(), m0_q[0]*m0_q[1]], -1))
    for left, right in zip(m0_p, s1["p"]):
        assert torch.allclose(left, right, rtol=1e-6, atol=1e-7)
    for left, right in zip(m0_q, s1["q"]):
        assert torch.allclose(left, right, rtol=1e-6, atol=1e-7)
    assert torch.allclose(m0_r, s1["r"], rtol=1e-6, atol=1e-7)
    assert torch.equal(m0_degree, s1["degree"])
    e0_functions = e0._expert_outputs(s0["h0"], s0["src"], s0["dst"])
    e01_functions = e01._expert_outputs(s1["h0"], s1["src"], s1["dst"])
    for index in range(3):
        assert all(torch.equal(a, b) for a, b in zip(e0_functions[index], e01_functions[index]))
    for modality in range(2):
        pi0 = torch.softmax(e0.router(s0["u"][modality]), -1)
        pi1 = torch.softmax(e01.router(s1["u"][modality]), -1)
        assert torch.equal(pi0, pi1)


def test_same_pi_e0_mixed_aggregate_matches_provenance_channel_sum():
    x, edge_index, info = graph()
    torch.manual_seed(93)
    e0 = E0Model(cfg("edge_mix", 3), info).eval()
    torch.manual_seed(93)
    e01 = Model(cfg("keep_edge", 3), info).eval()
    with torch.no_grad():
        _, _, _, _, e0_aux = e0(x, edge_index, return_diagnostics=True)
        _, _, _, _, e01_aux = e01(x, edge_index, return_diagnostics=True)
        state = e0.relation_state(x, edge_index)
        functions = e0._expert_outputs(state["h0"], state["src"], state["dst"])
        degree = state["degree"]
        pi = e0_aux["controls"]["pi"]
        for modality in range(2):
            message = (pi[modality][:, 1:2] * functions[0][modality]
                       + pi[modality][:, 2:3] * functions[1][modality]
                       + pi[modality][:, 3:4] * functions[2][modality])
            sums = torch.zeros_like(state["h0"][modality]).index_add(0, state["dst"], message)
            old_e0_mix = fixed_degree_mean(sums, degree)
            assert torch.allclose(old_e0_mix, e01_aux["c_mix"][modality], rtol=1e-6, atol=1e-6)


def test_cmix_is_sum_of_fixed_degree_channel_aggregates():
    model, x, edge_index = make_model("keep_edge", chunk=2)
    model.eval()
    aux = _contexts(model, x, edge_index)
    assert max(float((aux["c_mix"][m] - sum(aux["channel_contexts"][m])).abs().max())
               for m in range(2)) <= 1e-7
    _, src, dst = remove_self_messages(edge_index)
    degree = incoming_degree(dst, x.size(0))
    pi = aux["controls"]["pi"]
    state = model.relation_state(x, edge_index)
    functions = model._expert_outputs(state["h0"], src, dst)
    for modality in range(2):
        manual = []
        for channel_index, function in enumerate(functions[:3]):
            # E0 function return order is Smooth, Relational, Cross-Modal.
            messages = pi[modality][:, channel_index + 1:channel_index + 2] * function[modality]
            sums = torch.zeros_like(state["h0"][modality]).index_add(0, dst, messages)
            manual.append(fixed_degree_mean(sums, degree))
        for observed, expected in zip(aux["channel_contexts"][modality], manual):
            assert torch.allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_exact_parameter_count_and_bitwise_initialization_across_variants():
    x, edge_index, info = graph()
    models = {}
    for variant in E01_VARIANTS:
        torch.manual_seed(114)
        models[variant] = Model(cfg(variant), info)
    reference = models[E01_VARIANTS[0]].state_dict()
    counts = {sum(parameter.numel() for parameter in model.parameters()) for model in models.values()}
    assert len(counts) == 1
    for model in models.values():
        state = model.state_dict()
        assert state.keys() == reference.keys()
        assert all(torch.equal(state[key], reference[key]) for key in state)
        assert len(model.provenance_composers) == 2
        assert all(layer[0].in_features == 384 and layer[0].out_features == 128
                   and layer[2].out_features == 128 for layer in model.provenance_composers)


@pytest.mark.parametrize("variant", E01_VARIANTS)
def test_historical_e0_common_initialization_bitwise_equal(variant):
    x, edge_index, info = graph()
    torch.manual_seed(88)
    e0 = E0Model(cfg("edge_mix"), info)
    torch.manual_seed(88)
    e01 = Model(cfg(variant), info)
    for key, value in e0.state_dict().items():
        assert key in e01.state_dict()
        assert torch.equal(value, e01.state_dict()[key]), key


def test_composer_input_contract_and_residual_base():
    model, x, edge_index = make_model("keep_edge")
    model.eval()
    aux = _contexts(model, x, edge_index)
    for modality in range(2):
        cs, cr, cx = aux["channel_contexts"][modality]
        expected = torch.cat((cs, cr, cx), -1)
        assert aux["composer_input"][modality].shape == (x.size(0), 384)
        assert torch.equal(aux["composer_input"][modality], expected)
        assert torch.allclose(aux["delta"][modality], aux["c_mix"][modality]
                              + aux["composer_output"][modality])

    premix, _, _ = make_model("premix_edge_control")
    premix.eval()
    aux_p = _contexts(premix, x, edge_index)
    for modality in range(2):
        repeated = (aux_p["c_mix"][modality] / 3).repeat(1, 3)
        assert torch.equal(aux_p["composer_input"][modality], repeated)
    collapsed_output = premix(x, edge_index, composer_input_mode="collapse")[0]
    assert torch.equal(collapsed_output, premix(x, edge_index)[0])

    keep_edge, _, _ = make_model("keep_edge")
    keep_edge.eval()
    aux_k = _contexts(keep_edge, x, edge_index)
    for modality in range(2):
        assert torch.equal(aux_p["c_mix"][modality], aux_k["c_mix"][modality])
        for c_p, c_k in zip(aux_p["channel_contexts"][modality], aux_k["channel_contexts"][modality]):
            assert torch.equal(c_p, c_k)


def test_composer_initialization_is_small_but_receives_finite_nonzero_gradients():
    model, x, edge_index = make_model()
    model.train()
    z = model(x, edge_index)[0]
    head = nn.Linear(model.out_dim, 3)
    y = torch.arange(x.size(0)) % 3
    nn.functional.cross_entropy(head(z), y).backward()
    for layer_index in (0, 2):
        grads = [p.grad for name, p in model.named_parameters()
                 if name.startswith("provenance_composers.") and f".{layer_index}." in name]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert sum(float(grad.norm()) for grad in grads) > 0

    model.eval()
    aux = _contexts(model, x, edge_index)
    ratios = []
    for modality in range(2):
        comp, mix = aux["composer_output"][modality], aux["c_mix"][modality]
        ratio = (comp.square().mean(-1).sqrt()) / (mix.square().mean(-1).sqrt() + model.eps)
        ratios.extend(ratio.tolist())
    assert torch.tensor(ratios).median() < 1.0

    # In the repeated-input capacity control, each 128D input block is active.
    premix, _, _ = make_model("premix_edge_control")
    premix.train()
    nn.functional.cross_entropy(head(premix(x, edge_index)[0]), y).backward()
    first_weight = premix.provenance_composers[0][0].weight.grad.reshape(128, 3, 128)
    assert all(float(first_weight[:, block].norm()) > 0 for block in range(3))


def test_all_required_gradients_are_nonzero():
    model, x, edge_index = make_model()
    model.train()
    head = nn.Linear(128, 3)
    labels = torch.arange(x.size(0)) % 3
    nn.functional.cross_entropy(head(model(x, edge_index)[0]), labels).backward()
    prefixes = ("proj_t.", "proj_v.", "router.", "smooth_transforms.",
                "relational_source.", "relational_mlps.", "cross_modal_source.",
                "cross_modal_mlps.", "provenance_composers.0.0.",
                "provenance_composers.0.2.", "fusion.")
    for prefix in prefixes:
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.norm()) for g in grads) > 0, prefix


@pytest.mark.parametrize("variant", E01_VARIANTS)
def test_chunk_equivalence(variant):
    small, x, edge_index = make_model(variant, chunk=2)
    large = copy.deepcopy(small)
    large.edge_chunk_size = 1000
    small.eval(); large.eval()
    with torch.no_grad():
        z_small = small(x, edge_index)[0]
        z_large = large(x, edge_index)[0]
    assert torch.allclose(z_small, z_large, rtol=2e-5, atol=2e-6)


def test_static_target_and_premix_identity_sanity():
    x, edge_index, info = graph()
    for variant in ("keep_static", "keep_target", "premix_edge_control"):
        model = Model(cfg(variant), info).eval()
        with torch.no_grad():
            _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
        pi = aux["controls"]["pi"]
        dst = aux["edge_index_nonself"][1]
        if variant == "keep_static":
            assert all(torch.allclose(pi_global_mean(pi[m]), pi[m], atol=1e-6, rtol=0)
                       for m in range(2))
        elif variant == "keep_target":
            assert all(torch.allclose(pi_target_mean(pi[m], dst, x.size(0)), pi[m], atol=1e-6, rtol=0)
                       for m in range(2))
        else:
            collapsed = model(x, edge_index, composer_input_mode="collapse")[0]
            assert torch.equal(collapsed, model(x, edge_index)[0])


def test_provenance_collapse_permutations_and_composer_off_identities():
    model, x, edge_index = make_model("keep_edge")
    model.eval()
    normal = _contexts(model, x, edge_index)
    collapsed = model(x, edge_index, composer_input_mode="collapse", return_diagnostics=True)[4]
    assert all(torch.equal(normal["c_mix"][m], collapsed["c_mix"][m]) for m in range(2))
    off = model(x, edge_index, composer_off=True, return_diagnostics=True)[4]
    assert all(torch.equal(off["delta"][m], off["c_mix"][m]) for m in range(2))
    orders = ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    for order in orders:
        mode = "permute:" + ",".join(map(str, order))
        permuted = model(x, edge_index, composer_input_mode=mode, return_diagnostics=True)[4]
        assert all(torch.equal(normal["c_mix"][m], permuted["c_mix"][m]) for m in range(2))
        for modality in range(2):
            expected = torch.cat(tuple(normal["channel_contexts"][modality][i] for i in order), -1)
            assert torch.equal(permuted["composer_input"][modality], expected)


def test_routing_intervention_identities_and_smooth_only_composer_input():
    model, x, edge_index = make_model("keep_edge")
    model.eval()
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    pi = aux["controls"]["pi"]
    dst = aux["edge_index_nonself"][1]
    from src.models.provenance_executor_e01 import provenance_interventions
    rows = provenance_interventions(pi, dst, x.size(0))
    for name, repeat, pair in rows:
        for modality in range(2):
            assert torch.allclose(pair[modality].sum(-1), torch.ones(pair[modality].size(0)), atol=1e-6)
            if name == "pi_target_mean":
                assert all(torch.equal(pair[modality][dst == node], pair[modality][dst == node][:1].expand_as(pair[modality][dst == node]))
                           for node in torch.unique(dst))
            if name == "pi_shuffle_within_target":
                assert repeat in range(1001, 1006)
                for node in torch.unique(dst):
                    original = sorted(map(tuple, pi[modality][dst == node].tolist()))
                    shuffled = sorted(map(tuple, pair[modality][dst == node].tolist()))
                    assert original == shuffled
    tied = pi_modality_tied(pi[0], pi[1])
    assert torch.equal(tied[0], tied[1])
    assert torch.allclose(tied[0].sum(-1), torch.ones(tied[0].size(0)), atol=1e-6)
    global_pair = next(row[2] for row in rows if row[0] == "pi_global_mean")
    assert all(torch.equal(global_pair[m], pi_global_mean(pi[m])) for m in range(2))

    smooth = torch.zeros_like(pi[0]); smooth[:, 1] = 1
    aux_s = model(x, edge_index, control_overrides={"pi": (smooth, smooth)}, return_diagnostics=True)[4]
    for modality in range(2):
        cs, cr, cx = aux_s["channel_contexts"][modality]
        assert torch.count_nonzero(cr) == 0 and torch.count_nonzero(cx) == 0
        assert torch.equal(aux_s["composer_input"][modality], torch.cat((cs, cr, cx), -1))


def test_no_test_labels_or_indices_are_available_to_e01_protocol():
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
    assert cfg().task.evaluate_test is False
