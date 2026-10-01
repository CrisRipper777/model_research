from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.adaptive_prop_m0 import Model as M0Model, incoming_degree, remove_self_messages
from src.models.adaptive_prop_n1 import (
    MASTER_BLOCKS,
    MASTER_DIM,
    Model,
    _feature_cosine,
    common_parameter_names,
    copy_m0_uniform_common_weights,
    parameter_counts,
)
from src.data.types import MAGData
from scripts.run_n1_recipient_function_strength_mixer import (
    beta_off,
    build_classifier,
    make_config,
    restrict_labels_to_train_val,
    shuffle_validation_beta_tuples,
    tie_validation_beta_modalities,
    validation_global_mean_beta,
)


def cfg(variant: str, chunk: int = 100000, dropout: float = 0.0):
    return OmegaConf.create({"model": {
        "name": "adaptive_prop_n1", "variant": variant, "hidden_dim": 128,
        "dropout": dropout, "edge_chunk_size": chunk,
    }})


def sample():
    torch.manual_seed(131)
    x = torch.randn(9, 14)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 1, 4, 2, 6, 7, 8, 0],
        [1, 1, 1, 1, 2, 2, 4, 5, 5, 8, 0],
    ])
    info = {"input_dim": 14, "num_nodes": 9, "num_classes": 3, "text_dim": 6, "visual_dim": 8}
    return x, edge_index, info


def make_model(variant="cross_state_strength", chunk=100000):
    x, edge_index, info = sample()
    torch.manual_seed(17)
    return Model(cfg(variant, chunk), info), x, edge_index, info


def test_master_state_has_exactly_eight_ordered_128d_blocks():
    model, x, edge_index, _ = make_model()
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    assert MASTER_BLOCKS == ("h0_text", "h0_visual", "smooth_text", "absdiff_text",
                             "product_text", "smooth_visual", "absdiff_visual", "product_visual")
    assert MASTER_DIM == 1024
    assert aux["master_dim"] == 1024


def test_text_visual_split_and_self_loop_removal_keep_directed_order():
    model, x, edge_index, _ = make_model()
    text, visual = model.split_modalities(x)
    assert torch.equal(text, x[:, :6]) and torch.equal(visual, x[:, 6:])
    kept, src, dst = remove_self_messages(edge_index)
    assert torch.equal(kept, edge_index[:, edge_index[0] != edge_index[1]])
    assert not torch.any(src == dst)
    assert torch.equal(dst, kept[1])
    assert incoming_degree(dst, x.size(0)).tolist() == [0, 3, 2, 0, 1, 2, 0, 0, 0]


def test_channel_messages_match_exact_smooth_absdiff_product_formulas():
    model, x, edge_index, _ = make_model("cross_state_strength")
    model.eval()
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    src, dst = aux["edge_index_nonself"]
    degree = aux["degree"].clamp_min(1).to(x.dtype)
    for modality, (lo, hi) in enumerate(((0, 6), (6, 14))):
        h = (model.proj_t if modality == 0 else model.proj_v)(x[:, lo:hi])
        brute = [h.new_zeros((x.size(0), 128)) for _ in range(3)]
        for edge_id in range(src.numel()):
            i, j = int(dst[edge_id]), int(src[edge_id])
            messages = (
                model.w_s[modality](h[j]),
                model.w_d[modality](torch.abs(h[j] - h[i])),
                model.w_p[modality](h[i] * h[j]),
            )
            for c, message in enumerate(messages):
                brute[c][i] += message
        brute = [value / degree.clamp_min(1).unsqueeze(-1) for value in brute]
        for actual, expected in zip(aux["contexts"][modality], brute):
            assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-6)


def test_isolated_nodes_have_zero_contexts():
    model, x, edge_index, _ = make_model()
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    isolated = aux["degree"] == 0
    for modality in aux["contexts"]:
        for context in modality:
            assert torch.count_nonzero(context[isolated]) == 0


def test_chunked_and_unchunked_contexts_match():
    small, x, edge_index, _ = make_model(chunk=2)
    large = copy.deepcopy(small)
    large.edge_chunk_size = 1000
    small.eval(); large.eval()
    with torch.no_grad():
        a = small(x, edge_index, return_diagnostics=True)
        b = large(x, edge_index, return_diagnostics=True)
    assert torch.allclose(a[0], b[0], atol=2e-6, rtol=1e-6)
    for ma, mb in zip(a[4]["contexts"], b[4]["contexts"]):
        for ca, cb in zip(ma, mb):
            assert torch.allclose(ca, cb, atol=2e-6, rtol=1e-6)


def test_smoothonly_forward_matches_m0_uniform_after_common_weight_mapping():
    x, edge_index, info = sample()
    torch.manual_seed(71)
    m0 = M0Model(OmegaConf.create({"model": {
        "variant": "uniform", "hidden_dim": 128, "dropout": 0.0, "edge_chunk_size": 100000,
        "relation_dim": 32, "relation_state_dim": 64, "modality_embed_dim": 8,
    }}), info)
    torch.manual_seed(71)
    n1 = Model(cfg("smooth_only"), info)
    copy_m0_uniform_common_weights(m0, n1)
    m0.eval(); n1.eval()
    with torch.no_grad():
        z0 = m0(x, edge_index)[0]
        z1, _, _, _, aux = n1(x, edge_index, return_diagnostics=True)
    assert torch.allclose(aux["contexts"][0][0], _m0_context(m0, x, edge_index, 0), atol=1e-7, rtol=1e-6)
    assert torch.allclose(z0, z1, atol=1e-7, rtol=1e-6)


def test_residual_and_fusion_follow_the_declared_formula():
    model, x, edge_index, _ = make_model("cross_state_strength")
    model.eval()
    z, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    for modality in range(2):
        cs, cd, cp = aux["contexts"][modality]
        beta_d, beta_p = aux["betas"][modality].unbind(-1)
        update = cs + beta_d[:, None] * cd + beta_p[:, None] * cp
        expected_tilde = model.residual_norms[modality](aux["h0"][modality] + update)
        assert torch.allclose(expected_tilde, aux["h_tilde"][modality], atol=1e-7, rtol=1e-6)
    expected_z = model.fusion(torch.cat([
        aux["h0"][0], aux["h0"][1], aux["h_tilde"][0], aux["h_tilde"][1]
    ], dim=-1))
    assert torch.equal(z, expected_z)


def _m0_context(model, x, edge_index, modality):
    h0, _, src, dst, degree, *_ = model._prepare(x, edge_index)
    agg = h0[modality].new_zeros(h0[modality].shape)
    for begin in range(0, src.numel(), model.edge_chunk_size):
        end = min(begin + model.edge_chunk_size, src.numel())
        agg = agg.index_add(0, dst[begin:end], model.w0[modality](h0[modality][src[begin:end]]))
    return agg / degree.clamp_min(1).to(agg.dtype).unsqueeze(-1)


def test_variant_masks_are_exact_and_cross_uses_all_master_blocks():
    master = torch.randn(4, MASTER_DIM)
    for modality, expected_blocks in ((0, (0, 2, 3, 4)), (1, (1, 5, 6, 7))):
        masked = Model._mask_for_variant(master, "same_state_strength", modality)
        expected = torch.zeros_like(master)
        for block in expected_blocks:
            expected[:, block * 128:(block + 1) * 128] = master[:, block * 128:(block + 1) * 128]
        assert torch.equal(masked, expected)
    assert torch.equal(Model._mask_for_variant(master, "static_strength", 0), torch.zeros_like(master))
    assert torch.equal(Model._mask_for_variant(master, "cross_state_strength", 1), master)


def test_static_mask_passes_exactly_zero_to_both_mixers():
    model, x, edge_index, _ = make_model("static_strength")
    seen = []
    hooks = [mixer[0].register_forward_pre_hook(lambda _module, args: seen.append(args[0].detach().clone()))
             for mixer in model.strength_mixers]
    model(x, edge_index)
    for hook in hooks:
        hook.remove()
    assert len(seen) == 2
    assert all(torch.count_nonzero(value) == 0 for value in seen)


def test_four_variants_have_equal_params_and_bitwise_initialization():
    x, edge_index, info = sample()
    models = {}
    for variant in ("smooth_only", "static_strength", "same_state_strength", "cross_state_strength"):
        torch.manual_seed(29)
        models[variant] = Model(cfg(variant), info)
    reference = models["smooth_only"].state_dict()
    for variant, model in models.items():
        assert parameter_counts(model)["model_total"] == parameter_counts(models["smooth_only"])["model_total"]
        assert common_parameter_names(model) == common_parameter_names(models["smooth_only"])
        assert all(torch.equal(reference[name], model.state_dict()[name]) for name in reference)


def test_classifier_seed_isolated_from_variant_module_initialization():
    x, edge_index, info = sample()
    classifiers = []
    for variant in ("smooth_only", "static_strength", "same_state_strength", "cross_state_strength"):
        torch.manual_seed(42)
        _ = Model(cfg(variant), info)
        classifiers.append(build_classifier(128, 3, 42).state_dict())
    for other in classifiers[1:]:
        assert all(torch.equal(classifiers[0][name], other[name]) for name in classifiers[0])


def test_initial_strengths_are_about_point_one_and_nonnegative_finite():
    model, x, edge_index, _ = make_model("cross_state_strength")
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    for beta in aux["betas"]:
        assert torch.isfinite(beta).all()
        assert (beta >= 0).all()
        assert torch.allclose(beta, torch.full_like(beta, 0.1), atol=1e-3, rtol=2e-3)


def test_smooth_only_strengths_are_exact_zero_and_have_no_alternative_gradients():
    model, x, edge_index, _ = make_model("smooth_only")
    z = model(x, edge_index)[0]
    nn.functional.cross_entropy(nn.Linear(128, 3)(z), torch.arange(x.size(0)) % 3).backward()
    for parameter in list(model.w_d.parameters()) + list(model.w_p.parameters()) + list(model.strength_mixers.parameters()):
        assert parameter.grad is None
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    assert all(torch.count_nonzero(beta) == 0 for beta in aux["betas"])


def test_static_strength_is_node_constant_and_mixer_parameters_receive_gradients():
    model, x, edge_index, _ = make_model("static_strength")
    beta = model(x, edge_index, return_diagnostics=True)[4]["betas"]
    assert all(torch.equal(value, value[:1].expand_as(value)) for value in beta)
    z = model(x, edge_index)[0]
    nn.functional.cross_entropy(nn.Linear(128, 3)(z), torch.arange(x.size(0)) % 3).backward()
    for parameter in model.strength_mixers.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("variant", ["same_state_strength", "cross_state_strength"])
def test_dynamic_variants_allow_node_dependent_strengths(variant):
    model, x, edge_index, _ = make_model(variant)
    with torch.no_grad():
        for m, mixer in enumerate(model.strength_mixers):
            mixer[0].weight.normal_(0, 0.1)
            mixer[2].weight.normal_(0, 0.1)
    beta = model(x, edge_index, return_diagnostics=True)[4]["betas"]
    assert all(float(value.std(unbiased=False)) > 0 for value in beta)


def test_smooth_contexts_are_identical_across_variant_masks():
    x, edge_index, info = sample()
    outputs = {}
    for variant in ("smooth_only", "static_strength", "same_state_strength", "cross_state_strength"):
        torch.manual_seed(57)
        model = Model(cfg(variant), info).eval()
        outputs[variant] = model(x, edge_index, return_diagnostics=True)[4]["contexts"]
    for variant in tuple(outputs)[1:]:
        for m in range(2):
            assert torch.equal(outputs["smooth_only"][m][0], outputs[variant][m][0])


def test_beta_override_identity_and_intervention_controls():
    model, x, edge_index, _ = make_model("cross_state_strength")
    model.eval()
    base = model(x, edge_index, return_diagnostics=True)
    beta = base[4]["betas"]
    identical = model(x, edge_index, beta_overrides=beta)[0]
    assert torch.equal(base[0], identical)
    d_off = tuple(torch.stack([torch.zeros_like(b[:, 0]), b[:, 1]], dim=-1) for b in beta)
    p_off = tuple(torch.stack([b[:, 0], torch.zeros_like(b[:, 1])], dim=-1) for b in beta)
    dp_off = tuple(torch.zeros_like(b) for b in beta)
    for override in (d_off, p_off, dp_off):
        assert torch.isfinite(model(x, edge_index, beta_overrides=override)[0]).all()


def test_shuffle_preserves_full_beta_tuple_multiset():
    beta = torch.tensor([[1., 10.], [2., 20.], [3., 30.], [4., 40.]])
    perm = torch.tensor([2, 0, 3, 1])
    assert sorted(map(tuple, beta.tolist())) == sorted(map(tuple, beta[perm].tolist()))
    assert torch.equal(beta[:, 0][perm], beta[perm, 0])


def test_global_mean_and_modality_tying_values_are_exact():
    beta_t = torch.tensor([[1., 2.], [3., 4.]])
    beta_v = torch.tensor([[5., 6.], [7., 8.]])
    global_t, global_v = beta_t.mean(0, keepdim=True), beta_v.mean(0, keepdim=True)
    tied = (beta_t + beta_v) / 2
    assert torch.equal(global_t, torch.tensor([[2., 3.]]))
    assert torch.equal(global_v, torch.tensor([[6., 7.]]))
    assert torch.equal(tied, torch.tensor([[3., 4.], [5., 6.]]))


def test_intervention_helpers_preserve_tuple_multisets_and_nonvalidation_nodes():
    beta_t = torch.tensor([[0., 10.], [1., 11.], [2., 12.], [3., 13.], [4., 14.]])
    beta_v = torch.tensor([[5., 15.], [6., 16.], [7., 17.], [8., 18.], [9., 19.]])
    betas = (beta_t, beta_v)
    val_idx = torch.tensor([1, 2, 4])
    shuffled = shuffle_validation_beta_tuples(betas, val_idx, 3101)
    for modality in range(2):
        assert sorted(map(tuple, betas[modality][val_idx].tolist())) == sorted(
            map(tuple, shuffled[modality][val_idx].tolist()))
        nonval = torch.tensor([0, 3])
        assert torch.equal(betas[modality][nonval], shuffled[modality][nonval])
    means = validation_global_mean_beta(betas, val_idx)
    for modality in range(2):
        expected = betas[modality][val_idx].mean(0).expand(val_idx.numel(), -1)
        assert torch.equal(means[modality][val_idx], expected)
        assert torch.equal(means[modality][torch.tensor([0, 3])], betas[modality][torch.tensor([0, 3])])
    tied = tie_validation_beta_modalities(betas, val_idx)
    assert torch.equal(tied[0][val_idx], tied[1][val_idx])
    assert torch.equal(tied[0][torch.tensor([0, 3])], beta_t[torch.tensor([0, 3])])
    off = beta_off(betas, (0,))
    assert torch.count_nonzero(off[0][:, 0]) == 0 and torch.count_nonzero(off[1][:, 0]) == 0
    assert torch.equal(off[0][:, 1], beta_t[:, 1]) and torch.equal(off[1][:, 1], beta_v[:, 1])


def test_changing_validation_beta_does_not_change_other_node_representations():
    model, x, edge_index, _ = make_model("cross_state_strength")
    model.eval()
    base = model(x, edge_index, return_diagnostics=True)
    val = torch.tensor([1, 2, 5])
    overrides = [v.clone() for v in base[4]["betas"]]
    for value in overrides:
        value[val] = value[val].flip(0)
    changed = model(x, edge_index, tuple(overrides), return_diagnostics=True)
    nonval = torch.ones(x.size(0), dtype=torch.bool)
    nonval[val] = False
    assert torch.equal(base[4]["h_tilde"][0][nonval], changed[4]["h_tilde"][0][nonval])
    assert torch.equal(base[4]["h_tilde"][1][nonval], changed[4]["h_tilde"][1][nonval])


def test_channel_scale_and_distinctness_diagnostics_are_finite():
    model, x, edge_index, _ = make_model()
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    for modality in aux["contexts"]:
        for value in modality:
            assert torch.isfinite(value).all()
    cs, cd, cp = aux["contexts"][0]
    assert torch.isfinite(_feature_cosine(cs, cd)).all()
    assert torch.isfinite(_feature_cosine(cs, cp)).all()


def test_training_config_matches_frozen_unified_protocol_and_disables_test():
    config = make_config("Movies", 42, "cross_state_strength", "cpu")
    assert config.task.protocol_version == "unified_full_graph_nc_v1"
    assert config.task.training_mode == "full_graph"
    assert config.task.optimizer == "adamw"
    assert float(config.task.lr) == 1e-3
    assert float(config.task.weight_decay) == 1e-4
    assert int(config.task.epochs) == 300 and int(config.task.patience) == 30
    assert float(config.task.grad_clip) == 1.0
    assert not bool(config.task.evaluate_test)


def test_nc_label_mask_keeps_train_validation_labels_and_hides_test_labels():
    data = MAGData(
        name="synthetic", source="magb", task="nc", x=torch.zeros(5, 4),
        edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=5,
        y=torch.tensor([0, 1, 2, 1, 0]), train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2]), test_idx=torch.tensor([3, 4]), num_classes=3,
    )
    split = restrict_labels_to_train_val(data)
    assert data.test_idx is None
    assert split["train"].tolist() == [0, 1] and split["validation"].tolist() == [2]
    assert data.y.tolist()[:3] == [0, 1, 2]
    assert data.y[3:].tolist() == [-100, -100]
