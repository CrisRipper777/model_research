#!/usr/bin/env python3
"""Label-free V4A preflight and validation-selected checkpoint diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("R0_raw", "R1_global_residual", "R2_expert_residual", "R3_adaptive_residual")
PAIRS = (
    ("R1_global_residual", "R0_raw"),
    ("R2_expert_residual", "R1_global_residual"),
    ("R2_expert_residual", "R0_raw"),
    ("R3_adaptive_residual", "R2_expert_residual"),
    ("R3_adaptive_residual", "R0_raw"),
    ("R3_adaptive_residual", "R1_global_residual"),
)
EXPERT_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v4a_raw_anchored_residual_screen"
DATA_ROOT = RESEARCH_ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v4a_raw_anchored_residual_screen"
V3C_DATA = ROOT / "research" / "mvcge_mag_v3c_functional_role_screen" / "data"
V3C_OUTPUT = ROOT / "outputs" / "mvcge_mag_v3c_functional_role_screen"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _compose(dataset: str, seed: int):
    from scripts.analyze_mvcge_mag_v3a_screen import _compose_dataset_config

    return _compose_dataset_config(dataset, seed)


def load_features_and_edges(dataset: str, seed: int = 42):
    # This loader intentionally returns only frozen modality features and physical edges.
    from scripts.analyze_mvcge_mag_v3a_screen import load_features_and_edges as load

    return load(dataset, seed)


def _feature_dims(dataset: str, seed: int):
    from scripts.analyze_mvcge_mag_v3a_screen import _feature_dims

    return _feature_dims(dataset, seed)


def _data_info(dataset: str, seed: int, x: torch.Tensor) -> dict[str, int]:
    cfg = _compose(dataset, seed)
    input_dim, text_dim, visual_dim = _feature_dims(dataset, seed)
    if input_dim != x.size(1):
        raise ValueError(f"feature dimension mismatch for {dataset}: {input_dim} != {x.size(1)}")
    return {"input_dim": input_dim, "text_dim": text_dim, "visual_dim": visual_dim,
            "num_nodes": int(x.size(0)), "num_classes": int(cfg.dataset.num_classes)}


def _model_cfg(variant: str):
    cfg = OmegaConf.load(ROOT / "configs" / "model" / "mvcge_mag_v4a.yaml")
    cfg.variant = variant
    return OmegaConf.create({"model": OmegaConf.to_container(cfg, resolve=True)})


def _rms(x: torch.Tensor, active: torch.Tensor | None = None) -> float:
    if active is not None:
        x = x[active]
    if not x.numel():
        return 0.0
    return float(x.float().square().mean().sqrt().item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if not a.numel() or not b.numel():
        return None
    return float(F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0, eps=1e-8).clamp(-1, 1).item())


def _span_outside(basis: torch.Tensor, target: torch.Tensor, eps: float) -> tuple[float, float | None]:
    """Small Gram/pseudoinverse projection; never constructs a node-space projector."""
    gram = basis @ basis.T
    cross = basis @ target.T
    coef = torch.linalg.pinv(gram) @ cross
    residual_sq = (target.square().sum() - 2 * torch.trace(coef.T @ cross)
                   + torch.trace(coef.T @ gram @ coef)).clamp_min(0)
    ratio = float(residual_sq.sqrt().item() / (torch.linalg.vector_norm(target).item() + eps))
    cond = float(torch.linalg.cond(gram).item())
    return ratio, cond if math.isfinite(cond) else None


@torch.no_grad()
def preflight(device: str = "cuda:0") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Untrained feature/edge-only checks; no label or split object is loaded."""
    from src.models.mvcge_mag_v4a import Model

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device unavailable: {device}")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    initial_checks = []
    for dataset in DATASETS:
        x_cpu, edge_cpu = load_features_and_edges(dataset, 42)
        info = _data_info(dataset, 42, x_cpu)
        x, edge = x_cpu.to(device), edge_cpu.to(device)

        # Compare initialized forwards without retaining full graph diagnostics.
        z0_cpu = None
        aux0 = None
        for variant in VARIANTS:
            torch.manual_seed(44042)
            model = Model(_model_cfg(variant), info).to(device).eval()
            z, _, _, aux, _ = model(x, edge)
            z_cpu, aux_value = z.detach().cpu(), float(aux.item())
            if variant == "R0_raw":
                z0_cpu, aux0 = z_cpu, aux_value
            else:
                assert z0_cpu is not None and aux0 is not None
                initial_checks.append({"dataset": dataset, "variant": variant,
                    "z_exact": torch.equal(z0_cpu,z_cpu),"aux_exact": aux0==aux_value,
                    "z_allclose_atol_1e5":torch.allclose(z0_cpu,z_cpu,atol=1e-5,rtol=0),
                    "z_max_abs_diff":float((z0_cpu-z_cpu).abs().max().item()),"aux_max_abs_diff":abs(aux0-aux_value)})
            del model,z,z_cpu,aux
            if "cuda" in device: torch.cuda.empty_cache()

        # Stream raw/support/discrepant hops. Only the normalized Raw and contrast
        # bases needed for expert-profile RMS are retained, on host memory.
        torch.manual_seed(44042)
        model=Model(_model_cfg("R3_adaptive_residual"),info).to(device).eval()
        inputs=model._split_modalities(x)
        priors={m:model.projectors[m](inputs[m]) for m in ("text","visual")}
        src,dst,raw_weight,active,_=model._normalized_operator(edge,x.size(0),x.dtype)
        roles={m:model._role_partition(inputs[m],src,dst,raw_weight) for m in ("text","visual")}
        disagreement=float((roles["text"]["supportive_mask"]^roles["visual"]["supportive_mask"]).float().mean().item())
        for role in roles.values(): role.pop("semantic_features",None)
        del inputs
        alpha=model._effective_alpha(model.alpha_raw,model.eps).detach().cpu()
        gamma=model._normalized_gamma(model.residual_gamma_raw,model.eps).detach().cpu()
        active_cpu=active.detach().cpu()
        for modality in ("text","visual"):
            role=roles[modality]
            mask_s,mask_d=role["supportive_mask"],role["discrepant_mask"]
            role_rows.append({"dataset":dataset,"modality":modality,"physical_edge_count":int(mask_s.numel()),
                "support_edge_fraction":float(mask_s.float().mean().item()),"discrepant_edge_fraction":float(mask_d.float().mean().item()),
                "support_node_coverage":float((role["support_active"]&active).sum().item()/max(int(active.sum()),1)),
                "discrepant_node_coverage":float((role["discrepant_active"]&active).sum().item()/max(int(active.sum()),1)),
                "text_visual_role_disagreement_fraction":disagreement,"partition_weight_max_abs_error":float(role["partition_max_abs_error"].item()),
                "partition_valid":bool(torch.equal(mask_s|mask_d,torch.ones_like(mask_s)) and not bool((mask_s&mask_d).any())
                    and torch.allclose(role["support_norm_weight"]+role["discrepant_norm_weight"],raw_weight,atol=1e-7,rtol=0))})
            previous=priors[modality]
            raw_cpu=[]; contrast_cpu=[]
            for hop in range(4):
                raw_state=model._propagate(previous,src,dst,raw_weight)*active.to(previous.dtype).unsqueeze(-1)
                support=model._propagate(previous,src,dst,role["support_norm_weight"])*active.to(previous.dtype).unsqueeze(-1)
                discrepant=model._propagate(previous,src,dst,role["discrepant_norm_weight"])*active.to(previous.dtype).unsqueeze(-1)
                state_correction=raw_state-(support+discrepant)
                discrepant=discrepant+state_correction
                if bool(active.any()): denom=torch.sqrt(raw_state[active].square().mean(0)+model.eps)
                else: denom=raw_state.new_full((raw_state.size(-1),),math.sqrt(model.eps))
                raw_normed=model._active_rms_norm(raw_state,active)
                support_normed=torch.zeros_like(support); discrepant_normed=torch.zeros_like(discrepant)
                if bool(active.any()):
                    support_normed[active]=support[active]/denom; discrepant_normed[active]=discrepant[active]/denom
                normalized_correction=raw_normed-(support_normed+discrepant_normed)
                discrepant_normed=discrepant_normed+normalized_correction
                contrast=support_normed-discrepant_normed
                state_error=support+discrepant-raw_state
                norm_error=support_normed+discrepant_normed-raw_normed
                row={"dataset":dataset,"modality":modality,"hop":hop+1,"expert_id":"all",
                    "state_partition_max_abs_error":float(state_error.abs().max().item()),"normalized_partition_max_abs_error":float(norm_error.abs().max().item()),
                    "state_roundoff_correction_max_abs":float(state_correction.abs().max().item()),
                    "normalized_roundoff_correction_max_abs":float(normalized_correction.abs().max().item()),
                    "raw_state_rms":_rms(raw_state,active),"support_message_rms":_rms(support,active),"discrepant_message_rms":_rms(discrepant,active),
                    "raw_normalized_rms":_rms(raw_normed,active),"support_normalized_rms":_rms(support_normed,active),"discrepant_normalized_rms":_rms(discrepant_normed,active),
                    "role_contrast_rms":_rms(contrast,active),"role_contrast_to_raw_rms_ratio":_rms(contrast,active)/(_rms(raw_normed,active)+model.eps),
                    "raw_role_contrast_cosine":_cos(raw_normed[active],contrast[active]),
                    "support_discrepant_cosine":_cos(support_normed[active],discrepant_normed[active])}
                rows.append(row)
                raw_cpu.append(raw_normed.detach().cpu()); contrast_cpu.append(contrast.detach().cpu())
                previous=raw_state
                del raw_state,support,discrepant,raw_normed,support_normed,discrepant_normed,contrast,state_error,norm_error,denom,state_correction,normalized_correction
            for expert in range(4):
                raw_profile=sum(float(alpha[expert,k])*raw_cpu[k] for k in range(4))
                q_shared=sum(float(alpha[expert,k])*contrast_cpu[k] for k in range(4))
                q_adaptive=sum(float(gamma[expert,k])*contrast_cpu[k] for k in range(4))
                raw_rms_t=raw_profile[active_cpu].square().mean().add(model.eps).sqrt()
                qs_rms_t=q_shared[active_cpu].square().mean().add(model.eps).sqrt()
                qa_rms_t=q_adaptive[active_cpu].square().mean().add(model.eps).sqrt()
                scale_shared=min(1.0,float((raw_rms_t/(qs_rms_t+model.eps)).item()))
                scale_adaptive=min(1.0,float((raw_rms_t/(qa_rms_t+model.eps)).item()))
                rows.append({"dataset":dataset,"modality":modality,"hop":"profile","expert_id":expert,
                    "state_partition_max_abs_error":0.0,"normalized_partition_max_abs_error":0.0,
                    "raw_profile_rms":float(raw_profile[active_cpu].square().mean().sqrt().item()),
                    "Q_shared_rms":float(q_shared[active_cpu].square().mean().sqrt().item()),
                    "Q_adaptive_rms":float(q_adaptive[active_cpu].square().mean().sqrt().item()),
                    "Q_shared_adaptive_max_abs_diff":float((q_shared-q_adaptive).abs().max().item()),
                    "safety_scale_shared":scale_shared,"safety_scale_adaptive":scale_adaptive,
                    "shared_adaptive_equal":torch.allclose(q_shared,q_adaptive,atol=1e-6,rtol=1e-6)})
                del raw_profile,q_shared,q_adaptive
            del raw_cpu,contrast_cpu
        del model,priors,roles,x,edge,z0_cpu,active_cpu
        if "cuda" in device: torch.cuda.empty_cache()
    errors = [float(r.get(k, 0) or 0) for r in rows for k in ("state_partition_max_abs_error", "normalized_partition_max_abs_error")]
    all_contrast_small = all(float(r.get("role_contrast_rms", 0) or 0) < 1e-8 for r in rows if isinstance(r.get("hop"), int))
    checks = {
        "finite": all(math.isfinite(float(v)) for row in rows+role_rows for v in row.values() if isinstance(v, (float,int))),
        "decomposition_identity_max_abs_error": max(errors, default=0.0),
        "decomposition_identity_valid": max(errors, default=0.0) <= 1e-5,
        "role_partition_valid": all(r["partition_valid"] for r in role_rows),
        "role_contrast_non_degenerate": not all_contrast_small,
        "initial_r0_r3_equality": all(r["z_allclose_atol_1e5"] and r["aux_exact"] for r in initial_checks),
        "initial_variant_equality": all(r["z_allclose_atol_1e5"] and r["aux_exact"] for r in initial_checks),
        "initial_shared_adaptive_profiles_equal": all(r.get("shared_adaptive_equal", False) for r in rows if r.get("hop") == "profile"),
    }
    summary = {"label_free": True, "labels_or_splits_read": False, "datasets": list(DATASETS),
               "decomposition_rows": len(rows), "role_partition_rows": role_rows,
               "initial_variant_checks": initial_checks, "checks": checks,
               "implementation_degeneracy": not all(bool(v) for k,v in checks.items() if k not in {"decomposition_identity_max_abs_error"})}
    summary["passed"] = all(bool(v) for k,v in checks.items() if k != "decomposition_identity_max_abs_error")
    summary["implementation_degeneracy"] = not summary["passed"]
    return rows, summary


def _load_selected(row: dict[str, Any], device: str):
    from src.models.mvcge_mag_v4a import Model

    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(),4))

    run_dir = OUTPUT_ROOT / row.get("mode", "formal") / "runs" / row["dataset"] / f"seed_{row['seed']}" / row["variant"]
    ckpt_path = ROOT / row["checkpoint_path"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(run_dir / "hydra" / ".hydra" / "config.yaml")
    if cfg.task.get("evaluate_test", True) is not False:
        raise AssertionError(f"test evaluation enabled in {run_dir}")
    cfg.model.variant = row["variant"]
    model = Model(cfg, ckpt["data_info"]).to(device).eval()
    model.load_state_dict(ckpt["model_state"], strict=True)
    x_cpu, edge_cpu = load_features_and_edges(row["dataset"], int(row["seed"]))
    x, edge = x_cpu.to(device), edge_cpu.to(device)
    z, _, _, aux, info = model(x, edge, return_details=True)
    test_keys = [k for k in row.get("metrics", {}) if str(k).lower().startswith("test")]
    if test_keys:
        raise AssertionError(f"test metrics present: {test_keys}")
    return model, ckpt, run_dir, x, edge, z, aux, info


@torch.no_grad()
def audit_checkpoint(row: dict[str, Any], device: str = "cuda:0") -> dict[str, Any]:
    model, ckpt, run_dir, x, edge, z, aux, info = _load_selected(row, device)
    dataset, seed, variant = row["dataset"], int(row["seed"]), row["variant"]
    active = info["active_nodes"]
    finite = bool(torch.isfinite(z).all() and torch.isfinite(aux))
    role_partition_valid, decomposition_valid = True, True
    role_rows, decomposition_rows, contrast_rows, novelty_rows = [], [], [], []
    residual_rows, gate_rows, routing_rows, strength_rows, profile_rows, similarity_rows = [], [], [], [], [], []
    alpha = model._effective_alpha(model.alpha_raw, model.eps)
    gamma = model._normalized_gamma(model.residual_gamma_raw, model.eps)
    hadamard = model.residual_gamma_raw.new_tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]]) * .5
    alpha_init = hadamard / (hadamard.norm(dim=-1, keepdim=True) + model.eps)
    gamma_init = hadamard / (hadamard.norm(dim=-1, keepdim=True) + model.eps)
    beta_global, beta_expert = model._residual_betas()

    for modality in ("text", "visual"):
        item = info["details"][modality]
        role = info["role_channels"][modality]
        expected = ["prior", "raw_states", "support_messages", "discrepant_messages", "raw_trajectory",
                    "support_trajectory", "discrepant_trajectory", "role_contrast_basis", "raw_profiles",
                    "raw_expert_outputs", "residual_raw", "residual_safe", "expert_outputs", "route_weights", "strength"]
        finite &= all(bool(torch.isfinite(item[k]).all()) for k in expected)
        mask_s, mask_d = role["supportive_mask"], role["discrepant_mask"]
        weight_error = (role["support_norm_weight"] + role["discrepant_norm_weight"] - item["raw_operator_weight"]).abs().max().item()
        valid = (torch.equal(mask_s | mask_d, torch.ones_like(mask_s)) and not bool((mask_s & mask_d).any()) and weight_error <= 1e-7)
        role_partition_valid &= valid
        role_rows.append({"dataset":dataset,"seed":seed,"variant":variant,"modality":modality,
                          "physical_edge_count":int(mask_s.numel()),"support_edge_fraction":float(mask_s.float().mean().item()),
                          "discrepant_edge_fraction":float(mask_d.float().mean().item()),
                          "support_node_coverage":float((role["support_active"]&active).sum().item()/max(int(active.sum()),1)),
                          "discrepant_node_coverage":float((role["discrepant_active"]&active).sum().item()/max(int(active.sum()),1)),
                          "role_disagreement_fraction":None,"partition_weight_max_abs_error":float(weight_error),"partition_valid":valid})
        for hop in range(4):
            raw, sup, disc = item["raw_states"][hop], item["support_messages"][hop], item["discrepant_messages"][hop]
            rn, sn, dn = item["raw_trajectory"][hop], item["support_trajectory"][hop], item["discrepant_trajectory"][hop]
            state_error, norm_error = sup+disc-raw, sn+dn-rn
            state_max, norm_max = float(state_error.abs().max().item()), float(norm_error.abs().max().item())
            decomposition_valid &= state_max <= 1e-5 and norm_max <= 1e-5
            d = item["role_contrast_basis"][hop]
            decomposition_rows.append({"dataset":dataset,"seed":seed,"variant":variant,"modality":modality,"hop":hop+1,
                "raw_state_rms":_rms(raw,active),"support_message_rms":_rms(sup,active),"disc_message_rms":_rms(disc,active),
                "state_partition_max_abs_error":state_max,"state_partition_relative_rms_error":_rms(state_error)/(_rms(raw)+model.eps),
                "raw_normalized_rms":_rms(rn,active),"support_normalized_rms":_rms(sn,active),"disc_normalized_rms":_rms(dn,active),
                "normalized_partition_max_abs_error":norm_max,"normalized_partition_relative_rms_error":_rms(norm_error)/(_rms(rn)+model.eps),
                "support_disc_cosine":_cos(sn[active],dn[active]),"role_contrast_rms":_rms(d,active),
                "role_contrast_to_raw_rms_ratio":_rms(d,active)/(_rms(rn,active)+model.eps),"raw_role_contrast_cosine":_cos(rn[active],d[active])})
            contrast_rows.append({"dataset":dataset,"seed":seed,"variant":variant,"modality":modality,"hop":hop+1,
                                  "role_contrast_rms":_rms(d,active),"raw_rms":_rms(rn,active),
                                  "role_contrast_to_raw_rms_ratio":_rms(d,active)/(_rms(rn,active)+model.eps),
                                  "raw_role_contrast_cosine":_cos(rn[active],d[active]),
                                  "support_discrepant_cosine":_cos(sn[active],dn[active])})

        raw_basis = item["raw_trajectory"][:,active].float().reshape(4,-1)
        contrast_basis = item["role_contrast_basis"][:,active].float().reshape(4,-1)
        contrast_out, contrast_cond = _span_outside(raw_basis,contrast_basis,model.eps)
        raw_out, raw_cond = _span_outside(contrast_basis,raw_basis,model.eps)
        union = torch.cat([raw_basis,contrast_basis],dim=0)
        union_cond = float(torch.linalg.cond(union@union.T).item())
        novelty_rows.append({"dataset":dataset,"seed":seed,"variant":variant,"modality":modality,
                             "role_contrast_outside_raw_span_ratio":contrast_out,"raw_outside_role_contrast_span_ratio":raw_out,
                             "raw_gram_condition":contrast_cond,"contrast_gram_condition":raw_cond,
                             "combined_union_condition":union_cond if math.isfinite(union_cond) else None})
        route = info["router"][modality]
        top_pair = tuple(sorted(int(v) for v in route["top_indices"][0].tolist()))
        rr={"dataset":dataset,"seed":seed,"variant":variant,"modality":modality,"selected_expert_pair":"-".join(map(str,top_pair)),
            "strength":float(route["strength"][0].item()),"same_pair_text_visual":None,"union_experts_text_visual":None,"dead_slots_text_visual":None}
        for expert in range(4):
            rr[f"top2_weight_expert_{expert}"]=float(route["route_weights"][0,expert].item())
            rr[f"dense_prob_expert_{expert}"]=float(route["dense_probs"][0,expert].item())
        routing_rows.append(rr)
        q_raw, q_safe = item["residual_raw"], item["residual_safe"]
        for expert in range(4):
            if variant == "R1_global_residual":
                raw_gate=float(model.residual_global_raw[0].item()); beta=float(beta_global.item()); scope="global"
            elif variant in {"R2_expert_residual","R3_adaptive_residual"}:
                raw_gate=float(model.residual_expert_raw[expert].item()); beta=float(beta_expert[expert].item()); scope="expert"
            else:
                raw_gate=0.0; beta=0.0; scope="none"
            sign="positive" if beta>1e-8 else ("negative" if beta < -1e-8 else "near_zero")
            gate_rows.append({"dataset":dataset,"seed":seed,"variant":variant,"gate_scope":scope,"expert_id":expert,
                              "raw_gate":raw_gate,"beta":beta,"abs_beta":abs(beta),"sign":sign,
                              "near_zero":abs(beta)<.01,"near_cap":abs(beta)>.9*model.residual_beta_max,"beta_max":model.residual_beta_max})
            rp=item["raw_profiles"][expert]; q=q_raw[expert]; qs=q_safe[expert]; scale=float(item["safety_scale"][expert].item())
            scaled=beta*qs
            pr={"dataset":dataset,"seed":seed,"variant":variant,"modality":modality,"expert_id":expert,
                "raw_profile_rms":_rms(rp,active),"Q_raw_rms_before_safety":_rms(q,active),"safety_scale":scale,
                "Q_safe_rms":_rms(qs,active),"beta":beta,"scaled_residual_rms":_rms(scaled,active),
                "scaled_residual_to_raw_profile_rms":_rms(scaled,active)/(_rms(rp,active)+model.eps),
                "raw_profile_Q_cosine":_cos(rp[active],q[active]),"raw_expert_output_rms":_rms(item["raw_expert_outputs"][expert],active),
                "final_expert_output_rms":_rms(item["expert_outputs"][expert],active),
                "final_minus_raw_expert_rms":_rms(item["expert_outputs"][expert]-item["raw_expert_outputs"][expert],active)}
            if variant=="R3_adaptive_residual":
                gr=model.residual_gamma_raw[expert].detach(); gn=gamma[expert].detach()
                for k in range(4): pr[f"gamma_raw_{k+1}"]=float(gr[k].item()); pr[f"gamma_normalized_{k+1}"]=float(gn[k].item())
                pr.update({"gamma_drift_from_init":float(torch.linalg.vector_norm(gr-hadamard[expert]).item()),
                    "gamma_normalized_drift_from_init":float(torch.linalg.vector_norm(gn-gamma_init[expert]).item()),
                    "gamma_alpha_cosine":_cos(gn,alpha[expert]),
                    "gamma_sign_change_count":int(((gn>0)!=(gamma_init[expert]>0)).sum().item()),
                    "gamma_abs_low_order_mass":float(gn[:2].abs().sum().item()),"gamma_abs_high_order_mass":float(gn[2:].abs().sum().item()),
                    "gamma_abs_order_centroid":float(((torch.arange(1,5,device=gn.device)*gn.abs()).sum()/(gn.abs().sum()+model.eps)).item())})
            residual_rows.append(pr)
        for a,b in EXPERT_PAIRS:
            similarity_rows.append({"dataset":dataset,"seed":seed,"variant":variant,"modality":modality,"kind":"raw_expert_output",
                "expert_a":a,"expert_b":b,"flattened_cosine":_cos(item["raw_expert_outputs"][a][active],item["raw_expert_outputs"][b][active]),
                "mean_node_cosine":float(F.cosine_similarity(item["raw_expert_outputs"][a][active].float(),item["raw_expert_outputs"][b][active].float(),dim=-1,eps=model.eps).mean().item())})
            if variant=="R3_adaptive_residual":
                similarity_rows.append({"dataset":dataset,"seed":seed,"variant":variant,"modality":modality,"kind":"gamma",
                    "expert_a":a,"expert_b":b,"flattened_cosine":_cos(gamma[a],gamma[b]),"mean_node_cosine":None})
        for expert in range(4):
            rec={"dataset":dataset,"seed":seed,"variant":variant,"expert_id":expert,
                 "alpha_drift_from_init":float(torch.linalg.vector_norm(model.alpha_raw[expert].detach()-hadamard[expert]).item()),
                 "alpha_normalized_drift_from_init":float(torch.linalg.vector_norm(alpha[expert]-alpha_init[expert]).item())}
            for k in range(4): rec[f"alpha_raw_{k+1}"]=float(model.alpha_raw[expert,k].item()); rec[f"alpha_normalized_{k+1}"]=float(alpha[expert,k].item())
            profile_rows.append(rec)

        raw_mix=item["raw_mixture"]; residual_mix=item["residual_mixture"]; total_mix=item["mixture"]
        stren=route["strength"].unsqueeze(-1)
        raw_corr=stren*raw_mix; residual_corr=stren*residual_mix; total_corr=stren*total_mix
        strength_rows.append({"dataset":dataset,"seed":seed,"variant":variant,"modality":modality,
            "raw_expert_mixture_rms":_rms(raw_mix,active),"residual_expert_mixture_rms":_rms(residual_mix,active),
            "total_expert_mixture_rms":_rms(total_mix,active),"scaled_raw_correction_rms":_rms(raw_corr,active),
            "scaled_residual_correction_rms":_rms(residual_corr,active),"scaled_total_correction_rms":_rms(total_corr,active),
            "residual_to_raw_mixture_rms_ratio":_rms(residual_mix,active)/(_rms(raw_mix,active)+model.eps),
            "raw_residual_mixture_cosine":_cos(raw_mix[active],residual_mix[active]),
            "prior_total_correction_cosine":_cos(item["prior"][active],total_corr[active]),"strength":float(route["strength"][0].item())})

    # Fixed roles are independently computed by modality; report their disagreement on the same physical edges.
    text_mask=info["role_channels"]["text"]["supportive_mask"]
    visual_mask=info["role_channels"]["visual"]["supportive_mask"]
    disagreement=float((text_mask ^ visual_mask).float().mean().item())
    for r in role_rows: r["role_disagreement_fraction"]=disagreement

    # Static routing summary: compare modality expert pairs without node-router assumptions.
    pair_by={r["modality"]:r["selected_expert_pair"] for r in routing_rows}
    sets={m:set(map(int,pair_by[m].split("-"))) for m in pair_by}
    union=set.union(*sets.values())
    for r in routing_rows:
        r["same_pair_text_visual"]=sets["text"]==sets["visual"]
        r["union_experts_text_visual"]=len(union); r["dead_slots_text_visual"]=4-len(union)
    beta_values=[r["beta"] for r in gate_rows]
    result={"dataset":dataset,"seed":seed,"variant":variant,"finite":finite,"test_metrics_present":False,
            "task_evaluate_test":False,"role_partition_valid":role_partition_valid,"decomposition_valid":decomposition_valid,
            "role_partition_max_abs_error":max((r["partition_weight_max_abs_error"] for r in role_rows),default=0),
            "decomposition_max_abs_error":max((max(r["state_partition_max_abs_error"],r["normalized_partition_max_abs_error"]) for r in decomposition_rows),default=0),
            "static_routing_verified":all(torch.equal(info["router"][m]["selection_logits"],info["router"][m]["selection_logits"][:1].expand_as(info["router"][m]["selection_logits"])) for m in ("text","visual")),
            "top2_verified":all(int((info["router"][m]["route_weights"]>0).sum(-1).min().item())==2 and int((info["router"][m]["route_weights"]>0).sum(-1).max().item())==2 for m in ("text","visual")),
            "beta_min":min(beta_values,default=0),"beta_max":max(beta_values,default=0),"audit_device":device,
            "role_partition_rows":role_rows,"decomposition_rows":decomposition_rows,"role_contrast_rows":contrast_rows,
            "action_space_novelty_rows":novelty_rows,"residual_gate_rows":gate_rows,"residual_profile_rows":residual_rows,
            "routing_rows":routing_rows,"strength_rows":strength_rows,"expert_profile_rows":profile_rows,"expert_similarity_rows":similarity_rows}
    if not finite or not role_partition_valid or not decomposition_valid:
        raise AssertionError(f"selected checkpoint audit failed: {dataset}/{seed}/{variant}")
    del model,ckpt,x,edge,z
    if "cuda" in device: torch.cuda.empty_cache()
    return result


def _paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed={(r["dataset"],int(r["seed"]),r["variant"]):r for r in rows}
    result=[]
    for cand,base in PAIRS:
        pairs=[(keyed[(d,s,cand)],keyed[(d,s,base)]) for d in DATASETS for s in SEEDS]
        da=[100*(float(a["metrics"]["val_acc"])-float(b["metrics"]["val_acc"])) for a,b in pairs]
        df=[100*(float(a["metrics"]["val_macro_f1"])-float(b["metrics"]["val_macro_f1"])) for a,b in pairs]
        rec={"comparison":f"{cand} - {base}","candidate":cand,"baseline":base,
             "overall_accuracy_delta_pp":statistics.fmean(da),"overall_macro_f1_delta_pp":statistics.fmean(df),
             "positive_accuracy_pairs":sum(v>0 for v in da),"positive_macro_f1_pairs":sum(v>0 for v in df),"n_pairs":len(da)}
        ds_means=[]
        for dataset in DATASETS:
            sub=[(a,b) for a,b in pairs if a["dataset"]==dataset]
            xa=[100*(float(a["metrics"]["val_acc"])-float(b["metrics"]["val_acc"])) for a,b in sub]
            xf=[100*(float(a["metrics"]["val_macro_f1"])-float(b["metrics"]["val_macro_f1"])) for a,b in sub]
            rec[f"{dataset}_accuracy_delta_pp"]=statistics.fmean(xa); rec[f"{dataset}_accuracy_positive_seeds"]=sum(v>0 for v in xa)
            rec[f"{dataset}_macro_f1_delta_pp"]=statistics.fmean(xf); rec[f"{dataset}_macro_f1_positive_seeds"]=sum(v>0 for v in xf)
            ds_means.append(statistics.fmean(xa))
        rec["stable_positive_descriptive"]=(rec["overall_accuracy_delta_pp"]>0 and rec["overall_macro_f1_delta_pp"]>0 and rec["positive_accuracy_pairs"]>=6 and sum(v>0 for v in ds_means)>=2)
        rec["approximately_equal_descriptive"]=(abs(rec["overall_accuracy_delta_pp"])<=.15 and abs(rec["overall_macro_f1_delta_pp"])<=.50)
        result.append(rec)
    return result


def _historical_r0(rows: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    from src.models.mvcge_mag_v3c import Model as V3CModel
    old_rows=json.loads((V3C_DATA/"run_rows.json").read_text(encoding="utf-8"))
    results=[]
    for row in rows:
        if row["variant"]!="R0_raw": continue
        old=next(r for r in old_rows if r["dataset"]==row["dataset"] and int(r["seed"])==int(row["seed"]) and r["variant"]=="F0_raw")
        # Same checkpoint: load V4A shared tensors into V3C and compare full-graph forward.
        run_dir=OUTPUT_ROOT/row.get("mode","formal")/"runs"/row["dataset"]/f"seed_{row['seed']}"/row["variant"]
        ckpt=torch.load(ROOT/row["checkpoint_path"],map_location="cpu",weights_only=False)
        cfg=OmegaConf.load(run_dir/"hydra"/".hydra"/"config.yaml")
        v3cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)); v3cfg.model.name="mvcge_mag_v3c"; v3cfg.model.variant="F0_raw"
        v3=V3CModel(v3cfg,ckpt["data_info"]).to(device).eval()
        shared={k:v for k,v in ckpt["model_state"].items() if k not in {"residual_global_raw","residual_expert_raw","residual_gamma_raw"}}
        incompatible=v3.load_state_dict(shared,strict=True)
        x_cpu,e_cpu=load_features_and_edges(row["dataset"],int(row["seed"])); x,e=x_cpu.to(device),e_cpu.to(device)
        cfg.model.name="mvcge_mag_v4a"; cfg.model.variant="R0_raw"
        v4=__import__("src.models.mvcge_mag_v4a",fromlist=["Model"]).Model(cfg,ckpt["data_info"]).to(device).eval()
        v4.load_state_dict(ckpt["model_state"],strict=True)
        z3,_,_,a3,_=v3(x,e); z4,_,_,a4,_=v4(x,e)
        old_acc=float(old["metrics"]["val_acc"]); new_acc=float(row["metrics"]["val_acc"])
        old_f1=float(old["metrics"]["val_macro_f1"]); new_f1=float(row["metrics"]["val_macro_f1"])
        results.append({"dataset":row["dataset"],"seed":int(row["seed"]),"old_val_acc":old_acc,"new_val_acc":new_acc,
            "R0_minus_V3C_F0_accuracy_delta_pp":100*(new_acc-old_acc),"old_val_macro_f1":old_f1,"new_val_macro_f1":new_f1,
            "R0_minus_V3C_F0_macro_f1_delta_pp":100*(new_f1-old_f1),"old_best_epoch":int(old["metadata"]["best_epoch"]),
            "new_best_epoch":int(row["metadata"]["best_epoch"]),"best_epoch_delta":int(row["metadata"]["best_epoch"])-int(old["metadata"]["best_epoch"]),
            "same_checkpoint_v3c_v4a_state_compatible":not incompatible.missing_keys and not incompatible.unexpected_keys,
            "same_checkpoint_forward_exact":torch.equal(z3,z4) and torch.equal(a3,a4),
            "same_checkpoint_forward_allclose_1e7":torch.allclose(z3,z4,atol=1e-7,rtol=0) and torch.allclose(a3,a4,atol=1e-7,rtol=0),
            "same_checkpoint_z_max_abs_diff":float((z3-z4).abs().max().item()),"same_checkpoint_aux_max_abs_diff":float((a3-a4).abs().max().item())})
        del v3,v4,ckpt,x,e
        if "cuda" in device: torch.cuda.empty_cache()
    return results


def _report(rows, paired, audits, historical, preflight_summary, smoke):
    lines=["# MvCGE-MAG V4A: Raw-Anchored Adaptive Role Residual Screen","",
           "## Protocol","",
           "Validation-only full-graph node classification; 3 datasets × 3 seeds × 4 variants (36 runs). Checkpoints are selected by validation Accuracy; Macro-F1 is diagnostic. Test evaluation was disabled.","",
           f"Preflight passed: **{preflight_summary.get('passed')}**. Smoke: **{smoke.get('completed_runs',0)}/4**. Formal rows: **{len(rows)}/36**.","",
           "The fixed V3C role partition is used only to form a Support-minus-Discrepant residual. The complete Raw trajectory and its expert MLP remain the backbone. Reported role contrast is not described as a heterophily or high-pass signal.","",
           "## Validation summary","","| Dataset | Variant | Accuracy mean ± SD | Macro-F1 mean ± SD | Best epoch mean |","|---|---|---:|---:|---:|"]
    for d in (*DATASETS,"ALL"):
        for v in VARIANTS:
            group=[r for r in rows if r["variant"]==v and (d=="ALL" or r["dataset"]==d)]
            if not group: continue
            acc=[100*float(r["metrics"]["val_acc"]) for r in group]; f1=[100*float(r["metrics"]["val_macro_f1"]) for r in group]
            ep=[int(r["metadata"]["best_epoch"]) for r in group]
            lines.append(f"| {d} | {v} | {statistics.fmean(acc):.3f} ± {statistics.pstdev(acc):.3f}% | {statistics.fmean(f1):.3f} ± {statistics.pstdev(f1):.3f}% | {statistics.fmean(ep):.1f} |")
    lines += ["","## Paired descriptive comparisons","","Paired deltas use the same dataset and seed. No significance tests were run. Stable-positive and approximately-equal labels follow the preregistered descriptive cutoffs and are not inferential claims.","","| Comparison | Accuracy Δ (pp) | Macro-F1 Δ (pp) | Acc positive /9 | F1 positive /9 | Stable-positive | Approximately equal |","|---|---:|---:|---:|---:|:---:|:---:|"]
    for r in paired:
        lines.append(f"| {r['comparison']} | {r['overall_accuracy_delta_pp']:+.3f} | {r['overall_macro_f1_delta_pp']:+.3f} | {r['positive_accuracy_pairs']}/9 | {r['positive_macro_f1_pairs']}/9 | {r['stable_positive_descriptive']} | {r['approximately_equal_descriptive']} |")
    lines += ["","## Interpretation map","","Use the paired table together with beta, realized residual magnitude, safety scales, and R3 gamma diagnostics. A stable-positive R1/R2/R3 indicates evidence for the corresponding residual complexity. If all arms are approximately equal and beta stays near zero, the model declined the fixed residual. Nonzero beta with no gain indicates use without validation benefit. R2 vs R3 distinguishes a shared Raw hop profile from an independent residual profile. These outcomes guide a later decision; this report does not launch V4B.","",
              "## Diagnostics and provenance","",
              f"- Preflight max decomposition identity error: {preflight_summary['checks']['decomposition_identity_max_abs_error']:.3g}",
              f"- Selected checkpoint audits: {len(audits)}; finite={all(a['finite'] for a in audits)}; role partition valid={all(a['role_partition_valid'] for a in audits)}.",
              f"- Same-checkpoint V4A R0 / V3C F0 forward allclose pairs: {sum(bool(r['same_checkpoint_forward_allclose_1e7']) for r in historical)}/{len(historical)}.",
              "- Preflight reads frozen modality features and physical edges only; it does not load labels or split indices.",
              "- Formal validation metrics contain no Test metrics; `task.evaluate_test=false` was verified from each saved Hydra configuration.",""]
    return "\n".join(lines)


def analyze(device: str = "cuda:0") -> None:
    rows=json.loads((DATA_ROOT/"run_rows.json").read_text(encoding="utf-8"))
    manifest=json.loads((DATA_ROOT/"campaign_manifest.json").read_text(encoding="utf-8"))
    if len(rows)!=36 or any(str(k).lower().startswith("test") for r in rows for k in r.get("metrics",{})):
        raise RuntimeError("formal rows incomplete or Test metrics found")
    # Detailed full-graph diagnostics materialize several hop tensors; keep
    # these read-only audits on host memory so checkpoint selection/training
    # GPU state and device capacity stay outside the audit path.
    audit_device="cpu"
    audits=[audit_checkpoint(row,audit_device) for row in rows]
    gather={key:[r for a in audits for r in a[key]] for key in (
        "role_partition_rows","decomposition_rows","role_contrast_rows","action_space_novelty_rows",
        "residual_gate_rows","residual_profile_rows","routing_rows","strength_rows","expert_profile_rows","expert_similarity_rows")}
    historical=_historical_r0(rows,audit_device)
    paired=_paired(rows)
    # Compact per-run metric table, including the required validation-selected metadata.
    summary=[]
    for dataset in (*DATASETS,"ALL"):
        for variant in VARIANTS:
            group=[r for r in rows if r["variant"]==variant and (dataset=="ALL" or r["dataset"]==dataset)]
            if group:
                for metric in ("val_acc","val_macro_f1"):
                    vals=[float(r["metrics"][metric]) for r in group]
                    summary.append({"dataset":dataset,"variant":variant,"metric":metric,"mean":statistics.fmean(vals),"population_std":statistics.pstdev(vals),"n":len(vals)})
                eps=[float(r["metadata"]["best_epoch"]) for r in group]
                summary.append({"dataset":dataset,"variant":variant,"metric":"best_epoch","mean":statistics.fmean(eps),"population_std":statistics.pstdev(eps),"n":len(eps)})
    pre= json.loads((DATA_ROOT/"preflight_summary.json").read_text(encoding="utf-8"))
    smoke=json.loads((DATA_ROOT/"smoke_summary.json").read_text(encoding="utf-8"))
    tables={"summary.csv":summary,"paired_comparisons.csv":paired,
        "role_partition_diagnostics.csv":gather["role_partition_rows"],"decomposition_diagnostics.csv":gather["decomposition_rows"],
        "role_contrast_diagnostics.csv":gather["role_contrast_rows"],"action_space_novelty.csv":gather["action_space_novelty_rows"],
        "residual_gate_diagnostics.csv":gather["residual_gate_rows"],"residual_profile_diagnostics.csv":gather["residual_profile_rows"],
        "routing_diagnostics.csv":gather["routing_rows"],"strength_diagnostics.csv":gather["strength_rows"],
        "expert_profiles.csv":gather["expert_profile_rows"],"expert_similarity.csv":gather["expert_similarity_rows"],
        "historical_r0_regression.csv":historical}
    for filename,table in tables.items(): write_csv(DATA_ROOT/filename,table)
    (RESEARCH_ROOT/"REPORT.md").write_text(_report(rows,paired,audits,historical,pre,smoke),encoding="utf-8")
    manifest.update({"task_evaluate_test":False,"test_evaluation_verified_false":True,"test_metrics_absent":True,
        "selected_checkpoint_audits":len(audits),"all_selected_checkpoints_finite":all(a["finite"] for a in audits),
        "all_role_partitions_valid":all(a["role_partition_valid"] for a in audits),
        "all_decomposition_identities_valid":all(a["decomposition_valid"] for a in audits),
        "same_checkpoint_r0_compatibility_pairs":sum(bool(r["same_checkpoint_forward_allclose_1e7"]) for r in historical),
        "analysis_device":audit_device,"campaign_device":device,"artifacts_complete":True,"analyzed_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds")})
    write_json(DATA_ROOT/"campaign_manifest.json",manifest)
    print(f"[V4A analysis] audits={len(audits)}/36; wrote required diagnostics and REPORT.md",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--device",default="cuda:0"); parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    if args.preflight:
        drows,summary=preflight(args.device); write_csv(DATA_ROOT/"preflight_decomposition_diagnostics.csv",drows)
        write_csv(DATA_ROOT/"role_partition_preflight.csv",summary.pop("role_partition_rows"))
        write_json(DATA_ROOT/"preflight_summary.json",summary)
        print(f"[V4A preflight] passed={summary['passed']} rows={len(drows)} checks={summary['checks']}",flush=True)
        if not summary["passed"]: raise SystemExit(2)
    else: analyze(args.device)
