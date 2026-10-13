#!/usr/bin/env python3
"""Real-data V2.2 paired-init, M0 compatibility, and RNG preflight."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rmcge_v21 import module_hash  # noqa: E402
from scripts.run_rmcge_v22 import (  # noqa: E402
    CONDITIONS,
    DATASETS,
    DATA_DIR,
    SEEDS,
    build_config,
    environment_record,
    mask_test_labels,
    verify_cache_entries,
)
from src.data import load_mag_data  # noqa: E402
from src.models.rmcge_v21 import Model as V21Model  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("authorized CUDA preflight must run in yhf_env with physical GPU 1 visible")
    caches = verify_cache_entries()
    device = torch.device("cuda:0")
    records = []
    for dataset in DATASETS:
        for seed in SEEDS:
            cfg_data = build_config(dataset, "M0", seed, str(device), caches[dataset])
            raw = load_mag_data(cfg_data, "nc", seed)
            safe = mask_test_labels(raw)
            info = {
                "input_dim": raw.input_dim,
                "num_nodes": raw.num_nodes,
                "num_classes": raw.num_classes,
                "text_dim": int(raw.x_t.size(1)),
                "visual_dim": int(raw.x_i.size(1)),
            }
            models, heads, model_hashes, head_hashes = {}, {}, {}, {}
            for condition in CONDITIONS:
                cfg = build_config(dataset, condition, seed, str(device), caches[dataset])
                set_seed(seed)
                model = build_model(cfg, info)
                model.load_semantic_cache(
                    caches[dataset]["cache_path_abs"], expected_dataset=dataset,
                    expected_feature_fingerprint=caches[dataset]["feature_fingerprint"],
                    expected_candidate_fingerprint=caches[dataset]["candidate_fingerprint"],
                    expected_v7b_fingerprint=caches[dataset]["v7b_fingerprint"],
                )
                head = torch.nn.Linear(model.out_dim, raw.num_classes)
                model_hashes[condition] = module_hash(model)
                head_hashes[condition] = module_hash(head)
                models[condition] = model.to(device).eval()
                heads[condition] = head.to(device).eval()

            cfg_base = copy.deepcopy(build_config(dataset, "C0", seed, str(device), caches[dataset]))
            cfg_base.model.name = "rmcge_v21"
            cfg_base.model.condition = "C0"
            set_seed(seed)
            source = V21Model(cfg_base, info)
            source.load_semantic_cache(
                caches[dataset]["cache_path_abs"], expected_dataset=dataset,
                expected_feature_fingerprint=caches[dataset]["feature_fingerprint"],
                expected_candidate_fingerprint=caches[dataset]["candidate_fingerprint"],
                expected_v7b_fingerprint=caches[dataset]["v7b_fingerprint"],
            )
            source_head = torch.nn.Linear(source.out_dim, raw.num_classes)
            source_model_hash = module_hash(source)
            source_head_hash = module_hash(source_head)
            source = source.to(device).eval()
            source_head = source_head.to(device).eval()

            if len(set(model_hashes.values())) != 1 or len(set(head_hashes.values())) != 1:
                raise RuntimeError(f"V2.2 arms do not share identical initialization for {dataset}/seed_{seed}")
            common_keys = set(source.state_dict())
            for key in common_keys:
                left = models["M0"].state_dict()[key]
                right = source.state_dict()[key]
                if not torch.equal(left.detach().cpu(), right.detach().cpu()):
                    raise RuntimeError(f"M0/V2.1 C0 initial tensor differs at {dataset}/seed_{seed}/{key}")
            if head_hashes["M0"] != source_head_hash:
                raise RuntimeError(f"M0/V2.1 C0 classifier initialization differs for {dataset}/seed_{seed}")

            x, edge = safe.x.to(device), safe.edge_index.to(device)
            cpu_rng = torch.random.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state(device).clone()
            with torch.no_grad():
                v22_logits = models["M0"](x, edge)[0]
            unchanged_after_v22 = torch.equal(cpu_rng, torch.random.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
            cpu_rng = torch.random.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state(device).clone()
            with torch.no_grad():
                v21_logits = source(x, edge)[0]
            unchanged_after_v21 = torch.equal(cpu_rng, torch.random.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
            difference = (v22_logits.float() - v21_logits.float()).abs()
            max_abs = float(difference.max().cpu()) if difference.numel() else 0.0
            if max_abs > 5.0e-5:
                raise RuntimeError(f"real-data M0/C0 forward mismatch {dataset}/seed_{seed}: {max_abs}")
            record = {
                "dataset": dataset, "seed": seed,
                "v22_arm_model_hashes": model_hashes,
                "v22_arm_classifier_hashes": head_hashes,
                "v21_c0_model_state_hash": source_model_hash,
                "v21_c0_classifier_state_hash": source_head_hash,
                "initial_common_tensors_exact": True,
                "initial_classifier_exact": True,
                "m0_vs_v21_c0_forward_max_abs": max_abs,
                "initial_forward_atol": 5.0e-5,
                "m0_forward_rng_unchanged": unchanged_after_v22,
                "v21_c0_forward_rng_unchanged": unchanged_after_v21,
                "test_evaluation": False,
                "test_label_values_indexed": False,
            }
            records.append(record)
            print(f"INIT_OK {dataset}/seed_{seed}: max_abs={max_abs:.3e}", flush=True)
            del models, heads, source, source_head, safe, raw, x, edge, v22_logits, v21_logits
            torch.cuda.empty_cache()

    path = DATA_DIR / "initialization_audit.json"
    path.write_text(json.dumps({"environment": environment_record(), "records": records, "all_passed": True}, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    checks_path = DATA_DIR / "implementation_checks.json"
    previous = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    previous.update({"initialization_audit": str(path.relative_to(ROOT)), "real_dataset_seed_pairs": records, "formal_runs": previous.get("formal_runs", [])})
    checks_path.write_text(json.dumps(previous, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
