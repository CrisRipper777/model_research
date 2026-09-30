from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.p13_joint_readout_operator_probe import (  # noqa: E402
    DATASETS,
    run_correctness_smoke,
    run_one,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="P1.3 common joint-readout operator counterfactual probe")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data-root", default="/hdd1/DataInHere/YHF/data")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p13_joint_readout_operator_probe"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--smoke", action="store_true", help="Movies/42 correctness smoke, 3 epochs")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    print(f"[campaign] device={device} cuda_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}", flush=True)
    if args.smoke:
        result = run_correctness_smoke(args.output_dir, args.data_root, device)
        print(json.dumps(result, indent=2), flush=True)
        return

    smoke_path = args.output_dir / "smoke" / "Movies" / "seed_42" / "smoke_result.json"
    if not smoke_path.is_file():
        raise RuntimeError("run --smoke first; full campaign is gated on Movies/42 correctness")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "smoke_passed" or not smoke.get("frozen_encoder_unchanged"):
        raise RuntimeError(f"Movies/42 smoke did not pass: {smoke_path}")

    results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            result_path = args.output_dir / "runs" / dataset / f"seed_{seed}" / "run_result.json"
            if result_path.is_file():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if existing.get("status") == "completed" and existing.get("fast_vs_brute_logit_max_abs") and existing.get("edge_alignment", {}).get("edge_set_match"):
                    print(f"[skip-complete] {dataset} seed={seed}", flush=True)
                    results.append(existing)
                    continue
            print(f"[run] {dataset} seed={seed}", flush=True)
            result = run_one(dataset, seed, args.output_dir, args.data_root, device, max_epochs=args.max_epochs)
            results.append(result)
            print(f"[done] {dataset} seed={seed} edges={result['validation_target_messages']} perf={result['head_performance']}", flush=True)
    payload = {
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "device": str(device),
        "completed_dataset_seed_runs": sum(r.get("status") == "completed" for r in results),
        "results": [{"dataset": r["dataset"], "seed": r["seed"], "status": r["status"], "validation_target_messages": r["validation_target_messages"]} for r in results],
        "test_evaluation": False,
        "link_prediction": False,
    }
    write_json(args.output_dir / "campaign_result.json", payload)
    print(f"[complete] runs={payload['completed_dataset_seed_runs']} output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
