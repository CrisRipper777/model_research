from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.p11_p12_operator_rescue import (  # noqa: E402
    DATASETS,
    run_correctness_smoke,
    run_one,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-operator rescue probes on a frozen P0 semantic substrate")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data-root", default="/hdd1/DataInHere/YHF/data")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p11_p12_operator_rescue"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--smoke", action="store_true", help="Movies/42 correctness smoke, 3 epochs per head")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    print(f"[campaign] device={device} cuda_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}", flush=True)

    if args.smoke:
        summary = run_correctness_smoke(args.output_dir, args.data_root, device)
        print(json.dumps(summary, indent=2), flush=True)
        return

    smoke_path = args.output_dir / "smoke" / "Movies" / "seed_42" / "smoke_result.json"
    if not smoke_path.is_file():
        raise RuntimeError("run --smoke first; full campaign is gated on the Movies/42 correctness smoke")
    smoke_record = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke_record.get("status") != "smoke_passed" or not smoke_record.get("frozen_encoder_unchanged"):
        raise RuntimeError(f"Movies/42 correctness smoke did not pass: {smoke_path}")

    all_results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"[run] {dataset} seed={seed}", flush=True)
            result = run_one(dataset, seed, args.output_dir, args.data_root, device, max_epochs=args.max_epochs)
            all_results.append({"dataset": dataset, "seed": seed, "status": result["status"], "messages": result["validation_target_messages"]})
            print(f"[done] {dataset} seed={seed} messages={result['validation_target_messages']} perf={result['head_performance']}", flush=True)
    write_json(args.output_dir / "campaign_result.json", {
        "datasets": args.datasets,
        "seeds": args.seeds,
        "device": str(device),
        "results": all_results,
        "test_evaluation": False,
        "link_prediction": False,
    })
    print(f"[complete] runs={len(all_results)} output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
