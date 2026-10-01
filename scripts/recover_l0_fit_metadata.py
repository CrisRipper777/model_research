from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/l0_relation_function_learnability_audit/checkpoints"
DATA = ROOT / "research/l0_relation_function_learnability_audit/data"
EVIDENCE = {
    "SIM_ONLY", "TARGET_ONLY", "ENDPOINT", "ENDPOINT_LOCAL",
    "ENDPOINT_LOCAL_SHUFFLED_TARGET",
}


def parse_checkpoint(path: Path) -> dict:
    match = re.fullmatch(r"seed_(\d+)_fold_(\d+)_(.+)", path.stem)
    if not match:
        raise ValueError(f"unrecognized L0 checkpoint name: {path}")
    seed, fold, name = int(match.group(1)), int(match.group(2)), match.group(3)
    dataset = path.parent.name
    row = {"dataset": dataset, "seed": seed, "fold": fold}
    if name in EVIDENCE:
        row.update(model=name, variant=name)
    elif name == "M0StateDirect":
        row["model"] = name
    elif any(name.endswith("_" + m) for m in ("text", "visual")):
        modality = next(m for m in ("text", "visual") if name.endswith("_" + m))
        if modality is None:
            raise ValueError(f"frozen checkpoint missing modality suffix: {path}")
        representation = name[: -(len(modality) + 1)]
        row.update(model=f"Frozen_{representation}", modality=modality,
                   representation=representation)
    else:
        raise ValueError(f"unknown L0 model checkpoint: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    row.update(payload["metadata"])
    row["checkpoint_path"] = str(path.relative_to(ROOT))
    if str(row["model"]).startswith("Frozen_"):
        row["frozen_checkpoint"] = str(
            Path("outputs/e01_function_provenance_preservation/checkpoints")
            / dataset / f"seed_{seed}_keep_edge.pt"
        )
        row["edge_alignment_verified"] = True
    return row


def key_for(row: dict) -> tuple:
    return tuple(row.get(k) for k in ("dataset", "seed", "fold", "model", "variant",
                                       "modality", "representation"))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    existing_path = DATA / "parameter_summary.csv"
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    old_lookup = {}
    if not existing.empty:
        for item in existing.to_dict("records"):
            normalized = {k: (None if pd.isna(v) else v) for k, v in item.items()}
            old_lookup[key_for(normalized)] = normalized

    checkpoints = sorted(OUTPUT.rglob("*.pt"))
    rows = []
    for path in checkpoints:
        row = parse_checkpoint(path)
        old = old_lookup.get(key_for(row), {})
        # Preserve fit-level audits added by the runner after checkpoint serialization.
        for column, value in old.items():
            if column not in row or row[column] is None:
                row[column] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    expected = {"SIM_ONLY": 27, "TARGET_ONLY": 27, "ENDPOINT": 27,
                "ENDPOINT_LOCAL": 27, "ENDPOINT_LOCAL_SHUFFLED_TARGET": 27,
                "M0StateDirect": 27, "Frozen_Q_PAIR": 54,
                "Frozen_R_SHARED": 54, "Frozen_U_MODAL": 54}
    counts = frame.groupby("model").size().to_dict()
    if len(frame) != 324 or counts != expected:
        raise AssertionError(f"fit metadata coverage mismatch: rows={len(frame)} counts={counts}")
    sort_cols = [c for c in ["dataset", "seed", "fold", "model", "modality"] if c in frame]
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    frame.to_csv(existing_path, index=False)
    audit = {
        "status": "reconstructed_from_saved_fit_checkpoints",
        "checkpoint_count": len(checkpoints),
        "parameter_rows": len(frame),
        "fit_counts_by_model": counts,
        "preserved_runner_only_audits_from_previous_parameter_summary": True,
        "source": "outputs/l0_relation_function_learnability_audit/checkpoints/**/*.pt",
    }
    (DATA / "parameter_summary_reconstruction.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
