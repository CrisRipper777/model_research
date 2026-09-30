from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

from src.data.graph_utils import ensure_edge_index, preprocess_edge_index
from src.data.loaders import (
    PROJECT_ROOT,
    _load_dgl_graph,
    _load_numpy_feature,
)


DATASETS = ("Movies", "Grocery", "ele-fashion")
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 30
MIN_EPOCH = 30
EDGE_BATCH_SIZE = 65536


@dataclass
class ProbeData:
    """NC data visible to this analysis; deliberately has no test-index field."""

    name: str
    source: str
    x_t: torch.Tensor
    x_v: torch.Tensor
    edge_index: torch.Tensor
    labels: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    num_nodes: int
    num_classes: int
    paths: dict[str, str]


@dataclass
class ProbeFit:
    network: "NeutralOneHopProbe"
    z: torch.Tensor
    h_t: torch.Tensor
    h_v: torch.Tensor
    c_t: torch.Tensor | None
    c_v: torch.Tensor | None
    logits: torch.Tensor
    metrics: dict[str, float | int]


class TestIndexForbiddenSplit(Mapping[str, Any]):
    """Expose only requested split fields and fail if test indices are touched."""

    def __init__(self, payload: Mapping[str, Any]):
        self._payload = payload

    def __getitem__(self, key: str) -> Any:
        if key in {"test_idx", "test_index", "test"}:
            raise AssertionError("P0/P1 is validation-only; test split access is forbidden")
        return self._payload[key]

    def __iter__(self):
        # Do not expose the payload's key list, which could reveal held-out fields.
        return iter(("train_idx", "val_idx"))

    def __len__(self) -> int:
        return 2


def read_train_val_indices(payload: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    split = TestIndexForbiddenSplit(payload)
    train = torch.as_tensor(split["train_idx"], dtype=torch.long).reshape(-1).cpu()
    val = torch.as_tensor(split["val_idx"], dtype=torch.long).reshape(-1).cpu()
    if train.numel() == 0 or val.numel() == 0:
        raise ValueError("NC train and validation splits must both be nonempty")
    if torch.isin(train, val).any():
        raise ValueError("NC train and validation splits overlap")
    return train.contiguous(), val.contiguous()


def _path(value: str, data_root: str, dataset_root: str, seed: int) -> Path:
    resolved = str(value)
    resolved = resolved.replace("${paths.data_root}", data_root)
    resolved = resolved.replace("${paths.split_root}", str(Path(data_root).expanduser().resolve() / "MAGB_split"))
    resolved = resolved.replace("${dataset.root}", dataset_root)
    resolved = resolved.replace("${seed}", str(seed))
    result = Path(resolved).expanduser()
    if not result.is_absolute():
        result = PROJECT_ROOT / result
    return result.resolve()


def _read_split(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required frozen NC split is missing: {path}. This analysis will not generate or modify splits."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return read_train_val_indices(payload)


def _mask_unobserved_labels(
    raw_labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Copy only train/validation labels into the labels visible to the probe."""
    raw_labels = torch.as_tensor(raw_labels, dtype=torch.long).reshape(-1).cpu()
    visible = torch.full_like(raw_labels, -100)
    train_labels = raw_labels[train_idx]
    val_labels = raw_labels[val_idx]
    for name, values in (("train", train_labels), ("validation", val_labels)):
        if not ((values >= 0) & (values < num_classes)).all():
            raise ValueError(f"{name} split contains a label outside [0, {num_classes})")
    visible[train_idx] = train_labels
    visible[val_idx] = val_labels
    return visible


def load_probe_data(
    dataset: str,
    seed: int,
    data_root: str = "/hdd1/DataInHere/YHF/data",
) -> ProbeData:
    """Load the configured NC data without using the framework NC runner."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    ds_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "dataset" / f"{dataset}.yaml")
    ds = OmegaConf.to_container(ds_cfg, resolve=False)
    dataset_root = str(Path(data_root).expanduser().resolve() / dataset)
    num_classes = int(ds["num_classes"])
    if ds["source"] == "magb":
        graph_path = _path(ds["graph_path"], data_root, dataset_root, seed)
        text_path = _path(ds["text_feat_path"], data_root, dataset_root, seed)
        visual_path = _path(ds["image_feat_path"], data_root, dataset_root, seed)
        split_path = _path(ds["nc_split_path"], data_root, dataset_root, seed)
        raw_edges, raw_labels, num_nodes = _load_dgl_graph(graph_path)
        x_t = _load_numpy_feature(text_path, ds.get("feature_dtype", "float32"))
        x_v = _load_numpy_feature(visual_path, ds.get("feature_dtype", "float32"))
        if x_t.size(0) != num_nodes or x_v.size(0) != num_nodes:
            raise ValueError(f"{dataset}: feature rows do not match graph node count")
        train_idx, val_idx = _read_split(split_path)
        edge_index = preprocess_edge_index(
            raw_edges,
            num_nodes,
            make_undirected=bool(ds.get("make_undirected", True)),
            with_self_loops=False,
        )
        paths = {
            "graph": str(graph_path),
            "text_features": str(text_path),
            "visual_features": str(visual_path),
            "nc_split": str(split_path),
        }
    elif ds["source"] == "mmgraph":
        root = _path(ds["root"], data_root, dataset_root, seed)
        joint_path = _path(ds["joint_feat_path"], data_root, dataset_root, seed)
        edges_path = _path(ds["edge_path"], data_root, dataset_root, seed)
        labels_path = _path(ds["label_path"], data_root, dataset_root, seed)
        split_path = _path(ds["node_split_path"], data_root, dataset_root, seed)
        joint = torch.load(joint_path, map_location="cpu", weights_only=False).float().contiguous()
        x_t, x_v = slice_text_visual(
            joint,
            int(ds["text_dim"]) if ds.get("text_dim") else 0,
            int(ds["visual_dim"]) if ds.get("visual_dim") else 0,
        )
        raw_edges = torch.as_tensor(
            torch.load(edges_path, map_location="cpu", weights_only=False), dtype=torch.long
        )
        raw_labels = torch.as_tensor(
            torch.load(labels_path, map_location="cpu", weights_only=False), dtype=torch.long
        )
        num_nodes = int(joint.size(0))
        if raw_labels.numel() != num_nodes:
            raise ValueError(f"{dataset}: label rows do not match feature rows")
        train_idx, val_idx = _read_split(split_path)
        edge_index = preprocess_edge_index(
            ensure_edge_index(raw_edges),
            num_nodes,
            make_undirected=bool(ds.get("make_undirected", True)),
            with_self_loops=False,
        )
        paths = {
            "root": str(root),
            "joint_features": str(joint_path),
            "edges": str(edges_path),
            "labels": str(labels_path),
            "nc_split": str(split_path),
        }
    else:
        raise ValueError(f"Unsupported configured data source: {ds['source']}")

    if int(max(train_idx.max(), val_idx.max())) >= num_nodes:
        raise ValueError(f"{dataset}: split contains a node outside the graph")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError(f"{dataset}: graph edge contains an invalid node ID")
    if edge_index.numel() and (edge_index[0] == edge_index[1]).any():
        raise AssertionError("Physical graph must have no self-loops for this probe")
    labels = _mask_unobserved_labels(raw_labels, train_idx, val_idx, num_classes)
    return ProbeData(
        name=dataset,
        source=str(ds["source"]),
        x_t=x_t.float().contiguous(),
        x_v=x_v.float().contiguous(),
        edge_index=edge_index.long().contiguous(),
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        num_nodes=num_nodes,
        num_classes=num_classes,
        paths=paths,
    )


def slice_text_visual(x: torch.Tensor, text_dim: int, visual_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [Text, Visual] slices from the configured joint feature layout."""
    if text_dim <= 0 or visual_dim <= 0 or x.size(-1) < text_dim + visual_dim:
        raise ValueError("joint feature tensor is too small for the requested modality slices")
    return x[..., :text_dim].contiguous(), x[..., text_dim : text_dim + visual_dim].contiguous()


def neighbor_mean(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform incoming one-hop means, with self-loops excluded."""
    src, dst = edge_index
    if src.numel() and (src == dst).any():
        raise ValueError("neighbor_mean requires a graph with self-loops removed")
    degree = torch.bincount(dst, minlength=num_nodes)
    total = torch.zeros_like(h)
    if src.numel():
        total.index_add_(0, dst, h[src])
    context = total / degree.clamp_min(1).to(h.dtype).unsqueeze(-1)
    return context, degree


class NeutralOneHopProbe(nn.Module):
    """Independent modality projections, fixed uniform context, and a linear head."""

    def __init__(self, text_dim: int, visual_dim: int, num_classes: int, hidden_dim: int, contextual: bool):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.contextual = bool(contextual)
        self.proj_t = nn.Linear(text_dim, hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.proj_v = nn.Linear(visual_dim, hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim * (4 if contextual else 2), num_classes)

    def encode(
        self,
        x_t: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        h_t = self.norm_t(self.proj_t(x_t))
        h_v = self.norm_v(self.proj_v(x_v))
        if self.contextual:
            c_t, _ = neighbor_mean(h_t, edge_index, num_nodes)
            c_v, _ = neighbor_mean(h_v, edge_index, num_nodes)
            z = torch.cat([h_t, h_v, c_t, c_v], dim=-1)
            return h_t, h_v, c_t, c_v, z
        z = torch.cat([h_t, h_v], dim=-1)
        return h_t, h_v, None, None, z

    def forward(self, x_t: torch.Tensor, x_v: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
        encoded = self.encode(x_t, x_v, edge_index, num_nodes)
        return self.head(encoded[-1]), encoded


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    pred = logits.argmax(-1)
    return {
        "val_acc": float((pred == labels).float().mean().item()),
        "val_macro_f1": float(
            f1_score(
                labels.detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "val_ce": float(F.cross_entropy(logits, labels).item()),
    }


def train_probe(
    data: ProbeData,
    device: torch.device,
    seed: int,
    contextual: bool,
    hidden_dim: int = HIDDEN_DIM,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_epoch: int = MIN_EPOCH,
    checkpoint_path: Path | None = None,
) -> ProbeFit:
    set_all_seeds(seed)
    network = NeutralOneHopProbe(
        data.x_t.size(1), data.x_v.size(1), data.num_classes, hidden_dim, contextual
    ).to(device)
    x_t = data.x_t.to(device)
    x_v = data.x_v.to(device)
    edge_index = data.edge_index.to(device)
    labels = data.labels.to(device)
    train_idx = data.train_idx.to(device)
    val_idx = data.val_idx.to(device)
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float | int] = {}
    best_epoch = 0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = network(x_t, x_v, edge_index, data.num_nodes)
        loss = F.cross_entropy(logits[train_idx], train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()

        network.eval()
        with torch.no_grad():
            val_logits, _ = network(x_t, x_v, edge_index, data.num_nodes)
            current = _classification_metrics(val_logits[val_idx], val_labels, data.num_classes)
        if current["val_acc"] > best_acc:
            best_acc = current["val_acc"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            best_metrics = {**current, "best_epoch": epoch, "contextual": int(contextual)}
            wait = 0
        elif epoch >= min_epoch:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("Probe training produced no validation-selected checkpoint")
    network.load_state_dict(best_state)
    network.to(device).eval()
    with torch.no_grad():
        h_t, h_v, c_t, c_v, z = network.encode(x_t, x_v, edge_index, data.num_nodes)
        logits = network.head(z)
        val_metrics = _classification_metrics(logits[val_idx], val_labels, data.num_classes)
    best_metrics = {**best_metrics, **val_metrics, "best_epoch": best_epoch}
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dataset": data.name,
                "seed": seed,
                "contextual": contextual,
                "best_epoch": best_epoch,
                "selection": "validation_accuracy",
                "metrics": best_metrics,
                "model_state": best_state,
                "hidden_dim": hidden_dim,
            },
            checkpoint_path,
        )
    return ProbeFit(network, z, h_t, h_v, c_t, c_v, logits, best_metrics)


def relative_ce_utility(ce_full: torch.Tensor, ce_removed: torch.Tensor) -> torch.Tensor:
    """Signed relative CE change; positive helps, negative harms when removed."""
    return (ce_removed - ce_full) / ce_full.clamp_min(1e-12)


def margin_utility(logits_full: torch.Tensor, logits_removed: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    def margin(logits: torch.Tensor) -> torch.Tensor:
        true = logits.gather(1, labels.reshape(-1, 1)).squeeze(1)
        other = logits.clone()
        other.scatter_(1, labels.reshape(-1, 1), -torch.inf)
        return true - other.max(dim=1).values

    return margin(logits_full) - margin(logits_removed)


def _removed_logits_fast(
    network: NeutralOneHopProbe,
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    logits_full_by_node: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    modality: str,
) -> torch.Tensor:
    h = h_t if modality == "text" else h_v
    start = 2 * network.hidden_dim if modality == "text" else 3 * network.hidden_dim
    weight = network.head.weight[:, start : start + network.hidden_dim].double()
    effect = (h[src].double() / degree[dst].double().unsqueeze(-1)) @ weight.T
    return logits_full_by_node[dst] - effect


def _removed_logits_bruteforce(
    network: NeutralOneHopProbe,
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    c_t: torch.Tensor,
    c_v: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    modality: str,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for source, target in zip(src.tolist(), dst.tolist()):
        row_t = c_t[target].double().clone()
        row_v = c_v[target].double().clone()
        # Fixed original denominator: no degree decrement or re-normalization.
        if modality == "text":
            row_t = row_t - h_t[source].double() / degree[target].double()
        else:
            row_v = row_v - h_v[source].double() / degree[target].double()
        z = torch.cat([h_t[target].double(), h_v[target].double(), row_t, row_v]).unsqueeze(0)
        rows.append(F.linear(z, network.head.weight.double(), network.head.bias.double()).squeeze(0))
    return torch.stack(rows) if rows else h_t.new_empty((0, network.head.out_features))


@torch.no_grad()
def validate_fast_utility(
    network: NeutralOneHopProbe,
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    c_t: torch.Tensor,
    c_v: torch.Tensor,
    labels: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
) -> dict[str, float]:
    if src.numel() == 0:
        raise ValueError("Correctness validation needs at least one directed message")
    z_full = torch.cat([h_t, h_v, c_t, c_v], dim=-1).double()
    logits_precise = F.linear(z_full, network.head.weight.double(), network.head.bias.double())
    logits_edge = logits_precise[dst]
    y = labels[dst]
    errors = {}
    for modality in ("text", "visual"):
        fast = _removed_logits_fast(network, h_t, h_v, logits_precise, src, dst, degree, modality)
        brute = _removed_logits_bruteforce(network, h_t, h_v, c_t, c_v, src, dst, degree, modality)
        if not torch.allclose(fast, brute, rtol=1e-5, atol=1e-6):
            raise AssertionError(f"{modality}: fast and brute-force removed logits differ")
        ce_full = F.cross_entropy(logits_edge, y, reduction="none")
        fast_u = relative_ce_utility(ce_full, F.cross_entropy(fast, y, reduction="none"))
        brute_u = relative_ce_utility(ce_full, F.cross_entropy(brute, y, reduction="none"))
        # Both paths use the same float64 affine evaluation after float32 projection.
        if not torch.allclose(fast_u, brute_u, rtol=1e-8, atol=1e-9):
            raise AssertionError(f"{modality}: fast and brute-force utility differ")
        errors[f"{modality}_logit_max_abs"] = float((fast - brute).abs().max().item())
        errors[f"{modality}_utility_max_abs"] = float((fast_u - brute_u).abs().max().item())
    return errors


@torch.no_grad()
def build_edge_table(
    data: ProbeData,
    fit: ProbeFit,
    device: torch.device,
    batch_size: int = EDGE_BATCH_SIZE,
    correctness_check: bool = False,
) -> Any:
    """Return complete directed validation-target message evidence as a DataFrame."""
    import pandas as pd

    src_all, dst_all = data.edge_index.to(device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask[data.val_idx.to(device)] = True
    keep = val_mask[dst_all]
    src_all, dst_all = src_all[keep], dst_all[keep]
    degree = torch.bincount(data.edge_index[1], minlength=data.num_nodes).to(device)
    if src_all.numel() == 0:
        raise ValueError(f"{data.name}: no incoming edges to validation targets")
    assert fit.c_t is not None and fit.c_v is not None
    labels = data.labels.to(device)

    first = min(src_all.numel(), 24)
    if correctness_check:
        validate_fast_utility(
            fit.network,
            fit.h_t,
            fit.h_v,
            fit.c_t,
            fit.c_v,
            labels,
            src_all[:first],
            dst_all[:first],
            degree,
        )

    val_row = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
    val_row[data.val_idx.to(device)] = torch.arange(data.val_idx.numel(), device=device)
    classifier_weight = fit.network.head.weight.double()
    classifier_bias = fit.network.head.bias.double()
    logits_val = F.linear(fit.z[data.val_idx.to(device)].double(), classifier_weight, classifier_bias)
    logits_full_precise = torch.zeros(
        (data.num_nodes, data.num_classes), dtype=torch.float64, device=device
    )
    logits_full_precise[data.val_idx.to(device)] = logits_val
    x_t = data.x_t.to(device)
    x_v = data.x_v.to(device)
    chunks: list[dict[str, np.ndarray]] = []

    def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8)

    for start in range(0, src_all.numel(), batch_size):
        src = src_all[start : start + batch_size]
        dst = dst_all[start : start + batch_size]
        full_logits = logits_val[val_row[dst]]
        y = labels[dst]
        ce_full = F.cross_entropy(full_logits, y, reduction="none")
        # Compute the full margin from the high-precision affine logits.
        true_logit = full_logits.gather(1, y[:, None]).squeeze(1)
        other_logits = full_logits.clone()
        other_logits.scatter_(1, y[:, None], -torch.inf)
        margin_full = true_logit - other_logits.max(dim=1).values
        logits_removed_t = _removed_logits_fast(
            fit.network, fit.h_t, fit.h_v,
            logits_full_precise, src, dst, degree, "text"
        )
        logits_removed_v = _removed_logits_fast(
            fit.network, fit.h_t, fit.h_v,
            logits_full_precise, src, dst, degree, "visual"
        )
        ce_removed_t = F.cross_entropy(logits_removed_t, y, reduction="none")
        ce_removed_v = F.cross_entropy(logits_removed_v, y, reduction="none")
        u_t = relative_ce_utility(ce_full, ce_removed_t)
        u_v = relative_ce_utility(ce_full, ce_removed_v)
        um_t = margin_utility(full_logits, logits_removed_t, y)
        um_v = margin_utility(full_logits, logits_removed_v, y)
        chunks.append(
            {
                "src": src.cpu().numpy(),
                "dst": dst.cpu().numpy(),
                "dst_degree": degree[dst].cpu().numpy(),
                "raw_text_cosine": _cos(x_t[src], x_t[dst]).cpu().numpy(),
                "raw_visual_cosine": _cos(x_v[src], x_v[dst]).cpu().numpy(),
                "projected_text_cosine": _cos(fit.h_t[src], fit.h_t[dst]).cpu().numpy(),
                "projected_visual_cosine": _cos(fit.h_v[src], fit.h_v[dst]).cpu().numpy(),
                "utility_text_ce": u_t.cpu().numpy(),
                "utility_visual_ce": u_v.cpu().numpy(),
                "utility_text_margin": um_t.cpu().numpy(),
                "utility_visual_margin": um_v.cpu().numpy(),
                "dst_label": y.cpu().numpy(),
                "ce_full": ce_full.cpu().numpy(),
                "margin_full": margin_full.cpu().numpy(),
            }
        )
    table = pd.DataFrame({key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]})
    table.insert(0, "seed", int(getattr(fit.network, "seed", -1)))
    table.insert(0, "dataset", data.name)
    return table


def deterministic_edge_sample(table: Any, max_rows: int, seed: int) -> Any:
    """Take a reproducible bounded sample for a fixed input row order."""
    if len(table) <= max_rows:
        return table.copy().reset_index(drop=True)
    # A fixed torch generator avoids depending on Python's randomized hash seed.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(len(table), generator=generator)[:max_rows].sort().values.numpy()
    return table.iloc[indices].reset_index(drop=True)


@torch.no_grad()
def group_intervention_rows(
    data: ProbeData,
    fit: ProbeFit,
    edge_table: Any,
    seed: int,
    device: torch.device,
    repeats: int = 10,
) -> list[dict[str, Any]]:
    """Mask per-target utility tails or matched-count random messages."""
    import pandas as pd

    if fit.c_t is None or fit.c_v is None:
        raise ValueError("Group interventions require the contextual probe")
    table = edge_table
    table = table[table["dst_degree"] >= 5].reset_index(drop=True)
    if table.empty:
        raise ValueError(f"{data.name}: no degree>=5 validation messages for group intervention")
    by_dst = {int(dst): group.index.to_numpy() for dst, group in table.groupby("dst", sort=True)}
    val_nodes = sorted(by_dst)
    val_nodes_t = torch.tensor(val_nodes, dtype=torch.long, device=device)
    labels = data.labels.to(device)[val_nodes_t]
    base_logits = fit.logits[val_nodes_t]
    base_metrics = _classification_metrics(base_logits, labels, data.num_classes)
    src = torch.as_tensor(table["src"].to_numpy(copy=True), dtype=torch.long, device=device)
    dst = torch.as_tensor(table["dst"].to_numpy(copy=True), dtype=torch.long, device=device)
    degree = torch.as_tensor(table["dst_degree"].to_numpy(copy=True), dtype=torch.float32, device=device)
    rows: list[dict[str, Any]] = []

    def _run(modality: str, selected_indices: np.ndarray, policy: str, repeat: int | None):
        h = fit.h_t if modality == "text" else fit.h_v
        context = fit.c_t if modality == "text" else fit.c_v
        context_start = 2 * fit.network.hidden_dim if modality == "text" else 3 * fit.network.hidden_dim
        removed = torch.zeros_like(context)
        idx = torch.as_tensor(selected_indices, dtype=torch.long, device=device)
        if idx.numel():
            removed.index_add_(0, dst[idx], h[src[idx]])
        z_new = fit.z[val_nodes_t].clone()
        z_new[:, context_start : context_start + fit.network.hidden_dim] -= (
            removed[val_nodes_t] / torch.bincount(data.edge_index[1], minlength=data.num_nodes)
            .to(device)[val_nodes_t].to(h.dtype).unsqueeze(-1)
        )
        logits = fit.network.head(z_new)
        metrics = _classification_metrics(logits, labels, data.num_classes)
        rows.append(
            {
                "dataset": data.name,
                "seed": seed,
                "modality": modality,
                "policy": policy,
                "repeat": repeat,
                "target_nodes": len(val_nodes),
                "masked_messages": int(idx.numel()),
                "baseline_ce": base_metrics["val_ce"],
                "baseline_accuracy": base_metrics["val_acc"],
                "baseline_macro_f1": base_metrics["val_macro_f1"],
                "val_ce": metrics["val_ce"],
                "accuracy": metrics["val_acc"],
                "macro_f1": metrics["val_macro_f1"],
                "delta_val_ce": metrics["val_ce"] - base_metrics["val_ce"],
                "delta_accuracy": metrics["val_acc"] - base_metrics["val_acc"],
                "delta_macro_f1": metrics["val_macro_f1"] - base_metrics["val_macro_f1"],
            }
        )

    for modality, utility_col in (
        ("text", "utility_text_ce"),
        ("visual", "utility_visual_ce"),
    ):
        utility = table[utility_col].to_numpy()
        bottom: list[int] = []
        top: list[int] = []
        random_draws: list[list[int]] = [[] for _ in range(repeats)]
        rng = np.random.default_rng(seed + (101 if modality == "text" else 211))
        for dst_node, indices in by_dst.items():
            count = max(1, int(np.ceil(0.2 * len(indices))))
            ordered = indices[np.argsort(utility[indices], kind="stable")]
            bottom.extend(ordered[:count].tolist())
            top.extend(ordered[-count:].tolist())
            for repeat in range(repeats):
                random_draws[repeat].extend(
                    rng.choice(indices, size=count, replace=False).tolist()
                )
        _run(modality, np.asarray(bottom, dtype=np.int64), "bottom_utility_20pct", None)
        _run(modality, np.asarray(top, dtype=np.int64), "top_utility_20pct", None)
        for repeat, values in enumerate(random_draws):
            _run(modality, np.asarray(values, dtype=np.int64), "random_control", repeat)
    return rows
