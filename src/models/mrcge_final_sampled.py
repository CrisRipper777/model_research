"""LinkNeighborLoader adapter for the frozen MR-CGE Final encoder.

The target nodes keep every cached V7B.1 semantic candidate. The Late
Completion source nodes are added with their full semantic neighborhoods as
well. Their physical context is a documented fixed approximation: three
physical hops with fanout 1 per hop, sampled only from the official Train graph.
This avoids turning every LP mini-batch into a full physical-graph encoder pass.
The original LinkNeighborLoader fanouts and global-EID positive-edge mask for
the supervision endpoints remain unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.models.mrcge_final import Model as FinalModel


class Model(FinalModel):
    """Sampled LP adapter sharing the exact Final Encoder module definitions."""

    supports_link_prediction = True
    requires_global_semantic_candidates = False
    requires_v7b_semantic_cache = True
    requires_full_lp_sampler_depth = False

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__(cfg, data_info)
        self.semantic_context_physical_fanout = tuple(
            int(value)
            for value in cfg.model.get("semantic_context_physical_fanout", [1, 1, 1])
        )
        if len(self.semantic_context_physical_fanout) != self.physical_hops or any(
            value < 1 for value in self.semantic_context_physical_fanout
        ):
            raise ValueError("semantic_context_physical_fanout must define one positive fanout per physical hop")
        self._semantic_incoming_cpu: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._sampled_semantic_edge_flags: dict[str, torch.Tensor] | None = None
        self._lp_global_x: torch.Tensor | None = None
        self._lp_global_edge_index: torch.Tensor | None = None
        self._lp_global_graph: Data | None = None
        self._lp_sorted_edge_codes: torch.Tensor | None = None
        self._lp_sorted_edge_ids: torch.Tensor | None = None
        self.last_sampled_context: dict[str, int | list[int] | str] = {}

    def build_semantic_incoming_index(self) -> None:
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("load the validated V7B.1 cache before building sampled semantic context")
        self._semantic_incoming_cpu = {}
        directed_modalities: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for modality in ("text", "visual"):
            edge_index = getattr(self, f"semantic_{modality}_edge_index").detach().cpu().long().contiguous()
            weight = getattr(self, f"semantic_{modality}_weight").detach().cpu().float().contiguous()
            if edge_index.size(1) != weight.numel():
                raise ValueError(f"{modality} semantic edge and weight lengths differ")
            if edge_index.numel() and (
                int(edge_index.min()) < 0 or int(edge_index.max()) >= self.num_nodes
            ):
                raise ValueError(f"{modality} semantic cache contains an invalid node ID")
            directed_modalities[modality] = (edge_index, weight)

        # Recover the cache's canonical union-edge ordering so both original
        # source counts stay aligned with every directed modality operator.
        canonical_codes = []
        for edge_index, _ in directed_modalities.values():
            src, dst = edge_index
            canonical_codes.append(
                torch.minimum(src, dst) * self.num_nodes + torch.maximum(src, dst)
            )
        unique_codes = torch.unique(torch.cat(canonical_codes), sorted=True) if canonical_codes else torch.empty(0, dtype=torch.long)
        flags = self.semantic_source_flags_v7b.detach().cpu().to(torch.uint8).contiguous()
        if flags.shape != (unique_codes.numel(), 2):
            raise ValueError("V7B source flags do not align with the union semantic candidates")

        for modality, (edge_index, weight) in directed_modalities.items():
            src, dst = edge_index
            codes = torch.minimum(src, dst) * self.num_nodes + torch.maximum(src, dst)
            positions = torch.searchsorted(unique_codes, codes)
            if positions.numel() and not torch.equal(unique_codes[positions], codes):
                raise ValueError(f"{modality} operator edge is absent from V7B source flags")
            directed_flags = flags[positions]
            order = torch.argsort(edge_index[1], stable=True)
            sources = edge_index[0, order].contiguous()
            destinations = edge_index[1, order]
            counts = torch.bincount(destinations, minlength=self.num_nodes)
            ptr = torch.zeros(self.num_nodes + 1, dtype=torch.long)
            ptr[1:] = counts.cumsum(dim=0)
            self._semantic_incoming_cpu[modality] = (
                ptr, sources, weight[order].contiguous(), directed_flags[order].contiguous()
            )

    def bind_lp_graph(self, x: torch.Tensor, train_edge_index: torch.Tensor) -> None:
        """Bind only the loader's official Train-positive physical graph."""
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("load the validated V7B.1 cache before binding the LP graph")
        x_cpu = x.detach().cpu().contiguous()
        edge_cpu = train_edge_index.detach().cpu().long().contiguous()
        if x_cpu.ndim != 2 or x_cpu.size(0) != self.num_nodes or x_cpu.size(1) != self.input_dim:
            raise ValueError("LP global features do not match the model's global node order and dimensions")
        if edge_cpu.ndim != 2 or edge_cpu.size(0) != 2:
            raise ValueError("LP Train graph must have shape [2, E]")
        if edge_cpu.numel() and (int(edge_cpu.min()) < 0 or int(edge_cpu.max()) >= self.num_nodes):
            raise ValueError("LP Train graph contains an invalid node ID")
        self._lp_global_x = x_cpu
        self._lp_global_edge_index = edge_cpu
        self._lp_global_graph = Data(x=x_cpu, edge_index=edge_cpu)
        edge_codes = edge_cpu[0] * self.num_nodes + edge_cpu[1]
        self._lp_sorted_edge_codes, self._lp_sorted_edge_ids = edge_codes.sort()
        self.build_semantic_incoming_index()

    @staticmethod
    def _incoming_rows(
        destinations: torch.Tensor,
        ptr: torch.Tensor,
        sources: torch.Tensor,
        weights: torch.Tensor,
        source_flags: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        destinations = torch.unique(destinations.detach().cpu().long(), sorted=True)
        counts = ptr[destinations + 1] - ptr[destinations]
        total = int(counts.sum().item())
        if total == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty, torch.empty(0, dtype=weights.dtype), torch.empty((0, 2), dtype=torch.uint8)
        group = torch.repeat_interleave(torch.arange(destinations.numel()), counts)
        group_starts = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        offsets = torch.arange(total, dtype=torch.long) - group_starts
        positions = ptr[destinations[group]] + offsets
        src = sources[positions]
        dst = destinations[group]
        return (
            src.contiguous(), dst.contiguous(), weights[positions].contiguous(),
            source_flags[positions].contiguous(),
        )

    def _global_eids_for_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if self._lp_sorted_edge_codes is None or self._lp_sorted_edge_ids is None:
            raise RuntimeError("LP graph has not been bound")
        codes = torch.unique(codes.detach().cpu().long(), sorted=True)
        if codes.numel() == 0 or self._lp_sorted_edge_codes.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        starts = torch.searchsorted(self._lp_sorted_edge_codes, codes, right=False)
        ends = torch.searchsorted(self._lp_sorted_edge_codes, codes, right=True)
        counts = ends - starts
        present = counts > 0
        starts, counts = starts[present], counts[present]
        if starts.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        groups = torch.repeat_interleave(torch.arange(starts.numel()), counts)
        group_starts = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        offsets = torch.arange(int(counts.sum().item()), dtype=torch.long) - group_starts
        return self._lp_sorted_edge_ids[starts[groups] + offsets].unique(sorted=True)

    def prepare_sampled_lp_batch(self, batch, *, sample_seed: int) -> None:
        """Add semantic closure and sampled Train-only physical context in-place.

        Rows already returned by LinkNeighborLoader stay first, so the loader's
        local edge_label_index continues to address the same nodes.
        """
        if self._lp_global_graph is None or self._lp_global_x is None:
            raise RuntimeError("bind_lp_graph must be called before sampled LP batches are prepared")
        if not hasattr(batch, "n_id") or not hasattr(batch, "e_id"):
            raise ValueError("sampled LP batches require LinkNeighborLoader n_id/e_id mappings")
        if batch.n_id.device.type != "cpu" or batch.edge_index.device.type != "cpu":
            raise ValueError("sampled LP context must be expanded before moving the batch to the accelerator")
        original_n_id = batch.n_id.detach().cpu().long().contiguous()
        if original_n_id.numel() and (int(original_n_id.min()) < 0 or int(original_n_id.max()) >= self.num_nodes):
            raise ValueError("LinkNeighborLoader returned an invalid global node ID")
        seed_local = torch.unique(batch.edge_label_index.detach().cpu().long())
        target_global = torch.unique(original_n_id[seed_local], sorted=True)

        # Late Completion uses the opposite-modality semantic graph. The union
        # of both directed source sets supplies the exact candidate IDs for all
        # target endpoints, without a local top-K or candidate truncation.
        target_sources: list[torch.Tensor] = []
        for modality in ("text", "visual"):
            ptr, sources, weights, source_flags = self._semantic_incoming_cpu[modality]
            src, _, _, _ = self._incoming_rows(target_global, ptr, sources, weights, source_flags)
            target_sources.append(src)
        late_source_global = torch.unique(torch.cat(target_sources), sorted=True) if target_sources else target_global[:0]
        semantic_targets = torch.unique(torch.cat((target_global, late_source_global)), sorted=True)

        semantic_rows: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        semantic_source_nodes = []
        for modality in ("text", "visual"):
            ptr, sources, weights, source_flags = self._semantic_incoming_cpu[modality]
            src, dst, edge_weight, edge_flags = self._incoming_rows(
                semantic_targets, ptr, sources, weights, source_flags
            )
            semantic_rows[modality] = (src, dst, edge_weight, edge_flags)
            semantic_source_nodes.append(src)
        all_semantic_sources = torch.unique(torch.cat(semantic_source_nodes), sorted=True)

        # Sample the source nodes' physical representations from the official
        # Train graph only. Fanout 1 is a fixed, explicit approximation to the
        # encoder's full 3-hop physical receptive field for these extra nodes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(sample_seed) % (2**31 - 1))
            loader = NeighborLoader(
                self._lp_global_graph,
                input_nodes=late_source_global,
                num_neighbors=list(self.semantic_context_physical_fanout),
                batch_size=max(int(late_source_global.numel()), 1),
                shuffle=False,
                subgraph_type="bidirectional",
                num_workers=0,
            )
            extra = next(iter(loader)) if late_source_global.numel() else None

        positive_mask = batch.edge_label.detach().cpu() > 0.5
        positive_local = batch.edge_label_index[:, positive_mask].detach().cpu().long()
        positive_global = original_n_id[positive_local]
        positive_codes = torch.cat(
            (
                positive_global[0] * self.num_nodes + positive_global[1],
                positive_global[1] * self.num_nodes + positive_global[0],
            )
        )
        forbidden_eids = self._global_eids_for_codes(positive_codes)
        existing_eids = batch.e_id.detach().cpu().long().unique(sorted=True)

        combined_ids = original_n_id
        extra_edges_global = torch.empty((2, 0), dtype=torch.long)
        extra_eids = torch.empty(0, dtype=torch.long)
        if extra is not None:
            extra_n_id = extra.n_id.detach().cpu().long()
            context_nodes = torch.unique(
                torch.cat((extra_n_id, late_source_global, all_semantic_sources)), sorted=True
            )
            present = torch.zeros(self.num_nodes, dtype=torch.bool)
            present[original_n_id] = True
            add_ids = context_nodes[~present[context_nodes]].unique(sorted=True)
            combined_ids = torch.cat((original_n_id, add_ids))

            extra_eids_all = extra.e_id.detach().cpu().long()
            forbidden = torch.cat((forbidden_eids, existing_eids)).unique(sorted=True)
            keep = ~torch.isin(extra_eids_all, forbidden)
            # A sampled subgraph may repeat a global EID. Keep one copy only.
            kept_eids = extra_eids_all[keep]
            if kept_eids.numel():
                # Preserve NeighborLoader edge order while dropping duplicate EIDs.
                seen: set[int] = set()
                selected = []
                for index, eid in enumerate(kept_eids.tolist()):
                    if eid not in seen:
                        seen.add(eid)
                        selected.append(index)
                selected_tensor = torch.tensor(selected, dtype=torch.long)
                keep_positions = torch.nonzero(keep, as_tuple=False).flatten()[selected_tensor]
                extra_eids = extra_eids_all[keep_positions]
                extra_local_edges = extra.edge_index.detach().cpu().long()[:, keep_positions]
                edge_global_ids = extra_n_id[extra_local_edges]
                global_to_local = torch.full((self.num_nodes,), -1, dtype=torch.long)
                global_to_local[combined_ids] = torch.arange(combined_ids.numel())
                extra_edges_global = global_to_local[edge_global_ids]
                if bool((extra_edges_global < 0).any()):
                    raise RuntimeError("semantic-context physical edges reference an unmapped node")

        if combined_ids.numel() != original_n_id.numel():
            add_ids = combined_ids[original_n_id.numel():]
            batch.x = torch.cat((batch.x, self._lp_global_x[add_ids]), dim=0).contiguous()
        if extra_eids.numel():
            batch.edge_index = torch.cat((batch.edge_index, extra_edges_global), dim=1).contiguous()
            batch.e_id = torch.cat((batch.e_id, extra_eids), dim=0).contiguous()
        batch.n_id = combined_ids.contiguous()

        global_to_local = torch.full((self.num_nodes,), -1, dtype=torch.long)
        global_to_local[combined_ids] = torch.arange(combined_ids.numel())
        local_semantic: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        local_flags: dict[str, torch.Tensor] = {}
        for modality, (src, dst, weight, edge_flags) in semantic_rows.items():
            if src.numel() and (int(global_to_local[src].min()) < 0 or int(global_to_local[dst].min()) < 0):
                raise RuntimeError(f"{modality} semantic closure omitted a required source node")
            edge_index = torch.stack((global_to_local[src], global_to_local[dst]), dim=0).contiguous()
            local_semantic[modality] = (edge_index, weight)
            local_flags[modality] = edge_flags
        self._sampled_semantic_edges = local_semantic
        self._sampled_semantic_edge_flags = local_flags
        self._batch_n_id = batch.n_id
        self.last_sampled_context = {
            "target_nodes": int(target_global.numel()),
            "late_source_nodes": int(late_source_global.numel()),
            "semantic_context_nodes": int(all_semantic_sources.numel()),
            "local_nodes": int(batch.n_id.numel()),
            "semantic_edges_text": int(local_semantic["text"][0].size(1)),
            "semantic_edges_visual": int(local_semantic["visual"][0].size(1)),
            "extra_train_physical_edges": int(extra_eids.numel()),
            "semantic_source_physical_fanout": list(self.semantic_context_physical_fanout),
            "approximation": "full semantic candidates; semantic-source physical context sampled with fixed fanout 1 per hop from Train graph",
        }

    def clear_sampled_lp_context(self) -> None:
        self._batch_n_id = None
        self._sampled_semantic_edges = None
        self._sampled_semantic_edge_flags = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        # The inherited full-graph path is also used for Validation and final Test.
        if self._batch_n_id is None:
            self._sampled_semantic_edges = None
            return super().forward(x, edge_index, return_details=return_details)
        if self._sampled_semantic_edges is None:
            raise RuntimeError("sampled LP forward requires a prepared semantic context")
        self._sampled_semantic_edges = {
            key: (edges.to(x.device), weights.to(device=x.device, dtype=x.dtype))
            for key, (edges, weights) in self._sampled_semantic_edges.items()
        }
        return super().forward(x, edge_index, return_details=return_details)
