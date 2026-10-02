# CARE-MAG A0–A2 design and interface audit

Base: `origin/main` at `579d8dcde6d9bf6d39efb4f9d88bf70a78acc38e`.

## Model contract

`src/models/factory.py` dynamically imports `src.models.<cfg.model.name>` and constructs `Model(cfg, data_info)`. NC and LP call `forward(x, edge_index)` and unpack `(z, _, _, aux_loss, aux_info)`; `out_dim` is consumed by the classifier/LP decoder. The optional `inference(x, edge_index, device, batch_size)` contract returns CPU embeddings. CARE will return a scalar zero auxiliary loss and use an exact full forward for inference.

## NC full-graph path

`src/tasks/nc.py` trains graph models with one full-graph forward per epoch, computes loss only at `train_idx`, then obtains embeddings for validation. With full inference mode, `src/tasks/inference.py` calls the model once on all `data.x` and `data.edge_index`; layerwise mode calls the model's `inference` method. The selected checkpoint is based on validation accuracy. CARE's diffusion operator therefore sees the NC graph passed to that forward.

## Sampled LP path and positive-edge removal

`src/tasks/lp.py` builds a `LinkNeighborLoader` over the training message graph. Each batch contains sampled `x`, local `edge_index`, local `edge_label_index`, and global-node/global-edge mappings (`n_id`, `e_id`). Before moving the batch to the training device, the default `global_eid` path maps positive supervision endpoints through `n_id`, looks up both directed orientations (including duplicate global edges), and removes their `e_id`s from the sampled message graph. The model then receives `batch.x` and this filtered, local `batch.edge_index`; `edge_label_index` addresses the sampled node rows. A model with `requires_full_lp_sampler_depth = True` gets fanouts resolved to `num_layers`; with three layers and the LP config's `[5,5,5]`, all three encoder hops are sampled.

## CARE diffusion placement

The symmetric GCN-style normalization must be constructed inside each CARE forward from the supplied `edge_index`. In NC that is the current full graph; in LP it is the current sampled graph *after* positive-supervision-edge removal. Precomputing propagation on the full training graph would preserve messages from edges absent in an LP sample, use full-graph degrees instead of sampled-subgraph degrees, and could retain a positive edge explicitly removed by the LP protocol. That would violate the current message graph used by the batch and make NC and LP propagation inconsistent.

## Modality order

Loaders concatenate MAGB features as `torch.cat([text_feat, image_feat], dim=1)` and MM-Graph joint features are documented/stored as `[x_t, x_i]`. `data_info` exposes `text_dim` and `visual_dim`. CARE must split `x[:, :text_dim]` as text and the following `visual_dim` columns as visual, and reject a width mismatch rather than infer or swap the modalities.

## Frozen choices for this screen

The state encoder uses the specified two-layer `4h -> state_dim -> state_dim` MLP, GELU, and LayerNorm. This follows the explicit equations and keeps the same state encoder and router modules instantiated in every variant. No training has started before this audit.
