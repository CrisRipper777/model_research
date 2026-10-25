# Preflight report

CPU/data preflight: `pass` at 2026-10-25T18:02:53+00:00.
GPU preflight status: `pass`.
Failed CPU preflight attempts: 2; all are retained in `data/failures.json` and were corrected before the clean pass.
Base: `6d94c14d67e511aec34ff5fdcad3ce1a1cfd0701`; training source: `7704c4edd43cddf4f7418a732bf5a6805e10d6d9`.

## Data protocol

- **Grocery**: nodes=17074, edges=142262, classes=20, Text/Visual dims=768/768; split `7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056`, split-file SHA `3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e`, semantic-cache SHA `530dec5249874959e46d621e7a651a0f60097238c725a9e4548bf8d695c64bde`; Test masked=True; Test evaluation=False.
- **Movies**: nodes=16672, edges=160802, classes=20, Text/Visual dims=768/768; split `2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d`, split-file SHA `9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211`, semantic-cache SHA `666c5bfa1524dcbc9c265d14927740c7f05f7684e9289f122c56d49fb277b867`; Test masked=True; Test evaluation=False.
- **ele-fashion**: nodes=97766, edges=399172, classes=12, Text/Visual dims=512/512; split `04d43160594823ab35d62b1bde6547225dc816405e55eb1f9c44790eabfae0c5`, split-file SHA `e48e74afd502df5bc5e4d638e611fc4fbfdecee461aa920f3f47a6446826c52f`, semantic-cache SHA `f56216d70489dda7fd41587ddff66a9946d233b75776a033d7f74c8eca4cdab8`; Test masked=True; Test evaluation=False.

## Per-cell GPU preflight

- Movies seed 42: `pass`; peak reserved=2579496960 bytes; GPU=NVIDIA GeForce RTX 3090; reason=.
- Movies seed 43: `pass`; peak reserved=2579496960 bytes; GPU=NVIDIA GeForce RTX 3090; reason=.
- Movies seed 44: `pass`; peak reserved=2579496960 bytes; GPU=NVIDIA GeForce RTX 3090; reason=.
- Grocery seed 42: `pass`; peak reserved=2520776704 bytes; GPU=NVIDIA GeForce RTX 3090; reason=.
- Grocery seed 43: `pass`; peak reserved=2520776704 bytes; GPU=NVIDIA GeForce RTX 3090; reason=.
- ele-fashion seed 42: `pass`; peak reserved=13004439552 bytes; GPU=NVIDIA GeForce RTX 3090; reason=.
- ele-fashion seed 43: `pass`; peak reserved=13004439552 bytes; GPU=NVIDIA GeForce RTX 3090; reason=.
