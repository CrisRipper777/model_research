# Preflight report

- CPU preflight: `pass` at 2026-10-26T05:25:56+00:00.
- Physical GPU: 1; GPU preflight status: `pass`.
- GPU preflight passing cells: 7/7.
- Required launch free memory: at least 8192 MiB and each formal launch also cleared the measured peak-reserved plus 2048 MiB margin.
- Training source freeze SHA: `4ea0e0aeebdc39cc9329db47ef9c047485ee8916`.
- Audit/report commit: `b7ee36a3c19569c92f34652ebe9a1b90d56fce5c`.
- Test evaluation: disabled.
- Initial sandbox-only GPU preflight was blocked by `nvidia-smi` SIGKILL and retained in `failures.json`; after authorized device access was available, a fresh ownership guard passed all seven cells.
