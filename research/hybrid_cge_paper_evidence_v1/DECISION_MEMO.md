# Decision memo

## Current status

- Four framework-level ablations are implemented in a new model module; the historical H2 file is untouched.
- Unit tests cover skipped calls, inactive gradients, no-MoE parameter removal, matched initialization, three-hop propagation count and Train-only class weights.
- Five historical H2 checkpoints strict-load in the new wrapper; all hashes are preserved.
- Forty-five inherited SAGE/MMGCN/RoleMAG cells passed protocol and fingerprint audit.
- Real-data preflights and the 174 new training cells are not complete yet.
- Final Test runner is default-deny; the model lock is pending independent review. No Test metrics have been computed.

No paper-level mechanism or performance claim is finalized until the complete paired evidence is available.
