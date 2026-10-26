# Data protocol audit

All seven V1.3-A and three fixed-split Movies H2 checkpoint references were taken from the prior frozen baseline. Each source commit, historical model-source SHA, checkpoint SHA, V7B.1 cache SHA/fingerprints, raw feature and physical edge fingerprints, fixed split fingerprint and split file SHA were verified before workers. The seed44 V1.3-A record came from the fixed-split physical-basis campaign.

| Dataset | Nodes | Directed edges | Split fingerprint | Cache SHA |
|---|---:|---:|---|---|
| Movies | 16672 | 160802 | `2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d` | `666c5bfa1524dcbc9c265d14927740c7f05f7684e9289f122c56d49fb277b867` |
| Grocery | 17074 | 142262 | `7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056` | `530dec5249874959e46d621e7a651a0f60097238c725a9e4548bf8d695c64bde` |
| ele-fashion | 97766 | 399172 | `04d43160594823ab35d62b1bde6547225dc816405e55eb1f9c44790eabfae0c5` | `f56216d70489dda7fd41587ddff66a9946d233b75776a033d7f74c8eca4cdab8` |

`auto_generate_splits=false`; existing fixed split files; V7B.1 K=8; Test labels masked using the existing helper and checked as `-100` before any model forward; `evaluate_test=false`. The model call receives only features and physical edges. No label-derived quantities or Test output were written.
