# Data protocol audit

The runner resolves each dataset's configured fixed split path and verifies its loaded split fingerprint against the frozen study value and the same-seed V1.3-A record. Feature, physical-edge, split-file, V7B.1 candidate, V7B, and semantic-cache hashes are checked before training. Automatic split creation is disabled. Test labels are replaced with -100 before the Trainer; Test evaluation and LP are disabled.

| Dataset | Nodes | Text dim | Visual dim | Split fingerprint | Split SHA256 | Cache SHA256 |
|---|---:|---:|---:|---|---|---|
| Grocery | 17074 | 768 | 768 | `7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056` | `3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e` | `530dec5249874959e46d621e7a651a0f60097238c725a9e4548bf8d695c64bde` |
| Movies | 16672 | 768 | 768 | `2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d` | `9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211` | `666c5bfa1524dcbc9c265d14927740c7f05f7684e9289f122c56d49fb277b867` |
| ele-fashion | 97766 | 512 | 512 | `04d43160594823ab35d62b1bde6547225dc816405e55eb1f9c44790eabfae0c5` | `e48e74afd502df5bc5e4d638e611fc4fbfdecee461aa920f3f47a6446826c52f` | `f56216d70489dda7fd41587ddff66a9946d233b75776a033d7f74c8eca4cdab8` |
