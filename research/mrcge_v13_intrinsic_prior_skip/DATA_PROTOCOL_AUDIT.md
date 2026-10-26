# Data Protocol Audit — Intrinsic-Prior Skip

Base `184deaaf12b21c9ee01d3328aef15d86bcc50346`; source commit `c7f9209c60f18059d330ac23d7b3ab1b13af2e6e`; branch `exp/mrcge_v13_intrinsic_prior_skip_screen`.
Fixed split seed is 42, auto-generation is disabled, Test labels are masked to `-100` before Trainer entry, Test evaluation is off, and Link Prediction is off.

| Dataset | Nodes | Classes | Split fingerprint | Split SHA256 | Feature fingerprint | Physical graph fingerprint | Semantic cache SHA256 | Candidate fingerprint | V7B fingerprint |
|---|---:|---:|---|---|---|---|---|---|---|
| Movies | 16672 | 20 | `2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d` | `9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211` | `00be19e204173f0208207da407d2db8de7590db2a31815a8508cf1ff0d3bdd1b` | `98607c3f0f5ce0562d82d730a38d9523cf2717af68d8fb8e961a97b876b0f154` | `666c5bfa1524dcbc9c265d14927740c7f05f7684e9289f122c56d49fb277b867` | `99b5f1d73b1c0364f12ea023aa1a21366661ebed974e9b15c1174259142e4ba5` | `4d00be6008609847151bf9d02be30a29549284be9aa8d79b5732196e3764aceb` |
| Grocery | 17074 | 20 | `7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056` | `3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e` | `5684907da6c9e3a544626de41930301fdf0b5e4f483f89017f38713bd74a87e8` | `c92a92669ed89b3af689aada40fc2a8efb924c58fdaf1429e8703f9156ae2de2` | `530dec5249874959e46d621e7a651a0f60097238c725a9e4548bf8d695c64bde` | `77ca3e4ed4c9fe8901ee288dfdcf5224d2125382f19bd6fda53fda33be280225` | `9c9da367a33a840848d45ff76c2ad66ef0a86a5481f6471ce0cb8f3ff5ec2e37` |
| ele-fashion | 97766 | 12 | `04d43160594823ab35d62b1bde6547225dc816405e55eb1f9c44790eabfae0c5` | `e48e74afd502df5bc5e4d638e611fc4fbfdecee461aa920f3f47a6446826c52f` | `266b11e2265af97724a32bdcd8c6303b69e5cb55839572d82a76872189c6518a` | `8e354c68946eb0286ec5ebdab09bfe360f2a96bb6f53d8f3f17b02575b7ced8e` | `f56216d70489dda7fd41587ddff66a9946d233b75776a033d7f74c8eca4cdab8` | `c9ca5402ca64ccb74676954599ea1cec16cebf9d6b2e3eddc5f76c47d183c589` | `b20a39387d411dace61f4ce59e3c4cd1a70f44a597517ef30fff05c2ef09e144` |

## Historical reference checkpoints

Each same-seed Best-Accuracy and Best-F1 checkpoint hash was verified before comparison. Movies seed 44 uses the registered Physical-Basis fixed-split comparison record.

| Dataset | Seed | Source SHA | Best-Accuracy checkpoint SHA256 | Best-F1 checkpoint SHA256 |
|---|---:|---|---|---|
| Movies | 42 | `78284c82308b117e98f6c2ef59bf5acbe13369bf` | `b9e8a2a5ebee2dfc91529d62203137c0cb03f06e9bd001d4227f97bf5ff9aaaf` | `0ce77ddd1744afaf34f0cf2ae863b30477d9b985b3cff1b58179819d7423596f` |
| Movies | 43 | `78284c82308b117e98f6c2ef59bf5acbe13369bf` | `4cd86fe28afdefa80046a96830febd6593e3a249dfee2c4c80dce34888e339e3` | `68309f20f5c27244ddbd4620c89b70bb5dfd406426a94f0c1afb7d1cc6850f68` |
| Movies | 44 | `cb641c4947c13d2305cfe83110cfdf9e35d42517` | `a725d9cdee1ab1a3875f8b9e0515ca515fd9e0951b3893f9020a3150b1c4eb5b` | `197629f2d648d7b3672e54f821a824f8f940b71478e01bed7b0daacde29e6898` |
| Grocery | 42 | `78284c82308b117e98f6c2ef59bf5acbe13369bf` | `650c740a0e727db442fd7e3bbb2248cb43346d7d8c38dee9effd03989f4ea3fa` | `af24c6ea11688736e2660c2d7b4e055afce4b1cce1cff92b2dfaf1f3dbb74085` |
| Grocery | 43 | `78284c82308b117e98f6c2ef59bf5acbe13369bf` | `28b3e836aa6553c996e04fe63a7954180c0f421f464cb6c587d7acee1a6059a4` | `7298023a1ecc1c420bfe29bd105b17e7e5636c7b33dd7603e7ee63645f075ba8` |
| ele-fashion | 42 | `78284c82308b117e98f6c2ef59bf5acbe13369bf` | `01506139bb66baa649d05871dfc64ee8bce0b547e958001979ea8c7edaffc780` | `64ca49d1ee2f0f00f9825d07d34eb5e2f915c832eab228cf14e2debfb27df7e8` |
| ele-fashion | 43 | `78284c82308b117e98f6c2ef59bf5acbe13369bf` | `c9c18603b06494acb3d63cccd02b11bfe0ae0ffaa094e5bdad3f1e4f4419c5e3` | `41657980e022804d35118bc8a5f52bf79b3e2291abe9aa839a1120bf85b272f2` |

Protected historical files: 2096 hashes checked against the propagation-audit manifest; unchanged: `True`.

All formal run rows record `test_evaluation=false`; no Test metric is loaded or emitted. The only Test-index operation checks that labels are masked to `-100`.
