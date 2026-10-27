# Node-Conditioned vs View-Global Router Ablation Results

Branch `exp/mrcge_v13_view_global_router_ablation`; base `4ec2e83b630adf6a7752b33f1ffd9e988392638b`; frozen training source `bb77aad775a47045750e6eca400a1c2d2ec8f8be`.

This full-mechanism ablation changes only the router granularity. The original V1.3-A protocol is retained, including Private CE weight 0.3 and Balance Loss weight 0.01. Results use Validation only; Test labels were masked before training.

## Paired Best-Validation-Accuracy checkpoints

| Dataset | Seed | Global Acc | Global Macro-F1 | Global CE | Δ Acc (pp) | Δ Macro-F1 (pp) | Δ CE | Train−Val Acc gap (pp) | Train−Val CE gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | 55.729% | 48.164% | 1.42844 | -0.240 | -0.033 | +0.05806 | +22.40 | -0.78660 |
| Movies | 43 | 55.699% | 49.197% | 1.37219 | +0.480 | +0.026 | -0.00763 | +17.93 | -0.60408 |
| Movies | 44 | 55.969% | 48.165% | 1.41099 | +0.120 | +0.131 | +0.05678 | +23.63 | -0.78620 |
| Grocery | 42 | 83.221% | 76.452% | 0.67617 | -0.293 | -0.170 | +0.03694 | +13.19 | -0.53507 |
| Grocery | 43 | 82.987% | 76.481% | 0.61770 | -0.703 | -0.455 | -0.02789 | +7.88 | -0.29393 |

## Dataset summaries

| Dataset | n | Global Acc mean ± SD | Global Macro-F1 mean ± SD | Global CE mean ± SD | Δ Acc mean ± SD (pp) | Δ Macro-F1 mean ± SD (pp) | Δ CE mean ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 3 | 55.799% ± 0.121 | 48.509% ± 0.487 | 1.40387 ± 0.02351 | +0.120 ± 0.294 | +0.041 ± 0.068 | +0.03574 ± 0.03067 |
| Grocery | 2 | 83.104% ± 0.117 | 76.466% ± 0.015 | 0.64694 ± 0.02924 | -0.498 ± 0.205 | -0.312 ± 0.143 | +0.00453 ± 0.03241 |

## Best-Accuracy checkpoint Balance Loss

Raw Balance is the mean across the six routed views; scaled Balance is `0.01 × raw`.

| Dataset | Node Raw | View-Global Raw | Δ Raw | Node Scaled | View-Global Scaled | Δ Scaled |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 1.170594 | 1.250517 | +0.079923 | 0.011706 | 0.012505 | +0.000799 |
| Grocery | 1.299796 | 1.380540 | +0.080745 | 0.012998 | 0.013805 | +0.000807 |

## Route and Balance Loss audit

For each Best-Accuracy checkpoint, `data/router_summary.csv` records the six independent view calls for both the historical node-conditioned checkpoint and the new View-Global checkpoint. The View-Global rows contain exact shared Top-2 IDs and mixture weights. Historical rows use existing aggregate telemetry: selected-expert union, four-expert mean probabilities, selection shares, entropy, and raw/scaled Balance Loss; the old telemetry does not retain per-node mixture weights.

### Movies, seed 42

Historical raw/scaled Balance: 1.151956 / 0.011520; View-Global: 1.218603 / 0.012186.

Node Conditioned expert routing by view:
- `private_text_physical`: selected-expert union [0, 1, 2, 3]; shares [0.20372480154037476, 0.27570176124572754, 0.1856105923652649, 0.3349628150463104]; entropy 1.0154.
- `private_text_semantic`: selected-expert union [0, 1, 2, 3]; shares [0.17421424388885498, 0.15157149732112885, 0.21002279222011566, 0.4641914367675781]; entropy 1.0136.
- `private_visual_physical`: selected-expert union [0, 1, 2, 3]; shares [0.19628718495368958, 0.2779810428619385, 0.159638911485672, 0.36609283089637756]; entropy 1.0194.
- `private_visual_semantic`: selected-expert union [0, 1, 2, 3]; shares [0.20963291823863983, 0.21647071838378906, 0.22525791823863983, 0.3486384153366089]; entropy 1.0744.
- `collaborative_text_physical`: selected-expert union [0, 1, 2, 3]; shares [0.37556982040405273, 0.15088170766830444, 0.2963951528072357, 0.1771533042192459]; entropy 1.0601.
- `collaborative_visual_physical`: selected-expert union [0, 1, 2, 3]; shares [0.1479426473379135, 0.3433901071548462, 0.1966770589351654, 0.3119901418685913]; entropy 0.9100.

View Global expert routing by view:
- `private_text_physical`: Top-2 experts [2, 3], weights [0.2979617556822494, 0.7020382443177506]; shares [0.0, 0.0, 0.5, 0.5]; entropy 1.2973.
- `private_text_semantic`: Top-2 experts [2, 3], weights [0.3212583078198986, 0.6787416921801014]; shares [0.0, 0.0, 0.5, 0.5]; entropy 1.3200.
- `private_visual_physical`: Top-2 experts [1, 3], weights [0.41405923045810494, 0.5859407695418951]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.3656.
- `private_visual_semantic`: Top-2 experts [2, 3], weights [0.4135234348607551, 0.586476565139245]; shares [0.0, 0.0, 0.5, 0.5]; entropy 1.3726.
- `collaborative_text_physical`: Top-2 experts [2, 3], weights [0.3428806236895483, 0.6571193763104517]; shares [0.0, 0.0, 0.5, 0.5]; entropy 1.3371.
- `collaborative_visual_physical`: Top-2 experts [1, 3], weights [0.703149909902297, 0.296850090097703]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.2585.

### Movies, seed 43

Historical raw/scaled Balance: 1.156279 / 0.011563; View-Global: 1.357418 / 0.013574.

Node Conditioned expert routing by view:
- `private_text_physical`: selected-expert union [0, 1, 2, 3]; shares [0.16419744491577148, 0.2410028725862503, 0.24199256300926208, 0.35280710458755493]; entropy 1.1402.
- `private_text_semantic`: selected-expert union [0, 1, 2, 3]; shares [0.1756237894296646, 0.28319936990737915, 0.1581093966960907, 0.38306739926338196]; entropy 1.3080.
- `private_visual_physical`: selected-expert union [0, 1, 2, 3]; shares [0.15849927067756653, 0.3405410051345825, 0.18420104682445526, 0.3167586326599121]; entropy 0.9434.
- `private_visual_semantic`: selected-expert union [0, 1, 2, 3]; shares [0.148452490568161, 0.35064777731895447, 0.21425142884254456, 0.2866482734680176]; entropy 0.9968.
- `collaborative_text_physical`: selected-expert union [0, 1, 2, 3]; shares [0.42520391941070557, 0.0867922231554985, 0.3819877505302429, 0.10601606965065002]; entropy 0.9701.
- `collaborative_visual_physical`: selected-expert union [0, 1, 2, 3]; shares [0.10967490077018738, 0.37934860587120056, 0.17514394223690033, 0.33583253622055054]; entropy 0.8137.

View Global expert routing by view:
- `private_text_physical`: Top-2 experts [1, 3], weights [0.3030437565071296, 0.6969562434928703]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.2911.
- `private_text_semantic`: Top-2 experts [1, 3], weights [0.3077204302651617, 0.6922795697348383]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.2978.
- `private_visual_physical`: Top-2 experts [2, 3], weights [0.23808720533171476, 0.7619127946682852]; shares [0.0, 0.0, 0.5, 0.5]; entropy 1.2110.
- `private_visual_semantic`: Top-2 experts [2, 3], weights [0.2591059456607051, 0.7408940543392949]; shares [0.0, 0.0, 0.5, 0.5]; entropy 1.2309.
- `collaborative_text_physical`: Top-2 experts [1, 3], weights [0.33101527201735625, 0.6689847279826437]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.3251.
- `collaborative_visual_physical`: Top-2 experts [1, 3], weights [0.6967912568193643, 0.3032087431806358]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.1297.

### Movies, seed 44

Historical raw/scaled Balance: 1.203547 / 0.012035; View-Global: 1.175530 / 0.011755.

Node Conditioned expert routing by view:
- `private_text_physical`: selected-expert union [0, 1, 2, 3]; shares [0.13249759376049042, 0.2987643778324127, 0.15774951875209808, 0.4109884798526764]; entropy 0.9794.
- `private_text_semantic`: selected-expert union [0, 1, 2, 3]; shares [0.06705854088068008, 0.2552483081817627, 0.27729126811027527, 0.40040186047554016]; entropy 0.8725.
- `private_visual_physical`: selected-expert union [0, 1, 2, 3]; shares [0.14173464477062225, 0.33787187933921814, 0.18066218495368958, 0.33973127603530884]; entropy 1.0564.
- `private_visual_semantic`: selected-expert union [0, 1, 2, 3]; shares [0.1482725441455841, 0.32395631074905396, 0.21955972909927368, 0.3082113564014435]; entropy 1.1172.
- `collaborative_text_physical`: selected-expert union [0, 1, 2, 3]; shares [0.38828572630882263, 0.17427422106266022, 0.12964850664138794, 0.30779150128364563]; entropy 1.1156.
- `collaborative_visual_physical`: selected-expert union [0, 1, 2, 3]; shares [0.08028430491685867, 0.42724326252937317, 0.17130517959594727, 0.3211672306060791]; entropy 0.8719.

View Global expert routing by view:
- `private_text_physical`: Top-2 experts [2, 3], weights [0.49302306152425024, 0.5069769384757498]; shares [0.0, 0.0, 0.5, 0.5]; entropy 1.3861.
- `private_text_semantic`: Top-2 experts [0, 2], weights [0.4988094583627021, 0.5011905416372979]; shares [0.5, 0.0, 0.5, 0.0]; entropy 1.3840.
- `private_visual_physical`: Top-2 experts [1, 3], weights [0.28045358012910243, 0.7195464198708976]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.2599.
- `private_visual_semantic`: Top-2 experts [1, 3], weights [0.304193727400052, 0.695806272599948]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.2971.
- `collaborative_text_physical`: Top-2 experts [1, 2], weights [0.5206361939009166, 0.4793638060990834]; shares [0.0, 0.5, 0.5, 0.0]; entropy 1.3853.
- `collaborative_visual_physical`: Top-2 experts [1, 3], weights [0.6334627791480641, 0.3665372208519358]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.2999.

### Grocery, seed 42

Historical raw/scaled Balance: 1.296798 / 0.012968; View-Global: 1.400283 / 0.014003.

Node Conditioned expert routing by view:
- `private_text_physical`: selected-expert union [0, 1, 2, 3]; shares [0.21506384015083313, 0.29342859983444214, 0.06861309707164764, 0.4228944778442383]; entropy 0.9514.
- `private_text_semantic`: selected-expert union [0, 1, 2, 3]; shares [0.2923450767993927, 0.20563429594039917, 0.0837530791759491, 0.41826754808425903]; entropy 0.9449.
- `private_visual_physical`: selected-expert union [0, 1, 2, 3]; shares [0.19125571846961975, 0.31928664445877075, 0.09116200357675552, 0.39829567074775696]; entropy 0.8254.
- `private_visual_semantic`: selected-expert union [0, 1, 2, 3]; shares [0.13702119886875153, 0.36959704756736755, 0.12782593071460724, 0.3655558228492737]; entropy 0.8702.
- `collaborative_text_physical`: selected-expert union [0, 1, 2, 3]; shares [0.22121354937553406, 0.42210379242897034, 0.14164812862873077, 0.21503455936908722]; entropy 0.7544.
- `collaborative_visual_physical`: selected-expert union [0, 1, 2, 3]; shares [0.13365350663661957, 0.3942544162273407, 0.15526531636714935, 0.3168267607688904]; entropy 0.6424.

View Global expert routing by view:
- `private_text_physical`: Top-2 experts [1, 3], weights [0.32886602905641976, 0.6711339709435802]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.3187.
- `private_text_semantic`: Top-2 experts [0, 3], weights [0.39076976533848096, 0.609230234661519]; shares [0.5, 0.0, 0.0, 0.5]; entropy 1.3619.
- `private_visual_physical`: Top-2 experts [1, 3], weights [0.15878090986283855, 0.8412190901371615]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.0286.
- `private_visual_semantic`: Top-2 experts [1, 3], weights [0.26730147985813063, 0.7326985201418693]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.2511.
- `collaborative_text_physical`: Top-2 experts [1, 2], weights [0.8904592750378667, 0.10954072496213332]; shares [0.0, 0.5, 0.5, 0.0]; entropy 0.8500.
- `collaborative_visual_physical`: Top-2 experts [1, 3], weights [0.31108459545070416, 0.6889154045492959]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.2252.

### Grocery, seed 43

Historical raw/scaled Balance: 1.302794 / 0.013028; View-Global: 1.360798 / 0.013608.

Node Conditioned expert routing by view:
- `private_text_physical`: selected-expert union [0, 1, 2, 3]; shares [0.14885206520557404, 0.3050544857978821, 0.12050486356019974, 0.42558860778808594]; entropy 0.9413.
- `private_text_semantic`: selected-expert union [0, 1, 2, 3]; shares [0.2082405984401703, 0.24510952830314636, 0.14735856652259827, 0.39929133653640747]; entropy 0.9370.
- `private_visual_physical`: selected-expert union [0, 1, 2, 3]; shares [0.14317090809345245, 0.32444068789482117, 0.13295069336891174, 0.3994377553462982]; entropy 0.8494.
- `private_visual_semantic`: selected-expert union [0, 1, 2, 3]; shares [0.11649291217327118, 0.35360783338546753, 0.15172192454338074, 0.37817734479904175]; entropy 0.8602.
- `collaborative_text_physical`: selected-expert union [0, 1, 2, 3]; shares [0.1798934042453766, 0.43200188875198364, 0.13207216560840607, 0.2560325562953949]; entropy 0.7422.
- `collaborative_visual_physical`: selected-expert union [0, 1, 2, 3]; shares [0.11072391271591187, 0.38907110691070557, 0.14006677269935608, 0.3601382374763489]; entropy 0.7826.

View Global expert routing by view:
- `private_text_physical`: Top-2 experts [2, 3], weights [0.2821237255952235, 0.7178762744047765]; shares [0.0, 0.0, 0.5, 0.5]; entropy 1.2785.
- `private_text_semantic`: Top-2 experts [2, 3], weights [0.34474271211965873, 0.6552572878803412]; shares [0.0, 0.0, 0.5, 0.5]; entropy 1.3304.
- `private_visual_physical`: Top-2 experts [1, 3], weights [0.19885485500483718, 0.8011451449951628]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.1233.
- `private_visual_semantic`: Top-2 experts [1, 3], weights [0.26585552955506625, 0.7341444704449338]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.2518.
- `collaborative_text_physical`: Top-2 experts [1, 3], weights [0.8793921877325199, 0.1206078122674801]; shares [0.0, 0.5, 0.0, 0.5]; entropy 0.8962.
- `collaborative_visual_physical`: Top-2 experts [1, 3], weights [0.38963250417222456, 0.6103674958277755]; shares [0.0, 0.5, 0.0, 0.5]; entropy 1.3539.

## Interpretation

**MIXED** — The paired Accuracy direction differs by dataset.

Balance Loss is coupled to route granularity because each View-Global call uses only two experts. This is a practical whole-router ablation, not a strict causal isolation of node adaptation from balancing. The transductive full-graph route mean uses unlabeled node features and does not establish inductive online behavior. No expert frequency, class, or modality specialization claim is made.

## Protocol and reproducibility

Five serial runs used physical GPU 1, fixed seed-42 splits and the existing V7B.1 K=8 cache. AdamW, 1e-3 LR, 1e-4 weight decay, 300-epoch cap, patience 30, minimum epoch 30, min delta 1e-4, clip 1.0, dropout 0.2, fusion dropout 0.2, late dropout 0.15, no scheduler, Private CE 0.3 and Balance Loss 0.01 were retained. Both Best-Accuracy and Best-F1 checkpoints were reloaded and verified. Test evaluation and Link Prediction were off. See `data/manifest.json`, `data/runs.csv`, and `data/router_summary.csv` for hashes and per-run provenance.
