# Implementation Audit — Intrinsic-Prior Skip

Base commit: `184deaaf12b21c9ee01d3328aef15d86bcc50346`
Branch: `exp/mrcge_v13_intrinsic_prior_skip_screen`
Parent source SHA256: `be9272ad25296927fcb95720776a7fcdaf3f54d00936c96387262159b3770c90`
Candidate source SHA256: `7bb6a52a0328bb56ff5909ac001463597f7e7c5280045930e8ebd7a8953ccf4e`

## State and initialization

The only new trainable state is `prior_skip_raw`, a zero-initialized scalar appended after the frozen parent constructor. It has one element and consumes no random draw. It is not a module, so the external classifier initialization order remains unchanged. The model inherits `src.models.mrcge_v13.Model` directly and adds no Two-Hop or Expert-Decoupled modules.

## Mechanical `forward()` comparison

The following unified diff is generated directly from `inspect.getsource` for the frozen parent and the independent subclass. All statements before the final fusion are copied unchanged. Computational changes are the concatenation of already-computed `h` as `h0_fused`, the zero-initialized scalar coefficient, and use of `fusion_skip(fused + lambda_prior * h0_fused)` while keeping the MLP on `fused`. Added remaining lines only record the scalar and no-bias RMS/cosine diagnostics or expose the scalar in `info`.

```diff
--- src/models/mrcge_v13.py:Model.forward
+++ src/models/mrcge_v13_intrinsic_prior_skip.py:Model.forward
@@ -231,8 +231,63 @@
                 }
         self._last_route_snapshot = route_snapshots
         fused = torch.cat((final["text"], final["visual"]), dim=-1)
-        z = self.fusion_norm(
-            self.fusion_skip(fused) + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused))))
+        h0_fused = torch.cat((h["text"], h["visual"]), dim=-1)
+        lambda_prior = self.lambda_prior.to(dtype=fused.dtype)
+        skip_input = fused + lambda_prior * h0_fused
+        mlp_output = self.fusion_linear2(
+            self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
+        )
+        z = self.fusion_norm(self.fusion_skip(skip_input) + mlp_output)
+        self._last_intrinsic_prior_skip = {
+            "prior_skip_raw": self.prior_skip_raw.detach(),
+            "lambda_prior": self.lambda_prior.detach(),
+        }
+        prior_diag = None
+        if return_details:
+            with torch.no_grad():
+                weight = self.fusion_skip.weight.detach()
+                old_skip_linear = F.linear(fused.detach(), weight, bias=None)
+                prior_linear = F.linear(
+                    lambda_prior.detach() * h0_fused.detach(), weight, bias=None
+                )
+                old_skip_rms = self._rms(old_skip_linear)
+                prior_rms = self._rms(prior_linear)
+                cosine_denominator = (
+                    old_skip_linear.float().norm() * prior_linear.float().norm()
+                )
+                cosine = (
+                    (old_skip_linear.float() * prior_linear.float()).sum()
+                    / cosine_denominator.clamp_min(self.eps)
+                ) if float(cosine_denominator) > 0.0 else old_skip_rms.new_zeros(())
+                prior_diag = {
+                    "prior_skip_raw": float(self.prior_skip_raw.detach().cpu()),
+                    "lambda_prior": float(self.lambda_prior.detach().cpu()),
+                    "h0_fused_rms": float(self._rms(h0_fused).cpu()),
+                    "fused_rms": float(self._rms(fused).cpu()),
+                    "fused_minus_h0_rms": float(self._rms(fused - h0_fused).cpu()),
+                    "fusion_skip_weight_fused_rms": float(old_skip_rms.cpu()),
+                    "prior_skip_linear_rms": float(prior_rms.cpu()),
+                    "fusion_mlp_rms": float(self._rms(mlp_output).cpu()),
+                    "prior_to_old_skip_rms_ratio": float(
+                        (prior_rms / old_skip_rms.clamp_min(self.eps)).cpu()
+                    ),
+                    "prior_to_old_skip_cosine": float(cosine.cpu()),
+                }
+            if self._capture_state_audit:
+                self._last_state_audit["intrinsic_prior_skip"] = {
+                    "h0_fused": h0_fused,
+                    "fused": fused,
+                    "skip_input": skip_input,
+                    "prior_skip_linear": prior_linear,
+                    "old_skip_linear": old_skip_linear,
+                    "mlp_output": mlp_output,
+                    "z": z,
+                }
+        mechanism_forward["intrinsic_prior_raw"] = self._diagnostic_scalar(
+            self.prior_skip_raw, return_details
+        )
+        mechanism_forward["intrinsic_prior_lambda"] = self._diagnostic_scalar(
+            self.lambda_prior, return_details
         )
         info: dict[str, Any] = {
             "condition": self.condition,
@@ -242,6 +297,8 @@
             "input_self_loops_removed": loop_count,
             "balance_loss": balance.detach(),
             "aux_loss": aux_loss.detach(),
+            "prior_skip_raw": self.prior_skip_raw.detach(),
+            "lambda_prior": self.lambda_prior.detach(),
             "unimodal_logits": {m: self.unimodal_heads[m](private[m]) for m in _MODALITIES},
             "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
             "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
@@ -289,5 +346,6 @@
                 "fused_output_rms": float(self._rms(z).cpu()),
                 "v13_mechanism": dict(mechanism_forward),
             }
+            info["details"]["intrinsic_prior_skip"] = prior_diag
         self._last_diagnostics = info.get("details", {})
         return z, None, None, aux_loss, info
```

## Formula audit

`lambda_prior = tanh(prior_skip_raw)` and `z = fusion_norm(fusion_skip(F + lambda_prior * H0) + fusion_linear2(fusion_dropout(GELU(fusion_linear1(F)))))`. The existing skip is called once, so its bias is applied once. `H0` is not detached, normalized, dropped out, or recomputed. No Router, Private, Collaborative, Late, Loss, or initialization path depends on lambda.

## Test evidence

- Dedicated CPU tests: 5 passed, including exact initialization/Eval parity, Train loss/shared-gradient/RNG parity, nonzero scalar gradient and optimizer inclusion, final-fusion invariants, checkpoint reload, and finite empty/isolated/zero-cross cases.
- Existing MR-CGE regressions: 73 passed across V1.1, V1.2, V1.3, and Two-Hop Cross.
- GPU parity and dataset forward/backward are separately required by the guarded GPU preflight and are recorded in `data/gpu_preflight.json`.
