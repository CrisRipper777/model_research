# PIGPR-C1 design audit

## Provenance and scope

This audit is written before C1 implementation. The branch starts at PIGPR-C0 final SHA `66017c7a83af127d842e58238cd88cc2e0aa08ff`, with a clean worktree and no other experiment branch incorporated. C1 isolates direct GPR attribution on the existing PIGPR backbone. It does not add a MAG-specific module and does not use representation drift as a target or as an explanation for task quality.

## A. Direct and protected composition

C0's protected form is

\[
Z_{pre}=P+\lambda(G-P)=(1-\lambda)P+\lambda G.
\]

When `G` has unrestricted polynomial coefficients, this is not a simple function-class restriction. If `G=\sum_k d_k H_k`, the effective raw polynomial coefficients are `c=(1-\lambda)e_0+\lambda d`, where `e_0` selects `P=H_0`. For any target coefficient vector `c`, an interior choice such as `\lambda=1/2` and `d=2c-e_0` represents that same vector. Thus protected and direct forms can reach the same polynomial functions, while their parameterization and optimization paths differ. C1 interprets direct-versus-protected comparisons as tests of optimization and inductive bias, not as proof that the direct model has greater expressive capacity.

## B. Raw and anchored polynomial bases

For `H_k=\hat A^kP`, the anchored recurrence is `S_0=P` and `S_k=(1-\alpha)\hat A S_{k-1}+\alpha P`. At `\alpha=0.1` and `K=3`:

\[
\begin{bmatrix}S_0\\S_1\\S_2\\S_3\end{bmatrix}
=M\begin{bmatrix}H_0\\H_1\\H_2\\H_3\end{bmatrix},\qquad
M=\begin{bmatrix}
1&0&0&0\\
0.1&0.9&0&0\\
0.1&0.09&0.81&0\\
0.1&0.09&0.081&0.729
\end{bmatrix}.
\]

`M` is lower triangular with nonzero diagonal, hence invertible. For `G=\gamma^T S`, its equivalent raw coefficients are `c=M^T\gamma`. Conversely, every raw coefficient vector has an anchored representation by solving `M^T\gamma=c`. Therefore unrestricted RGD and AGD represent the same polynomial function family. Their comparison tests basis-dependent optimization and implicit regularization, not a difference in polynomial capacity.

The CoSI restart prior is constructed from `restart=0.15`, `order=2`: `c_prior=[0.15, 0.1275, 0.7225, 0]`. Solving `M^T gamma_prior=c_prior` gives approximately `[0.05555556, 0.05246914, 0.89197531, 0]`. This construction will be tested against the recurrence and used to initialize RFD, RGD, and AGD consistently.

## C. Attribution and interpretation boundary

C1 compares protected versus direct uniform composition, adding the intrinsic order-0 state, a fixed PPR-informed profile, learned raw coefficients, and raw versus anchored coefficient bases. The principal questions are whether task-adaptive raw GPR improves over its fixed prior and whether anchored coordinates change optimization outcomes for an equivalent polynomial family. The analysis will describe validation metrics, learned coefficients, actual term contributions, state similarities, and checkpoint interventions. It will not argue that lower semantic drift is inherently better.

All task comparisons use the fixed NC splits and validation-selected checkpoints. Same-seed repeats estimate execution noise only. No pseudo-IID p-values, NC test results, formal LP results, or claims beyond these fixed-split validation experiments will be reported.
