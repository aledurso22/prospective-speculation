# Critical review before the GPU experiment

Reviewed 23 November 2026. **Ready for a guarded cluster experiment is not the same as evidence of a GPU discovery.** No CUDA execution has been validated in the development environment.

## What the evidence currently says

The saved CPU experiment contains 30 paired contexts: three trained model seeds, five sequence seeds, and two regimes. On identical target-generated prefixes, prospective correction increases mean total variation (TV) distance from the target in **all 30 contexts**. Mean TV increases are approximately 0.000618 in the smooth regime and 0.003839 in the abrupt regime. These are small but consistently unfavorable differences.

For a single speculative proposal at a given prefix, expected acceptance is `sum(min(p, q)) = 1 - TV(p, q)`. Consequently those paired fidelity measurements favor the ordinary drafter. Free-running tokens per verification can still vary the other way because different methods consume random draws differently and follow different token paths. The two measurements answer different questions; free-running counts do not overturn the paired fidelity result.

`audit_evidence.py` reproduces this analysis from the saved observations and tests perfect-draft controls. Its exit status checks data integrity and mathematical invariants; a successful exit does **not** mean the proposed improvement is supported. See `results/AUDIT_EVIDENCE.md` and `results/audit_results.json`.

The original CPU timings favored ordinary speculation over prospective correction, and target-only generation over both. Those timings are historical: this audit strengthened the ordinary baseline by removing stateless adapter snapshots and unused bonus-prefix draft evaluations. The original results and model assets remain unchanged. New timing claims must use the revised code; the GPU runner fingerprints it and the collector refuses to pool incompatible versions.

## What is actually implemented

The draft is an ordinarily distilled, feedforward synthetic model. The intervention is a causal filter on its already-computed logits. It does not implement prospective teaching signals, NLA plasticity, or prospective learning of a recurrent drafter.

With equilibrium initialization, the ideal control's residual obeys `u[n]-f[n] = tau/(tau+dt) * (u[n-1]-f[n-1])`, so it is identically zero. This control recovers the ordinary draft; it does not create a better prediction. The finite-adaptation version has approximately 6% overshoot on a step at the default parameters. A perfect ordinary draft already has expected acceptance one, so changing its probabilities can only tie or worsen that control. This does not rule out useful filtering of an imperfect drafter, but such a benefit needs evidence.

The ordinary baseline now avoids work that a stateless draft does not need. Stateful adapters retain their necessary rollback and committed-prefix updates. This strengthens this Python implementation's comparison; it does not turn the baseline into an optimized production serving engine.

## Why a GPU could change timing, but cannot guarantee success

Batching candidate prefixes may reduce target evaluation cost per committed token. For a method, the relevant measured quantity is approximately

`time/token = (draft work + target verification + sampling/transfers/state work) / committed tokens`.

The prospective method must improve the denominator enough, or reduce numerator work enough, to cover its additional processing. Accelerating the common target can also make CPU drafting and synchronization more prominent. No timing outcome follows from the name of the mechanism or from a higher proposal count alone.

This target has about 17 MB of float32 weights, potentially small enough for GPU cache. The benchmark measures one synthetic request with a CPU draft and CPU sampling. It cannot establish large-LLM memory-bandwidth gains, KV-cache benefits, multi-request throughput, or benefits for GPUs in general.

## Decide success before inspecting GPU results

1. **Execution gate:** the test suite, immutable asset checks, actual CUDA numerical preflight, count invariants, ideal-control equality, and repeat reproducibility must pass. Failure leaves an incomplete result, not a speedup report.
2. **Primary comparison:** use the existing `k=4` setting, prospective versus ordinary (`instant`) speculation, separately in both regimes and for all three models. Report target-only performance too. Keep all five sequences and all timing repeats; repeats are not independent trained models.
3. **Practical screening target:** require at least **1.10x** median paired speed versus ordinary speculation in every model/regime group, and improvement over target-only generation in those groups. This 10% margin is a chosen engineering screening target, not a significance test or a definition of scientific discovery. Report exact effects even when the screen fails.
4. **Exploration:** report all `k=1,2,4,8` results. A best setting selected after seeing this sweep needs a separate confirmation dataset; do not relabel it a prespecified primary result. Each method ultimately deserves comparable tuning on separate validation data.
5. **Confirmation:** repeat promising results in fresh jobs, use new held-out contexts and more independently trained models, and report uncertainty at model and context levels. Preserve failures and slow runs unless a documented, method-independent hardware fault invalidates the entire affected comparison. The present three-model sample is small.
6. **Discovery claim:** only consider it after matched trained drafters and real language-model targets show a repeatable end-to-end benefit against competitive speculative decoders, with equal output-quality guarantees, relevant batch sizes, and the claimed GPU configurations. A successful toy smoke run or one favorable device is insufficient.

Failure of the inference filter is a reason to reject or revise this particular mechanism. It is not a disproof of prospective learning in the original papers. Do not tune parameters on the reported test set just to force a pass.

## Relevant prior work and a better research question

[Leviathan et al. (2023)](https://proceedings.mlr.press/v202/leviathan23a.html) already establish exact speculative sampling with parallel verification. [EAGLE (2024)](https://proceedings.mlr.press/v235/li24bt.html) uses learned feature extrapolation; [EAGLE-3 (2025)](https://arxiv.org/abs/2503.01840) changes that design to direct token prediction with fused multi-layer features. Predicting cheap candidates for verification is established prior work.

[Rawat et al., version 2, 7 November 2026](https://arxiv.org/pdf/2609.04134v2) study prospective input in trained recurrent networks and diagonal state-space models. Their two-tap input correction preserves the recurrent transition and parallel scan. Their [associated repository](https://github.com/Sequel-Institute/prospective-rqf) documents fused Triton CUDA kernels. This is relevant prior work, not evidence that our frozen output filter accelerates speculative decoding. Scan parallelism also assumes inputs are available; future sampled tokens remain unknown during autoregressive drafting.

A more substantive next hypothesis is: **can a trained prospective recurrent drafter yield more accepted tokens per unit drafting cost than a matched non-prospective drafter?** Test that against equal parameter/training budgets and ordinary filtering/extrapolation controls, then measure end-to-end GPU decoding. This is a research direction, not an established novel result; the literature check is not exhaustive.
