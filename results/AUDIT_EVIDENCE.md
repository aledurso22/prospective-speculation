# Scientific evidence audit

**Correctness controls: PASS. Discovery evidence: not established.**

Prospective fidelity is worse in every paired context in these saved results.

Passing implementation tests checks the intended behavior in the tested cases. It does not establish that prospective coding improves GPU performance. This audit uses the saved CPU evidence and deterministic controls; it does not change the experiment to obtain a favorable result.

Paired contexts: 30. Prospective fidelity wins/losses/ties: 0/30/0. Comparison tolerance: 1e-12.

## Identical-prefix fidelity

| Regime | Contexts | Prospective wins / losses / ties | Mean TV increase | Instant expected acceptance | Prospective expected acceptance |
|---|---:|---|---:|---:|---:|
| abrupt | 15 | 0 / 15 / 0 | +0.00383930 | 0.922190 | 0.918350 |
| smooth | 15 | 0 / 15 / 0 | +0.00061845 | 0.940870 | 0.940252 |

For one proposal at the same prefix, expected acceptance is exactly `sum(min(p,q)) = 1 − TV(p,q)` in real arithmetic. The saved TV values are already averaged over common target-generated prefixes, so the identity also holds for these averages. This is not the free-running tokens-per-verification statistic: different samplers consume random numbers differently and visit different output prefixes.

## Model-level differences

| Regime | Model seed | Mean TV increase | Wins / losses / ties |
|---|---:|---:|---|
| abrupt | 17 | +0.00373401 | 0 / 5 / 0 |
| abrupt | 23 | +0.00434187 | 0 / 5 / 0 |
| abrupt | 41 | +0.00344203 | 0 / 5 / 0 |
| smooth | 17 | +0.00044546 | 0 / 5 / 0 |
| smooth | 23 | +0.00061100 | 0 / 5 / 0 |
| smooth | 41 | +0.00079890 | 0 / 5 / 0 |

Tokens within a sequence and contexts sharing model weights are correlated. No token-level p-values or claims of statistical independence are made.

## Perfect-drafter controls

Here the target is defined as the softmax of the raw draft logits. Ordinary instantaneous drafting is therefore already perfect. The smooth, alternating, and constant signals are deterministic and all adapters use the saved time constants. No parameter search is performed.

| Signal | Adapter | Mean TV | Expected acceptance |
|---|---|---:|---:|
| smooth_sinusoid | instant | 0.00000000 | 1.00000000 |
| smooth_sinusoid | leaky | 0.12213168 | 0.87786832 |
| smooth_sinusoid | prospective | 0.01081399 | 0.98918601 |
| smooth_sinusoid | ideal | 0.00000000 | 1.00000000 |
| abrupt_alternating | instant | 0.00000000 | 1.00000000 |
| abrupt_alternating | leaky | 0.25521028 | 0.74478972 |
| abrupt_alternating | prospective | 0.00427409 | 0.99572591 |
| abrupt_alternating | ideal | 0.00000000 | 1.00000000 |
| constant | instant | 0.00000000 | 1.00000000 |
| constant | leaky | 0.00000000 | 1.00000000 |
| constant | prospective | 0.00000000 | 1.00000000 |
| constant | ideal | 0.00000000 | 1.00000000 |

Hard guards passed: ideal and instantaneous probabilities agree; the instantaneous draft is perfect; prospective compensation never improves on that perfect reference beyond numerical tolerance. A benefit over the leaky control alone is repair of the introduced lag.

## What this means

The current inference adapter receives logits that the ordinary drafter has already computed. Ideal compensation recovers those logits; finite adaptation filters or extrapolates them. The implementation does not reproduce prospective learning, skip the draft network forward pass, or remove GPU layer dependencies. These negative controls do not rule out a differently trained prospective predictor, but the current results do not support a discovery claim.

A future performance claim must beat the ordinary speculative baseline in measured end-to-end GPU time with comparable correctness, fair implementation effort, and independently held-out tasks. A small synthetic MLP benchmark cannot establish performance for a production LLM.

The JSON companion contains each pair, model-level values, source/result hashes, parameters, and numerical guard outcomes.
