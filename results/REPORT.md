# Prospective coding with speculative decoding: controlled prototype

This is a synthetic autoregressive neural-network experiment, not a pretrained LLM benchmark. It tests an inference-time prospective adapter, not teaching-signal synchronization during learning.

Observed: prospective compensation improved draft fidelity over the leaky control in both regimes. It did not improve fidelity or median runtime over the ordinary instantaneous drafter. This pilot provides no evidence of an added performance benefit from prospective coding.

Hardware/runtime: macOS-26.5.1-arm64-arm-64bit-Mach-O; device `cpu`; NumPy 2.5.5.
Model seeds: [17, 23, 41]; 5 independent sequence seeds per model and regime; 256 committed tokens per run; draft length 4.

## Measured results

| Context | Method | TV to target ↓ | Tokens / verification ↑ | ms / token ↓ | Speed vs target alone ↑ | Speed vs ordinary speculation ↑ |
|---|---|---:|---:|---:|---:|---:|
| smooth | target_only | 0.0000 | 1.000 | 0.111 | 1.000× | 1.465× |
| smooth | instant | 0.0591 | 4.374 | 0.163 | 0.683× | 1.000× |
| smooth | leaky | 0.1026 | 4.051 | 0.186 | 0.616× | 0.895× |
| smooth | prospective | 0.0597 | 4.439 | 0.166 | 0.670× | 0.968× |
| smooth | ideal | 0.0591 | 4.374 | 0.169 | 0.658× | 0.974× |
| abrupt | target_only | 0.0000 | 1.000 | 0.110 | 1.000× | 1.537× |
| abrupt | instant | 0.0778 | 4.215 | 0.166 | 0.650× | 1.000× |
| abrupt | leaky | 0.2807 | 2.801 | 0.256 | 0.435× | 0.668× |
| abrupt | prospective | 0.0816 | 4.224 | 0.175 | 0.629× | 0.960× |
| abrupt | ideal | 0.0778 | 4.215 | 0.175 | 0.632× | 0.957× |

Times are measured end to end, including CPU drafting, array operations, verification, sampling, and rollback. Speed is the median of paired target-only/runtime ratios. TV is total variation on identical held-out target-generated prefixes. Sequence repeats within each model share weights; the distinct model seeds are the independent model replications.

## Design and controls

* Target: a frozen, randomly initialized, nonlinear MLP over a persistent prefix memory and known clock features. It defines a valid synthetic autoregressive categorical distribution. It has not learned natural language.
* Draft: a 64-hidden-unit MLP distilled from target logits on a separate training set using ordinary Adam/MSE. Its tiny forward pass runs on the CPU using NumPy. The wider target runs on the reported device.
* `instant`: ordinary drafter without an integration delay.
* `leaky`: the same draft with a first-order low-pass transmission stage.
* `prospective`: the same draft with the finite-adaptation construction in Teaching Signal Synchronization, Eqs. 6–7.
* `ideal`: the matched derivative-compensation reference. It cancels the introduced lag and should recover the instantaneous drafter; equality is a sanity check, not a research gain.
* Persistent memory is separate from every transmission adapter. Smooth context retains 94% of the previous memory at each token. Abrupt context is replaced by the latest token embedding.
* Adapters evolve in continuous synthetic time with piecewise constant causal draft logits between token events; each interval is integrated analytically. This clock is a modeling device, not GPU wall time.
* All adapters initialize to the first current draft output; there is no selectively penalized initial transient.
* All model weights, time constants, and draft length are fixed before test evaluation. No selection of the best test setting is performed.

## Correctness

Each proposed token retains its actual normalized q distribution. Acceptance is min(1,p/q); the rejection correction is normalized max(p-q,0). Full target probabilities are computed on candidate prefixes in one batched neural forward pass. Rejected suffix state is restored; the correction or bonus token is consumed exactly once. Verification never uses a future true token.

Tests cover the analytic sampling identity, all-accept behavior, state rollback against replay, causal prefix construction, and the adapter solution against numerical integration. The sampling identity is exact in real arithmetic. This implementation preserves the target distribution up to floating-point differences between single-row and batched evaluation, not identical random-seed token sequences across different samplers.

## GPU interpretation

These results measure CPU execution only. The system reports an Apple M5 Pro GPU, but a direct Metal device probe returned unavailable in this execution session. The benchmark does not establish results for an NVIDIA data-center GPU, a Transformer, KV-cache traffic, large serving batches, or distributed training. The target is small and host/dispatch overhead can dominate.

A gain over `leaky` alone demonstrates lag compensation in a deliberately dynamical draft. A useful new GPU result would require beating `instant` in end-to-end runtime at comparable output distribution. The prospective construction has no automatic reason to beat a model that has no integration delay.

## Target batch microbenchmark

| Model seed | Single-position call (ms) | Block call (ms) |
|---|---:|---:|
| 17 | 0.094 | 0.581 |
| 23 | 0.100 | 0.608 |
| 41 | 0.086 | 0.589 |

Maximum observed single-row/batch probability discrepancy: 7.66e-07; all passed the 1e-5 tolerance. Ideal and instantaneous controls produced identical tokens and acceptance counts in every paired run.

These include array operations and probability conversion and are not standalone kernel timings. The block contains draft length + 1 positions. They are a diagnostic, not a speedup prediction.

## Reproduce

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python run_experiment.py --device auto --output cpu_rerun
```

Run `python run_experiment.py --help` for model size, seed, and sequence controls. Run configuration and all raw run statistics are in results.json. Saved draft weights are in draft_seed_*.npz. Target weights can be regenerated from the recorded seed, architecture, logit scale and NumPy version. Reruns may differ in floating-point behavior and timing.

## Primary references

* [Teaching signal synchronization in deep neural networks with prospective neurons](https://arxiv.org/abs/2511.14917)
* [A neuronal least-action principle for real-time learning in cortical circuits](https://pmc.ncbi.nlm.nih.gov/articles/PMC11661794/)
* [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
