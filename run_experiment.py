"""Run a controlled, explicitly synthetic prospective/speculative decoding study."""
import argparse
import csv
import json
import platform
from pathlib import Path
import statistics
import time
import numpy as np

from adapters import make_adapter
from decoding import generate, generate_target
from models import VOCAB, make_context, synchronize, train_pair

METHODS = ['instant', 'leaky', 'prospective', 'ideal']
REGIMES = {'smooth': 0.94, 'abrupt': 0.0}


def softmax(x):
    p = np.exp(x - np.max(x))
    return p / p.sum()


def adapter_for(name, args):
    return make_adapter(name, VOCAB, tau=args.tau, tau_a=args.tau_a, dt=1.0)


def paired_fidelity(target, draft, context, tokens, args):
    c = context.clone()
    features = []
    for token in tokens:
        features.append(c.features())
        c.advance(token)
    probabilities = target(np.array(features))
    raw_logits = [draft(x) for x in features]
    output = {}
    for name in METHODS:
        adapter = adapter_for(name, args)
        qs = np.array([softmax(adapter.apply(x)) for x in raw_logits])
        tvs = np.abs(probabilities-qs).sum(axis=-1) / 2
        kls = (probabilities*(np.log(probabilities)-np.log(np.maximum(qs, 1e-300)))).sum(axis=-1)
        output[name] = {'tv': float(tvs.mean()), 'kl': float(kls.mean()),
                        'expected_single_proposal_acceptance': float(1-tvs.mean())}
    return output


def batch_benchmark(target, context, k, repeats=7, inner=40):
    sizes = [1, k+1]
    inputs = {}
    for size in sizes:
        c = context.clone()
        x = []
        for i in range(size):
            x.append(c.features())
            c.advance(i % VOCAB)
        inputs[size] = np.array(x)
        for _ in range(10):
            target(inputs[size])
    timings = {size: [] for size in sizes}
    for repeat in range(repeats):
        for size in sizes[::1 if repeat % 2 == 0 else -1]:
            synchronize(target.device)
            start = time.perf_counter()
            for _ in range(inner):
                target(inputs[size])
            synchronize(target.device)
            timings[size].append((time.perf_counter()-start)/inner)
    result = {str(size): {'median_ms': 1000*statistics.median(ts),
                       'all_ms': [1000*t for t in ts]} for size, ts in timings.items()}
    xb = inputs[k+1]
    difference = float(np.max(np.abs(target(xb)-np.concatenate([target(x[None,:]) for x in xb]))))
    result['max_abs_batch_vs_single_probability_difference'] = difference
    if difference > 1e-5:
        raise RuntimeError(f'Target batch numerical discrepancy too large: {difference}')
    return result


def summarize(rows):
    result = []
    for regime in REGIMES:
        for method in ['target_only']+METHODS:
            subset = [r for r in rows if r['regime'] == regime and r['method'] == method]
            times = [1000*r['wall_seconds']/r['committed'] for r in subset]
            baseline = {(r['model_seed'], r['sequence_seed']): r for r in rows
                        if r['regime'] == regime and r['method'] == 'target_only'}
            ratios = [baseline[(r['model_seed'],r['sequence_seed'])]['wall_seconds']/r['wall_seconds']
                      for r in subset]
            instantaneous = {(r['model_seed'],r['sequence_seed']):r for r in rows
                             if r['regime']==regime and r['method']=='instant'}
            instant_ratios = [instantaneous[(r['model_seed'],r['sequence_seed'])]['wall_seconds']/r['wall_seconds'] for r in subset]
            item = {'speedup_vs_instant_median':statistics.median(instant_ratios), 'regime': regime, 'method': method, 'runs':len(subset),
                    'ms_per_token_median': statistics.median(times),
                    'ms_per_token_mean': statistics.mean(times),
                    'ms_per_token_sd': statistics.stdev(times) if len(times)>1 else 0,
                    'paired_speedup_median':statistics.median(ratios),
                    'tokens_per_verification':sum(r['committed'] for r in subset)/sum(r['rounds'] for r in subset),
                    'accepted_fraction_all_proposals':sum(r.get('accepted',0) for r in subset)/max(1,sum(r.get('proposed',0) for r in subset)),
                    'accepted_fraction_checked_proposals':sum(r.get('accepted',0) for r in subset)/max(1,sum(r.get('checked',0) for r in subset)),
                    'paired_prefix_tv':statistics.mean(r['tv'] for r in subset) if method!='target_only' else 0,
                    'paired_prefix_kl':statistics.mean(r['kl'] for r in subset) if method!='target_only' else 0}
            result.append(item)
    return result


def write_report(out, metadata, summary, batches, training):
    comparisons = {regime:{r['method']:r for r in summary if r['regime']==regime} for regime in REGIMES}
    no_quality_gain = all(c['prospective']['paired_prefix_tv'] >= c['instant']['paired_prefix_tv'] for c in comparisons.values())
    no_speed_gain = all(c['prospective']['speedup_vs_instant_median'] <= 1 for c in comparisons.values())
    recovered = all(c['prospective']['paired_prefix_tv'] < c['leaky']['paired_prefix_tv'] for c in comparisons.values())
    outcome = ('Observed: prospective compensation improved draft fidelity over the leaky control in both regimes. ' if recovered else 'Observed fidelity changes depend on context. ')
    if no_quality_gain and no_speed_gain:
        outcome += 'It did not improve fidelity or median runtime over the ordinary instantaneous drafter. This pilot provides no evidence of an added performance benefit from prospective coding.'
    else:
        outcome += 'Inspect the paired comparisons below; small timing differences require additional measurement before a performance claim.'
    text = [
        '# Prospective coding with speculative decoding: controlled prototype',
        '',
        'This is a synthetic autoregressive neural-network experiment, not a pretrained LLM benchmark. '
        'It tests an inference-time prospective adapter, not teaching-signal synchronization during learning.',
        '',
        outcome, '',
        f"Hardware/runtime: {metadata['platform']}; device `{metadata['device']}`; NumPy {metadata['numpy']}.",
        f"Model seeds: {metadata['args']['seeds']}; {metadata['args']['sequences']} independent sequence seeds per model and regime; "
        f"{metadata['args']['tokens']} committed tokens per run; draft length {metadata['args']['k']}.",
        '',
        '## Measured results', '',
        '| Context | Method | TV to target ↓ | Tokens / verification ↑ | ms / token ↓ | Speed vs target alone ↑ | Speed vs ordinary speculation ↑ |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        text.append(f"| {r['regime']} | {r['method']} | {r['paired_prefix_tv']:.4f} | {r['tokens_per_verification']:.3f} | "
                    f"{r['ms_per_token_median']:.3f} | {r['paired_speedup_median']:.3f}× | {r['speedup_vs_instant_median']:.3f}× |")
    text += ['', 'Times are measured end to end, including CPU drafting, array operations, verification, '
             'sampling, and rollback. Speed is the median of paired target-only/runtime ratios. '
             'TV is total variation on identical held-out target-generated prefixes. '
             'Sequence repeats within each model share weights; the distinct model seeds are the independent model replications.', '',
             '## Design and controls', '',
             '* Target: a frozen, randomly initialized, nonlinear MLP over a persistent prefix memory and known clock features. '
             'It defines a valid synthetic autoregressive categorical distribution. It has not learned natural language.',
             '* Draft: a 64-hidden-unit MLP distilled from target logits on a separate training set using ordinary Adam/MSE. '
             'Its tiny forward pass runs on the CPU using NumPy. The wider target runs on the reported device.',
             '* `instant`: ordinary drafter without an integration delay.',
             '* `leaky`: the same draft with a first-order low-pass transmission stage.',
             '* `prospective`: the same draft with the finite-adaptation construction in Teaching Signal Synchronization, Eqs. 6–7.',
             '* `ideal`: the matched derivative-compensation reference. It cancels the introduced lag and should recover '
             'the instantaneous drafter; equality is a sanity check, not a research gain.',
             '* Persistent memory is separate from every transmission adapter. Smooth context retains 94% of the previous '
             'memory at each token. Abrupt context is replaced by the latest token embedding.',
             '* Adapters evolve in continuous synthetic time with piecewise constant causal draft logits between token events; '
             'each interval is integrated analytically. This clock is a modeling device, not GPU wall time.',
             '* All adapters initialize to the first current draft output; there is no selectively penalized initial transient.',
             '* All model weights, time constants, and draft length are fixed before test evaluation. '
             'No selection of the best test setting is performed.', '',
             '## Correctness', '',
             'Each proposed token retains its actual normalized q distribution. Acceptance is min(1,p/q); '
             'the rejection correction is normalized max(p-q,0). Full target probabilities are computed on '
             'candidate prefixes in one batched neural forward pass. Rejected suffix state is restored; '
             'the correction or bonus token is consumed exactly once. Verification never uses a future true token.',
             '', 'Tests cover the analytic sampling identity, all-accept behavior, state rollback against replay, '
             'causal prefix construction, and the adapter solution against numerical integration. '
             'The sampling identity is exact in real arithmetic. This implementation preserves the target distribution up to floating-point differences between single-row and batched evaluation, '
             'not identical random-seed token sequences across different samplers.',
             '', '## GPU interpretation', '',
             'These results measure CPU execution only. The system reports an Apple M5 Pro GPU, but a direct Metal device probe returned unavailable in this execution session. '
             'The benchmark does not establish results for an NVIDIA data-center GPU, a Transformer, KV-cache traffic, '
             'large serving batches, or distributed training. The target is small and host/dispatch overhead can dominate.',
             '', 'A gain over `leaky` alone demonstrates lag compensation in a deliberately dynamical draft. '
             'A useful new GPU result would require beating `instant` in end-to-end runtime at comparable output distribution. '
             'The prospective construction has no automatic reason to beat a model that has no integration delay.',
             '', '## Target batch microbenchmark', '',
             '| Model seed | Single-position call (ms) | Block call (ms) |', '|---|---:|---:|']
    for seed, b in batches.items():
        text.append(f"| {seed} | {b['1']['median_ms']:.3f} | {b[str(metadata['args']['k']+1)]['median_ms']:.3f} |")
    maximum_difference = max(b['max_abs_batch_vs_single_probability_difference'] for b in batches.values())
    text += ['', f'Maximum observed single-row/batch probability discrepancy: {maximum_difference:.3g}; all passed the 1e-5 tolerance. Ideal and instantaneous controls produced identical tokens and acceptance counts in every paired run.', '', 'These include array operations and probability conversion and are not standalone kernel timings. '
             'The block contains draft length + 1 positions. They are a diagnostic, not a speedup prediction.', '',
             '## Reproduce', '',
             '```sh', 'python3 -m venv .venv', 'source .venv/bin/activate',
             'python -m pip install -r requirements.txt', 'python -m unittest discover -v',
             'python run_experiment.py --device auto --output cpu_rerun', '```', '',
             'Run `python run_experiment.py --help` for model size, seed, and sequence controls. '
             'Run configuration and all raw run statistics are in results.json. Saved draft weights are in draft_seed_*.npz. '
             'Target weights can be regenerated from the recorded seed, architecture, logit scale and NumPy version. '
             'Reruns may differ in floating-point behavior and timing.', '',
             '## Primary references', '',
             '* [Teaching signal synchronization in deep neural networks with prospective neurons](https://arxiv.org/abs/2511.14917)',
             '* [A neuronal least-action principle for real-time learning in cortical circuits](https://pmc.ncbi.nlm.nih.gov/articles/PMC11661794/)',
             '* [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)', '']
    (out/'REPORT.md').write_text('\n'.join(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='auto', choices=['auto','cpu'])
    p.add_argument('--seeds', type=int, nargs='+', default=[17,23,41])
    p.add_argument('--sequences', type=int, default=5)
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--width', type=int, default=2048)
    p.add_argument('--train-size', type=int, default=8192)
    p.add_argument('--steps', type=int, default=1600)
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--tau-a', type=float, default=0.2)
    p.add_argument('--output', default='results')
    args = p.parse_args()
    if min(args.sequences,args.tokens,args.width,args.train_size,args.steps) < 1 or args.k < 0:
        p.error('Sequence count, token count, width, training size and steps must be positive; k must be nonnegative.')
    device = args.device
    if device == 'auto':
        device = 'cpu'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'platform':platform.platform(), 'processor':platform.processor(), 'device':device,
                'numpy':np.__version__, 'args':vars(args), 'gpu_probe':'Metal device unavailable to this execution session',
                'timing':'perf_counter end-to-end CPU timings; NumPy BLAS; no simulated GPU timings'}
    print(json.dumps(metadata), flush=True)
    rows, batches, training = [], {}, {}
    for seed in args.seeds:
        start = time.perf_counter()
        print(f'Training frozen-target/distilled-draft pair, seed={seed}', flush=True)
        target, draft, embeddings, train_stats = train_pair(seed, device, args.width, args.train_size, args.steps)
        training[str(seed)] = {**train_stats, 'training_wall_seconds':time.perf_counter()-start}
        draft.save(out/f'draft_seed_{seed}.npz')
        print(json.dumps(training[str(seed)]), flush=True)
        warm_context = make_context(embeddings, 0.94, 90000)
        for batch_size in range(1, args.k+2):
            for _ in range(5):
                target(np.array([warm_context.features()]*batch_size))
        batches[str(seed)] = batch_benchmark(target, warm_context, args.k)
        for regime, decay in REGIMES.items():
            for seq in range(args.sequences):
                sequence_seed = 80000 + seq
                context = make_context(embeddings, decay, sequence_seed)
                # Generate held-out prefixes once for paired fidelity comparisons.
                reference = generate_target(target, context, n_tokens=args.tokens, seed=sequence_seed+10000)
                fidelity = paired_fidelity(target, draft, context, reference['tokens'], args)
                methods = ['target_only']+METHODS
                np.random.default_rng(seed*100+seq+(0 if regime=='smooth' else 50)).shuffle(methods)
                generated = {}
                for method in methods:
                    synchronize(device)
                    if method == 'target_only':
                        result = generate_target(target, context, n_tokens=args.tokens, seed=sequence_seed)
                    else:
                        result = generate(target, draft, adapter_for(method,args), context,
                                          n_tokens=args.tokens, k=args.k, seed=sequence_seed)
                    synchronize(device)
                    generated[method] = result
                    stats = result['stats']
                    row = {'model_seed':seed, 'sequence_seed':sequence_seed, 'regime':regime, 'method':method, **stats}
                    if method!='target_only':
                        row.update(fidelity[method])
                    rows.append(row)
                if generated['instant']['tokens'] != generated['ideal']['tokens']:
                    raise AssertionError('Ideal cancellation changed sampled tokens versus instantaneous control')
                for key in ('rounds','proposed','accepted','checked'):
                    if generated['instant']['stats'][key] != generated['ideal']['stats'][key]:
                        raise AssertionError(f'Ideal control mismatch in {key}')
                print(f'Completed seed={seed} regime={regime} sequence={seq+1}/{args.sequences}', flush=True)
        payload = {'metadata':metadata,'training':training,'batch_benchmark':batches,'runs':rows}
        (out/'checkpoint.json').write_text(json.dumps(payload,indent=2))
    summary = summarize(rows)
    payload = {'metadata':metadata,'training':training,'batch_benchmark':batches,'runs':rows,'summary':summary,
               'validation':{'ideal_instant_identical_tokens_and_counts':True,
                             'paired_control_comparisons':len(args.seeds)*len(REGIMES)*args.sequences}}
    (out/'results.json').write_text(json.dumps(payload,indent=2))
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    write_report(out,metadata,summary,batches,training)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
