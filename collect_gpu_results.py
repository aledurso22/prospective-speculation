"""Summarize completed measured GPU runs without mixing in partial checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect(root: Path):
    root = Path(root)
    paths = sorted(root.rglob('gpu_results.json'))
    if not paths:
        raise ValueError('No completed gpu_results.json files found. Preflight/checkpoint files are not timing results.')
    rows, metadata = [], []
    incomplete = []
    for checkpoint in root.rglob('gpu_checkpoint.json'):
        if not (checkpoint.parent / 'gpu_results.json').is_file():
            incomplete.append(str(checkpoint.parent.relative_to(root)))
    signatures = set()
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get('status') != 'complete':
            raise ValueError(f'{path} is not marked complete; refusing to summarize it as a completed run.')
        info = payload['metadata']
        if info['device'] not in ('cuda', 'mps') or not info.get('benchmark_executed'):
            raise ValueError(f'{path} is not a measured GPU benchmark.')
        provenance = info.get('source_provenance', {})
        signature = (info.get('device'), info.get('hardware'), info.get('torch_version'),
                     info.get('numpy_version'), info.get('teacher_width'), info.get('tokens'),
                     info.get('sequences'), info.get('repeats'), tuple(info.get('seeds', [])),
                     info.get('cuda_runtime_version'), info.get('tau'), info.get('tau_a'),
                     info.get('python_version'), info.get('platform'), info.get('cpu_model'),
                     info.get('torch_threads'), info.get('torch_interop_threads'),
                     json.dumps(info.get('precision', {}), sort_keys=True),
                     json.dumps(info.get('cpu_thread_environment', {}), sort_keys=True),
                     json.dumps(info.get('driver', {}), sort_keys=True),
                     json.dumps(info.get('source_code_sha256', {}), sort_keys=True),
                     provenance.get('source_results_sha256'), provenance.get('asset_manifest_sha256'),
                     provenance.get('runtime_model_source_sha256'))
        signatures.add(signature)
        key = (info['k'], signature)
        if key in seen:
            raise ValueError('Duplicate draft-length run; use a separate output root for independent sweeps.')
        seen.add(key)
        metadata.append(info)
        for original in payload['summary']:
            row = {'draft_length': info['k'], 'source': str(path.relative_to(root)), **original}
            rows.append(row)
    if len(signatures) != 1:
        raise ValueError('Runs use different devices, package versions, models, or test configurations; summarize them separately.')
    rows.sort(key=lambda row: (row['draft_length'], row['regime'], row['method']))
    return rows, metadata[0], incomplete


def write_summary(root: Path, rows, info, incomplete):
    columns = list(rows[0])
    with (root / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    text = [
        '# GPU experiment results', '',
        f"Device: {info['hardware']} (`{info['device']}`). PyTorch {info['torch_version']}; NumPy {info['numpy_version']}.",
        '',
        '**Primary comparison: prospective versus instant.** A speed ratio above 1 means faster than the ordinary speculative drafter. '
        'Inspect per-model results and timing dispersion before interpreting small differences.',
        '',
        '**Completion is an execution result, not a scientific pass.** The prespecified primary draft length is k=4; '
        'the other lengths are exploratory. See RESEARCH_REVIEW.md for the practical screening target and confirmation requirements.',
        '',
        'The small synthetic target runs on GPU; drafting, state management, sampling, and transfers remain in end-to-end timing. '
        'This is not a pretrained LLM or serving-throughput benchmark.', '',
        '| Draft length | Context | Method | ms/token | Tokens/verification | Speed vs ordinary speculation |',
        '|---:|---|---|---:|---:|---:|',
    ]
    for row in rows:
        text.append(f"| {row['draft_length']} | {row['regime']} | {row['method']} | "
                    f"{row['ms_per_token_median']:.4f} | {row['tokens_per_verification']:.3f} | "
                    f"{row['paired_speedup_vs_instant_median']:.3f}× |")
    text += ['', 'Ratios pair the same model, context, and timing repeat. Repeats of one context estimate timing noise; '
             'they do not increase the number of independently trained models. Each run directory retains numerical checks, '
             'raw observations, hardware metadata, and per-model summaries.']
    if incomplete:
        text += ['', '**Incomplete directories excluded:** ' + ', '.join(incomplete) + '. This is a partial sweep.']
    (root / 'SUMMARY.md').write_text('\n'.join(text) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='One run_gpu.sh output root with k*/gpu_results.json.')
    args = parser.parse_args()
    try:
        rows, info, incomplete = collect(args.root)
        write_summary(args.root, rows, info, incomplete)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Summarized {len(rows)} completed method/context rows in {args.root / 'SUMMARY.md'}")
    if incomplete:
        print('Excluded incomplete runs:', ', '.join(incomplete))


if __name__ == '__main__':
    main()
