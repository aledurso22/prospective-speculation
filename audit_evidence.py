"""Audit scientific evidence separately from implementation correctness.

This script does not retrain, tune parameters, modify existing results, or test
GPU speed. It pairs the saved CPU fidelity measurements and runs deterministic
controls in which the unfiltered draft distribution is already the target.

    python audit_evidence.py --results results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from adapters import make_adapter


METHODS = ("instant", "leaky", "prospective", "ideal")
SCHEMA = "prospective_speculation.evidence_audit"


def _tolerance(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Tolerance must be finite and nonnegative.")
    return value


def _tv(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Saved TV must be a number.")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Saved TV must be finite and lie in [0, 1].")
    return value


def _aggregate(pairs, tolerance):
    deltas = [row["tv_prospective_minus_instant"] for row in pairs]
    instant = [row["instant_tv"] for row in pairs]
    prospective = [row["prospective_tv"] for row in pairs]
    return {
        "paired_contexts": len(pairs),
        "model_seeds": len({row["model_seed"] for row in pairs}),
        "prospective_fidelity_wins": sum(delta < -tolerance for delta in deltas),
        "prospective_fidelity_losses": sum(delta > tolerance for delta in deltas),
        "fidelity_ties": sum(abs(delta) <= tolerance for delta in deltas),
        "mean_tv_prospective_minus_instant": statistics.mean(deltas),
        "min_tv_prospective_minus_instant": min(deltas),
        "max_tv_prospective_minus_instant": max(deltas),
        "mean_instant_tv": statistics.mean(instant),
        "mean_prospective_tv": statistics.mean(prospective),
        "mean_instant_expected_single_proposal_acceptance": 1 - statistics.mean(instant),
        "mean_prospective_expected_single_proposal_acceptance": 1 - statistics.mean(prospective),
        "mean_expected_acceptance_prospective_minus_instant": -statistics.mean(deltas),
    }


def aggregate_pairs(rows, tolerance=1e-12):
    """Pair identical held-out contexts, without treating tokens as replicates.

    The input is the CPU experiment's ``runs`` list. Its ``tv`` measurements
    were computed on one common target-generated prefix sequence per context,
    not on each decoder's different free-running sampled output.
    """
    tolerance = _tolerance(tolerance)
    groups = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each saved run must be a JSON object.")
        method = row.get("method")
        if method not in ("instant", "prospective"):
            continue
        try:
            key = (row["model_seed"], row["regime"], row["sequence_seed"])
            tv = _tv(row["tv"])
        except KeyError as error:
            raise ValueError(f"Missing paired fidelity field: {error}") from error
        if (isinstance(key[0], bool) or not isinstance(key[0], int)
                or not isinstance(key[1], str) or not key[1]
                or isinstance(key[2], bool) or not isinstance(key[2], int)):
            raise ValueError("Pair keys must contain integer model/sequence seeds and a regime name.")
        group = groups.setdefault(key, {})
        if method in group:
            raise ValueError(f"Duplicate {method} fidelity row for context {key}; repeats are not new contexts.")
        group[method] = (tv, row)
    if not groups:
        raise ValueError("No instantaneous/prospective fidelity pairs were found.")
    pairs = []
    for key, group in sorted(groups.items()):
        if set(group) != {"instant", "prospective"}:
            raise ValueError(f"Incomplete instantaneous/prospective pair for context {key}.")
        instant, instant_row = group["instant"]
        prospective, prospective_row = group["prospective"]
        if ("committed" in instant_row and "committed" in prospective_row
                and instant_row["committed"] != prospective_row["committed"]):
            raise ValueError(f"Paired runs have different committed-token counts for {key}.")
        delta = prospective - instant
        pairs.append({
            "model_seed": key[0], "regime": key[1], "sequence_seed": key[2],
            "instant_tv": instant, "prospective_tv": prospective,
            "tv_prospective_minus_instant": delta,
            "instant_expected_single_proposal_acceptance": 1 - instant,
            "prospective_expected_single_proposal_acceptance": 1 - prospective,
            "expected_acceptance_prospective_minus_instant": -delta,
            "outcome": "prospective_win" if delta < -tolerance else (
                "prospective_loss" if delta > tolerance else "tie"),
        })
    regimes = sorted({row["regime"] for row in pairs})
    models = []
    for regime in regimes:
        for seed in sorted({row["model_seed"] for row in pairs if row["regime"] == regime}):
            selected = [row for row in pairs if row["regime"] == regime and row["model_seed"] == seed]
            models.append({"regime": regime, "model_seed": seed, **_aggregate(selected, tolerance)})
    return {
        "comparison_tolerance": tolerance,
        "direction": "Positive TV delta means prospective is worse; negative means better.",
        "acceptance_identity": "For a single proposal at a fixed prefix, E[accept]=sum_x min(p_x,q_x)=1-TV(p,q).",
        "replication": "Context-level averages are paired within models. Tokens and timing repeats are not independent model samples; no p-values are computed.",
        "overall": _aggregate(pairs, tolerance),
        "by_regime": [{"regime": regime, **_aggregate([row for row in pairs if row["regime"] == regime], tolerance)}
                      for regime in regimes],
        "by_model_and_regime": models,
        "paired_contexts": pairs,
    }


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    weights = np.exp(values - values.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def perfect_drafter_controls(steps=512, tau=2.0, tau_a=0.2, dt=1.0, tolerance=1e-12):
    """Controlled counterexamples, with no model fitting or stochastic sampling."""
    tolerance = _tolerance(tolerance)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("At least two control steps are required.")
    t = np.arange(steps)
    sequences = {
        "smooth_sinusoid": np.stack((2*np.sin(0.3*t), -2*np.sin(0.3*t)), axis=1),
        "abrupt_alternating": np.stack((2*np.where(t % 2 == 0, 1., -1.),
                                         -2*np.where(t % 2 == 0, 1., -1.)), axis=1),
        "constant": np.tile(np.array([1., -1.]), (steps, 1)),
    }
    output = []
    for scenario, inputs in sequences.items():
        target = _softmax(inputs)
        probabilities, tvs = {}, {}
        for method in METHODS:
            adapter = make_adapter(method, 2, tau=tau, tau_a=tau_a, dt=dt)
            q = _softmax(np.array([adapter.apply(row) for row in inputs]))
            if not np.isfinite(q).all() or not np.allclose(q.sum(axis=1), 1, atol=tolerance, rtol=0):
                raise AssertionError(f"Invalid adapter probabilities in {scenario}/{method}.")
            probabilities[method] = q
            tvs[method] = np.abs(target-q).sum(axis=1)/2
        ideal_difference = float(np.max(np.abs(probabilities["ideal"]-probabilities["instant"])))
        if ideal_difference > tolerance:
            raise AssertionError(f"Ideal adapter differs from instantaneous control in {scenario}.")
        if float(tvs["instant"].max()) > tolerance:
            raise AssertionError(f"The deliberately perfect instantaneous draft is not perfect in {scenario}.")
        if np.any(tvs["prospective"] < tvs["instant"]-tolerance):
            raise AssertionError(f"Prospective appears to beat a perfect instantaneous draft in {scenario}.")
        output.append({
            "scenario": scenario, "steps": steps,
            "target_definition": "p_t=softmax(f_t); the raw instantaneous draft is exactly the target.",
            "guards": {"passed": True, "ideal_instant_max_abs_probability_difference": ideal_difference,
                       "instant_is_perfect": True, "prospective_does_not_beat_perfect_instant": True},
            "methods": [{"method": method, "mean_tv": float(tvs[method].mean()),
                         "max_tv": float(tvs[method].max()),
                         "expected_single_proposal_acceptance": float(1-tvs[method].mean())}
                        for method in METHODS],
        })
    return {"parameters": {"tau":tau,"tau_a":tau_a,"dt":dt,"tolerance":tolerance},
            "results": output,
            "interpretation": "Lag compensation can repair an introduced filter while degrading an already correct ordinary drafter. This counterexample disproves a universal improvement claim, not every possible prospective architecture."}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(results_dir, tolerance=1e-12):
    directory = Path(results_dir)
    source_path = directory / "results.json"
    source = json.loads(source_path.read_text())
    if source.get("metadata", {}).get("device") != "cpu":
        raise ValueError("This audit expects the saved CPU experiment's paired fidelity results.")
    pairs = aggregate_pairs(source["runs"], tolerance)
    args = source["metadata"]["args"]
    controls = perfect_drafter_controls(tau=args["tau"], tau_a=args["tau_a"], tolerance=tolerance)
    overall = pairs["overall"]
    if overall["prospective_fidelity_wins"] == 0 and overall["fidelity_ties"] == 0:
        finding = "Prospective fidelity is worse in every paired context in these saved results."
    elif overall["prospective_fidelity_losses"] == 0 and overall["fidelity_ties"] == 0:
        finding = "Prospective fidelity is better in every paired context in these saved results; this alone does not establish a GPU benefit."
    else:
        finding = "Paired fidelity outcomes are mixed or tied; inspect the context and model-level results."
    return {
        "schema":SCHEMA,"version":1,
        # The source lives beside the audit outputs; retain a portable label.
        "source":{"path":source_path.name,"sha256":_sha256(source_path),
                  "numpy_version":np.__version__,"audit_code_sha256":_sha256(__file__),
                  "adapters_code_sha256":_sha256(Path(__file__).with_name("adapters.py"))},
        "correctness_controls_status":"PASS",
        "discovery_status":"NOT_ESTABLISHED",
        "finding":finding,
        "scope":"No GPU measurement, retraining, hyperparameter tuning, or significance test is performed by this audit.",
        "paired_saved_evidence":pairs,"perfect_drafter_controls":controls,
    }


def render_markdown(result):
    evidence = result["paired_saved_evidence"]
    overall = evidence["overall"]
    lines = [
        '# Scientific evidence audit', '',
        '**Correctness controls: PASS. Discovery evidence: not established.**', '',
        result["finding"], '',
        'Passing implementation tests checks the intended behavior in the tested cases. '
        'It does not establish that prospective coding improves GPU performance. '
        'This audit uses the saved CPU evidence and deterministic controls; it does not change the experiment to obtain a favorable result.', '',
        f"Paired contexts: {overall['paired_contexts']}. Prospective fidelity wins/losses/ties: "
        f"{overall['prospective_fidelity_wins']}/{overall['prospective_fidelity_losses']}/{overall['fidelity_ties']}. "
        f"Comparison tolerance: {evidence['comparison_tolerance']:g}.", '',
        '## Identical-prefix fidelity', '',
        '| Regime | Contexts | Prospective wins / losses / ties | Mean TV increase | Instant expected acceptance | Prospective expected acceptance |',
        '|---|---:|---|---:|---:|---:|',
    ]
    for row in evidence["by_regime"]:
        lines.append(f"| {row['regime']} | {row['paired_contexts']} | "
                     f"{row['prospective_fidelity_wins']} / {row['prospective_fidelity_losses']} / {row['fidelity_ties']} | "
                     f"{row['mean_tv_prospective_minus_instant']:+.8f} | "
                     f"{row['mean_instant_expected_single_proposal_acceptance']:.6f} | "
                     f"{row['mean_prospective_expected_single_proposal_acceptance']:.6f} |")
    lines += ['', 'For one proposal at the same prefix, expected acceptance is exactly '
              '`sum(min(p,q)) = 1 − TV(p,q)` in real arithmetic. The saved TV values are already averaged over '
              'common target-generated prefixes, so the identity also holds for these averages. '
              'This is not the free-running tokens-per-verification statistic: different samplers consume random numbers '
              'differently and visit different output prefixes.', '',
              '## Model-level differences', '',
              '| Regime | Model seed | Mean TV increase | Wins / losses / ties |', '|---|---:|---:|---|']
    for row in evidence["by_model_and_regime"]:
        lines.append(f"| {row['regime']} | {row['model_seed']} | {row['mean_tv_prospective_minus_instant']:+.8f} | "
                     f"{row['prospective_fidelity_wins']} / {row['prospective_fidelity_losses']} / {row['fidelity_ties']} |")
    lines += ['', 'Tokens within a sequence and contexts sharing model weights are correlated. '
              'No token-level p-values or claims of statistical independence are made.', '',
              '## Perfect-drafter controls', '',
              'Here the target is defined as the softmax of the raw draft logits. Ordinary instantaneous drafting '
              'is therefore already perfect. The smooth, alternating, and constant signals are deterministic and '
              'all adapters use the saved time constants. No parameter search is performed.', '',
              '| Signal | Adapter | Mean TV | Expected acceptance |', '|---|---|---:|---:|']
    for scenario in result["perfect_drafter_controls"]["results"]:
        for row in scenario["methods"]:
            lines.append(f"| {scenario['scenario']} | {row['method']} | {row['mean_tv']:.8f} | "
                         f"{row['expected_single_proposal_acceptance']:.8f} |")
    lines += ['', 'Hard guards passed: ideal and instantaneous probabilities agree; the instantaneous draft is perfect; '
              'prospective compensation never improves on that perfect reference beyond numerical tolerance. '
              'A benefit over the leaky control alone is repair of the introduced lag.', '',
              '## What this means', '',
              'The current inference adapter receives logits that the ordinary drafter has already computed. '
              'Ideal compensation recovers those logits; finite adaptation filters or extrapolates them. '
              'The implementation does not reproduce prospective learning, skip the draft network forward pass, '
              'or remove GPU layer dependencies. These negative controls do not rule out a differently trained '
              'prospective predictor, but the current results do not support a discovery claim.', '',
              'A future performance claim must beat the ordinary speculative baseline in measured end-to-end GPU time '
              'with comparable correctness, fair implementation effort, and independently held-out tasks. '
              'A small synthetic MLP benchmark cannot establish performance for a production LLM.', '',
              'The JSON companion contains each pair, model-level values, source/result hashes, parameters, and numerical guard outcomes.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--tolerance', type=float, default=1e-12)
    args = parser.parse_args(argv)
    try:
        result = audit(args.results, args.tolerance)
        (args.results/'audit_results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        (args.results/'AUDIT_EVIDENCE.md').write_text(render_markdown(result))
    except (ValueError, KeyError, OSError, AssertionError) as error:
        parser.error(str(error))
    print(json.dumps({'correctness_controls_status':result['correctness_controls_status'],
                      'discovery_status':result['discovery_status'], 'finding':result['finding'],
                      'paired_summary':result['paired_saved_evidence']['overall']}, indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
