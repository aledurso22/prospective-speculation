"""Exact speculative sampling for a small, causal, categorical target.

The target receives the entire proposed-prefix batch in one call. It must return
the same conditional distribution for each row that it would return when called
on that row alone. This module measures CPU wall time; batching this toy target
does not establish a GPU speedup.

A stateful adapter is evaluated once at every prefix from which a committed token
is emitted. Speculative evaluations beyond that prefix are rolled back. An
explicitly stateless adapter is evaluated only for actual draft proposals. The
caller's context is cloned, while a stateful adapter retains its committed state.
"""

from __future__ import annotations

import copy
from time import perf_counter
from typing import Any, Callable

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Draft logits must be a nonempty vector.")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Draft logits must be finite or negative infinity.")
    maximum = np.max(values)
    if not np.isfinite(maximum):
        raise ValueError("At least one draft logit must be finite.")
    weights = np.exp(values - maximum)
    return weights / weights.sum()


def _probability_rows(probabilities: np.ndarray, rows: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0:
        raise ValueError("Target must return a [batch, vocabulary] array.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Target probabilities must be finite and nonnegative.")
    totals = values.sum(axis=1, keepdims=True)
    if not np.allclose(totals, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Each target probability row must sum to one.")
    # Remove floating-point summation drift before categorical sampling.
    return values / totals


def _residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Conditional correction distribution, used only after a rejection."""
    residual = np.maximum(p - q, 0.0)
    mass = float(residual.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("A rejected draft must leave positive residual mass.")
    return residual / mass


def _finish_stats(stats: dict[str, Any], started: float) -> None:
    stats["wall_seconds"] = perf_counter() - started
    stats["other_seconds"] = max(
        0.0,
        stats["wall_seconds"] - stats["draft_seconds"] - stats["verify_seconds"],
    )
    stats["total_committed"] = stats["committed"]


def _stats() -> dict[str, Any]:
    return {
        "rounds": 0,
        "proposed": 0,
        "checked": 0,
        "accepted": 0,
        "committed": 0,
        "draft_seconds": 0.0,
        "verify_seconds": 0.0,
    }


def generate(
    target: Callable[[np.ndarray], np.ndarray],
    draft: Callable[[np.ndarray], np.ndarray],
    adapter: Any,
    context: Any,
    n_tokens: int = 256,
    k: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample exactly from the target using a draft and optional adapter state.

    ``context`` provides ``features()``, ``advance(token)``, and ``clone()``.
    ``adapter`` provides ``apply(logits)``, ``snapshot()``, and ``restore(state)``.
    At a rejection, its state includes the evaluation at the corrected token's
    prefix, but none of the evaluations at later speculative prefixes.

    The draft is a pure function of prefix features; temporal draft state belongs
    in the adapter. An adapter may explicitly set ``requires_state_updates=False``
    when it has no per-prefix state. Only then are snapshots, rollback, and
    otherwise unused bonus-prefix draft evaluations omitted. Custom adapters
    without that explicit declaration retain the stateful behavior.

    ``accepted`` counts only accepted draft tokens. Every round also commits one
    target correction or bonus token. The draft block is shortened at the end so
    no output beyond the requested length is generated or credited.
    """
    if not isinstance(n_tokens, (int, np.integer)) or n_tokens < 0:
        raise ValueError("n_tokens must be a nonnegative integer.")
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer.")
    started = perf_counter()
    rng = np.random.default_rng(seed)
    committed_context = context.clone()
    output: list[int] = []
    stats = _stats()
    requires_state_updates = getattr(adapter, "requires_state_updates", True) is not False

    while len(output) < n_tokens:
        block_size = min(k, n_tokens - len(output) - 1)
        proposed: list[int] = []
        draft_probabilities: list[np.ndarray] = []
        states_after_prefix: list[Any] = []
        feature_rows: list[np.ndarray] = []

        draft_started = perf_counter()
        proposed_context = committed_context.clone()
        for _ in range(block_size):
            features = np.asarray(proposed_context.features()).copy()
            feature_rows.append(features)
            q = _softmax(adapter.apply(draft(features)))
            if requires_state_updates:
                # Copy the snapshot so adapters may return a mutable object.
                states_after_prefix.append(copy.deepcopy(adapter.snapshot()))
            token = int(rng.choice(q.size, p=q))
            proposed.append(token)
            draft_probabilities.append(q)
            proposed_context.advance(token)
        feature_rows.append(np.asarray(proposed_context.features()).copy())
        feature_batch = np.stack(feature_rows)
        stats["draft_seconds"] += perf_counter() - draft_started

        verify_started = perf_counter()
        p_rows = _probability_rows(target(feature_batch), block_size + 1)
        stats["verify_seconds"] += perf_counter() - verify_started
        if any(q.shape[0] != p_rows.shape[1] for q in draft_probabilities):
            raise ValueError("Draft and target vocabularies must match.")
        stats["rounds"] += 1
        stats["proposed"] += block_size

        rejected = False
        for position, token in enumerate(proposed):
            stats["checked"] += 1
            p = p_rows[position]
            q = draft_probabilities[position]
            # q[token] > 0 because token was sampled from q. Multiplication
            # avoids division by tiny probabilities in the acceptance ratio.
            if rng.random() * q[token] < p[token]:
                output.append(token)
                committed_context.advance(token)
                stats["accepted"] += 1
            else:
                correction = int(rng.choice(p.size, p=_residual_distribution(p, q)))
                if requires_state_updates:
                    adapter.restore(copy.deepcopy(states_after_prefix[position]))
                output.append(correction)
                committed_context.advance(correction)
                rejected = True
                break

        if not rejected:
            # The bonus needs only the target probabilities already computed.
            # A stateful adapter must nevertheless advance its clock at this
            # prefix, including k=0 and the final single-token round.
            if requires_state_updates:
                draft_started = perf_counter()
                bonus_q = _softmax(adapter.apply(draft(feature_rows[-1])))
                if bonus_q.shape[0] != p_rows.shape[1]:
                    raise ValueError("Draft and target vocabularies must match.")
                stats["draft_seconds"] += perf_counter() - draft_started
            bonus = int(rng.choice(p_rows.shape[1], p=p_rows[-1]))
            output.append(bonus)
            committed_context.advance(bonus)

        stats["committed"] = len(output)

    _finish_stats(stats, started)
    return {"tokens": output, "stats": stats}


def generate_target(
    target: Callable[[np.ndarray], np.ndarray],
    context: Any,
    n_tokens: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    """Ordinary sequential sampling from exactly the same target."""
    if not isinstance(n_tokens, (int, np.integer)) or n_tokens < 0:
        raise ValueError("n_tokens must be a nonnegative integer.")
    started = perf_counter()
    rng = np.random.default_rng(seed)
    current = context.clone()
    output: list[int] = []
    stats = _stats()
    for _ in range(n_tokens):
        features = np.asarray(current.features()).copy()[None, :]
        verify_started = perf_counter()
        p = _probability_rows(target(features), 1)[0]
        stats["verify_seconds"] += perf_counter() - verify_started
        token = int(rng.choice(p.size, p=p))
        output.append(token)
        current.advance(token)
        stats["rounds"] += 1
    stats["committed"] = len(output)
    _finish_stats(stats, started)
    return {"tokens": output, "stats": stats}
