"""Distributional and causal-state checks for speculative sampling."""

import itertools
import unittest
from unittest import mock

import numpy as np

from decoding import _residual_distribution, generate, generate_target


COUNT_FIELDS = ("rounds", "proposed", "checked", "accepted", "committed", "total_committed")


class Context:
    def __init__(self, tokens=()):
        self.tokens = list(tokens)

    def features(self):
        code = 0
        for token in self.tokens:
            code = 3 * code + token + 1
        return np.array([len(self.tokens), code], dtype=np.float64)

    def advance(self, token):
        self.tokens.append(int(token))

    def clone(self):
        return Context(self.tokens)


class RecordingAdapter:
    """Records causal inputs so correct rollback is directly observable."""

    def __init__(self):
        self.history = []

    def apply(self, logits):
        self.history.append(tuple(float(x) for x in logits))
        return np.asarray(logits)

    def snapshot(self):
        # Deliberately return a live list: the sampler must preserve snapshots.
        return self.history

    def restore(self, state):
        self.history = list(state)


def deterministic_draft(features):
    return np.array([1000 + features[1], -1000 - features[0]])


def constant_target(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    return lambda features: np.tile(p, (len(features), 1))


def reconstructed_marginal(p, q):
    # Enumerate each possible first proposal. Zero-q entries must still receive
    # their target mass through the correction branch.
    accepted_mass = np.array([
        q[i] * min(1.0, p[i] / q[i]) if q[i] > 0 else 0.0
        for i in range(len(p))
    ])
    rejection_mass = 1.0 - float(accepted_mass.sum())
    if rejection_mass <= 1e-14:
        return accepted_mass
    return accepted_mass + rejection_mass * _residual_distribution(p, q)


class DecodingTests(unittest.TestCase):
    def assert_same_tokens_and_counts(self, actual, expected):
        self.assertEqual(actual["tokens"], expected["tokens"])
        self.assertEqual({key: actual["stats"][key] for key in COUNT_FIELDS},
                         {key: expected["stats"][key] for key in COUNT_FIELDS})

    def test_stateless_identity_matches_legacy_and_ideal_with_less_work(self):
        from adapters import make_adapter

        cases = (
            # Exact equal distributions, with nontrivial random output tokens.
            ("all_accept", constant_target([0.5, 0.5]), lambda _: np.log([0.5, 0.5]), 11, 3),
            ("all_reject", constant_target([0.0, 1.0]), deterministic_draft, 9, 4),
            # A real stochastic mixture exercises partial accept/reject paths.
            ("mixed", constant_target([0.2, 0.8]), lambda _: np.log([0.65, 0.35]), 31, 4),
        )
        for name, target, draft, count, block in cases:
            with self.subTest(path=name):
                adapter = make_adapter("instant", 2)
                optimized_draft = mock.Mock(side_effect=draft)
                # An explicitly stateless adapter must never be checkpointed.
                with mock.patch.object(adapter, "snapshot", side_effect=AssertionError("unused snapshot")), \
                     mock.patch.object(adapter, "restore", side_effect=AssertionError("unused restore")):
                    optimized = generate(target, optimized_draft, adapter, Context(),
                                         n_tokens=count, k=block, seed=7)
                legacy_adapter = RecordingAdapter()  # No capability flag: default stays stateful.
                legacy_draft = mock.Mock(side_effect=draft)
                legacy = generate(target, legacy_draft, legacy_adapter, Context(),
                                  n_tokens=count, k=block, seed=7)
                ideal = generate(target, draft, make_adapter("ideal", 2), Context(),
                                 n_tokens=count, k=block, seed=7)
                self.assert_same_tokens_and_counts(optimized, legacy)
                self.assert_same_tokens_and_counts(optimized, ideal)
                self.assertEqual(optimized_draft.call_count, optimized["stats"]["proposed"])
                self.assertGreater(legacy_draft.call_count, optimized_draft.call_count)
                self.assertEqual(len(legacy_adapter.history), count)
                if name == "all_accept":
                    self.assertEqual(optimized["stats"]["accepted"], optimized["stats"]["proposed"])
                    self.assertEqual(legacy_draft.call_count, count)
                elif name == "all_reject":
                    self.assertEqual(optimized["stats"]["accepted"], 0)
                else:
                    self.assertGreater(optimized["stats"]["accepted"], 0)
                    self.assertGreater(optimized["stats"]["checked"], optimized["stats"]["accepted"])

    def test_stateless_zero_block_and_single_token_never_call_unused_draft(self):
        from adapters import make_adapter

        target = constant_target([0.3, 0.7])
        for count, block in ((13, 0), (1, 4), (1, 0), (0, 4)):
            with self.subTest(n_tokens=count, k=block):
                unused_draft = mock.Mock(side_effect=AssertionError("no proposal requires the draft"))
                adapter = make_adapter("instant", 2)
                with mock.patch.object(adapter, "apply", side_effect=AssertionError("unused adapter")), \
                     mock.patch.object(adapter, "snapshot", side_effect=AssertionError("unused snapshot")), \
                     mock.patch.object(adapter, "restore", side_effect=AssertionError("unused restore")):
                    optimized = generate(target, unused_draft, adapter, Context(),
                                         n_tokens=count, k=block, seed=8)
                legacy_draft = mock.Mock(side_effect=lambda _: np.log([0.65, 0.35]))
                legacy = generate(target, legacy_draft, RecordingAdapter(), Context(),
                                  n_tokens=count, k=block, seed=8)
                self.assert_same_tokens_and_counts(optimized, legacy)
                self.assertEqual(optimized["tokens"], generate_target(target, Context(), n_tokens=count, seed=8)["tokens"])
                unused_draft.assert_not_called()
                self.assertEqual(legacy_draft.call_count, count)

    def test_only_explicit_false_opts_out_of_state_updates(self):
        adapter = RecordingAdapter()
        adapter.requires_state_updates = 0  # A falsey custom value is not an explicit boolean opt-in.
        draft = mock.Mock(side_effect=lambda _: np.log([0.5, 0.5]))
        generate(constant_target([0.5, 0.5]), draft, adapter, Context(), n_tokens=4, k=0)
        self.assertEqual(draft.call_count, 4)
        self.assertEqual(len(adapter.history), 4)

    def test_one_step_exactness_including_zero_draft_support(self):
        cases = [
            ([0.2, 0.3, 0.5], [0.8, 0.1, 0.1]),
            ([0.2, 0.3, 0.5], [1.0, 0.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.5, 0.0, 0.5]),
            ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
        ]
        for p, q in cases:
            p, q = np.array(p), np.array(q)
            np.testing.assert_allclose(reconstructed_marginal(p, q), p, atol=1e-14)

    def test_full_joint_enumeration_of_two_token_round(self):
        p0 = np.array([0.25, 0.35, 0.4])
        q0 = np.array([0.6, 0.0, 0.4])
        conditional = np.array([[0.1, 0.7, 0.2], [0.3, 0.2, 0.5], [0.8, 0.1, 0.1]])
        residual = _residual_distribution(p0, q0)
        actual = np.zeros((3, 3))
        # k=1: accepted proposal followed by the target bonus, or correction
        # followed by a final k=0 target draw. Enumerate every outcome.
        for proposal in range(3):
            acceptance = min(1.0, p0[proposal] / q0[proposal]) if q0[proposal] else 0
            for second in range(3):
                actual[proposal, second] += q0[proposal] * acceptance * conditional[proposal, second]
            for first, second in itertools.product(range(3), repeat=2):
                actual[first, second] += (
                    q0[proposal] * (1 - acceptance) * residual[first] * conditional[first, second]
                )
        np.testing.assert_allclose(actual, p0[:, None] * conditional, atol=1e-14)
        self.assertAlmostEqual(float(actual.sum()), 1.0)

    def test_equal_distributions_accept_every_proposal_and_clock_bonus(self):
        context = Context()
        adapter = RecordingAdapter()
        result = generate(constant_target([1.0, 0.0]), deterministic_draft, adapter, context,
                          n_tokens=11, k=3, seed=3)
        self.assertEqual(result["tokens"], [0] * 11)
        self.assertEqual(result["stats"]["accepted"], result["stats"]["proposed"])
        self.assertEqual(result["stats"]["checked"], result["stats"]["proposed"])
        self.assertEqual(result["stats"]["committed"], 11)
        self.assertEqual(result["stats"]["rounds"], 3)
        self.assertEqual(len(adapter.history), 11)
        replay = Context()
        expected = []
        for token in result["tokens"]:
            expected.append(tuple(deterministic_draft(replay.features())))
            replay.advance(token)
        self.assertEqual(adapter.history, expected)
        self.assertEqual(context.tokens, [])

    def test_rejected_states_match_replay_of_committed_prefix(self):
        adapter = RecordingAdapter()
        result = generate(constant_target([0.0, 1.0]), deterministic_draft, adapter, Context(),
                          n_tokens=9, k=4, seed=7)
        self.assertEqual(result["tokens"], [1] * 9)
        self.assertEqual(result["stats"]["accepted"], 0)
        self.assertEqual(result["stats"]["rounds"], 9)
        self.assertEqual(result["stats"]["checked"], 8)
        replay = Context()
        expected = []
        for token in result["tokens"]:
            expected.append(tuple(deterministic_draft(replay.features())))
            replay.advance(token)
        self.assertEqual(adapter.history, expected)

    def test_partial_acceptance_then_rejection_restores_correct_prefix(self):
        def target(features):
            # At even prefix lengths draft zero is correct; at odd lengths a
            # one is mandatory, so rounds accept a prefix and then reject.
            return np.array([[1.0, 0.0] if int(row[0]) % 2 == 0 else [0.0, 1.0]
                             for row in features])

        adapter = RecordingAdapter()
        result = generate(target, deterministic_draft, adapter, Context(), n_tokens=10, k=4, seed=4)
        self.assertEqual(result["tokens"], [0, 1] * 5)
        self.assertEqual(result["stats"]["accepted"], 5)
        replay = Context()
        expected = []
        for token in result["tokens"]:
            expected.append(tuple(deterministic_draft(replay.features())))
            replay.advance(token)
        self.assertEqual(adapter.history, expected)

    def test_target_batch_contains_only_proposed_prefixes(self):
        batches = []

        def target(features):
            batches.append(features.copy())
            return np.tile([0.0, 1.0], (len(features), 1))

        result = generate(target, deterministic_draft, RecordingAdapter(), Context(),
                          n_tokens=6, k=3, seed=2)
        self.assertEqual(len(batches), result["stats"]["rounds"])
        # Every round rejects the first proposed zero and commits a one.
        actual_prefix = Context()
        for index, batch in enumerate(batches):
            proposed_prefix = actual_prefix.clone()
            expected_rows = []
            for _ in range(min(3, 6 - index - 1) + 1):
                expected_rows.append(proposed_prefix.features())
                proposed_prefix.advance(0)
            np.testing.assert_array_equal(batch, np.stack(expected_rows))
            actual_prefix.advance(1)

    def test_zero_block_matches_target_sampler_and_updates_adapter(self):
        target = constant_target([0.3, 0.7])
        adapter = RecordingAdapter()
        result = generate(target, deterministic_draft, adapter, Context(), n_tokens=31, k=0, seed=8)
        reference = generate_target(target, Context(), n_tokens=31, seed=8)
        self.assertEqual(result["tokens"], reference["tokens"])
        self.assertEqual(result["stats"]["proposed"], 0)
        self.assertEqual(result["stats"]["accepted"], 0)
        self.assertEqual(len(adapter.history), 31)

    def test_every_real_adapter_matches_committed_prefix_replay(self):
        from adapters import make_adapter

        def draft(features):
            return np.array([np.sin(features[1] * 0.03), np.cos(features[0] * 0.2)])

        for name in ("instant", "leaky", "prospective", "ideal"):
            with self.subTest(adapter=name):
                adapter = make_adapter(name, 2)
                result = generate(constant_target([0.4, 0.6]), draft, adapter, Context(),
                                  n_tokens=23, k=4, seed=57)
                replay_adapter = make_adapter(name, 2)
                replay_context = Context()
                for token in result["tokens"]:
                    replay_adapter.apply(draft(replay_context.features()))
                    replay_context.advance(token)
                state = adapter.snapshot()["state"]
                expected = replay_adapter.snapshot()["state"]
                if expected is None:
                    self.assertIsNone(state)
                else:
                    self.assertEqual(set(state), set(expected))
                    for key in expected:
                        np.testing.assert_array_equal(state[key], expected[key])

    def test_batch_probabilities_match_single_causal_prefixes(self):
        # Contract check with a small row-separable neural target. The experiment
        # separately checks this tolerance on its actual accelerator backend.
        batches = []
        weights = np.array([[0.1, -0.3], [0.015, 0.025]])

        def probabilities(features):
            logits = np.tanh(features @ weights)
            p = np.exp(logits - logits.max(axis=-1, keepdims=True))
            return p / p.sum(axis=-1, keepdims=True)

        def target(features):
            batches.append(features.copy())
            return probabilities(features)

        generate(target, lambda _: np.log([0.6, 0.4]), RecordingAdapter(), Context(),
                 n_tokens=13, k=3, seed=81)
        for batch in batches:
            separate = np.stack([probabilities(row[None, :])[0] for row in batch])
            np.testing.assert_allclose(probabilities(batch), separate, rtol=1e-13, atol=1e-14)

    def test_empty_generation_and_invalid_arguments(self):
        target = constant_target([0.3, 0.7])
        for method in (generate_target,):
            result = method(target, Context(), n_tokens=0)
            self.assertEqual(result["tokens"], [])
            self.assertEqual(result["stats"]["rounds"], 0)
        result = generate(target, deterministic_draft, RecordingAdapter(), Context(), n_tokens=0)
        self.assertEqual(result["tokens"], [])
        with self.assertRaises(ValueError):
            generate(target, deterministic_draft, RecordingAdapter(), Context(), k=-1)

    def test_real_sampler_marginals_for_conditional_target(self):
        # Repeated independent short sequences exercise both accept/reject paths
        # and the actual random sampler, separately from the algebra above.
        def target(features):
            return np.array([[0.2, 0.8] if row[1] % 2 == 0 else [0.75, 0.25]
                             for row in features], dtype=np.float64)

        def draft(_features):
            return np.log([0.65, 0.35])

        counts = np.zeros((2, 2))
        trials = 3000
        for seed in range(trials):
            tokens = generate(target, draft, RecordingAdapter(), Context(), n_tokens=2, k=1, seed=seed)["tokens"]
            counts[tuple(tokens)] += 1
        expected = np.array([[0.2 * 0.75, 0.2 * 0.25], [0.8 * 0.2, 0.8 * 0.8]])
        np.testing.assert_allclose(counts / trials, expected, atol=0.035, rtol=0)


if __name__ == "__main__":
    unittest.main()
