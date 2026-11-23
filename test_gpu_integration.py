"""CPU algorithm integration tests for the GPU runner's orchestration.

Every target here runs with NumPy on the CPU. Synchronization is replaced by an
explicit CPU-only assertion; no Torch tensor backend, accelerator, GPU timing,
or persistent benchmark-result file is used. These tests validate bookkeeping
and guards, not CUDA readiness or GPU performance.
"""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gpu_benchmark import check_generation_counts, check_repeat_control, run_context
from models import VOCAB, NeuralTarget, NumpyDraft, TargetEvaluator, make_context, make_embeddings


class CountedCPUTarget:
    device = "cpu_algorithm_test_only"

    def __init__(self, model):
        self.evaluate = TargetEvaluator(model)
        self.forward_calls = 0
        self.positions_evaluated = 0

    def __call__(self, inputs):
        self.forward_calls += 1
        self.positions_evaluated += len(inputs)
        return self.evaluate(inputs)


class CPUIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model = NeuralTarget(17, width=24)
        self.model.calibrate(2.0)
        self.draft = NumpyDraft(17, width=8)
        self.embeddings = make_embeddings(17)
        self.saved = {"tau": 2.0, "tau_a": 0.2}

    def synchronize_cpu_algorithm(self, backend, device):
        self.assertIsNone(backend, "Algorithm tests must not invoke a Torch backend")
        self.assertEqual(device, "cpu_algorithm_test_only")

    def run_case(self, target, context, *, tokens, k, repeats=2, regime="smooth"):
        with patch("gpu_benchmark.synchronize", self.synchronize_cpu_algorithm):
            return run_context(None, target, self.draft, context, 17, regime, 80000,
                               SimpleNamespace(tokens=tokens, repeats=repeats), self.saved, k, VOCAB)

    def test_complete_contexts_conserve_counts_and_preserve_inputs(self):
        cases = ((1, 0), (1, 4), (2, 1), (9, 0), (9, 4), (17, 8))
        for regime, decay in (("smooth", 0.94), ("abrupt", 0.0)):
            for tokens, k in cases:
                with self.subTest(regime=regime, tokens=tokens, k=k):
                    context = make_context(self.embeddings, decay, 80000)
                    original_memory = context.memory.copy()
                    original_index = context.index
                    rows = self.run_case(CountedCPUTarget(self.model), context, tokens=tokens, k=k, regime=regime)
                    self.assertEqual(len(rows), 10)
                    for row in rows:
                        self.assertEqual(row["committed"], tokens)
                        self.assertEqual(row["accepted"] + row["rounds"], tokens)
                        self.assertLessEqual(row["accepted"], row["checked"])
                        self.assertLessEqual(row["checked"], row["proposed"])
                        self.assertEqual(row["target_forward_calls"], row["rounds"])
                        self.assertEqual(row["target_positions_evaluated"], row["proposed"] + row["rounds"])
                        if k == 0 or tokens == 1:
                            self.assertEqual(row["proposed"], 0)
                            self.assertEqual(row["rounds"], tokens)
                    for method in ("target_only", "instant", "leaky", "prospective", "ideal"):
                        repeated = [row for row in rows if row["method"] == method]
                        self.assertEqual(len(repeated), 2)
                        for key in ("rounds", "accepted", "proposed", "checked", "committed"):
                            self.assertEqual(repeated[0][key], repeated[1][key])
                    np.testing.assert_array_equal(context.memory, original_memory)
                    self.assertEqual(context.index, original_index)

    def test_changed_target_between_repeats_aborts_the_group(self):
        class ChangingCPU(CountedCPUTarget):
            def __call__(self, inputs):
                # tokens=1 makes one target call per method: exactly five in
                # each repeat. Change the deterministic target only in repeat 2.
                token = 0 if self.forward_calls < 5 else 1
                self.forward_calls += 1
                self.positions_evaluated += len(inputs)
                probabilities = np.zeros((len(inputs), VOCAB))
                probabilities[:, token] = 1.0
                return probabilities

        context = make_context(self.embeddings, 0.94, 80000)
        with self.assertRaisesRegex(RuntimeError, "Repeat-determinism.*tokens changed"):
            self.run_case(ChangingCPU(self.model), context, tokens=1, k=0)

    def test_independent_target_position_counter_is_checked(self):
        class BadCounterCPU(CountedCPUTarget):
            def __call__(self, inputs):
                probabilities = super().__call__(inputs)
                self.positions_evaluated += 1
                return probabilities

        context = make_context(self.embeddings, 0.94, 80000)
        with self.assertRaisesRegex(RuntimeError, "Target position count"):
            self.run_case(BadCounterCPU(self.model), context, tokens=9, k=4)

    def test_conservation_guard_detects_corrupted_sampler_statistics(self):
        valid = {"tokens": [1, 2, 3], "stats": {
            "rounds": 1, "accepted": 2, "checked": 2, "proposed": 2, "committed": 3,
            "total_committed": 3,
        }}
        check_generation_counts(valid, 3, 1, 3)
        for key, bad_value, message in (("accepted", 1, "token conservation"),
                                         ("checked", 1, "proposal conservation"),
                                         ("total_committed", 4, "aliases disagree")):
            invalid = copy.deepcopy(valid)
            invalid["stats"][key] = bad_value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, message):
                check_generation_counts(invalid, 3, 1, 3)

    def test_repeat_guard_checks_counts_even_when_tokens_are_equal(self):
        first = {"tokens": [1, 2], "stats": {"rounds": 1, "accepted": 1, "checked": 1,
                                              "proposed": 1, "committed": 2}}
        later = copy.deepcopy(first)
        later["stats"]["checked"] = 2
        with self.assertRaisesRegex(RuntimeError, "Repeat-determinism.*counts changed"):
            check_repeat_control(first, later, "prospective")


if __name__ == "__main__":
    unittest.main()
