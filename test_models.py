"""Check manual training gradients and reproducible model checkpoints."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from .models import INPUT_DIM, VOCAB, NeuralTarget, NumpyDraft, TargetEvaluator, train_pair
except ImportError:
    from models import INPUT_DIM, VOCAB, NeuralTarget, NumpyDraft, TargetEvaluator, train_pair


class ModelTests(unittest.TestCase):
    def test_analytic_gradients_match_central_finite_differences(self):
        rng = np.random.default_rng(27)
        draft = NumpyDraft(seed=8, width=7)
        inputs = rng.normal(size=(5, INPUT_DIM))
        targets = rng.normal(size=(5, VOCAB))
        targets -= targets.mean(axis=1, keepdims=True)
        _, gradients = draft.loss_and_gradients(inputs, targets)
        epsilon = 1e-6
        for parameter_index, (parameter, gradient) in enumerate(zip(draft.parameters, gradients)):
            for flat_index in rng.choice(parameter.size, size=min(8, parameter.size), replace=False):
                with self.subTest(parameter=parameter_index, index=int(flat_index)):
                    original = parameter.flat[flat_index]
                    parameter.flat[flat_index] = original + epsilon
                    plus, _ = draft.loss_and_gradients(inputs, targets)
                    parameter.flat[flat_index] = original - epsilon
                    minus, _ = draft.loss_and_gradients(inputs, targets)
                    parameter.flat[flat_index] = original
                    numerical = (plus - minus) / (2 * epsilon)
                    self.assertAlmostEqual(float(gradient.flat[flat_index]), numerical, delta=2e-9)

    def test_distillation_reduces_held_out_error_and_checkpoint_round_trips(self):
        target, draft, embeddings, metrics = train_pair(seed=11, width=32, train_size=512, steps=160)
        self.assertLess(metrics["validation_logit_mse"], 0.3 * metrics["initial_validation_logit_mse"])
        self.assertEqual(embeddings.shape, (VOCAB, 16))
        inputs = np.random.default_rng(7).normal(size=(9, INPUT_DIM)).astype(np.float32)
        probabilities = target(inputs)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=2e-15)
        self.assertTrue((probabilities > 0).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.npz"
            draft.save(path)
            restored = NumpyDraft.load(path)
            np.testing.assert_array_equal(restored(inputs), draft(inputs))
        reconstructed = NeuralTarget(seed=11, width=32)
        reconstructed.calibrate(metrics["teacher_logit_scale"])
        np.testing.assert_array_equal(reconstructed(inputs), target.model(inputs))

    def test_calibration_and_batched_target_preserve_the_same_mapping(self):
        model = NeuralTarget(seed=4, width=32)
        inputs = np.random.default_rng(33).normal(size=(11, INPUT_DIM)).astype(np.float32)
        before = model(inputs)
        model.calibrate(1.7)
        target = TargetEvaluator(model)
        np.testing.assert_allclose(target.logits(inputs), before * 1.7, rtol=3e-6, atol=1e-6)
        np.testing.assert_allclose(target.logits(inputs, batch_size=3), target.logits(inputs), rtol=3e-6, atol=1e-6)
        self.assertEqual(target.logits(np.empty((0, INPUT_DIM))).shape, (0, VOCAB))


if __name__ == "__main__":
    unittest.main()
