"""Identity, malformed-asset, and numerical gates for portable GPU inputs."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import benchmark_assets as assets
from models import NumpyDraft


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = {
            "metadata": {"numpy": np.__version__, "args": {"seeds": [17, 23], "width": 16}},
            "training": {"17": {"teacher_logit_scale": 1.2}, "23": {"teacher_logit_scale": 1.4}},
        }
        (self.directory / "results.json").write_text(json.dumps(self.source))
        for seed in (17, 23):
            NumpyDraft(seed=seed, width=8).save(self.directory / f"draft_seed_{seed}.npz")
        assets.create_model_assets(self.directory)

    def test_valid_bundle_verifies_identity_and_cpu_numerics_without_retraining(self):
        original_files = {p.name: assets.sha256_file(p) for p in self.directory.iterdir()}
        with mock.patch("models.train_pair", side_effect=AssertionError("must not retrain")):
            source, seeds, provenance = assets.load_source_metadata(self.directory, [23])
            model, draft, embeddings, checks = assets.load_model_assets(self.directory, 23)
        self.assertEqual(source, self.source)
        self.assertEqual(seeds, [23])
        self.assertEqual(set(provenance["selected_asset_files"]), {"23"})
        self.assertTrue(checks["teacher_weights_exact_hash_match"])
        self.assertTrue(checks["cpu_target_probabilities"]["passed"])
        self.assertEqual(embeddings.shape, (16, 16))
        self.assertEqual(model.weights[0].shape, (24, 16))
        self.assertEqual(draft.w1.shape, (8, 24))
        self.assertEqual(original_files, {p.name: assets.sha256_file(p) for p in self.directory.iterdir()})
        json.dumps(checks, allow_nan=False)
        json.dumps(provenance, allow_nan=False)

    def test_verify_cli_and_creation_refuse_overwrite(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = assets.main(["--verify", "--results", str(self.directory), "--seeds", "17"])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(output.getvalue())["gpu_executed"])
        with self.assertRaisesRegex(assets.AssetValidationError, "already exists"):
            assets.create_model_assets(self.directory)

    def test_selection_rejects_duplicate_unknown_empty_and_noninteger_seeds(self):
        for seeds in ([17, 17], [999], [], [True], [17.0], [-1]):
            with self.subTest(seeds=seeds):
                with self.assertRaises(assets.AssetValidationError):
                    assets.load_source_metadata(self.directory, seeds)

    def test_changed_source_or_selected_npz_is_detected_by_lightweight_recheck(self):
        for filename in ("results.json", "draft_seed_17.npz", "target_probes_seed_17.npz"):
            path = self.directory / filename
            original = path.read_bytes()
            with self.subTest(filename=filename):
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(assets.AssetValidationError, "SHA-256 mismatch"):
                    assets.load_source_metadata(self.directory, [17])
            path.write_bytes(original)

    def test_missing_manifest_or_reference_probes_never_silently_skips(self):
        manifest = self.directory / assets.MANIFEST_NAME
        original = manifest.read_bytes()
        manifest.unlink()
        with self.assertRaisesRegex(assets.AssetValidationError, "verification never regenerates"):
            assets.load_model_assets(self.directory, 17)
        manifest.write_bytes(original)
        (self.directory / "target_probes_seed_17.npz").unlink()
        with self.assertRaisesRegex(assets.AssetValidationError, "Missing"):
            assets.load_reference_probes(self.directory, 17)

    def test_regenerated_weight_mismatch_fails_even_when_small(self):
        original_constructor = assets.NeuralTarget

        def altered_constructor(*args, **kwargs):
            model = original_constructor(*args, **kwargs)
            model.weights[0].flat[0] = np.nextafter(model.weights[0].flat[0], np.float32(np.inf))
            return model

        with mock.patch.object(assets, "NeuralTarget", side_effect=altered_constructor):
            with self.assertRaisesRegex(assets.AssetValidationError, "teacher weight SHA-256 mismatch"):
                assets.load_model_assets(self.directory, 17)

    def test_probe_check_rejects_changed_shape_nonfinite_and_changed_probabilities(self):
        probes = assets.load_reference_probes(self.directory, 17)
        expected = probes["target_probabilities"]
        self.assertTrue(assets.check_reference_probabilities(expected.copy(), probes)["passed"])
        bad_values = [expected[:-1], expected * np.nan, np.full_like(expected, 1.0 / 16)]
        for values in bad_values:
            with self.assertRaises(assets.AssetValidationError):
                assets.check_reference_probabilities(values, probes)
        with self.assertRaises(assets.AssetValidationError):
            assets.check_reference_probabilities(expected, probes, atol=np.inf)

    def test_different_numpy_version_requires_same_bytes_and_warns(self):
        with mock.patch.object(np, "__version__", "test-other-version"):
            _, _, _, diagnostics = assets.load_model_assets(self.directory, 17)
        self.assertTrue(diagnostics["teacher_weights_exact_hash_match"])
        self.assertTrue(any("NumPy version differs" in text for text in diagnostics["warnings"]))

    def test_manifest_fingerprint_and_path_traversal_are_rejected(self):
        path = self.directory / assets.MANIFEST_NAME
        manifest = json.loads(path.read_text())
        manifest["models"]["17"]["logit_scale"] = 7.0
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(assets.AssetValidationError, "fingerprint mismatch"):
            assets.load_source_metadata(self.directory)
        with self.assertRaises(assets.AssetValidationError):
            assets._asset_path(self.directory, "../draft.npz")

    def test_hash_checked_but_malformed_probe_arrays_are_rejected(self):
        path = self.directory / assets.MANIFEST_NAME
        manifest = json.loads(path.read_text())
        entry = manifest["models"]["17"]
        probe_path = self.directory / entry["probes"]["path"]
        probes = assets.load_reference_probes(self.directory, 17)
        probes["inputs"] = probes["inputs"][:, :-1]
        np.savez_compressed(probe_path, **probes)
        entry["probes"]["sha256"] = assets.sha256_file(probe_path)
        del entry["fingerprint_sha256"]
        entry["fingerprint_sha256"] = assets._canonical_hash(entry)
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(assets.AssetValidationError, "input dimensions"):
            assets.load_reference_probes(self.directory, 17)


if __name__ == "__main__":
    unittest.main()
