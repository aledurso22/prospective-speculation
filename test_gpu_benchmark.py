"""CPU-only tests of GPU-runner bookkeeping and failure guards.

These do not emulate GPU execution or constitute a GPU performance result.
Device availability is mocked only to test device-selection policy.
"""

import copy
import io
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import gpu_benchmark as benchmark


def availability(cuda=False, mps=False):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                           backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


def paired_rows(seeds=(17, 23), repeats=2):
    rows = []
    for seed in seeds:
        for repeat in range(repeats):
            times = {"target_only": 10.0, "instant": 5.0, "leaky": 6.0,
                     "prospective": 4.0 if seed == 17 else 2.0, "ideal": 5.0}
            for method in benchmark.METHODS:
                ordinary = method == "target_only"
                rows.append({"model_seed": seed, "regime": "smooth", "sequence_seed": 80000,
                             "repeat": repeat, "method": method,
                             "wall_seconds": times[method] * (1 + repeat * 0.1),
                             "committed": 100, "rounds": 100 if ordinary else 20,
                             "proposed": 0 if ordinary else 100,
                             "checked": 0 if ordinary else 90,
                             "accepted": 0 if ordinary else 80})
    return rows


class ArgumentAndDeviceTests(unittest.TestCase):
    def test_defaults_require_explicit_output_and_prefer_cuda(self):
        parser = benchmark.build_parser()
        args = parser.parse_args(["--output", "fresh-run", "--preflight-only"])
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.repeats, 5)
        self.assertTrue(args.preflight_only)
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_counts_and_seed_arguments_are_validated(self):
        parser = benchmark.build_parser()
        for flag, value in (("--tokens", "0"), ("--sequences", "-1"), ("--repeats", "0"),
                            ("--k", "-1"), ("--torch-threads", "0"), ("--seeds", "-2")):
            with self.subTest(flag=flag), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--output", "fresh-run", flag, value])
        args = parser.parse_args(["--output", "fresh-run", "--seeds", "17", "17"])
        with self.assertRaisesRegex(ValueError, "unique"):
            benchmark.prepare_assets(args)
        self.assertEqual(parser.parse_args(["--output", "fresh-run", "--k", "0"]).k, 0)

    def test_explicit_cuda_never_silently_falls_back(self):
        with self.assertRaisesRegex(RuntimeError, "No usable cuda"):
            benchmark.select_device(availability(mps=True), "cuda")
        self.assertEqual(benchmark.select_device(availability(mps=True), "auto"), "mps")
        self.assertEqual(benchmark.select_device(availability(cuda=True, mps=True), "auto"), "cuda")
        with self.assertRaisesRegex(RuntimeError, "No usable auto"):
            benchmark.select_device(availability(), "auto")

    def test_missing_assets_fail_before_gpu_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "not-created"
            with patch("sys.stderr", new=io.StringIO()) as error:
                code = benchmark.main(["--results", str(root / "missing"), "--output", str(output),
                                       "--preflight-only"])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            self.assertIn("stopped", error.getvalue())

    @unittest.skipIf(importlib.util.find_spec("torch") is not None, "Exercises the real missing-Torch failure only")
    def test_missing_torch_leaves_explicit_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch("sys.stderr", new=io.StringIO()) as error:
                code = benchmark.main(["--output", str(output), "--preflight-only"])
            self.assertEqual(code, 2)
            self.assertIn("PyTorch installation", error.getvalue())
            state = json.loads((output / "gpu_checkpoint.json").read_text())
            self.assertEqual(state["status"], "incomplete")
            self.assertFalse(state["metadata"]["benchmark_executed"])
            self.assertFalse(state["metadata"]["gpu_preflight_executed"])
            self.assertEqual([p.name for p in output.iterdir()], ["gpu_checkpoint.json"])


class OutputAndAssetGuardTests(unittest.TestCase):
    def test_completed_or_partial_output_is_never_reused(self):
        for filename in ("gpu_results.json", "gpu_checkpoint.json", "gpu_preflight.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                prior = path / filename
                prior.write_text("prior run")
                with self.assertRaisesRegex(FileExistsError, "new run directory"):
                    benchmark.reserve_output_directory(path)
                self.assertEqual(prior.read_text(), "prior run")

    def test_empty_output_reservation_excludes_another_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            lock = benchmark.reserve_output_directory(path)
            self.assertTrue(lock.exists())
            with self.assertRaises(FileExistsError):
                benchmark.reserve_output_directory(path)
            lock.unlink()

    def test_atomic_checkpoint_preserves_last_good_version_on_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu_checkpoint.json"
            state = {"status": "incomplete", "completed_context_groups": 1, "runs": []}
            benchmark.atomic_write_json(path, state)
            with self.assertRaises(ValueError):
                benchmark.atomic_write_json(path, {"invalid": float("nan")})
            self.assertEqual(json.loads(path.read_text()), state)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), [path.name])
            state["completed_context_groups"] = 2
            benchmark.atomic_write_json(path, state)
            self.assertEqual(json.loads(path.read_text())["completed_context_groups"], 2)

    def test_final_asset_guard_compares_full_provenance(self):
        args = SimpleNamespace(results=Path("unused-in-mock"))
        source, seeds, provenance = {"metadata": {}}, [17], {"sha256": "original"}
        with patch("benchmark_assets.load_source_metadata", return_value=(source, seeds, provenance)):
            benchmark.check_source_unchanged(args, source, seeds, provenance)
        changed = {"sha256": "changed"}
        with patch("benchmark_assets.load_source_metadata", return_value=(source, seeds, changed)):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                benchmark.check_source_unchanged(args, source, seeds, provenance)

    def test_algorithm_fingerprints_cover_all_execution_sources(self):
        hashes = benchmark.source_code_hashes()
        self.assertEqual(set(hashes), {"gpu_benchmark.py", "decoding.py", "adapters.py", "models.py", "benchmark_assets.py"})
        for digest in hashes.values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)


class SummaryAndControlTests(unittest.TestCase):
    def test_primary_instant_comparison_and_per_model_dispersion(self):
        rows = paired_rows()
        summary = benchmark.summarize(rows)
        prospective = next(r for r in summary if r["method"] == "prospective")
        self.assertEqual(prospective["paired_speedup_vs_instant_median"], 1.875)
        self.assertEqual(prospective["paired_speedup_vs_target_median"], 3.75)
        self.assertEqual(prospective["runs"], 4)
        self.assertEqual(prospective["model_seeds"], 2)
        self.assertGreater(prospective["paired_speedup_vs_instant_between_model_sd"], 0)
        per_model = benchmark.summarize(rows, by_model=True)
        values = {r["model_seed"]: r["paired_speedup_vs_instant_median"]
                  for r in per_model if r["method"] == "prospective"}
        self.assertEqual(values, {17: 1.25, 23: 2.5})

    def test_ratios_are_paired_before_taking_median(self):
        rows = paired_rows(seeds=(17,), repeats=2)
        for row in rows:
            if row["method"] == "prospective":
                row["wall_seconds"] = (1.0, 100.0)[row["repeat"]]
            elif row["method"] == "instant":
                row["wall_seconds"] = (2.0, 100.0)[row["repeat"]]
        prospective = next(r for r in benchmark.summarize(rows) if r["method"] == "prospective")
        self.assertEqual(prospective["paired_speedup_vs_instant_median"], 1.5)

    def test_incomplete_duplicate_and_nonfinite_groups_are_rejected(self):
        rows = paired_rows(seeds=(17,), repeats=1)
        for invalid in (rows[:-1], rows + [rows[0]], [{**rows[0], "wall_seconds": float("inf")}] + rows[1:]):
            with self.assertRaises(ValueError):
                benchmark.summarize(invalid)
        self.assertEqual(benchmark.summarize([]), [])

    def test_ideal_guard_checks_outputs_and_all_counts(self):
        result = {"tokens": [0, 1], "stats": {key: 2 for key in benchmark.COUNT_FIELDS}}
        controls = {"instant": copy.deepcopy(result), "ideal": copy.deepcopy(result)}
        benchmark.check_ideal_control(controls)
        controls["ideal"]["tokens"] = [1, 0]
        with self.assertRaisesRegex(RuntimeError, "tokens"):
            benchmark.check_ideal_control(controls)
        controls["ideal"] = copy.deepcopy(result)
        controls["ideal"]["stats"]["checked"] += 1
        with self.assertRaisesRegex(RuntimeError, "counts"):
            benchmark.check_ideal_control(controls)

    def test_probability_diagnostic_rejects_batch_dependent_cpu_function(self):
        # Pure diagnostic test, not an emulated GPU run or timing measurement.
        features = np.zeros((3, 2))
        stable = lambda x: np.tile([0.2, 0.8], (len(x), 1))
        self.assertTrue(benchmark.batch_consistency(stable, features)["passed"])
        dependent = lambda x: np.tile([0.3, 0.7] if len(x) > 1 else [0.2, 0.8], (len(x), 1))
        with self.assertRaisesRegex(RuntimeError, "check failed"):
            benchmark.batch_consistency(dependent, features)


if __name__ == "__main__":
    unittest.main()
