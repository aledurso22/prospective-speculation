"""Verify frozen benchmark identities and small CPU reference probes.

Creation is explicit and belongs in the environment that produced the saved
CPU results. Verification never retrains, rewrites assets, or imports PyTorch.
Matching hashes establish byte identity with this manifest, not authenticity.
Reference probes check numerics; they are not a proof over every possible input.

    python benchmark_assets.py --create --results results
    python benchmark_assets.py --verify --results results
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np

from models import (
    INPUT_DIM, MEMORY_DIM, VOCAB, NeuralTarget, NumpyDraft, TargetEvaluator,
    make_context, make_embeddings, training_inputs,
)


SCHEMA = "prospective_speculation.model_assets"
VERSION = 1
MANIFEST_NAME = "model_assets.json"
PROBE_KEYS = {"inputs", "target_logits", "target_probabilities", "draft_logits", "embeddings"}
LOGIT_ATOL = 5e-5
LOGIT_RTOL = 1e-5
PROBABILITY_ATOL = 1e-5
PROBABILITY_RTOL = 1e-5


class AssetValidationError(ValueError):
    """Saved artifacts or regenerated models failed validation."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_fingerprint(value):
    """Hash canonical little-endian C-order bytes with dtype and shape."""
    array = np.asarray(value)
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    header = {"dtype": array.dtype.str, "shape": list(array.shape)}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return {**header, "sha256": digest.hexdigest()}


def _json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as error:
        raise AssetValidationError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise AssetValidationError(f"Expected a JSON object in {path}")
    return value


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AssetValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _seed_list(value, label):
    if not isinstance(value, (list, tuple)) or not value:
        raise AssetValidationError(f"{label} must be a nonempty sequence of model seeds")
    seeds = [_integer(seed, label) for seed in value]
    if len(set(seeds)) != len(seeds):
        raise AssetValidationError(f"{label} contains duplicate model seeds")
    return seeds


def _source_configuration(source):
    try:
        args = source["metadata"]["args"]
        seeds = _seed_list(args["seeds"], "saved seeds")
        width = _integer(args["width"], "target width", minimum=1)
        training = source["training"]
        scales = {}
        for seed in seeds:
            scale = training[str(seed)]["teacher_logit_scale"]
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
                raise AssetValidationError(f"Invalid target calibration for seed {seed}")
            scales[seed] = float(scale)
    except (KeyError, TypeError) as error:
        raise AssetValidationError(f"Incomplete saved CPU experiment metadata: {error}") from error
    return seeds, width, scales


def _asset_path(directory, name):
    # Manifests can only select direct files in their own results directory.
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise AssetValidationError("Asset paths must be plain filenames inside the results directory")
    directory = Path(directory).resolve()
    path = (directory / name).resolve()
    if path.parent != directory or not path.is_file():
        raise AssetValidationError(f"Missing or out-of-directory benchmark asset: {name}")
    return path


def _verify_file(directory, description):
    if not isinstance(description, dict):
        raise AssetValidationError("Malformed asset file description")
    expected = description.get("sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise AssetValidationError("Malformed asset SHA-256 digest")
    path = _asset_path(directory, description.get("path"))
    observed = sha256_file(path)
    if observed != expected:
        raise AssetValidationError(f"SHA-256 mismatch for {path.name}; the saved benchmark artifact changed")
    return path, observed


def _manifest(directory):
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise AssetValidationError(
            f"Missing {path}. Create assets explicitly in the original CPU environment with "
            "python benchmark_assets.py --create --results RESULTS_DIR; verification never regenerates them."
        )
    manifest = _json(path)
    if manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
        raise AssetValidationError("Unsupported benchmark asset manifest schema/version")
    seeds = _seed_list(manifest.get("seeds"), "manifest seeds")
    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != {str(seed) for seed in seeds}:
        raise AssetValidationError("Manifest models do not match its seed list")
    for seed in seeds:
        entry = models[str(seed)]
        if not isinstance(entry, dict):
            raise AssetValidationError(f"Malformed model entry for seed {seed}")
        content = {key: value for key, value in entry.items() if key != "fingerprint_sha256"}
        if entry.get("fingerprint_sha256") != _canonical_hash(content):
            raise AssetValidationError(f"Model fingerprint mismatch for seed {seed}")
    return manifest


def load_source_metadata(results_dir, requested_seeds=None):
    """Return (CPU source JSON, validated seed selection, JSON provenance).

    A manifest and reference probes are required for bundled benchmark assets.
    This function verifies source identity, selected draft/probe file hashes,
    and selection without allocating a teacher. It can be repeated after a
    benchmark to detect changes to any selected source artifact.
    """
    directory = Path(results_dir).resolve()
    manifest = _manifest(directory)
    if manifest.get("source_results", {}).get("path") != "results.json":
        raise AssetValidationError("Manifest source must be results.json")
    source_path, source_hash = _verify_file(directory, manifest["source_results"])
    source = _json(source_path)
    saved, width, scales = _source_configuration(source)
    if saved != manifest["seeds"]:
        raise AssetValidationError("Manifest seeds differ from saved CPU experiment seeds")
    seeds = saved if requested_seeds is None else _seed_list(requested_seeds, "requested seeds")
    if any(seed not in saved for seed in seeds):
        raise AssetValidationError(f"Requested model seeds must be a subset of saved seeds {saved}")
    for seed in saved:
        entry = manifest["models"][str(seed)]
        if entry.get("model_seed") != seed or entry.get("width") != width or entry.get("logit_scale") != scales[seed]:
            raise AssetValidationError(f"Manifest architecture/calibration differs from saved seed {seed}")
    asset_files = {}
    for seed in seeds:
        entry = manifest["models"][str(seed)]
        files = {}
        for kind in ("draft", "probes"):
            path, digest = _verify_file(directory, entry.get(kind))
            files[kind] = {"path": str(path), "sha256": digest}
        asset_files[str(seed)] = files
    model_source = Path(__file__).resolve().with_name("models.py")
    provenance = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_results_path": str(source_path),
        "source_results_sha256": source_hash,
        "asset_manifest_path": str(directory / MANIFEST_NAME),
        "asset_manifest_sha256": sha256_file(directory / MANIFEST_NAME),
        "source_numpy": source["metadata"].get("numpy"),
        "asset_creation_numpy": manifest.get("created_with", {}).get("numpy"),
        "runtime_numpy": np.__version__,
        "recorded_model_source_sha256": manifest.get("model_source_sha256"),
        "runtime_model_source_sha256": sha256_file(model_source),
        "selected_seeds": list(seeds),
        "selected_asset_files": asset_files,
    }
    return source, list(seeds), provenance


def load_reference_probes(results_dir, seed):
    """Return required, hash-checked arrays; never silently omit bad probes."""
    _, _, _ = load_source_metadata(results_dir, [seed])
    entry = _manifest(results_dir)["models"][str(seed)]
    path, _ = _verify_file(results_dir, entry.get("probes"))
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != PROBE_KEYS:
                raise AssetValidationError(f"Incorrect probe array names for seed {seed}")
            probes = {key: np.array(stored[key], copy=True) for key in PROBE_KEYS}
    except (OSError, ValueError) as error:
        if isinstance(error, AssetValidationError):
            raise
        raise AssetValidationError(f"Cannot load reference probes for seed {seed}: {error}") from error
    inputs = probes["inputs"]
    if inputs.ndim != 2 or inputs.shape[1] != INPUT_DIM or len(inputs) < 16:
        raise AssetValidationError("Invalid reference input dimensions")
    expected = {
        "inputs": (len(inputs), INPUT_DIM),
        "target_logits": (len(inputs), VOCAB),
        "target_probabilities": (len(inputs), VOCAB),
        "draft_logits": (len(inputs), VOCAB),
        "embeddings": (VOCAB, MEMORY_DIM),
    }
    for key, shape in expected.items():
        value = probes[key]
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise AssetValidationError(f"Invalid reference array {key}")
    probabilities = probes["target_probabilities"]
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise AssetValidationError("Reference probabilities are not normalized")
    return probes


def _compare(actual, reference, label, atol, rtol):
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise AssetValidationError(f"{label} has incorrect dimensions or nonfinite outputs")
    difference = float(np.max(np.abs(actual - reference)))
    passed = bool(np.allclose(actual, reference, atol=atol, rtol=rtol))
    diagnostic = {"passed": passed, "max_abs_difference": difference, "atol": atol, "rtol": rtol}
    if not passed:
        raise AssetValidationError(
            f"{label} differs from saved CPU reference probes (max absolute difference {difference:.3g}). "
            "Stop benchmarking; use the original NumPy environment or export the original full target weights."
        )
    return diagnostic


def check_reference_probabilities(probabilities, probes, *, atol=PROBABILITY_ATOL, rtol=PROBABILITY_RTOL):
    """Check a CPU/GPU probability matrix against the saved CPU reference."""
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise AssetValidationError("Reference tolerances must be finite and nonnegative")
    probabilities = np.asarray(probabilities)
    reference = probes["target_probabilities"]
    diagnostic = _compare(probabilities, reference, "Target probabilities", atol, rtol)
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-10, rtol=1e-10):
        raise AssetValidationError("Target probabilities must be nonnegative and normalized")
    diagnostic.update({
        "positions": len(reference),
        "max_total_variation": float(np.max(0.5 * np.abs(probabilities - reference).sum(axis=1))),
    })
    return diagnostic


def load_model_assets(results_dir, seed):
    """Return (model, draft, saved embeddings, JSON diagnostics), CPU verified.

    Exact regenerated target hashes are required. A different NumPy version is
    allowed only when it preserves those bytes and passes numerical probes.
    """
    source, _, provenance = load_source_metadata(results_dir, [seed])
    manifest = _manifest(results_dir)
    entry = manifest["models"][str(seed)]
    draft_path, draft_hash = _verify_file(results_dir, entry.get("draft"))
    try:
        draft = NumpyDraft.load(draft_path)
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError) as error:
        raise AssetValidationError(f"Invalid saved draft for seed {seed}: {error}") from error
    probes = load_reference_probes(results_dir, seed)
    model = NeuralTarget(seed, width=entry["width"])
    model.calibrate(entry["logit_scale"])
    regenerated = [array_fingerprint(weight) for weight in model.weights]
    if regenerated != entry.get("weights"):
        original = manifest.get("created_with", {}).get("numpy", "2.5.5")
        raise AssetValidationError(
            f"Regenerated teacher weight SHA-256 mismatch for seed {seed}. "
            f"Use the original NumPy {original} environment, or export original full weights and build "
            "a separately versioned full-weight asset bundle. Current assets were not modified."
        )
    embeddings = make_embeddings(seed)
    if array_fingerprint(probes["embeddings"]) != entry.get("embeddings"):
        raise AssetValidationError(f"Saved embedding fingerprint mismatch for seed {seed}")
    embedding_check = _compare(embeddings, probes["embeddings"], "Prefix embeddings", 1e-12, 1e-12)
    evaluator = TargetEvaluator(model)
    actual_logits = evaluator.logits(probes["inputs"])
    logit_check = _compare(actual_logits, probes["target_logits"], "CPU target logits", LOGIT_ATOL, LOGIT_RTOL)
    probability_check = check_reference_probabilities(evaluator(probes["inputs"]), probes)
    draft_check = _compare(draft(probes["inputs"]), probes["draft_logits"], "CPU draft logits", 1e-10, 1e-10)
    notices = []
    if np.__version__ != provenance["asset_creation_numpy"]:
        notices.append("NumPy version differs from asset creation; exact target weight hashes and CPU probes passed.")
    if provenance["runtime_model_source_sha256"] != provenance["recorded_model_source_sha256"]:
        notices.append("models.py source hash differs from asset creation; exact target weight hashes and CPU probes passed.")
    diagnostics = {
        "model_seed": seed,
        "model_fingerprint_sha256": entry["fingerprint_sha256"],
        "teacher_weights_exact_hash_match": True,
        "regenerated_weight_fingerprints": regenerated,
        "draft_sha256": draft_hash,
        "probes_sha256": entry["probes"]["sha256"],
        "cpu_target_logits": logit_check,
        "cpu_target_probabilities": probability_check,
        "cpu_draft_logits": draft_check,
        "embeddings": embedding_check,
        "warnings": notices,
        "provenance": provenance,
    }
    # Saved embeddings retain the original exact prefix features across hosts.
    return model, draft, probes["embeddings"].copy(), diagnostics


def _probe_inputs(embeddings, seed):
    parts = [training_inputs(embeddings, 32, seed + 70000)]
    for regime, decay in enumerate((0.94, 0.0)):
        context = make_context(embeddings, decay, seed + 190000 + regime)
        features = []
        for position in range(16):
            features.append(context.features())
            context.advance((position * 7 + seed) % VOCAB)
        parts.append(np.asarray(features, dtype=np.float32))
    return np.concatenate(parts)


def create_model_assets(results_dir):
    """Explicitly create a new immutable manifest and probes, without training."""
    directory = Path(results_dir).resolve()
    if (directory / MANIFEST_NAME).exists():
        raise AssetValidationError("Asset manifest already exists; verification never overwrites saved identities")
    source_path = _asset_path(directory, "results.json")
    source = _json(source_path)
    seeds, width, scales = _source_configuration(source)
    recorded_numpy = source.get("metadata", {}).get("numpy")
    if recorded_numpy != np.__version__:
        raise AssetValidationError(
            f"Create reference assets in the original NumPy {recorded_numpy} environment; "
            f"current NumPy is {np.__version__}. This command must not bless unverified regenerated models."
        )
    for seed in seeds:
        if (directory / f"target_probes_seed_{seed}.npz").exists():
            raise AssetValidationError(f"Reference probes already exist for seed {seed}; refusing to overwrite")
        _asset_path(directory, f"draft_seed_{seed}.npz")
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "created_with": {"numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform()},
        "source_results": {"path": "results.json", "sha256": sha256_file(source_path)},
        "model_source_sha256": sha256_file(Path(__file__).resolve().with_name("models.py")),
        "seeds": seeds,
        "regeneration": "NeuralTarget(seed,width), then calibrate(saved teacher_logit_scale); no retraining",
        "validation_policy": "Exact target weight hashes plus CPU reference probes required; GPU execution not implied",
        "models": {},
    }
    # Prepare everything in memory before writing new files, so invalid saved
    # drafts cannot leave behind a partially created bundle.
    prepared = []
    for seed in seeds:
        model = NeuralTarget(seed, width=width)
        model.calibrate(scales[seed])
        draft_path = directory / f"draft_seed_{seed}.npz"
        draft = NumpyDraft.load(draft_path)
        embeddings = make_embeddings(seed)
        inputs = _probe_inputs(embeddings, seed)
        evaluator = TargetEvaluator(model)
        arrays = {
            "inputs": inputs,
            "target_logits": evaluator.logits(inputs),
            "target_probabilities": evaluator(inputs),
            "draft_logits": draft(inputs),
            "embeddings": embeddings,
        }
        entry = {
            "schema": SCHEMA, "version": VERSION, "model_seed": seed,
            "width": width, "logit_scale": scales[seed],
            "draft": {"path": draft_path.name, "sha256": sha256_file(draft_path)},
            "weights": [array_fingerprint(w) for w in model.weights],
            "embeddings": array_fingerprint(embeddings),
        }
        prepared.append((seed, arrays, entry))
    for seed, arrays, entry in prepared:
        path = directory / f"target_probes_seed_{seed}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        entry["probes"] = {"path": path.name, "sha256": sha256_file(path), "positions": len(arrays["inputs"])}
        entry["fingerprint_sha256"] = _canonical_hash(entry)
        manifest["models"][str(seed)] = entry
    with (directory / MANIFEST_NAME).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true", help="Create probes in the original CPU environment; refuses overwrite")
    action.add_argument("--verify", action="store_true", help="Verify assets and CPU reconstruction without PyTorch")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seeds", type=int, nargs="+", help="Verify a unique subset of saved model seeds")
    args = parser.parse_args(argv)
    try:
        if args.create:
            if args.seeds is not None:
                parser.error("--create records every saved model; --seeds applies only to --verify")
            create_model_assets(args.results)
        _, seeds, provenance = load_source_metadata(args.results, args.seeds)
        checks = {str(seed): load_model_assets(args.results, seed)[3] for seed in seeds}
    except (AssetValidationError, OSError) as error:
        print(f"Asset validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"verified": True, "gpu_executed": False, "provenance": provenance, "models": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
