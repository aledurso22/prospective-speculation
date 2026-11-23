"""Optional GPU benchmark using an already trained synthetic draft.

This accelerator path was not executed when the initial prototype was built.
It needs a separately installed, hardware-compatible PyTorch and a real CUDA or
Apple MPS device. It never substitutes CPU measurements for GPU measurements.
There is no retraining: teacher weights are regenerated from the saved seed and
calibration, and draft weights are loaded from the CPU experiment's NPZ files.

Example:
    python gpu_benchmark.py --results results --output runs/cuda-001 --repeats 5

The default five repeats estimate timing variability for each context. This is a
small synthetic neural target, not a language-model or KV-cache benchmark.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from adapters import make_adapter
from decoding import generate, generate_target


METHODS = ("target_only", "instant", "leaky", "prospective", "ideal")
REGIMES = {"smooth": 0.94, "abrupt": 0.0}
COUNT_FIELDS = ("rounds", "proposed", "checked", "accepted", "committed")
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def source_code_hashes():
    root = Path(__file__).resolve().parent
    filenames = ("gpu_benchmark.py", "decoding.py", "adapters.py", "models.py", "benchmark_assets.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in filenames}


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_integer(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def atomic_write_text(path, text):
    """Replace one file atomically; incomplete temp files never look complete."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def reserve_output_directory(path):
    """Reserve a new or empty directory. Resuming/overwriting is not implicit."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new run directory.")
    lock = path / ".gpu_benchmark.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(f"pid={os.getpid()}\n")
    return lock


def check_ideal_control(results):
    """Check the matched compensation control against the ordinary draft."""
    instant, ideal = results["instant"], results["ideal"]
    if instant["tokens"] != ideal["tokens"]:
        raise RuntimeError("Ideal-control guard failed: output tokens differ from the instantaneous draft.")
    differences = [key for key in COUNT_FIELDS if instant["stats"][key] != ideal["stats"][key]]
    if differences:
        raise RuntimeError(f"Ideal-control guard failed: counts differ for {differences}.")


def check_generation_counts(result, expected_tokens, forward_calls, positions_evaluated):
    """Check conservation laws using independently counted target evaluations."""
    stats = result["stats"]
    for key in COUNT_FIELDS:
        value = stats[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise RuntimeError(f"Decoder count {key} must be a nonnegative integer.")
    if expected_tokens <= 0 or len(result["tokens"]) != expected_tokens or stats["committed"] != expected_tokens:
        raise RuntimeError("Decoder committed an unexpected number of tokens.")
    if stats.get("total_committed", stats["committed"]) != stats["committed"]:
        raise RuntimeError("Decoder committed-token count aliases disagree.")
    if stats["accepted"] + stats["rounds"] != stats["committed"]:
        raise RuntimeError("Decoder token conservation failed: accepted + rounds must equal committed.")
    if not (0 <= stats["accepted"] <= stats["checked"] <= stats["proposed"]):
        raise RuntimeError("Decoder proposal conservation failed: accepted <= checked <= proposed is required.")
    if forward_calls != stats["rounds"]:
        raise RuntimeError("Target forward-call count differs from verifier rounds.")
    if positions_evaluated != stats["proposed"] + stats["rounds"]:
        raise RuntimeError("Target position count differs from proposed + verifier rounds.")


def check_repeat_control(reference, current, method):
    """Repeated timing of one path must not silently change the workload."""
    if reference["tokens"] != current["tokens"]:
        raise RuntimeError(f"Repeat-determinism guard failed for {method}: output tokens changed.")
    differences = [key for key in COUNT_FIELDS if reference["stats"][key] != current["stats"][key]]
    if differences:
        raise RuntimeError(f"Repeat-determinism guard failed for {method}: counts changed for {differences}.")


def driver_metadata():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5, check=True)
        return {"status": "available", "driver_versions": sorted(set(result.stdout.split()))}
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "reason": str(error)[:300]}


def runtime_metadata(torch, device):
    cpu_model = platform.processor()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name') and ':' in line:
                cpu_model = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    metadata = {
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "cpu_model": cpu_model, "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu_thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "slurm_environment": {name: os.environ.get(name) for name in
                              ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": device,
        "precision": {"weights_and_matmul": "float32", "softmax_and_sampler": "float64",
                      "cuda_matmul_tf32_enabled": False if device == "cuda" else None},
    }
    if device == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["gpu"] = {
            "logical_device_index": index, "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessors": int(properties.multi_processor_count),
            "l2_cache_bytes": getattr(properties, "L2_cache_size", getattr(properties, "l2_cache_size", None)),
        }
        metadata["driver"] = driver_metadata()
    else:
        metadata["gpu"] = {"name": "Apple GPU via PyTorch MPS", "total_memory_bytes": None}
        metadata["driver"] = {"status": "not_applicable"}
    metadata["hardware"] = metadata["gpu"]["name"]
    return metadata


def synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    else:
        raise ValueError("This benchmark requires a CUDA or MPS device.")


def select_device(torch, requested):
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = mps_backend is not None and bool(mps_backend.is_available())
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
    elif requested == "cuda" and cuda_available:
        return "cuda"
    elif requested == "mps" and mps_available:
        return "mps"
    raise RuntimeError(
        f"No usable {requested} GPU is available to this PyTorch installation. "
        "No GPU benchmark was run and no GPU results were written."
    )


class TorchTarget:
    """Frozen row-input MLP; weights remain on the device across all calls."""

    def __init__(self, torch, numpy_model, device):
        self.torch = torch
        self.device = device
        self.weights = tuple(
            torch.as_tensor(np.ascontiguousarray(w, dtype=np.float32), device=device)
            for w in numpy_model.weights
        )
        self.forward_calls = 0
        self.positions_evaluated = 0
        self.weight_bytes = sum(int(w.nbytes) for w in numpy_model.weights)
        synchronize(torch, device)

    def __call__(self, inputs):
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim != 2:
            raise ValueError("Target inputs must have shape [positions, features].")
        torch = self.torch
        with torch.inference_mode():
            x = torch.as_tensor(np.ascontiguousarray(inputs), device=self.device)
            hidden = torch.tanh(x @ self.weights[0])
            hidden = torch.tanh(hidden @ self.weights[1])
            # Copying results to CPU waits for the GPU before verification uses
            # them. Softmax stays in float64, matching the CPU sampler contract.
            logits = (hidden @ self.weights[2]).cpu().numpy().astype(np.float64)
        if not np.isfinite(logits).all():
            raise FloatingPointError("GPU target returned nonfinite logits.")
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        self.forward_calls += 1
        self.positions_evaluated += len(inputs)
        return probabilities


def causal_features(context, count, vocab):
    current = context.clone()
    features = []
    for position in range(count):
        features.append(current.features())
        current.advance(position % vocab)
    return np.asarray(features, dtype=np.float32)


def batch_consistency(target, features, tolerance=1e-5):
    together = target(features)
    separate = np.vstack([target(row[None, :]) for row in features])
    finite = bool(np.isfinite(together).all() and np.isfinite(separate).all())
    difference = float(np.max(np.abs(together - separate)))
    diagnostic = {
        "positions": len(features),
        "finite": finite,
        "max_abs_probability_difference": difference,
        "tolerance": tolerance,
        "passed": finite and difference <= tolerance,
    }
    if not diagnostic["passed"]:
        raise RuntimeError(
            "GPU batch/single-prefix probability check failed: "
            f"max absolute difference {difference:.3g}, tolerance {tolerance:.3g}. "
            "Timing stopped; no distribution-preservation claim is made."
        )
    return diagnostic


def batch_microbenchmark(torch, target, features, k, repeats):
    sizes = sorted(set((1, k + 1)))
    for size in sizes:
        for _ in range(10):
            target(features[:size])
    measurements = {size: [] for size in sizes}
    inner = 20
    for repeat in range(repeats):
        order = sizes if repeat % 2 == 0 else sizes[::-1]
        for size in order:
            synchronize(torch, target.device)
            started = time.perf_counter()
            for _ in range(inner):
                target(features[:size])
            synchronize(torch, target.device)
            measurements[size].append((time.perf_counter() - started) / inner)
    return {
        str(size): {
            "median_ms_per_call": statistics.median(times) * 1000,
            "all_ms_per_call": [value * 1000 for value in times],
            "calls_per_measurement": inner,
            "includes_host_transfers_and_probability_conversion": True,
        }
        for size, times in measurements.items()
    }


def summarize(rows, *, by_model=False):
    """Paired ratios for complete groups; repeats are timing, not new samples."""
    if not rows:
        return []
    groups = {}
    for row in rows:
        key = (row["model_seed"], row["regime"], row["sequence_seed"], row["repeat"])
        group = groups.setdefault(key, {})
        if row["method"] not in METHODS or row["method"] in group:
            raise ValueError("Summary contains an unknown or duplicate method in a paired group.")
        if not np.isfinite(row["wall_seconds"]) or row["wall_seconds"] <= 0:
            raise ValueError("Summary times must be positive and finite.")
        if row["committed"] <= 0 or row["rounds"] <= 0:
            raise ValueError("Summary token and round counts must be positive.")
        group[row["method"]] = row
    for group in groups.values():
        if set(group) != set(METHODS):
            raise ValueError("Cannot summarize an incomplete paired method group.")
        if len({r["committed"] for r in group.values()}) != 1:
            raise ValueError("Paired methods must generate the same number of tokens.")
    seed_groups = sorted({row["model_seed"] for row in rows}) if by_model else [None]
    result = []
    for model_seed in seed_groups:
        for regime in REGIMES:
            for method in METHODS:
                selected = [r for r in rows if r["regime"] == regime and r["method"] == method
                            and (model_seed is None or r["model_seed"] == model_seed)]
                if not selected:
                    continue
                milliseconds = [1000 * r["wall_seconds"] / r["committed"] for r in selected]
                item = {
                    "scope": "per_model" if by_model else "aggregate", "model_seed": model_seed,
                    "regime": regime, "method": method, "runs": len(selected),
                    "model_seeds": len({r["model_seed"] for r in selected}),
                    "ms_per_token_median": statistics.median(milliseconds),
                    "ms_per_token_mean": statistics.mean(milliseconds),
                    "ms_per_token_sd": statistics.stdev(milliseconds) if len(milliseconds) > 1 else 0.0,
                    "tokens_per_verification": sum(r["committed"] for r in selected) / sum(r["rounds"] for r in selected),
                    "accepted_fraction_all_proposals": sum(r["accepted"] for r in selected) / max(1, sum(r["proposed"] for r in selected)),
                    "accepted_fraction_checked_proposals": sum(r["accepted"] for r in selected) / max(1, sum(r["checked"] for r in selected)),
                }
                for label, baseline in (("target", "target_only"), ("instant", "instant")):
                    ratios, per_seed = [], {}
                    for row in selected:
                        key = (row["model_seed"], regime, row["sequence_seed"], row["repeat"])
                        ratio = groups[key][baseline]["wall_seconds"] / row["wall_seconds"]
                        ratios.append(ratio)
                        per_seed.setdefault(row["model_seed"], []).append(ratio)
                    prefix = f"paired_speedup_vs_{label}"
                    item[prefix + "_median"] = statistics.median(ratios)
                    item[prefix + "_mean"] = statistics.mean(ratios)
                    item[prefix + "_sd"] = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
                    item[prefix + "_p25"] = float(np.quantile(ratios, 0.25))
                    item[prefix + "_p75"] = float(np.quantile(ratios, 0.75))
                    medians = [statistics.median(values) for values in per_seed.values()]
                    item[prefix + "_between_model_sd"] = statistics.stdev(medians) if len(medians) > 1 else 0.0
                result.append(item)
    return result



def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results",
                        help="Directory containing the verified source assets and results.json.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New or empty run directory. Existing results/partials are never overwritten.")
    parser.add_argument("--device", choices=("cuda", "auto", "mps"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true", help="Validate device, assets and numerics; do not benchmark.")
    parser.add_argument("--seeds", type=nonnegative_integer, nargs="+", help="Unique subset of saved model seeds.")
    parser.add_argument("--sequences", type=positive_integer, default=5, help="Independent contexts per seed and regime.")
    parser.add_argument("--tokens", type=positive_integer, default=256, help="Committed tokens per run.")
    parser.add_argument("--repeats", type=positive_integer, default=5, help="Timing repeats on the same sampled paths.")
    parser.add_argument("--k", type=nonnegative_integer, help="Draft block size; default: saved setting. Zero disables proposals.")
    parser.add_argument("--torch-threads", type=positive_integer, default=1)
    parser.add_argument("--torch-interop-threads", type=positive_integer, default=1)
    return parser


def prepare_assets(args):
    from benchmark_assets import load_source_metadata
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique; duplicate seeds are not independent replications.")
    source, seeds, provenance = load_source_metadata(args.results, args.seeds)
    saved = source["metadata"]["args"]
    k = saved["k"] if args.k is None else args.k
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("The saved/requested draft length must be a nonnegative integer.")
    for name in ("tau", "tau_a"):
        if not np.isfinite(saved[name]) or saved[name] <= 0:
            raise ValueError(f"Saved {name} must be positive and finite.")
    return source, seeds, provenance, k


def check_source_unchanged(args, source, seeds, provenance):
    from benchmark_assets import load_source_metadata
    current, current_seeds, current_provenance = load_source_metadata(args.results, seeds)
    if current != source or current_seeds != seeds or current_provenance != provenance:
        raise RuntimeError("Source assets changed during this run. Results are left incomplete.")


def gpu_numerics(target, embeddings, seed, args, k, vocab):
    from benchmark_assets import check_reference_probabilities, load_reference_probes
    maximum = min(k + 1, args.tokens)
    diagnostics = {"by_regime": {}, "maximum_used_batch": maximum}
    for regime, decay in REGIMES.items():
        context = make_context_for_benchmark(embeddings, decay, 90000)
        features = causal_features(context, max(maximum, 8), vocab)
        diagnostics["by_regime"][regime] = {
            str(size): batch_consistency(target, features[:size]) for size in range(1, maximum + 1)
        }
        for size in range(1, maximum + 1):
            for _ in range(5):
                target(features[:size])
    probes = load_reference_probes(args.results, seed)
    diagnostics["saved_cpu_reference"] = check_reference_probabilities(target(probes["inputs"]), probes)
    return diagnostics


def make_context_for_benchmark(embeddings, decay, seed):
    from models import make_context
    return make_context(embeddings, decay, seed)


def run_context(torch, target, draft, context, seed, regime, sequence_seed, args, saved, k, vocab):
    """Return a group only when all repeats, methods and control guards pass."""
    pending_rows = []
    repeat_references = {}
    for repeat in range(args.repeats):
        methods = list(METHODS)
        np.random.default_rng(np.random.SeedSequence([seed, sequence_seed, repeat,
                                                     int(regime == "abrupt"), 1701])).shuffle(methods)
        controls = {}
        for method in methods:
            adapter = None if method == "target_only" else make_adapter(
                method, vocab, tau=saved["tau"], tau_a=saved["tau_a"], dt=1.0
            )
            calls_before, positions_before = target.forward_calls, target.positions_evaluated
            synchronize(torch, target.device)
            started = time.perf_counter()
            if method == "target_only":
                result = generate_target(target, context, n_tokens=args.tokens, seed=sequence_seed)
            else:
                result = generate(target, draft, adapter, context, n_tokens=args.tokens, k=k, seed=sequence_seed)
            synchronize(torch, target.device)
            elapsed = time.perf_counter() - started
            stats = dict(result["stats"])
            stats["sampler_wall_seconds"] = stats["wall_seconds"]
            stats["wall_seconds"] = elapsed
            stats["other_seconds"] = max(0.0, elapsed - stats["draft_seconds"] - stats["verify_seconds"])
            calls = target.forward_calls - calls_before
            positions = target.positions_evaluated - positions_before
            check_generation_counts(result, args.tokens, calls, positions)
            if method in repeat_references:
                check_repeat_control(repeat_references[method], result, method)
            else:
                repeat_references[method] = result
            pending_rows.append({
                "model_seed": seed, "regime": regime, "sequence_seed": sequence_seed,
                "repeat": repeat, "method": method, **stats,
                "target_forward_calls": calls,
                "target_positions_evaluated": positions,
            })
            if method in ("instant", "ideal"):
                controls[method] = result
        check_ideal_control(controls)
    return pending_rows


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    lock = None
    checkpoint = args.output / "gpu_checkpoint.json"
    state = {"status": "incomplete", "phase": "source_validation", "metadata": {},
             "runs": [], "asset_diagnostics": {}, "batch_consistency": {},
             "batch_microbenchmark": {}, "completed_context_groups": 0}
    try:
        source, seeds, provenance, k = prepare_assets(args)
        saved = source["metadata"]["args"]
        lock = reserve_output_directory(args.output)
        state["metadata"] = {
            "benchmark_executed": False, "gpu_preflight_executed": False,
            "source_cpu_results": str((args.results / "results.json").resolve()),
            "source_provenance": provenance,
            "source_code_sha256": source_code_hashes(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": seeds, "sequences": args.sequences, "tokens": args.tokens,
            "repeats": args.repeats, "k": k, "tau": saved["tau"], "tau_a": saved["tau_a"],
            "teacher_width": saved["width"], "target_weight_bytes": {},
            "regimes": REGIMES, "preflight_only": args.preflight_only,
            "timing": "perf_counter around complete generation; device synchronized before/after; CPU drafting, sampling and all transfers included",
            "method_order": "randomized per model, context and timing repeat",
            "replication": "repeats reuse identical sampled paths and measure timing noise; model seeds, contexts and repeats are distinct replication levels",
            "sampling": "same context and integer seed across methods; methods consume random draws differently and can return different token paths",
            "scope": "one request; small synthetic MLP GPU target with CPU drafter and CPU/GPU transfers; no LLM or KV cache",
            "memory_caveat": "Default target weights are about 17 MB and may fit in GPU L2. These timings do not establish an HBM-bandwidth benefit.",
            "primary_comparison": "prospective versus instant, paired by model seed/context/repeat",
            "execution_guards": ["token/proposal/target-call conservation", "identical token paths and counts across timing repeats",
                                 "ideal-control tokens and counts equal instantaneous control"],
            "initial_development_validation": "GPU execution was not performed while preparing this prototype",
        }
        state["phase"] = "device_initialization"
        atomic_write_json(checkpoint, state)
        try:
            import torch
        except (ImportError, OSError) as error:
            raise RuntimeError("A working GPU-compatible PyTorch installation is required. No GPU benchmark ran: " + str(error)) from error
        device = select_device(torch, args.device)
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(args.torch_interop_threads)
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")
        state["metadata"].update(runtime_metadata(torch, device))
        print(json.dumps(state["metadata"]), flush=True)
        from benchmark_assets import load_model_assets
        from models import VOCAB
        for seed in seeds:
            state["phase"] = f"model_{seed}_preflight"
            model, draft, embeddings, asset_diagnostics = load_model_assets(args.results, seed)
            target = TorchTarget(torch, model, device)
            state["metadata"]["target_weight_bytes"][str(seed)] = target.weight_bytes
            state["asset_diagnostics"][str(seed)] = asset_diagnostics
            state["batch_consistency"][str(seed)] = gpu_numerics(target, embeddings, seed, args, k, VOCAB)
            guard_context = make_context_for_benchmark(embeddings, 0.94, 90001)
            guard_results = {
                name: generate(target, draft, make_adapter(name, VOCAB, tau=saved["tau"], tau_a=saved["tau_a"], dt=1.0),
                               guard_context, n_tokens=min(args.tokens, 16), k=k, seed=90001)
                for name in ("instant", "ideal")
            }
            check_ideal_control(guard_results)
            state["batch_consistency"][str(seed)]["ideal_control_guard"] = {
                "passed": True, "tokens": min(args.tokens, 16), "timings_reported": False
            }
            state["metadata"]["gpu_preflight_executed"] = True
            atomic_write_json(checkpoint, state)
            print(f"Verified assets, GPU probabilities and all used batch shapes for model {seed}.", flush=True)
            if not args.preflight_only:
                warm_context = make_context_for_benchmark(embeddings, 0.94, 90000)
                effective_k = min(k, args.tokens - 1)
                features = causal_features(warm_context, effective_k + 1, VOCAB)
                state["batch_microbenchmark"][str(seed)] = batch_microbenchmark(torch, target, features, effective_k, args.repeats)
                for regime, decay in REGIMES.items():
                    for sequence in range(args.sequences):
                        sequence_seed = 80000 + sequence
                        state["phase"] = f"model_{seed}_{regime}_context_{sequence}"
                        context = make_context_for_benchmark(embeddings, decay, sequence_seed)
                        group = run_context(torch, target, draft, context, seed, regime, sequence_seed,
                                            args, saved, k, VOCAB)
                        # No incomplete method/context groups enter a saved checkpoint.
                        state["runs"].extend(group)
                        state["completed_context_groups"] += 1
                        state["metadata"]["benchmark_executed"] = True
                        atomic_write_json(checkpoint, state)
                        print(f"Saved complete group: model={seed} regime={regime} context={sequence + 1}/{args.sequences}", flush=True)
            del target
            if device == "cuda":
                torch.cuda.empty_cache()
            elif hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        state["phase"] = "final_source_validation"
        check_source_unchanged(args, source, seeds, provenance)
        if source_code_hashes() != state["metadata"]["source_code_sha256"]:
            raise RuntimeError("Benchmark source code changed during the run; results remain incomplete.")
        state["metadata"]["completed_utc"] = datetime.now(timezone.utc).isoformat()
        state["metadata"]["source_hashes_rechecked_at_completion"] = True
        if args.preflight_only:
            state.update(status="complete", phase="preflight_only", result_kind="preflight")
            atomic_write_json(args.output / "gpu_preflight.json", state)
            print("GPU preflight passed. No timed benchmark or speedup results were produced.", flush=True)
        else:
            state["summary"] = summarize(state["runs"])
            state["per_model_summary"] = summarize(state["runs"], by_model=True)
            summary_rows = state["summary"] + state["per_model_summary"]
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
            atomic_write_text(args.output / "gpu_results.csv", stream.getvalue())
            state.update(status="complete", phase="complete", result_kind="benchmark")
            # JSON is the completion marker; CSV alone never denotes a complete run.
            atomic_write_json(args.output / "gpu_results.json", state)
            print(json.dumps(state["summary"], indent=2), flush=True)
            print(f"Saved complete measured GPU results to {args.output.resolve()}", flush=True)
        # Failure to clean up a redundant checkpoint must not invalidate a
        # successfully committed completion marker.
        try:
            checkpoint.unlink(missing_ok=True)
        except OSError as cleanup_error:
            print(f"Completed results are saved; checkpoint cleanup failed: {cleanup_error}", file=sys.stderr)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        state["status"] = "incomplete"
        state["failed_during"] = state["phase"]
        state["phase"] = "interrupted" if interrupted else "failed"
        state["error"] = {"type": type(error).__name__, "message": str(error) or "Interrupted by user"}
        if lock is not None:
            try:
                atomic_write_json(checkpoint, state)
            except OSError as save_error:
                print(f"Could not preserve the partial checkpoint: {save_error}", file=sys.stderr)
        print(f"GPU benchmark stopped: {state['error']['message']}", file=sys.stderr)
        if lock is not None:
            print(f"Incomplete progress: {checkpoint}. Choose a new output directory for another run.", file=sys.stderr)
        return 130 if interrupted else 2
    finally:
        if lock is not None:
            try:
                lock.unlink(missing_ok=True)
            except OSError as cleanup_error:
                print(f"Run lock cleanup failed: {cleanup_error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
