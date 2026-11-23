# Run on an NVIDIA cluster

The folder is self-contained: trained draft checkpoints, target reconstruction metadata, reference probes, tests, and launch scripts are included. No retraining or model download is needed. Create a fresh environment on Linux; do not copy the Mac virtual environment.

This package has been checked locally on CPU. CUDA execution and performance must be validated on your allocated GPU. It uses a **small synthetic neural target**, not a pretrained language model.

## 1. Environment on a login/setup node

Use Python **3.12 or newer** with the pinned `numpy==2.5.5`. Prefer your cluster's maintained CUDA PyTorch environment if it has these dependencies. Otherwise, from this project directory:

```bash
# Load your site's Python 3.12+ module first, if required.
PYTHON_BIN=python3 bash cluster/setup_env.sh cu128
```

The provided recipe pins `torch==2.9.1` and uses its official CUDA 12.8 wheel. `cu126` and `cu130` are explicit alternatives supported for this pinned PyTorch release. Select the build your GPU and cluster driver support; the default is not universal. `TORCH_VERSION` and `BENCHMARK_VENV` can override the recipe. Install dependencies on a node with network access, before submitting jobs.

No NVIDIA driver or system CUDA toolkit is installed by this script. A sufficiently compatible host driver is still necessary. The CUDA version displayed by `nvidia-smi` is driver capability information, not proof that a particular wheel works. The allocated-node preflight performs actual CUDA computations.

For an existing site environment, skip setup and use its Python:

```bash
export BENCHMARK_PYTHON=/absolute/path/to/your/environment/bin/python
```

The scripts otherwise use `.venv/bin/python` inside this project. When using `BENCHMARK_VENV` at setup, set `BENCHMARK_PYTHON` to that environment's `bin/python` for running.

## 2. CPU checks before requesting a GPU

```bash
.venv/bin/python -m unittest discover -v
.venv/bin/python benchmark_assets.py --verify --results results
```

Use `$BENCHMARK_PYTHON` instead if you selected a site environment. These checks do not require CUDA. They validate sampling, state rollback, model reconstruction and checkpoint provenance. The model asset manifest is supplied; do not regenerate it on the cluster to silence a mismatch.

## 3. Submit a short smoke job first

Submit **from this project directory**. Supply your site's partition and account (omit account if your site does not require it):

```bash
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT \
  --time=00:10:00 cluster/gpu.slurm smoke
```

The template requests one GPU, four CPU cores, 16 GB of host memory, and one node. It uses `--gres=gpu:1`; sites using `--gpus=1` should replace that directive. Do not request both forms. Add your site's module loads to the template if required. Let Slurm set `CUDA_VISIBLE_DEVICES`.

The wrapper first runs the CPU test suite. The smoke run then checks CUDA execution, model assets, numerical agreement and one short decoding experiment with two timing repeats to check reproducibility. It writes into a new `runs/<job>-<timestamp>-smoke/` directory. Inspect its Slurm log and `SUMMARY.md` before submitting the full job.

For an already allocated GPU, the equivalent direct command is:

```bash
bash cluster/run_gpu.sh smoke
```

## 4. Full experiment

```bash
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT cluster/gpu.slurm full
```

This runs a sequential sweep over draft lengths **1, 2, 4, 8** on the same allocated GPU. Each setting uses all three saved model seeds, two context regimes, five sequences per model/regime, five timing repeats and 512 output tokens per run. The five methods are target-only, ordinary speculation (`instant`), leaky speculation, prospective correction, and ideal lag cancellation. No hyperparameter is selected using these test results.

The default one-hour reservation is a configurable allowance, not a measured runtime estimate. Override the workload through exported environment variables if needed:

```bash
export BENCHMARK_TOKENS=256
export BENCHMARK_SEQUENCES=3
export BENCHMARK_REPEATS=3
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT cluster/gpu.slurm full
```

One CPU math thread is used by default for the small drafter; `BENCHMARK_CPU_THREADS` changes this consistently. The wrapper records `nvidia-smi`, `pip freeze`, and runner metadata. It makes no network requests. Do not run competing benchmarks on the same GPU while comparing timings.

## 5. Outputs and interpretation

The run root contains:

| File | Meaning |
|---|---|
| `preflight/gpu_preflight.json` | Real-device and numerical checks; not a speed benchmark |
| `k*/gpu_results.json` | Completed runs, raw observations, hardware/package metadata, numerical checks and summaries |
| `k*/gpu_results.csv` | Aggregate and per-model results, distinguished by the `scope` column |
| `k*/gpu_checkpoint.json` | Progress checkpoint; incomplete files are not completed results |
| `comparison.csv` and `SUMMARY.md` | All completed draft lengths together |
| `pip-freeze.txt`, `nvidia-smi.txt`, `*.log` | Runtime and diagnostic records |

Every run requires a new output directory. Partial checkpoints are for inspection; automatic resume is not implemented. Rerun an interrupted setting into a new directory. Do not overwrite the original CPU `results/` directory. To summarize available completed settings after an interruption:

```bash
.venv/bin/python collect_gpu_results.py runs/YOUR_RUN_DIRECTORY
```

The primary metric is **paired speedup of prospective versus instantaneous drafting**. A ratio above 1 favors prospective correction. Also inspect tokens per verification, per-model summaries, and runtime variability. Repeating one context five times estimates timing noise; it does not create five new models or tasks.

Read [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md) before interpreting the run. The primary setting is `k=4`; the remaining sweep is exploratory. A completed job is not a scientific pass. The current CPU fidelity evidence is unfavorable, and the revised ordinary baseline removes unnecessary stateless bookkeeping. The supplied historical CPU timings are not measurements of this revised baseline. Numerical and reproducibility failures must be investigated rather than bypassed to obtain a speed report.

The 2048-wide target contains approximately 17 MB of float32 weights, which may fit GPU cache. Drafting and sampling remain on CPU, and host/device transfers are included. Results apply to this implementation; they do not establish benefits for a large Transformer's HBM traffic, KV cache, multi-request throughput, or prospective learning during training. This first cluster run checks whether the CPU performance conclusion changes on actual GPU hardware.

## Troubleshooting

- **No GPU / CPU-only PyTorch:** run inside an allocation; verify the selected Python and `torch.version.cuda`. The runner never falls back to CPU timing.
- **Unsupported kernel / driver:** use the site's compatible PyTorch module or choose an appropriate official CUDA wheel. Do not change system drivers through this project.
- **Asset mismatch:** copy the original checkpoints and asset manifest again; use the recorded NumPy version. Never regenerate the baseline manifest merely to make the check pass.
- **Numerical preflight failure:** retain the diagnostic output and package/device versions. TF32 is disabled for the controlled float32 comparison. Do not raise tolerance blindly.
- **Job stops partway:** retain completed `k*` directories and checkpoints, increase the wall-time allowance if needed, and start the failed setting in a fresh directory.

## Official installation and scheduler references

- [PyTorch CUDA wheel recipes](https://pytorch.org/get-started/previous-versions/)
- [NumPy 2.5.5 package requirements](https://pypi.org/project/numpy/2.5.5/)
- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)
- [Slurm sbatch options](https://slurm.schedmd.com/sbatch.html)
