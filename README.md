# Prospective coding + speculative decoding

A controlled, reproducible **synthetic** experiment. It uses a fixed nonlinear neural target, a smaller distilled draft, and exact speculative rejection sampling. Four otherwise identical drafts compare ordinary inference, leaky transmission, the finite-adaptation prospective mechanism, and an ideal lag-cancellation control.

For the NVIDIA cluster run, start with **[CLUSTER.md](CLUSTER.md)**. It contains environment setup, a short GPU smoke job, the full Slurm experiment, and result collection. The saved drafts and verified model assets are included; no retraining is needed.

To publish this folder on GitHub and clone it on the cluster, follow **[GITHUB.md](GITHUB.md)**. Public visibility is optional; private cloning requires authentication on the cluster.

Read `results/REPORT.md` for the original CPU findings and limitations. `results/results.json` contains every original CPU run; `results/summary.csv` provides aggregate measurements.

**Critical audit:** the current prospective filter has worse paired draft fidelity than the ordinary draft in all 30 saved test contexts. See [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md) and [the reproducible evidence audit](results/AUDIT_EVIDENCE.md). The ordinary stateless baseline has since been strengthened, so the original CPU timings are historical rather than timings of the revised code. Passing tests establishes correctness checks, not a performance discovery.

## CPU reproduction

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python run_experiment.py --device cpu --output cpu_rerun
```

Keep the original `results/` unchanged: its hashes and probes anchor the GPU comparison. The pinned NumPy version records the measured environment. Other versions must pass the model asset checks before use.

## NVIDIA cluster quick start

Copy or clone this folder including `results/`. Do not copy a local virtual environment.

```bash
# On the login/setup node, using Python 3.12+:
bash cluster/setup_env.sh cu128
.venv/bin/python -m unittest discover -v

# Replace partition/account with your site's settings:
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT --time=00:10:00 cluster/gpu.slurm smoke
# Inspect the smoke result, then:
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT cluster/gpu.slurm full
```

The default installation recipe is pinned PyTorch 2.9.1 / CUDA 12.8; choose a CUDA build compatible with the cluster or use its existing environment. See [CLUSTER.md](CLUSTER.md). Account, partition, module names and GPU-request syntax are site-specific.

The full job compares draft lengths 1, 2, 4, and 8 sequentially on one GPU. It includes GPU synchronization, checkpoint validation, numerical preflight, paired comparisons against ordinary speculation, hardware metadata, and progress checkpoints. Outputs are separate from the original CPU results.

CUDA execution remains untested locally. The runner fails clearly when GPU access is unavailable and never reports CPU timing as GPU performance.

## Scope

- This is not a pretrained language-model experiment. The target defines a synthetic autoregressive token distribution.
- The draft is trained with ordinary distillation. NLA plasticity and prospective teaching-signal synchronization during training are not implemented.
- The adapter follows the adaptive-input mechanism in the synchronization paper, but the input is held constant between token events. The simulation clock is distinct from execution time.
- Improving on the deliberately leaky control alone is insufficient evidence of a useful improvement over ordinary speculative decoding.
- No external dataset, paid compute service, or model download is needed for the CPU experiment.

## Files

- `models.py`: synthetic persistent context, neural target, manual NumPy distillation.
- `adapters.py`: instantaneous, leaky, prospective and ideal adapters.
- `decoding.py`: exact rejection sampling with adapter-state rollback.
- `run_experiment.py`: paired controls, multiple seeds and CPU timing.
- `gpu_benchmark.py`: real-device benchmark and preflight.
- `benchmark_assets.py`: immutable model provenance and reference checks.
- `cluster/`: Linux setup, GPU wrapper, and Slurm job.
- `collect_gpu_results.py`: comparison CSV and Markdown summary.
- `audit_evidence.py`: paired fidelity analysis and perfect-draft controls, independent of GPU timing.
- `RESEARCH_REVIEW.md`: evidence, limitations, prior work, and criteria for the GPU experiment.
- `test_*.py`: mathematical, numerical, state and training checks.
