#!/usr/bin/env bash
# Execute only after a GPU has been allocated. Never installs anything.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-full}"
python_bin="${BENCHMARK_PYTHON:-$project_dir/.venv/bin/python}"
if [[ "$mode" == "--help" || "$mode" == "-h" ]]; then
  cat <<'HELP'
Usage: bash cluster/run_gpu.sh [smoke|full]

smoke: preflight + seed 17, 32 tokens, one sequence, two repeats, k=4.
full:  preflight + all saved seeds, 512 tokens, five sequences/repeats,
       sequential k=1,2,4,8 sweep on one allocated GPU.

Overrides: BENCHMARK_PYTHON (executable), OUTPUT_ROOT (new directory),
BENCHMARK_CPU_THREADS (default 1), BENCHMARK_TOKENS (full default 512),
BENCHMARK_SEQUENCES (full default 5), BENCHMARK_REPEATS (full default 5).
CUDA_VISIBLE_DEVICES is left under scheduler control.
HELP
  exit 0
fi
case "$mode" in smoke|full) ;; *) echo "Mode must be smoke or full." >&2; exit 2 ;; esac
if [[ ! -x "$python_bin" ]]; then
  echo "Python not found: $python_bin. Run cluster/setup_env.sh or set BENCHMARK_PYTHON." >&2
  exit 2
fi

run_label="${SLURM_JOB_ID:-manual}-$(date -u +%Y%m%dT%H%M%SZ)-$mode"
output_root="${OUTPUT_ROOT:-$project_dir/runs/$run_label}"
if [[ "$output_root" != /* ]]; then output_root="$project_dir/$output_root"; fi
if [[ -e "$output_root" ]]; then
  echo "Output already exists: $output_root. Choose a fresh OUTPUT_ROOT." >&2
  exit 2
fi
mkdir -p "$output_root"

cpu_threads="${BENCHMARK_CPU_THREADS:-1}"
if [[ ! "$cpu_threads" =~ ^[1-9][0-9]*$ ]]; then
  echo "BENCHMARK_CPU_THREADS must be a positive integer." >&2
  exit 2
fi
export OMP_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$project_dir"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$output_root/nvidia-smi.txt" 2>&1 || true
fi
"$python_bin" -m pip freeze > "$output_root/pip-freeze.txt" 2>&1 || true
"$python_bin" -m unittest discover -v 2>&1 | tee "$output_root/test-results.log"
"$python_bin" benchmark_assets.py --verify --results results
"$python_bin" gpu_benchmark.py --device cuda --preflight-only --output "$output_root/preflight" \
  2>&1 | tee "$output_root/preflight.log"

if [[ "$mode" == "smoke" ]]; then
  "$python_bin" gpu_benchmark.py --device cuda --seeds 17 --sequences 1 --tokens 32 --repeats 2 --k 4 \
    --output "$output_root/k4" 2>&1 | tee "$output_root/k4.log"
else
  for k in 1 2 4 8; do
    "$python_bin" gpu_benchmark.py --device cuda --sequences "${BENCHMARK_SEQUENCES:-5}" \
      --tokens "${BENCHMARK_TOKENS:-512}" --repeats "${BENCHMARK_REPEATS:-5}" --k "$k" \
      --output "$output_root/k$k" 2>&1 | tee "$output_root/k$k.log"
  done
fi
"$python_bin" collect_gpu_results.py "$output_root"
echo "Completed $mode run: $output_root"
