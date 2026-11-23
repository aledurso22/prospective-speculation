#!/usr/bin/env bash
# Submit FROM the prospective_speculation directory; see CLUSTER.md.
#SBATCH --job-name=prospective-spec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=prospective-%j.out
#SBATCH --error=prospective-%j.err

set -euo pipefail
# Supply --partition and --account to sbatch as required by your site.
# If your site uses --gpus=1, replace --gres=gpu:1; do not request both.
# Load your site's Python/PyTorch modules here if using its environment.
project_dir="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit this job from the project directory}}"
cd "$project_dir"
[[ -f gpu_benchmark.py ]] || { echo "PROJECT_DIR must be the prospective_speculation folder." >&2; exit 2; }
srun --ntasks=1 bash cluster/run_gpu.sh "${1:-full}"
