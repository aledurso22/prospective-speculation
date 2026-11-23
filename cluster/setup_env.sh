#!/usr/bin/env bash
# Run on a network-enabled login/setup node, not inside the benchmark job.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
venv_dir="${BENCHMARK_VENV:-$project_dir/.venv}"
torch_version="${TORCH_VERSION:-2.9.1}"
cuda_build="${1:-cu128}"

if [[ "$cuda_build" == "--help" || "$cuda_build" == "-h" ]]; then
  cat <<'HELP'
Usage: bash cluster/setup_env.sh [cu126|cu128|cu130]

Creates a Linux virtual environment and installs numpy==2.5.5 and a pinned
PyTorch CUDA wheel. The default recipe is torch==2.9.1 / cu128, not a claim
that this CUDA build fits every cluster. Select a build supported by your
GPU and site driver; see CLUSTER.md. No driver/toolkit is installed.

Overrides: PYTHON_BIN, BENCHMARK_VENV, TORCH_VERSION.
If your site already provides a working CUDA PyTorch environment, skip this
script and set BENCHMARK_PYTHON to that environment's Python executable.
HELP
  exit 0
fi
case "$cuda_build" in cu126|cu128|cu130) ;; *) echo "Choose cu126, cu128, or cu130." >&2; exit 2 ;; esac

"$python_bin" - <<'PY'
import platform, sys
if sys.version_info < (3, 12):
    raise SystemExit('Python 3.12+ is required by the pinned NumPy 2.5.5. Load a newer Python module first.')
if platform.system() != 'Linux':
    raise SystemExit('This installer is for the Linux NVIDIA cluster. It does not install CUDA wheels on macOS.')
print('Setup Python:', sys.version.split()[0], platform.machine())
PY

if [[ ! -x "$venv_dir/bin/python" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi
"$venv_dir/bin/python" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit('The selected virtual environment uses Python older than 3.12. Choose a new BENCHMARK_VENV with Python 3.12+.')
PY
"$venv_dir/bin/python" -m pip install -r "$project_dir/requirements.txt"
"$venv_dir/bin/python" -m pip install "torch==$torch_version" --index-url "https://download.pytorch.org/whl/$cuda_build"
"$venv_dir/bin/python" -m pip check
"$venv_dir/bin/python" - <<'PY'
import sys, numpy, torch
print('Python:', sys.version.split()[0])
print('NumPy:', numpy.__version__)
print('PyTorch:', torch.__version__, 'wheel CUDA:', torch.version.cuda)
if torch.version.cuda is None:
    raise SystemExit('The installed PyTorch is CPU-only. Select a CUDA wheel or your site environment.')
print('GPU availability is checked later, inside the GPU allocation.')
PY
"$venv_dir/bin/python" "$project_dir/benchmark_assets.py" --verify --results "$project_dir/results"
echo "Environment ready at $venv_dir. Run the CPU tests, then the GPU smoke job in CLUSTER.md."
