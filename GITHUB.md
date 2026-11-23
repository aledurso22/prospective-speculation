# Publish this folder and clone it on the cluster

Use `prospective_speculation/` as the repository root. It includes the code, tests, launch scripts, saved drafts, target reference probes, and original CPU results. The enclosing workspace and local virtual environments are not part of the repository.

This is the validated synthetic inference-filter experiment. The separately trained prospective recurrent drafter discussed as a next experiment is not implemented in this release. GPU execution still needs the allocated-node smoke test.

## Publish from your computer

The local repository uses the `main` branch. Inspect the tracked files with `git ls-files`, then create a new GitHub repository and push using GitHub CLI:

```bash
cd /absolute/path/to/prospective_speculation
gh repo create prospective-speculation --public --source=. --remote=origin --push
```

GitHub CLI must be authenticated to the intended account (`gh auth status`). This command creates a new public repository. If that name already exists, inspect the existing repository and connect its remote deliberately instead of overwriting it.

Public visibility permits unauthenticated HTTPS cloning. It is optional: use `--private` if preferred, then authenticate the cluster through your site's supported GitHub method before cloning. Never put access tokens in a clone URL or tracked file.

The supplied `.gitignore` excludes the virtual environment, caches, new benchmark runs, intermediate checkpoints, and common local credential files. The committed `results/` files are required baseline assets; do not exclude that directory. The existing NPZ files are small and do not require Git LFS.

## Clone and install on the cluster login node

Replace `YOUR_GITHUB_USERNAME` with the account owning the new repository:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/prospective-speculation.git
cd prospective-speculation

# Load your site's Python 3.12+ module first if required.
bash cluster/setup_env.sh cu128
.venv/bin/python -m unittest discover -v
.venv/bin/python benchmark_assets.py --verify --results results
git rev-parse HEAD
```

Record the commit alongside the run outputs. Install dependencies on a login/setup node with network access, then execute CUDA work inside a GPU allocation. If the cluster provides a compatible environment, follow the `BENCHMARK_PYTHON` instructions in [CLUSTER.md](CLUSTER.md).

## Submit the initial single-GPU test

Use the partition and account names supplied by your cluster. The node below is the RTX 4090 example from your available-node list:

```bash
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT \
  --nodelist=pgi15-gpu9 --time=00:10:00 cluster/gpu.slurm smoke
```

Inspect the job log and its new `runs/...-smoke/SUMMARY.md`. After a successful smoke run:

```bash
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT \
  --nodelist=pgi15-gpu9 cluster/gpu.slurm full
```

The template requests one GPU. Other idle RTX 4090 nodes `pgi15-gpu10` through `pgi15-gpu13` are alternatives, subject to current scheduler availability. See [CLUSTER.md](CLUSTER.md) for site-specific GPU request syntax and output interpretation.

## Bring later code changes to the cluster

After committing and pushing changes from your computer, run `git pull --ff-only` in the cluster checkout **between jobs**. Do not change source files during a benchmark; the runner checks their hashes. Results under `runs/` remain local and ignored by Git. Copy them back separately for analysis.

## References

- [GitHub: adding locally hosted code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [GitHub CLI: create a repository](https://cli.github.com/manual/gh_repo_create)
- [GitHub: clone a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
