# Running the benchmark on the IFI SLURM cluster

Seven registered configs, **3,780 independent tasks** in total.
The queue manager (`slurm/submit_all.py`) submits all of them while staying
within the Krater/NvidiaAll partition limit of **30 jobs** (15 running + 15
pending). Task counts are always recomputed from the config
(`slurm/task_grid.py:build_all_runs` is the single source of truth) — nothing
below is hardcoded.

| Config key | File | Tasks | Partition | RQ |
|------------|------|-------|-----------|----|
| `accuracy` | `configs/RQ1-accuracy/config-accuracy.yaml` | 200 | Krater | Approximation accuracy vs. background size |
| `dimensionality` | `configs/RQ2-dimensionality/config-dimensionality.yaml` | 480 | Krater | Scalability with feature count |
| `tree` | `configs/RQ4-tree/config-tree.yaml` | 1050 | Krater | Tree-native backends vs. model-agnostic |
| `tree-fasttreeshap` | `configs/RQ4-tree/config-tree-fasttreeshap.yaml` | 700 | Krater | fasttreeshap-only repair sweep (needs its own venv, see step 4b) |
| `nn` | `configs/RQ3-neural-networks/config-neural-networks-gpu.yaml` | 150 | NvidiaAll | Gradient-based backends for neural networks (device=cuda) |
| `nn-cpu` | `configs/RQ3-neural-networks/config-neural-networks-cpu.yaml` | 150 | Krater | Same sweep, device=cpu |
| `tree-gpu` | `configs/RQ5-gpu/config-tree-gpu.yaml` | 1050 | NvidiaAll | woodelf CPU vs. GPU (cupy) backends |

The full config key list is defined in `slurm/submit_all.py`'s `CONFIG_REGISTRY`
— check there if this table looks stale.

---

## 0. Prerequisites — CIP account & SSH activation

You need a **CIP Kennung** (separate from your LMU Campus account — different
username and password).

1. Go to <https://conf.cip.ifi.lmu.de/> and log in with your CIP credentials.
   If you don't have a CIP account yet, log in with your LMU Campus account and
   request one.
2. In CipConf, find **"Remote Enabler"** (SSH/RDP access) and activate it.
3. Wait a few minutes for the change to propagate.

Without step 2 you get `Permission denied (publickey,password)` even with the
correct password.

---

## 1. Login

```bash
ssh <cip-kennung>@remote.cip.ifi.lmu.de
```

`remote.cip.ifi.lmu.de` is a load-balanced pool of login nodes. Do **not** run
computations here — submit everything through SLURM. Sessions are lost on weekly
reboots, so don't use it for long interactive work.

> **Tip — avoid typing your password every time:**
> ```bash
> # On your Mac, generate a key if you don't have one yet
> ssh-keygen -t ed25519
> # Copy it to the cluster
> ssh-copy-id <cip-kennung>@remote.cip.ifi.lmu.de
> ```

---

## 2. First-time setup on the cluster

Your home directory is **NFS-mounted** across all nodes, so install once and it
works everywhere.

```bash
# Clone the repo
git clone <your-repo-url> ~/PR_ModeAgnostic
cd ~/PR_ModeAgnostic

# Install uv (downloads to ~/.local/bin/uv — no sudo needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv manages Python itself — Ubuntu 24.04 ships 3.12 but the project needs >=3.13
uv python install 3.13

# Install all dependencies from uv.lock into .venv/
uv sync
```

---

## 3. Pre-download datasets (do this once on the login node)

Compute nodes may have no outbound internet access. Cache all datasets first.

To check what is already cached (never downloads anything — it blocks network
access and reports any dataset that would need it):

```bash
uv run python scripts/check_dataset_cache.py                                # all configs/*/*.yaml
uv run python scripts/check_dataset_cache.py configs/RQ4-tree/config-tree.yaml
```

Exit code 0 means every dataset loads offline; 1 lists the missing ones, which
the snippet below then downloads. Caching is per **dataset** only — the
`n_features`/`n_samples` sweep values in a config's `datasets:` section are
applied by subsampling/feature-selecting the already-loaded data in memory
(`datasets/load_datasets.py:_load_spec`), so there's no need to fetch a
separate copy per `(n_features, n_samples)` combination. Run once per unique
dataset key across **every** config in `configs/*/*.yaml` (not just the
configs in the table above — this glob matches what
`check_dataset_cache.py` checks by default, so it can't go stale as configs
are added):

```bash
cd ~/PR_ModeAgnostic
uv run python - <<'EOF'
import glob, yaml
from models.dataset_and_models import Dataset

seen = set()
for cfg_path in sorted(glob.glob("configs/*/*.yaml")):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["benchmark"]["seed"]
    seed = seed[0] if isinstance(seed, list) else seed
    for ds_key in cfg["datasets"]:
        if ds_key in seen:
            continue
        seen.add(ds_key)
        print(f"Fetching {ds_key} ...")
        Dataset[ds_key.upper()].load_dataset(seed=seed)
print(f"Done — {len(seen)} unique datasets cached.")
EOF
```

Re-run `scripts/check_dataset_cache.py` afterwards to confirm everything is
cached before submitting.

---

## 4. Check available partitions

```bash
sinfo
```

`accuracy`, `dimensionality`, `tree`, `tree-fasttreeshap`, and `nn-cpu` are
CPU-only and run on `Krater` (or any equivalent standard partition, e.g.
`Gesteine_A`). `nn` and `tree-gpu` need a GPU node — the `NvidiaAll` partition.
`slurm/submit_all.py` routes each config key to the right partition
automatically via `sbatch_args` in `CONFIG_REGISTRY`; `slurm/select_array_script.sh`
does the equivalent routing for `submit.sh` (legacy single-config path).

If the partition names differ on your allocation, edit the `--partition=`
line in `slurm/single_task.sh` (used by `submit_all.py`) and in
`slurm/bench_array.sh`, `slurm/bench_array_gpu.sh`, `slurm/bench_array_nn.sh`,
`slurm/bench_array_nn_cpu.sh` (used by `submit.sh`) before submitting.

> **Note on GPU jobs:** the CIP cluster defines no GPU GRES — `sinfo -o "%P %G"`
> shows `GRES=(null)` on every partition — so `--gres=gpu:1` / `--gpus=1` are
> rejected with `Invalid generic resource specification`. Do not add them.
> Requesting `--partition=NvidiaAll` alone is sufficient: the node's GPU is
> directly visible to the job.

### 4b. Extra prerequisite for `tree-fasttreeshap`

The fasttreeshap backend needs `numpy<2`, which conflicts with this project's
main `numpy>=2` environment, so it runs out of its own dedicated venv. Provision
it once on the login node before submitting `tree-fasttreeshap`:

```bash
bash scripts/setup_fasttreeshap_env.sh
```

This creates `~/.cache/pr-modeagnostic/.venv-fasttreeshap` (needs `python3.10`
on PATH, or set `FASTTREESHAP_PYTHON_BIN`). No action needed for the other
configs.

---

## 5. Submit

### Run all configs at once (recommended)

Open a **tmux or screen session** first — the queue manager is a long-running
process that must stay alive until all tasks finish:

```bash
tmux new -s bench       # or: screen -S bench
cd ~/PR_ModeAgnostic
uv run python slurm/submit_all.py --configs all
```

The script prints a task summary, then enters a poll loop:

```
Config task counts:
  accuracy         200 tasks  (configs/RQ1-accuracy/config-accuracy.yaml)
  dimensionality   480 tasks  (configs/RQ2-dimensionality/config-dimensionality.yaml)
  tree            1050 tasks  (configs/RQ4-tree/config-tree.yaml)
  tree-fasttreeshap 700 tasks  (configs/RQ4-tree/config-tree-fasttreeshap.yaml)
  nn               150 tasks  (configs/RQ3-neural-networks/config-neural-networks-gpu.yaml)
  nn-cpu           150 tasks  (configs/RQ3-neural-networks/config-neural-networks-cpu.yaml)
  tree-gpu        1050 tasks  (configs/RQ5-gpu/config-tree-gpu.yaml)

Total: 3780 tasks | MAX_JOBS=30 | poll every 60s
```

It submits up to 30 jobs, then wakes every 60 seconds, detects completions via
`squeue`, and tops up the queue. When all tasks are done it submits one merge
job per config automatically.

### Run a subset of configs

```bash
# Only accuracy and dimensionality
uv run python slurm/submit_all.py --configs accuracy dimensionality

# Only the tree sweeps
uv run python slurm/submit_all.py --configs tree tree-fasttreeshap

# Only neural networks (both device variants)
uv run python slurm/submit_all.py --configs nn nn-cpu
```

Valid config keys: `accuracy`, `dimensionality`, `tree`, `tree-fasttreeshap`,
`nn`, `nn-cpu`, `tree-gpu` (see `slurm/submit_all.py --help`).

### Submit a single config (legacy, no queue management)

Use `submit.sh` when you only need one config and don't need queue throttling —
it submits the full SLURM array in one shot. It accepts either a full path or
just the filename (searched under `configs/*/`), and picks the right array
script automatically via `slurm/select_array_script.sh`:

```bash
bash slurm/submit.sh configs/RQ1-accuracy/config-accuracy.yaml
bash slurm/submit.sh configs/RQ2-dimensionality/config-dimensionality.yaml
bash slurm/submit.sh configs/RQ4-tree/config-tree.yaml
bash slurm/submit.sh config-tree-fasttreeshap.yaml   # bare filename also works
bash slurm/submit.sh configs/RQ3-neural-networks/config-neural-networks-gpu.yaml
bash slurm/submit.sh configs/RQ3-neural-networks/config-neural-networks-cpu.yaml
bash slurm/submit.sh configs/RQ5-gpu/config-tree-gpu.yaml
```

> **Warning:** `submit.sh` submits all tasks at once with no concurrency limit.
> If the task count exceeds 30, SLURM may reject or heavily queue the
> submission. Use `submit_all.py` when in doubt — especially for `tree`,
> `tree-fasttreeshap`, and `tree-gpu`, which each have 700+ tasks.

Run any subset — each config gets its own output directory and merged CSV, so
they never collide.

Each submission:
1. Counts the task grid via `slurm/count_tasks.py` (same `build_all_runs` logic
   `submit_all.py` uses).
2. Submits a SLURM **array job** — one task per grid cell, each writing to its
   own CSV under `benchmarking/slurm_results/<config_name>/` so there are no
   race conditions.
3. Submits a **merge job** that runs automatically after all array tasks succeed
   (`--dependency=afterok`), combining results into
   `benchmarking/results_<config_name>.csv`.

---

## 6. Monitor

```bash
squeue -u $USER                           # all your running/pending jobs

# Live output for a task submitted via submit_all.py
tail -f slurm/logs/task_<JOBID>.out

# Live output for a task submitted via submit.sh (array jobs)
tail -f slurm/logs/bench_<ARRAYJOBID>_<TASKID>.out       # accuracy/dimensionality/tree/tree-fasttreeshap/tree-gpu
tail -f slurm/logs/bench_nn_<ARRAYJOBID>_<TASKID>.out    # nn/nn-cpu
```

Cancel everything if needed:

```bash
scancel -u $USER
```

---

## 7. Retrieve results

After the merge job(s) finish, copy the merged CSVs back to your Mac:

```bash
# Run this on your Mac
scp '<cip-kennung>@remote.cip.ifi.lmu.de:~/PR_ModeAgnostic/benchmarking/results_config-*.csv' \
    benchmarking/
```

Or with rsync:

```bash
rsync -avz <cip-kennung>@remote.cip.ifi.lmu.de:~/PR_ModeAgnostic/benchmarking/ \
    benchmarking/ --include='results_*.csv' --exclude='*'
```

The merged files produced (one per config key, named after the config file):

| File | Config key |
|------|------------|
| `benchmarking/results_config-accuracy.csv` | `accuracy` |
| `benchmarking/results_config-dimensionality.csv` | `dimensionality` |
| `benchmarking/results_config-tree.csv` | `tree` |
| `benchmarking/results_config-tree-fasttreeshap.csv` | `tree-fasttreeshap` |
| `benchmarking/results_config-neural-networks-gpu.csv` | `nn` |
| `benchmarking/results_config-neural-networks-cpu.csv` | `nn-cpu` |
| `benchmarking/results_config-tree-gpu.csv` | `tree-gpu` |

---

## File overview

```
slurm/
├── submit_all.py           ← main entry point: queue manager for all registered configs
├── submit.sh                ← legacy single-config submitter (no queue throttling)
├── select_array_script.sh   ← maps a config path to the right array script (used by submit.sh)
├── single_task.sh           ← generic sbatch wrapper used by submit_all.py
├── bench_array.sh            ← SLURM array script, CPU (accuracy/dimensionality/tree/tree-fasttreeshap)
├── bench_array_gpu.sh        ← SLURM array script, GPU, non-NN (tree-gpu)
├── bench_array_nn.sh         ← SLURM array script, GPU, NN (nn, device=cuda)
├── bench_array_nn_cpu.sh     ← SLURM array script, CPU, NN (nn-cpu, device=cpu)
├── merge.sh                  ← SLURM merge job (auto-triggered after all tasks/array job)
├── run_benchmark.py           ← worker: one benchmark cell, non-NN configs
├── run_benchmark_nn.py         ← worker: NN-specific gradient-based + model-agnostic backends
├── task_grid.py                ← build_all_runs — single source of truth for the task grid
├── count_tasks.py               ← prints len(build_all_runs(config)) for a given config
├── merge_results.py              ← merges per-task CSVs into results_<config_name>.csv
└── logs/                          ← per-task stdout/stderr (gitignored)

configs/
├── RQ1-accuracy/config-accuracy.yaml                    ← accuracy vs. background size
├── RQ2-dimensionality/config-dimensionality.yaml        ← scalability with feature count
├── RQ2-dimensionality/config-dimensionality-extreme.yaml← larger extreme-scale variant (not in submit_all.py registry; submit manually via submit.sh)
├── RQ3-neural-networks/config-neural-networks-gpu.yaml ← NN sweep, device=cuda
├── RQ3-neural-networks/config-neural-networks-cpu.yaml ← NN sweep, device=cpu
├── RQ4-tree/config-tree.yaml                            ← tree-native backends vs. model-agnostic
├── RQ4-tree/config-tree-fasttreeshap.yaml               ← fasttreeshap-only repair sweep (needs its own venv, step 4b)
└── RQ5-gpu/config-tree-gpu.yaml                         ← woodelf CPU vs. GPU (cupy) backends

benchmarking/
├── runner.py             ← BenchmarkRunner — oracle + approximators per cell
├── metrics.py             ← mean_abs_diff, sign_agreement, mean_sample_rho, runtime
├── backends/               ← approximators/, trees/, true_value/ — one class per (library, mode)
├── results_<config_name>.csv  ← merged results per config (after step 5/7)
└── slurm_results/
    └── <config_name>/          ← per-task CSVs, one dir per config (gitignored)

models/
├── dataset_and_models.py ← Dataset/Model enums; Model.is_tree gates the tree-specific sweep
├── config_parser.py      ← load_config / load_dataset_config — expand a config.yaml into parameter lists
├── architectures.py      ← neural network architectures
└── trainers.py            ← SklearnTrainer / PytorchTrainer

datasets/
└── load_datasets.py      ← dataset download/caching helpers (used by step 3); Dataset enum lives here
                             and is re-exported from models/dataset_and_models.py

scripts/
├── check_dataset_cache.py         ← verifies datasets are cached offline (step 3)
└── setup_fasttreeshap_env.sh      ← provisions the dedicated venv fasttreeshap needs (numpy<2);
                                       run once before submitting tree-fasttreeshap (step 4b)

tests/                    ← pytest suite — run with `uv run pytest tests/` before submitting
pyproject.toml            ← project metadata and dependencies
uv.lock                   ← locked dependency versions (synced in step 2)
```

---

## Updating a config

If you add models, datasets, or budgets, the task count updates automatically —
`submit_all.py` and `submit.sh` always recompute it from the config via
`slurm/task_grid.py:build_all_runs`. No hardcoded numbers need updating. If you
add a brand-new config file, also register it in `CONFIG_REGISTRY` in
`slurm/submit_all.py` so `submit_all.py --configs all` picks it up.

---

## If a task fails

Find its log. For `submit_all.py` jobs, logs are named by SLURM job ID:

```bash
cat slurm/logs/task_<JOBID>.out
```

For `submit.sh` array jobs:

```bash
cat slurm/logs/bench_<ARRAYJOBID>_<TASKID>.out       # non-NN configs
cat slurm/logs/bench_nn_<ARRAYJOBID>_<TASKID>.out    # nn / nn-cpu
```

Re-run just that task manually to debug:

```bash
# Non-NN configs (accuracy / dimensionality / tree / tree-fasttreeshap / tree-gpu)
uv run python slurm/run_benchmark.py \
    --task-id <TASKID> \
    --config configs/RQ1-accuracy/config-accuracy.yaml \
    --output-dir benchmarking/slurm_results/config-accuracy

# NN configs (nn / nn-cpu)
uv run python slurm/run_benchmark_nn.py \
    --task-id <TASKID> \
    --config configs/RQ3-neural-networks/config-neural-networks-gpu.yaml \
    --output-dir benchmarking/slurm_results/config-neural-networks-gpu
```

Replace the `--config`/`--output-dir` pair with the relevant config from the
table at the top for the other configs.

When all tasks are done (including any reruns), merge manually:

```bash
uv run python slurm/merge_results.py \
    --input-dir benchmarking/slurm_results/config-accuracy \
    --output-csv benchmarking/results_config-accuracy.csv
```

Replace `config-accuracy` / `results_config-accuracy` with the relevant config
name for the other configs.