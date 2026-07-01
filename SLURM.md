# Running the benchmark on the IFI SLURM cluster

Four Research Questions, four configs, **75 independent tasks** in total.
The queue manager (`slurm/submit_all.py`) submits all of them while staying
within the Krater partition limit of **30 jobs** (15 running + 15 pending).

| Config key | File | Tasks | RQ |
|------------|------|-------|----|
| `accuracy` | `configs/config-accuracy.yaml` | 12 | Approximation accuracy vs. background size |
| `dimensionality` | `configs/config-dimensionality.yaml` | 36 | Scalability with feature count |
| `tree` | `configs/config-tree.yaml` | 18 | Tree-native backends vs. model-agnostic |
| `nn` | `configs/config-neural-networks-RQ3.yaml` | 9 | Gradient-based backends for neural networks |

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
Run once per unique `(dataset, n_features, n_samples)` combination across all
four configs:

```bash
cd ~/PR_ModeAgnostic
uv run python - <<'EOF'
import yaml, itertools
from Models.dataset_and_models import Dataset

configs = [
    "configs/config-accuracy.yaml",
    "configs/config-dimensionality.yaml",
    "configs/config-tree.yaml",
    "configs/config-neural-networks-RQ3.yaml",
]
seen = set()
for cfg_path in configs:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["benchmark"]["seed"]
    for ds_key, params in cfg["datasets"].items():
        for nf in params.get("n_features", [None]):
            for ns in params.get("n_samples", [None]):
                key = (ds_key, nf, ns)
                if key in seen:
                    continue
                seen.add(key)
                print(f"Fetching {ds_key} nf={nf} ns={ns} ...")
                kw = {}
                if nf is not None:
                    kw["n_features"] = nf
                if ns is not None:
                    kw["n_samples"] = ns
                Dataset[ds_key.upper()].load_dataset(**kw, seed=seed)
print(f"Done — {len(seen)} unique (dataset, n_features, n_samples) combinations cached.")
EOF
```

---

## 4. Check available partitions

```bash
sinfo
```

For `configs/config.yaml` / `configs/config-tree.yaml` (CPU only) any standard
partition works — `Krater`, `Gesteine_A`, etc. `configs/config-tree-gpu.yaml`
needs a GPU node — Abaki — requested via `slurm/bench_array_gpu.sh`'s
`--gres=gpu:1`.

If the partition name differs from what is set in `slurm/bench_array.sh` /
`slurm/bench_array_gpu.sh`, edit the `--partition=` line there before submitting.
The benchmark scripts target the `Krater` partition. If the partition name
differs on your allocation, edit the `--partition=` line in
`slurm/bench_array.sh`, `slurm/bench_array_nn.sh`, and `slurm/single_task.sh`
before submitting.

> **Note on GPU for NN jobs:** `config-neural-networks-RQ3.yaml` sets
> `device: cuda`. If Krater does not auto-assign a GPU, add `--gres=gpu:1` to
> the `sbatch` call inside `submit_all.py` for `nn` config entries (look for the
> comment in `slurm/single_task.sh`).

---

## 5. Submit

### Run all four Research Questions at once (recommended)

Open a **tmux or screen session** first — the queue manager is a long-running
process that must stay alive until all 75 tasks finish:

```bash
tmux new -s bench       # or: screen -S bench
cd ~/PR_ModeAgnostic
uv run python slurm/submit_all.py --configs all
```

The script prints a task summary, then enters a poll loop:

```
Config task counts:
  accuracy         12 tasks  (configs/config-accuracy.yaml)
  dimensionality   36 tasks  (configs/config-dimensionality.yaml)
  tree             18 tasks  (configs/config-tree.yaml)
  nn                9 tasks  (configs/config-neural-networks-RQ3.yaml)

Total: 75 tasks | MAX_JOBS=30 | poll every 60s
```

It submits up to 30 jobs, then wakes every 60 seconds, detects completions via
`squeue`, and tops up the queue. When all 75 tasks are done it submits one merge
job per config automatically.

### Run a subset of configs

```bash
# Only accuracy and dimensionality
uv run python slurm/submit_all.py --configs accuracy dimensionality

# Only tree
uv run python slurm/submit_all.py --configs tree

# Only neural networks
uv run python slurm/submit_all.py --configs nn
```

### Submit a single config (legacy, no queue management)

Use `submit.sh` when you only need one config and don't need queue throttling —
it submits the full SLURM array in one shot:

```bash
bash slurm/submit.sh configs/config-accuracy.yaml
bash slurm/submit.sh configs/config-dimensionality.yaml
bash slurm/submit.sh configs/config-tree.yaml
bash slurm/submit.sh configs/config-neural-networks-RQ3.yaml
```

> **Warning:** `submit.sh` submits all tasks at once with no concurrency limit.
> If the task count exceeds 30, SLURM may reject the submission. Use
> `submit_all.py` when in doubt.
bash slurm/submit.sh                            # model-agnostic sweep (configs/config.yaml)
bash slurm/submit.sh configs/config-tree.yaml      # tree-specific sweep
bash slurm/submit.sh configs/config-tree-gpu.yaml  # woodelf cpu-vs-gpu sweep (needs a GPU node)
```

Run any subset — each gets its own output directory and merged CSV, so they
never collide. `submit.sh` picks the array script automatically:
`slurm/bench_array_gpu.sh` (requests `--gres=gpu:1` on Abaki) for any config
whose name contains "gpu", otherwise `slurm/bench_array.sh`. This script:
1. Counts the `(dataset × model)` combinations from the given config.
2. Submits a SLURM **array job** — one task per combination, each writing to its
   own CSV under `Benchmarking/slurm_results/<config_name>/` so there are no
   race conditions.
3. Submits a **merge job** that runs automatically after all array tasks succeed,
   combining results into `Benchmarking/results_<config_name>.csv`.

---

## 6. Monitor

```bash
squeue -u $USER                           # all your running/pending jobs

# Live output for a task submitted via submit_all.py
tail -f slurm/logs/task_<JOBID>.out

# Live output for a task submitted via submit.sh (array jobs)
tail -f slurm/logs/bench_<ARRAYJOBID>_<TASKID>.out
```

Cancel everything if needed:

```bash
scancel -u $USER
```

---

## 7. Retrieve results

After the merge job finishes, copy the merged CSV(s) back to your Mac
(`results_config.csv` for the model-agnostic run, `results_config-tree.csv`
for the tree run, `results_config-tree-gpu.csv` for the woodelf cpu-vs-gpu run):
After all merge jobs finish, copy the four result CSVs back to your Mac:

```bash
# Run this on your Mac
scp '<cip-kennung>@remote.cip.ifi.lmu.de:~/PR_ModeAgnostic/Benchmarking/results_config-*.csv' \
    Benchmarking/
```

Or with rsync:

```bash
rsync -avz <cip-kennung>@remote.cip.ifi.lmu.de:~/PR_ModeAgnostic/Benchmarking/ \
    Benchmarking/ --include='results_*.csv' --exclude='*'
```

The four merged files produced are:

| File | Config |
|------|--------|
| `Benchmarking/results_config-accuracy.csv` | accuracy |
| `Benchmarking/results_config-dimensionality.csv` | dimensionality |
| `Benchmarking/results_config-tree.csv` | tree |
| `Benchmarking/results_config-neural-networks-RQ3.csv` | nn |

---

## File overview

```
slurm/
├── submit_all.py       ← main entry point: queue manager for all 4 configs
├── submit.sh           ← legacy single-config submitter (no queue throttling)
├── single_task.sh      ← generic sbatch wrapper used by submit_all.py
├── bench_array.sh      ← SLURM array script for model-agnostic / tree configs
├── bench_array_nn.sh   ← SLURM array script for NN config
├── merge.sh            ← SLURM merge job (auto-triggered after all tasks)
├── run_benchmark.py    ← worker: one (dataset, model) cell, first-order +
│                          order-2 tree interactions; sweeps n_background if list
├── run_benchmark_nn.py ← worker: NN-specific gradient-based + model-agnostic backends
├── submit.sh           ← entry point: run this to submit everything (picks the
│                          array script below based on the config name)
├── bench_array.sh      ← SLURM array job definition (CPU); takes (config, output_dir) args
├── bench_array_gpu.sh  ← GPU counterpart (--gres=gpu:1, Abaki); used for any
│                          config whose name contains "gpu"
├── merge.sh            ← SLURM merge job (auto-triggered after array)
├── run_benchmark.py    ← worker: runs one (dataset, model) cell, both first-order
│                          and (for tree models) the order-2 interaction sweep
├── merge_results.py    ← merges per-task CSVs into results_<config_name>.csv
├── count_tasks.py      ← prints the number of task combinations for a given config
└── logs/               ← per-task stdout/stderr (gitignored)

configs/
├── config-accuracy.yaml            ← RQ: accuracy vs. background size (12 tasks)
├── config-dimensionality.yaml      ← RQ: scalability with feature count (36 tasks)
├── config-tree.yaml                ← RQ: tree-native backends (18 tasks)
└── config-neural-networks-RQ3.yaml ← RQ: gradient-based NN backends (9 tasks)
├── config.yaml           ← model-agnostic sweep (libraries, approximators, models)
├── config-tree.yaml      ← tree-specific sweep (tree backends, interactions)
└── config-tree-gpu.yaml  ← woodelf cpu-vs-gpu sweep (path_dependent/interventional,
                             each in a CPU and a GPU=True variant)

Benchmarking/
├── runner.py            ← BenchmarkRunner — oracle + approximators per cell
├── metrics.py           ← mean_abs_diff, sign_agreement, mean_sample_rho, runtime
├── backends/            ← one class per (library, mode)
├── results_config-accuracy.csv           ← merged after step 5/7
├── results_config-dimensionality.csv     ← merged after step 5/7
├── results_config-tree.csv               ← merged after step 5/7
├── results_config-neural-networks-RQ3.csv ← merged after step 5/7
├── runner.py            ← BenchmarkRunner — runs one oracle + backends/approximations per cell
├── metrics.py            ← mean_abs_diff, sign_agreement, mean_sample_rho, runtime
├── backends/             ← one class per (library, mode); tree_*.py / woodelf_backend.py /
│                            fasttreeshap_backend.py / gputreeshap_backend.py are tree-specific
├── results_config.csv         ← merged model-agnostic results (after step 5/7)
├── results_config-tree.csv    ← merged tree results (after step 5/7)
├── results_config-tree-gpu.csv ← merged woodelf cpu-vs-gpu results (after step 5/7)
└── slurm_results/
    ├── config-accuracy/          ← per-task CSVs (gitignored)
    ├── config-dimensionality/    ← per-task CSVs (gitignored)
    ├── config-tree/              ← per-task CSVs (gitignored)
    └── config-neural-networks-RQ3/ ← per-task CSVs (gitignored)
    ├── config/             ← model-agnostic run's per-task CSVs (gitignored)
    ├── config-tree/        ← tree run's per-task CSVs (gitignored)
    └── config-tree-gpu/    ← gpu run's per-task CSVs (gitignored)

Models/
├── dataset_and_models.py ← Dataset/Model enums; Model.is_tree gates the tree-specific sweep
├── config_parser.py      ← load_config / load_dataset_config — expand a config.yaml into parameter lists
└── trainers.py            ← SklearnTrainer / PytorchTrainer

Datasets/
└── load_datasets.py      ← dataset download/caching helpers (used by step 3)

scripts/
└── setup_fasttreeshap_env.sh  ← provisions the dedicated venv fasttreeshap needs (numpy<2);
                                   run once before submitting a tree-config job that uses it

tests/                    ← pytest suite — run with `uv run pytest tests/` before submitting
pyproject.toml            ← project metadata and dependencies
uv.lock                   ← locked dependency versions (synced in step 2)
```

---

## Updating a config

If you add models, datasets, or budgets, the task count updates automatically —
`submit_all.py` and `submit.sh` always recompute it from the config. No hardcoded
numbers need updating.

---

## If a task fails

Find its log. For `submit_all.py` jobs, logs are named by SLURM job ID:

```bash
cat slurm/logs/task_<JOBID>.out
```

For `submit.sh` array jobs:

```bash
cat slurm/logs/bench_<ARRAYJOBID>_<TASKID>.out
```

Re-run just that task manually to debug:

```bash
# Model-agnostic / tree / accuracy / dimensionality configs
uv run python slurm/run_benchmark.py \
    --task-id <TASKID> \
    --config configs/config-accuracy.yaml \
    --output-dir Benchmarking/slurm_results/config-accuracy

# NN config
uv run python slurm/run_benchmark_nn.py \
    --task-id <TASKID> \
    --config configs/config-neural-networks-RQ3.yaml \
    --output-dir Benchmarking/slurm_results/config-neural-networks-RQ3
```

When all tasks are done (including any reruns), merge manually:

```bash
uv run python slurm/merge_results.py \
    --input-dir Benchmarking/slurm_results/config-accuracy \
    --output-csv Benchmarking/results_config-accuracy.csv
```

Replace `config-accuracy` / `results_config-accuracy` with the relevant config
name for the other three RQs.
