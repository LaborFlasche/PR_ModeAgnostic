# Running the benchmark on the IFI SLURM cluster

The queue manager (`slurm/submit_all.py`) submits all runs while staying
within the slurm limit of **30 jobs** (15 running + 15 pending).

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

Compute nodes may have no outbound internet access. Cache all datasets first,
on the login node (its NFS home is shared with every compute node, so one run
caches for all):

```bash
cd ~/PR_ModeAgnostic
uv run python scripts/cache_datasets.py                          # all configs/*/*.yaml
uv run python scripts/cache_datasets.py configs/RQ4-tree/config-tree.yaml   # or specific configs
```

This loads each dataset **once** — the fetch that populates the cache downloads
the full dataset before any subsampling, so `n_features`/`n_samples`/`seed`
don't affect what's cached. Re-running is cheap: already-cached datasets are
read from disk without touching the network.

After running it, you can also verify that everything loads offline with
`check_dataset_cache.py` (never downloads — it blocks network access and reports
any dataset that would still need it). Exit code 0 means all cached; 1 lists the
missing ones:

```bash
uv run python scripts/check_dataset_cache.py                     # all configs/*/*.yaml
uv run python scripts/check_dataset_cache.py configs/RQ4-tree/config-tree.yaml
```

---

## 4. Check available partitions

```bash
sinfo
```

For `configs/config.yaml` / `configs/RQ4-tree/config-tree.yaml` (CPU only) any standard
partition works — `Krater`, `Gesteine_A`, etc. `configs/RQ5-gpu/config-tree-gpu.yaml`
and the NN config need a GPU node — the `NvidiaAll` partition, targeted by
`slurm/bench_array_gpu.sh` and `submit_all.py`'s `nn`/`tree-gpu` entries.

If the partition name differs from what is set in `slurm/bench_array.sh` /
`slurm/bench_array_gpu.sh`, edit the `--partition=` line there before submitting.
The benchmark scripts target the `Krater` partition. If the partition name
differs on your allocation, edit the `--partition=` line in
`slurm/bench_array.sh`, `slurm/bench_array_nn.sh`, and `slurm/single_task.sh`
before submitting.

> **Note on GPU jobs:** the CIP cluster defines no GPU GRES — `sinfo -o "%P %G"`
> shows `GRES=(null)` on every partition — so `--gres=gpu:1` / `--gpus=1` are
> rejected with `Invalid generic resource specification`. Do not add them.
> Requesting `--partition=NvidiaAll` alone is sufficient: the node's GPU is
> directly visible to the job.

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
  accuracy         12 tasks  (configs/RQ1-accuracy/config-accuracy.yaml)
  dimensionality   36 tasks  (configs/RQ2-dimensionality/config-dimensionality.yaml)
  tree             18 tasks  (configs/RQ4-tree/config-tree.yaml)
  nn                9 tasks  (configs/RQ3-neural-networks/config-neural-networks-RQ3.yaml)

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
bash slurm/submit.sh configs/RQ1-accuracy/config-accuracy.yaml
bash slurm/submit.sh configs/RQ2-dimensionality/config-dimensionality.yaml
bash slurm/submit.sh configs/RQ4-tree/config-tree.yaml
bash slurm/submit.sh configs/RQ3-neural-networks/config-neural-networks-RQ3.yaml
```

> **Warning:** `submit.sh` submits all tasks at once with no concurrency limit.
> If the task count exceeds 30, SLURM may reject the submission. Use
> `submit_all.py` when in doubt.
bash slurm/submit.sh                            # model-agnostic sweep (configs/config.yaml)
bash slurm/submit.sh configs/RQ4-tree/config-tree.yaml      # tree-specific sweep
bash slurm/submit.sh configs/RQ5-gpu/config-tree-gpu.yaml  # woodelf cpu-vs-gpu sweep (needs a GPU node)
```

Run any subset — each gets its own output directory and merged CSV, so they
never collide. `submit.sh` picks the array script automatically:
`slurm/bench_array_gpu.sh` (targets the `NvidiaAll` partition) for any config
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
├── submit_all.py        ← main entry point: queue manager that submits every config
│                           while respecting the 30-job Krater limit (see CONFIG_REGISTRY)
├── submit.sh            ← legacy single-config submitter (full array at once, no throttling)
├── single_task.sh       ← generic sbatch wrapper: runs one (worker, task-id, config) job
├── bench_array.sh       ← SLURM array script (CPU) for model-agnostic / tree configs
├── bench_array_gpu.sh   ← array script counterpart targeting the NvidiaAll partition
├── bench_array_nn.sh    ← array script for the NN configs
├── merge.sh             ← SLURM merge job (auto-triggered after all tasks succeed)
├── run_benchmark.py     ← worker: one (seed, dataset, model) cell — first-order plus
│                           order-2 tree interactions; sweeps seed / n_background if lists
├── run_benchmark_nn.py  ← worker: NN-specific gradient-based + model-agnostic backends
├── merge_results.py     ← merges per-task CSVs into results_<config_name>.csv
├── count_tasks.py       ← prints the task count for a given config
└── logs/                ← per-task stdout/stderr (gitignored)

configs/                 ← one folder per Research Question; scripts glob configs/*/*.yaml
├── RQ1-accuracy/
│   └── config-accuracy.yaml               ← accuracy vs. background size
├── RQ2-dimensionality/
│   ├── config-dimensionality.yaml         ← scalability with feature count
│   └── config-dimensionality-extreme.yaml ← extreme high-dimensional variant
├── RQ3-neural-networks/
│   ├── config-neural-networks-RQ3.yaml     ← gradient-based NN backends (GPU)
│   ├── config-neural-networks-RQ3-cpu.yaml ← CPU variant
│   └── config-test-nn-RQ3.yaml             ← small smoke-test config
├── RQ4-tree/
│   ├── config-tree.yaml                    ← tree-native backends vs. model-agnostic
│   └── config-tree-fasttreeshap.yaml       ← fasttreeshap-only repair sweep
└── RQ5-gpu/
    └── config-tree-gpu.yaml                ← woodelf cpu-vs-gpu sweep (needs a GPU node)

Benchmarking/
├── runner.py            ← BenchmarkRunner — runs one oracle + backends/approximators per cell
├── metrics.py           ← mean_abs_diff, sign_agreement, mean_sample_rho, runtime
├── eval_counter.py      ← counts model evaluations per backend
├── timeout.py           ← per-backend wall-clock timeout wrapper
├── backends/            ← one class per (library, mode)
│   ├── base_backend.py      ← shared backend interface
│   ├── approximators/       ← model-agnostic: captum / dalex / lightshap / shap(+nn) / shapiq(+nn)
│   ├── trees/               ← tree-native: tree_shap / tree_shapiq / woodelf / fasttreeshap
│   └── true_value/          ← exact-value oracles: dalex / lightshap / shap / shapiq
├── results_config-accuracy.csv            ← merged after step 5/7
├── results_config-dimensionality.csv      ← merged after step 5/7
├── results_config-tree.csv                ← merged after step 5/7
├── results_config-neural-networks-RQ3.csv ← merged after step 5/7
└── slurm_results/
    ├── config-accuracy/            ← per-task CSVs (gitignored)
    ├── config-dimensionality/      ← per-task CSVs (gitignored)
    ├── config-tree/                ← per-task CSVs (gitignored)
    └── config-neural-networks-RQ3/ ← per-task CSVs (gitignored)

Models/
├── dataset_and_models.py ← Dataset/Model enums; Model.is_tree gates the tree-specific sweep
├── config_parser.py      ← load_config / load_dataset_config — expand a config into param lists
├── architectures.py      ← neural-network architecture definitions
├── load_and_train.py     ← model construction + training entry points
└── trainers.py           ← SklearnTrainer / PytorchTrainer

Datasets/
├── load_datasets.py      ← Dataset enum + shared loader pipeline (fetch → impute/encode →
│                           subsample → variance feature-select); step 3's scripts cache
│                           these datasets by calling Dataset.load_dataset()
└── dataset.md            ← reference table of the nine datasets (features, task, domain)

scripts/
├── cache_datasets.py            ← step 3: pre-download every config's datasets into the
│                                   shared cache — one load per dataset key (no internet
│                                   needed on compute nodes afterwards)
├── check_dataset_cache.py       ← step 3: verify every dataset loads offline (blocks the
│                                   network, never downloads); exit 0 = all cached, 1 = missing
├── merge_fasttreeshap_repair.py ← merges the fasttreeshap repair-sweep CSVs (see BUGS_TO_FIX)
└── setup_fasttreeshap_env.sh    ← provisions the dedicated venv fasttreeshap needs (numpy<2);
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
    --config configs/RQ1-accuracy/config-accuracy.yaml \
    --output-dir Benchmarking/slurm_results/config-accuracy

# NN config
uv run python slurm/run_benchmark_nn.py \
    --task-id <TASKID> \
    --config configs/RQ3-neural-networks/config-neural-networks-RQ3.yaml \
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
