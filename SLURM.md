# Running the benchmark on the IFI SLURM cluster

The `(dataset × model)` benchmark cells (36 model-agnostic, 18 tree-specific —
see step 5) are fully independent and take hours sequentially. In parallel on
SLURM they finish in ~15–20 min.

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

Compute nodes may have no outbound internet access. Cache all datasets first:

```bash
cd ~/PR_ModeAgnostic
uv run python -c "
import yaml
from Models.dataset_and_models import Dataset

with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

for ds_key, params in cfg['datasets'].items():
    for nf in params.get('n_features', [4]):
        for ns in params.get('n_samples', [1000]):
            print(f'Fetching {ds_key} nf={nf} ns={ns} ...')
            Dataset[ds_key.upper()].load_dataset(n_features=nf, n_samples=ns)
print('All datasets cached.')
"
```

---

## 4. Check available partitions

```bash
sinfo
```

For this benchmark (CPU only, no GPU needed) any standard partition works —
`Krater`, `Gesteine_A`, etc. No application needed; Abaki (GPU nodes) is only
required if you add GPU workloads.

If the partition name differs from what is set in `slurm/bench_array.sh`, edit
the `--partition=` line there before submitting.

---

## 5. Submit

```bash
cd ~/PR_ModeAgnostic
bash slurm/submit.sh                       # model-agnostic sweep (configs/config.yaml)
bash slurm/submit.sh configs/config-tree.yaml  # tree-specific sweep
```

Run either or both — each gets its own output directory and merged CSV, so
they never collide. This script:
1. Counts the `(dataset × model)` combinations from the given config.
2. Submits a SLURM **array job** — one task per combination, each writing to its
   own CSV under `Benchmarking/slurm_results/<config_name>/` so there are no
   race conditions.
3. Submits a **merge job** that runs automatically after all array tasks succeed,
   combining results into `Benchmarking/results_<config_name>.csv`.

---

## 6. Monitor

```bash
squeue -u $USER                              # all your running/pending jobs
tail -f slurm/logs/bench_<JOBID>_<N>.out    # live output for task N
```

Cancel everything if needed:

```bash
scancel -u $USER
```

---

## 7. Retrieve results

After the merge job finishes, copy the merged CSV(s) back to your Mac
(`results_config.csv` for the model-agnostic run, `results_config-tree.csv`
for the tree run):

```bash
# Run this on your Mac
scp '<cip-kennung>@remote.cip.ifi.lmu.de:~/PR_ModeAgnostic/Benchmarking/results_*.csv' \
    Benchmarking/
```

Or if you prefer rsync:

```bash
rsync -avz <cip-kennung>@remote.cip.ifi.lmu.de:~/PR_ModeAgnostic/Benchmarking/ \
    Benchmarking/ --include='results_*.csv' --exclude='*'
```

---

## File overview

```
slurm/
├── submit.sh           ← entry point: run this to submit everything
├── bench_array.sh      ← SLURM array job definition; takes (config, output_dir) args
├── merge.sh            ← SLURM merge job (auto-triggered after array)
├── run_benchmark.py    ← worker: runs one (dataset, model) cell, both first-order
│                          and (for tree models) the order-2 interaction sweep
├── merge_results.py    ← merges per-task CSVs into results_<config_name>.csv
├── count_tasks.py      ← prints the number of task combinations for a given config
└── logs/               ← per-task stdout/stderr (gitignored)

configs/
├── config.yaml          ← model-agnostic sweep (libraries, approximators, models)
└── config-tree.yaml     ← tree-specific sweep (tree backends, interactions)

Benchmarking/
├── runner.py            ← BenchmarkRunner — runs one oracle + backends/approximations per cell
├── metrics.py            ← mean_abs_diff, sign_agreement, mean_sample_rho, runtime
├── backends/             ← one class per (library, mode); tree_*.py / woodelf_backend.py /
│                            fasttreeshap_backend.py / gputreeshap_backend.py are tree-specific
├── results_config.csv       ← merged model-agnostic results (after step 5/7)
├── results_config-tree.csv  ← merged tree results (after step 5/7)
└── slurm_results/
    ├── config/         ← model-agnostic run's per-task CSVs (gitignored)
    └── config-tree/    ← tree run's per-task CSVs (gitignored)

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

## Updating the config

If you add more models, datasets, or budgets, the task count changes
automatically — `submit.sh` always recomputes it from the config. No need to
update any hardcoded number. This applies to both `configs/config.yaml` and
`configs/config-tree.yaml`.

---

## If a task fails

Check its log:

```bash
cat slurm/logs/bench_<JOBID>_<TASKID>.out
```

Re-run just that task manually to debug (add `--config configs/config-tree.yaml`
if it's a tree-run task):

```bash
uv run python slurm/run_benchmark.py --task-id <TASKID>
```

When all tasks are done (including any reruns), merge manually:

```bash
uv run python slurm/merge_results.py --input-dir Benchmarking/slurm_results/config --output-csv Benchmarking/results_config.csv
```
