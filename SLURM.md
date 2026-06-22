# Running the benchmark on the IFI SLURM cluster

The 24 `(dataset × model)` benchmark cells are fully independent and take 2–3 h
sequentially. In parallel on SLURM they finish in ~15–20 min.

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
bash slurm/submit.sh
```

This script:
1. Counts the `(dataset × model)` combinations from `configs/config.yaml`
   (currently **24**).
2. Submits a SLURM **array job** — one task per combination, each writing to its
   own CSV so there are no race conditions.
3. Submits a **merge job** that runs automatically after all array tasks succeed,
   combining results into `Benchmarking/results.csv`.

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

After the merge job finishes, copy `results.csv` back to your Mac:

```bash
# Run this on your Mac
scp <cip-kennung>@remote.cip.ifi.lmu.de:~/PR_ModeAgnostic/Benchmarking/results.csv \
    Benchmarking/results.csv
```

Or if you prefer rsync:

```bash
rsync -avz <cip-kennung>@remote.cip.ifi.lmu.de:~/PR_ModeAgnostic/Benchmarking/results.csv \
    Benchmarking/results.csv
```

---

## File overview

```
slurm/
├── submit.sh          ← entry point: run this to submit everything
├── bench_array.sh     ← SLURM array job definition
├── merge.sh           ← SLURM merge job (auto-triggered after array)
├── run_benchmark.py   ← worker: runs one (dataset, model) cell
├── merge_results.py   ← merges per-task CSVs into results.csv
├── count_tasks.py     ← prints the number of task combinations
└── logs/              ← per-task stdout/stderr (gitignored)

Benchmarking/
└── slurm_results/     ← per-task CSVs written during the run (gitignored)
```

---

## Updating the config

If you add more models, datasets, or budgets, the task count changes
automatically — `submit.sh` always recomputes it from the config. No need to
update any hardcoded number.

---

## If a task fails

Check its log:

```bash
cat slurm/logs/bench_<JOBID>_<TASKID>.out
```

Re-run just that task manually to debug:

```bash
uv run python slurm/run_benchmark.py --task-id <TASKID>
```

When all tasks are done (including any reruns), merge manually:

```bash
uv run python slurm/merge_results.py
```
