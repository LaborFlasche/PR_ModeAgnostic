#!/usr/bin/env python3
"""
submit_all.py — submit all benchmark configs while respecting the 30-job
SLURM limit on Krater (15 running + 15 pending = 30 max).

Maintains a local queue of every (config, task_id) pair. Polls squeue every
POLL_INTERVAL seconds and submits new jobs whenever total user-job count drops
below MAX_JOBS. When all tasks complete, submits one merge job per config.

Usage (run from repo root, in a persistent session such as tmux/screen):
    python slurm/submit_all.py --configs all
    python slurm/submit_all.py --configs accuracy dimensionality
    python slurm/submit_all.py --configs tree nn

Available config keys: see CONFIG_REGISTRY below.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from task_grid import build_all_runs

MAX_JOBS = 30
POLL_INTERVAL = 60  # seconds between squeue polls

# "sbatch_args" override slurm/single_task.sh's #SBATCH directives (CLI options
# take precedence): the nn config runs with device=cuda and the tree-gpu config
# exercises woodelf's cupy path, so both need a GPU node instead of Krater.
# The CIP cluster defines no GPU GRES (sinfo shows GRES=(null) on every
# partition), so --gres/--gpus flags are rejected with "Invalid generic
# resource specification" — requesting the NvidiaAll partition alone is both
# necessary and sufficient; the node's GPU is directly visible to the job.
CONFIG_REGISTRY = {
    "accuracy": {
        "config": "configs/RQ1-accuracy/config-accuracy.yaml",
        "worker": "slurm/run_benchmark.py",
    },
    "dimensionality": {
        "config": "configs/RQ2-dimensionality/config-dimensionality.yaml",
        "worker": "slurm/run_benchmark.py",
    },
    "tree": {
        "config": "configs/RQ4-tree/config-tree.yaml",
        "worker": "slurm/run_benchmark.py",
    },
    # fasttreeshap-only repair sweep (see BUGS_TO_FIX.md Bug 5); requires
    # scripts/setup_fasttreeshap_env.sh to have been run on the cluster first.
    "tree-fasttreeshap": {
        "config": "configs/RQ4-tree/config-tree-fasttreeshap.yaml",
        "worker": "slurm/run_benchmark.py",
    },
    "nn": {
        "config": "configs/RQ3-neural-networks/config-neural-networks-RQ3-gpu.yaml",
        "worker": "slurm/run_benchmark_nn.py",
        "sbatch_args": ["--partition=NvidiaAll"],
    },
    "nn-cpu": {
        "config": "configs/RQ3-neural-networks/config-neural-networks-RQ3-cpu.yaml",
        "worker": "slurm/run_benchmark_nn.py",
    },
    "tree-gpu": {
        "config": "configs/RQ5-gpu/config-tree-gpu.yaml",
        "worker": "slurm/run_benchmark.py",
        "sbatch_args": ["--partition=NvidiaAll"],
    },

}


# ---------------------------------------------------------------------------
# Task counting
# ---------------------------------------------------------------------------

def count_tasks(config_path: str) -> int:
    """Number of array tasks — exactly the grid the workers index into via
    --task-id (task_grid.build_all_runs is the single source of truth)."""
    return len(build_all_runs(os.path.join(REPO_ROOT, config_path)))


def result_paths(config_key: str) -> tuple[str, str, str]:
    """(config_path, per-task output dir, merged CSV path) for one config."""
    config_path = CONFIG_REGISTRY[config_key]["config"]
    config_name = os.path.basename(config_path).replace(".yaml", "")
    return (config_path,
            f"Benchmarking/slurm_results/{config_name}",
            f"Benchmarking/results_{config_name}.csv")


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

def count_user_jobs() -> int:
    """Return total running+pending SLURM jobs for the current user."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME", "")
    result = subprocess.run(
        ["squeue", "-u", user, "--noheader"],
        capture_output=True, text=True,
    )
    return sum(1 for line in result.stdout.splitlines() if line.strip())


def get_active_job_ids(job_ids: set[int]) -> set[int]:
    """Return the subset of job_ids still present in the SLURM queue."""
    if not job_ids:
        return set()
    ids_str = ",".join(str(j) for j in sorted(job_ids))
    result = subprocess.run(
        ["squeue", "-j", ids_str, "--noheader", "-o", "%i"],
        capture_output=True, text=True,
    )
    active = set()
    for token in result.stdout.split():
        try:
            active.add(int(token))
        except ValueError:
            pass
    return active


def submit_task(config_key: str, task_id: int) -> int | None:
    """Submit one (config, task_id) as a single SLURM job. Returns job ID or None."""
    spec = CONFIG_REGISTRY[config_key]
    config_path, output_dir, _ = result_paths(config_key)

    os.makedirs(os.path.join(REPO_ROOT, output_dir), exist_ok=True)

    result = subprocess.run(
        [
            "sbatch",
            # full key, not a [:3] prefix: "tree" and "tree-gpu" would both
            # truncate to "tre" and be indistinguishable in squeue
            f"--job-name=bench_{config_key}_{task_id}",
            f"--chdir={REPO_ROOT}",
            *spec.get("sbatch_args", []),
            "slurm/single_task.sh",
            spec["worker"],
            str(task_id),
            config_path,
            output_dir,
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )

    if result.returncode != 0:
        print(f"  ERROR submitting {config_key}/{task_id}: {result.stderr.strip()}", file=sys.stderr)
        return None
    try:
        return int(result.stdout.strip().split()[-1])
    except (ValueError, IndexError):
        print(f"  ERROR parsing job ID from: {result.stdout!r}", file=sys.stderr)
        return None


def submit_merge(config_key: str) -> None:
    """Submit the merge job for a config (blocks until sbatch returns)."""
    _, input_dir, output_csv = result_paths(config_key)

    result = subprocess.run(
        [
            "sbatch",
            f"--chdir={REPO_ROOT}",
            "slurm/merge.sh",
            "--input-dir", input_dir,
            "--output-csv", output_csv,
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode == 0:
        print(f"  [{config_key}] merge job submitted: {result.stdout.strip()}"
              f"  ({input_dir} → {output_csv})")
    else:
        print(f"  [{config_key}] merge ERROR: {result.stderr.strip()}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main queue loop
# ---------------------------------------------------------------------------

def run(selected: list[str]) -> None:
    # Build pending queue and report task counts
    pending: list[tuple[str, int]] = []
    print("Config task counts:")
    for key in selected:
        cfg, output_dir, _ = result_paths(key)
        n = count_tasks(cfg)
        print(f"  {key:15s} {n:3d} tasks  ({cfg})")
        # Start from a clean per-task output dir (same as submit.sh): workers only
        # overwrite their own results_<task_id>.csv, so a previous sweep with a
        # larger grid would leave higher-numbered files behind and leak stale
        # rows into the merged CSV.
        output_dir = os.path.join(REPO_ROOT, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for stale in glob.glob(os.path.join(output_dir, "results_*.csv")):
            os.remove(stale)
        pending += [(key, task_id) for task_id in range(n)]

    total = len(pending)
    print(f"\nTotal: {total} tasks | MAX_JOBS={MAX_JOBS} | poll every {POLL_INTERVAL}s")
    print("Running in foreground — keep this session alive (tmux/screen recommended).\n")

    os.makedirs(os.path.join(REPO_ROOT, "slurm", "logs"), exist_ok=True)

    submitted: dict[int, tuple[str, int]] = {}  # job_id -> (config_key, task_id)
    failed: list[tuple[str, int]] = []
    completed = 0

    while pending or submitted:
        # Detect completions
        if submitted:
            active = get_active_job_ids(set(submitted))
            for jid in list(submitted):
                if jid not in active:
                    config_key, task_id = submitted.pop(jid)
                    completed += 1
                    print(f"[{_ts()}] done  job={jid} {config_key}/{task_id}"
                          f"  [{completed}/{total}]")

        # Fill up to MAX_JOBS
        if pending:
            slots = MAX_JOBS - count_user_jobs()
            n_submitted = 0
            while pending and slots > 0:
                config_key, task_id = pending.pop(0)
                jid = submit_task(config_key, task_id)
                if jid is not None:
                    submitted[jid] = (config_key, task_id)
                    slots -= 1
                    n_submitted += 1
                else:
                    failed.append((config_key, task_id))
            if n_submitted:
                print(f"[{_ts()}] submitted {n_submitted} job(s)"
                      f" | running={len(submitted)} pending_local={len(pending)}")

        if pending or submitted:
            time.sleep(POLL_INTERVAL)

    print(f"\n[{_ts()}] All {total} tasks dispatched.")
    if failed:
        print(f"WARNING: {len(failed)} submission(s) failed:")
        for ck, tid in failed:
            print(f"  {ck}/{tid}")

    print("\nSubmitting merge jobs...")
    for key in selected:
        submit_merge(key)
    print("Done.")


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    valid = list(CONFIG_REGISTRY)
    parser = argparse.ArgumentParser(
        description="Submit all benchmark configs with a 30-job SLURM queue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Valid config keys: {', '.join(valid)}",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["all"],
        metavar="KEY",
        help='Config key(s) to run, or "all" for every registered config. Default: all',
    )
    args = parser.parse_args()

    if "all" in args.configs:
        selected = valid
    else:
        unknown = [c for c in args.configs if c not in CONFIG_REGISTRY]
        if unknown:
            parser.error(f"Unknown config key(s): {unknown}. Valid: {valid}")
        selected = args.configs

    run(selected)


if __name__ == "__main__":
    main()
