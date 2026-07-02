#!/bin/bash
# Maps a config path to the sbatch array script that runs it — sourced by
# submit.sh (tested by tests/test_select_array_script.sh). Keep the precedence:
# GPU configs must win over the default even though they use run_benchmark.py.
select_array_script() {
    local config_name
    config_name="$(basename "$1" .yaml)"
    if [[ "$config_name" == *gpu* ]]; then
        echo "slurm/bench_array_gpu.sh"
    elif [[ "$config_name" == *neural-networks*cpu* ]]; then
        # device: cpu NN configs run on the CPU partition — same worker, no
        # GPU allocated (see slurm/bench_array_nn_cpu.sh).
        echo "slurm/bench_array_nn_cpu.sh"
    elif [[ "$config_name" == *neural-networks* ]]; then
        echo "slurm/bench_array_nn.sh"
    else
        echo "slurm/bench_array.sh"
    fi
}
