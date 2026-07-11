#!/bin/bash
# Maps a config path to the SLURM array script that runs it. Sourced by
# submit.sh.
#
# NN configs need the NN worker; the "-cpu" NN config must stay on the CPU
# partition (no GPU wasted), every other NN config trains with device=cuda and
# needs a GPU node. Non-NN "gpu" configs (woodelf's cupy path) also need a GPU
# node but run the tree worker.
select_array_script() {
    local config="$1"
    local name
    name="$(basename "$config" .yaml)"
    if [[ "$config" == *neural-networks* ]]; then
        if [[ "$name" == *cpu* ]]; then
            echo "slurm/bench_array_nn_cpu.sh"
        else
            echo "slurm/bench_array_nn.sh"
        fi
    elif [[ "$name" == *gpu* ]]; then
        echo "slurm/bench_array_gpu.sh"
    else
        echo "slurm/bench_array.sh"
    fi
}
