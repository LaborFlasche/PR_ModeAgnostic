# Picks the right SLURM array script for a config. Sourced by submit.sh (and
# tested directly by tests/test_select_array_script.sh) so there is exactly
# one place that decides GPU vs. NN vs. default dispatch.
select_array_script() {
    local config="$1"
    if [[ "$config" == *neural-networks* ]]; then
        echo "slurm/bench_array_nn.sh"
    elif [[ "$config" == *gpu* ]]; then
        echo "slurm/bench_array_gpu.sh"
    else
        echo "slurm/bench_array.sh"
    fi
}
