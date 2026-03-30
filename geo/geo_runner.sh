#!/usr/bin/env bash

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv310

PARAM_FILE="geo_params.yaml"
notebook_file="paper_notebook.ipynb"
runs=$(yq -r '.dataset.general.num_runs' "$PARAM_FILE")
datasets=($(yq -r '.dataset.general.dataset_list[]' "$PARAM_FILE"))

total_runs=$(( ${#datasets[@]} * runs ))
counter=0

for ds in "${datasets[@]}"; do
    export DATASET="$ds"
    echo "=== Starting runs for dataset: $ds ==="

    for ((i=1; i<=runs; i++)); do
        counter=$((counter + 1))
        echo ">>> Run $counter/$total_runs: dataset=$ds, iter $i/$runs <<<"

        # if ! python -m paper_notebook; then
        if ! jupyter nbconvert --to notebook --execute "$notebook_file" --output "output_${ds}_run${i}.ipynb"; then
            echo "Run $i failed for dataset $ds"
            continue
        fi
    done
done


# # ===============================
# #!/bin/bash

# # Move to project root
# cd "$(dirname "$0")/../.."

# # Activate conda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate venv310

# # Resolve param file path
# PARAM_FILE="src/param_config/baseline_params.yaml"

# # Read num_runs and datasets
# runs=$(yq -r '.basics.num_runs' "$PARAM_FILE")
# datasets=($(yq -r '.basics.dataset_list[]' "$PARAM_FILE"))

# total_runs=$(( ${#datasets[@]} * runs ))
# counter=0

# export BASH_RUN="1"

# for ds in "${datasets[@]}"; do
#     export DATASET="$ds"
#     echo "=== Starting runs for dataset: $ds ==="

#     for ((i=1; i<=runs; i++)); do
#         counter=$((counter + 1))
#         echo ">>> Run $counter/$total_runs: dataset=$ds, iter $i/$runs <<<"

#         if ! python -m src.scripts.road_runner; then
#             echo "Run $i failed for dataset $ds"
#             continue
#         fi
#     done
# done


