#!/bin/bash

set -euo pipefail
# set -x  # print every command as it runs

WEBHOOK_URL=$(yq -r '.webhook_url' param_config/messager.yaml)
datasets=($(yq -r '.basics.dataset_list[]' param_config/baseline_params.yaml))
runs=$(yq -r '.basics.num_runs' param_config/baseline_params.yaml)
total_runs=$(( ${#datasets[@]} * runs ))
counter=0

# set env var to let notebook know it is run from bash
export BASH_RUN="1"

for ds in "${datasets[@]}"; do
    echo "=== Starting runs for dataset: $ds ==="
    
    for ((i=1; i<=runs; i++)); do
        counter=$((counter + 1))
        echo ">>>>> Run $counter/$total_runs: $ds, iter $i/$runs <<<<<<"

        export DATASET="$ds"

        if ! jupyter nbconvert --to notebook \
                --ClearOutputPreprocessor.enabled=True \
                --execute dataset_notebook.ipynb \
                --output "run_${ds}_$i.ipynb" \
                --ExecutePreprocessor.allow_errors=False
        then
            echo "Run $i failed for $ds, skipping..."
            curl -H "Content-Type: application/json" \
                 -X POST \
                 -d "{\"content\": \"Notebook run #$i failed for $ds.\"}" \
                 "$WEBHOOK_URL"
            continue
        else
            curl -H "Content-Type: application/json" \
                 -X POST \
                 -d "{\"content\": \"Notebook run #$i successful for $ds.\"}" \
                 "$WEBHOOK_URL"
        fi
    done
done

curl -H "Content-Type: application/json" \
     -X POST \
     -d "{\"content\": \"All notebook runs finished.\"}" \
     "$WEBHOOK_URL"
