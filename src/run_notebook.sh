#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

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
    echo -e "${PURPLE}=== Starting runs for dataset: $ds ===${NC}"

    for ((i=1; i<=runs; i++)); do
        counter=$((counter + 1))
        echo -e "${GREEN}>>>>> Run $counter/$total_runs: $ds, iter $i/$runs <<<<<<${NC}"

        export DATASET="$ds"

        if ! jupyter nbconvert --to notebook \
                --ClearOutputPreprocessor.enabled=True \
                --execute dataset_notebook.ipynb \
                --output "run_${ds}_$i.ipynb" \
                --ExecutePreprocessor.allow_errors=False
        then
            echo -e "${RED}Run $i failed for $ds, skipping...${NC}"
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
