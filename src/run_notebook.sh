#!/bin/bash
# WEBHOOK_URL="https://canary.discord.com/api/webhooks/1426232166441811978/88Gy60hLuqc6u0UNDo9RCO-Itn0YYJoMG2lfpC4UHt-uxBIFx_w3v00IFoiotLcxEbrB"

WEBHOOK_URL=$(yq -r '.webhook_url' param_config/messager.yaml)
datasets=($(yq -r '.basics.dataset_list[]' param_config/baseline_params.yaml))
runs=$(yq -r '.basics.runs' param_config/baseline_params.yaml)
total_runs=$(( ${#datasets[@]} * runs ))
counter=0

for ds in "${datasets[@]}"; do
    echo "=== Starting runs for dataset: $ds ==="
    
    # update YAML to current dataset
    yq ".basics.dataset = \"$ds\"" param_config/baseline_params.yaml > param_config/tmp.yaml

    for ((i=1; i<=runs; i++)); do
        counter=$((counter + 1))
        echo ">>>>> Run $counter/$total_runs: $ds, iter $i/$runs <<<<<<"

        jupyter nbconvert --to notebook \
            --ClearOutputPreprocessor.enabled=True \
            --execute dataset_notebook.ipynb \
            --output "run_${ds}_$i.ipynb" \
            --ExecutePreprocessor.allow_errors=False || RUN_FAILED=true

        if [ "$RUN_FAILED" = true ]; then
            echo "Run $i failed for $ds, skipping..."
            curl -H "Content-Type: application/json" \
                 -X POST \
                 -d "{\"content\": \"Notebook run #$i failed for $ds.\"}" \
                 "$WEBHOOK_URL"
            RUN_FAILED=false
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
