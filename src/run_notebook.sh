#!/bin/bash
WEBHOOK_URL="https://canary.discord.com/api/webhooks/1426232166441811978/88Gy60hLuqc6u0UNDo9RCO-Itn0YYJoMG2lfpC4UHt-uxBIFx_w3v00IFoiotLcxEbrB"

runs=$(yq '.basics.runs' params2.yaml)
for ((i=1; i<=runs; i++)); do
# for i in {1..4}; do
    echo ">>> Starting notebook run #$i"

    jupyter nbconvert --to notebook \
        --ClearOutputPreprocessor.enabled=True \
        --execute dataset_notebook.ipynb \
        --output "run_$i.ipynb" \
        --ExecutePreprocessor.allow_errors=False || RUN_FAILED=true

    if [ "$RUN_FAILED" = true ]; then
        echo "Run $i failed, skipping..."
        curl -H "Content-Type: application/json" \
             -X POST \
             -d "{\"content\": \"Notebook run #$i failed.\"}" \
             $WEBHOOK_URL
        RUN_FAILED=false
        continue
    else
        curl -H "Content-Type: application/json" \
             -X POST \
             -d "{\"content\": \"Notebook run #$i successful\"}" \
             $WEBHOOK_URL
    fi
done

curl -H "Content-Type: application/json" \
     -X POST \
     -d "{\"content\": \"All notebook runs finished.\"}" \
     $WEBHOOK_URL

