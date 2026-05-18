#!/bin/bash
#SBATCH --job-name=PROTO_FEWSHOT
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

set -euo pipefail

cd /mnt/homeGPU/pbovera/

ENV_PY="/mnt/homeGPU/pbovera/envs/tfg/bin/python"
SHOTS_LIST=(1 5 10)
UPDATE_TYPE="${UPDATE_TYPE:-remove}"
RESULTS_DIR="/mnt/homeGPU/pbovera/results/class_removal_prototypical_fewshot"

"$ENV_PY" -c "import sys; print(sys.executable)"
"$ENV_PY" -V
"$ENV_PY" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

for shots in "${SHOTS_LIST[@]}"; do
    echo "Running Prototypical few-shot | update_type=${UPDATE_TYPE} | shots_per_class=${shots}"
    "$ENV_PY" -m experiments.retrain_prototypical_fewshot \
        --all-datasets \
        --overwrite \
        --update-type "$UPDATE_TYPE" \
        --shots-per-class "$shots" \
        --output-dir "$RESULTS_DIR"
done

mail -s "tfg_train finalizado" pauboverfemenias@gmail.com <<< "retrain_prototypical_fewshot.py ha terminado"
