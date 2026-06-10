#!/bin/bash
#SBATCH --job-name=PROTO_ADD
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

set -euo pipefail

ulimit -n 65536 || true

cd /mnt/homeGPU/pbovera/

ENV_PY="/mnt/homeGPU/pbovera/envs/tfg/bin/python"
MODELS=("ResNet18" "MobileNetV3-Small" "EfficientNet-B0")
SHOTS_LIST=(1 5 10)
RESULTS_DIR="/mnt/homeGPU/pbovera/results/class_addition_prototypical_fewshot"
REFERENCE_BASE_DIR="/mnt/homeGPU/pbovera/results/full_training_reference_add"
OVERWRITE_FLAG="--overwrite"

echo "Python executable:"
"$ENV_PY" -c "import sys; print(sys.executable)"
"$ENV_PY" -V
"$ENV_PY" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

mapfile -t DATASET_NAMES < <(
    "$ENV_PY" - <<'PY'
from src.experiments_config.class_to_add import CLASSES_TO_ADD_BY_DATASET

for dataset_name in CLASSES_TO_ADD_BY_DATASET:
    print(dataset_name)
PY
)

for dataset_name in "${DATASET_NAMES[@]}"; do
    for shots in "${SHOTS_LIST[@]}"; do
        echo "Running class-addition prototypical few-shot | dataset=${dataset_name} | shots=${shots}"
        "$ENV_PY" "/mnt/homeGPU/pbovera/experiments/retrain_prototypical_fewshot.py" \
            --models "${MODELS[@]}" \
            --update-type add \
            --shots-per-class "$shots" \
            --output-dir "$RESULTS_DIR" \
            --reference-dir "$REFERENCE_BASE_DIR" \
            "$OVERWRITE_FLAG" \
            --dataset "$dataset_name"
    done
done

echo "Finished. Summary table:"
echo "  $RESULTS_DIR/experiments_summary.csv"
