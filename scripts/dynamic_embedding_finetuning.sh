#!/bin/bash
#SBATCH --job-name=DYNAMIC_EMB_FT
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

set -euo pipefail

ulimit -n 65536 || true

cd /mnt/homeGPU/pbovera/

# Configuracion fija del barrido
ENV_PY="/mnt/homeGPU/pbovera/envs/tfg/bin/python"
UPDATE_TYPE="remove"
MODELS=("ResNet18" "MobileNetV3-Small" "EfficientNet-B0")
EPOCHS=5
PORC="${PORC:-10}"
RESULTS_BASE_DIR="/mnt/homeGPU/pbovera/results/dynamic_embedding_finetuning"
OVERWRITE_FLAG="--overwrite"

COMMON_ARGS=(
    --models "${MODELS[@]}"
    --update-type "$UPDATE_TYPE"
    --epochs "$EPOCHS"
    --porc "$PORC"
    "$OVERWRITE_FLAG"
)

echo "Python executable:"
"$ENV_PY" -c "import sys; print(sys.executable)"
"$ENV_PY" -V
"$ENV_PY" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

mapfile -t DATASET_CLASS_PAIRS < <(
    "$ENV_PY" - <<'PY'
from src.experiments_config.class_removal_baseline_config import CLASSES_TO_REMOVE_BY_DATASET

for dataset_name, class_list in CLASSES_TO_REMOVE_BY_DATASET.items():
    for class_name in class_list:
        print(f"{dataset_name}\t{class_name}")
PY
)

for pair in "${DATASET_CLASS_PAIRS[@]}"; do
    IFS=$'\t' read -r dataset_name modified_class <<< "$pair"

    echo "Running precomputed-embeddings dynamic fine-tuning | dataset=${dataset_name} | modified_class=${modified_class}"
    "$ENV_PY" "/mnt/homeGPU/pbovera/experiments/finetune_precomputed_embeddings.py" \
        "${COMMON_ARGS[@]}" \
        --dataset "$dataset_name" \
        --modified-class "$modified_class" \
        --output-dir "$RESULTS_BASE_DIR/precompute_embeddings_then_finetune/early_stopping"

    echo "Running epoch1-embeddings dynamic fine-tuning | dataset=${dataset_name} | modified_class=${modified_class}"
    "$ENV_PY" "/mnt/homeGPU/pbovera/experiments/finetune_epoch1_embeddings.py" \
        "${COMMON_ARGS[@]}" \
        --dataset "$dataset_name" \
        --modified-class "$modified_class" \
        --output-dir "$RESULTS_BASE_DIR/epoch1_embeddings_dynamic_finetune/early_stopping"
done

echo "Finished. Comparison table:"
echo "  $RESULTS_BASE_DIR/comparison_summary.csv"
