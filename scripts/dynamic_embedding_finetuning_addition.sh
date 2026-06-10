#!/bin/bash
#SBATCH --job-name=DYN_FT_ADD
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

set -euo pipefail

ulimit -n 65536 || true

cd /mnt/homeGPU/pbovera/

ENV_PY="/mnt/homeGPU/pbovera/envs/tfg/bin/python"
UPDATE_TYPE="add"
MODELS=("ResNet18" "MobileNetV3-Small" "EfficientNet-B0")
EPOCHS="${EPOCHS:-5}"
PERCENTAGES=(100 50 20 10)
SELECTION_STRATEGY="${SELECTION_STRATEGY:-composite_score}"
SCORE_ALPHA="${SCORE_ALPHA:-0.3}"
SCORE_BETA="${SCORE_BETA:-0.4}"
SCORE_GAMMA="${SCORE_GAMMA:-0.3}"
RESULTS_BASE_DIR="/mnt/homeGPU/pbovera/results/dynamic_embedding_finetuning"
REFERENCE_BASE_DIR="/mnt/homeGPU/pbovera/results/full_training_reference_add"
OVERWRITE_FLAG="--overwrite"

COMMON_ARGS=(
    --models "${MODELS[@]}"
    --update-type "$UPDATE_TYPE"
    --epochs "$EPOCHS"
    --selection-strategy "$SELECTION_STRATEGY"
    --score-alpha "$SCORE_ALPHA"
    --score-beta "$SCORE_BETA"
    --score-gamma "$SCORE_GAMMA"
    --reference-dir "$REFERENCE_BASE_DIR"
    "$OVERWRITE_FLAG"
)

echo "Python executable:"
"$ENV_PY" -c "import sys; print(sys.executable)"
"$ENV_PY" -V
"$ENV_PY" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

mapfile -t DATASET_CLASS_PAIRS < <(
    "$ENV_PY" - <<'PY'
from src.experiments_config.class_to_add import CLASSES_TO_ADD_BY_DATASET

for dataset_name, class_list in CLASSES_TO_ADD_BY_DATASET.items():
    for class_name in class_list:
        print(f"{dataset_name}\t{class_name}")
PY
)

for pair in "${DATASET_CLASS_PAIRS[@]}"; do
    IFS=$'\t' read -r dataset_name modified_class <<< "$pair"

    for porc in "${PERCENTAGES[@]}"; do
        echo "Running precomputed dynamic fine-tuning for addition | dataset=${dataset_name} | added_class=${modified_class} | porc=${porc}"
        "$ENV_PY" "/mnt/homeGPU/pbovera/experiments/finetune_precomputed_embeddings.py" \
            "${COMMON_ARGS[@]}" \
            --dataset "$dataset_name" \
            --modified-class "$modified_class" \
            --porc "$porc" \
            --output-dir "$RESULTS_BASE_DIR/precompute_embeddings_then_finetune/early_stopping"

        echo "Running epoch1 dynamic fine-tuning for addition | dataset=${dataset_name} | added_class=${modified_class} | porc=${porc}"
        "$ENV_PY" "/mnt/homeGPU/pbovera/experiments/finetune_epoch1_embeddings.py" \
            "${COMMON_ARGS[@]}" \
            --dataset "$dataset_name" \
            --modified-class "$modified_class" \
            --porc "$porc" \
            --output-dir "$RESULTS_BASE_DIR/epoch1_embeddings_dynamic_finetune/early_stopping"
    done
done

echo "Finished. Comparison table:"
echo "  $RESULTS_BASE_DIR/comparison_summary.csv"
