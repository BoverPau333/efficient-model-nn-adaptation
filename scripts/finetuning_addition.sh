#!/bin/bash
#SBATCH --job-name=FT_ADD
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

set -euo pipefail

ulimit -n 65536 || true

cd /mnt/homeGPU/pbovera/

ENV_PY="/mnt/homeGPU/pbovera/envs/tfg/bin/python"
MODELS=("ResNet18" "MobileNetV3-Small" "EfficientNet-B0")
EPOCHS="${EPOCHS:-5}"
FROZEN_EPOCHS="${FROZEN_EPOCHS:-5}"
UNFROZEN_EPOCHS="${UNFROZEN_EPOCHS:-10}"
LR="${LR:-0.001}"
PERCENTAGES=(100 50 20 10)
RESULTS_HEAD_ONLY_DIR="/mnt/homeGPU/pbovera/results/class_addition_finetuning_head_only"
RESULTS_TWO_STAGE_DIR="/mnt/homeGPU/pbovera/results/class_addition_finetuning_two_stage"
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
    for porc in "${PERCENTAGES[@]}"; do
        echo "Running class-addition fine-tuning | mode=head_only | dataset=${dataset_name} | porc=${porc}"
        "$ENV_PY" "/mnt/homeGPU/pbovera/experiments/finetuning_addition_after_class_introduction.py" \
            --models "${MODELS[@]}" \
            --epochs "$EPOCHS" \
            --frozen-epochs "$FROZEN_EPOCHS" \
            --unfrozen-epochs "$UNFROZEN_EPOCHS" \
            --lr "$LR" \
            --porc "$porc" \
            --output-dir "$RESULTS_HEAD_ONLY_DIR" \
            --reference-dir "$REFERENCE_BASE_DIR" \
            "$OVERWRITE_FLAG" \
            --dataset "$dataset_name"

        echo "Running class-addition fine-tuning | mode=two_stage_finetuning | dataset=${dataset_name} | porc=${porc}"
        "$ENV_PY" "/mnt/homeGPU/pbovera/experiments/finetuning_addition_after_class_introduction.py" \
            --models "${MODELS[@]}" \
            --epochs "$EPOCHS" \
            --frozen-epochs "$FROZEN_EPOCHS" \
            --unfrozen-epochs "$UNFROZEN_EPOCHS" \
            --lr "$LR" \
            --porc "$porc" \
            --output-dir "$RESULTS_TWO_STAGE_DIR" \
            --reference-dir "$REFERENCE_BASE_DIR" \
            "$OVERWRITE_FLAG" \
            --two-stage-finetuning \
            --dataset "$dataset_name"
    done
done

echo "Finished. Summary table:"
echo "  $RESULTS_HEAD_ONLY_DIR/all_experiments_summary.csv"
echo "  $RESULTS_TWO_STAGE_DIR/all_experiments_summary.csv"
