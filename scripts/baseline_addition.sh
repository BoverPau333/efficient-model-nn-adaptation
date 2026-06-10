#!/bin/bash
#SBATCH --job-name=BASE_ADD
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

set -euo pipefail

ulimit -n 65536 || true

cd /mnt/homeGPU/pbovera/

ENV_PY="/mnt/homeGPU/pbovera/envs/tfg/bin/python"
MODELS=("ResNet18" "MobileNetV3-Small" "EfficientNet-B0")
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-5}"
LR="${LR:-0.001}"
REFERENCE_LR="${REFERENCE_LR:-0.001}"
RESULTS_BASE_DIR="/mnt/homeGPU/pbovera/results/class_addition_baseline"
REFERENCE_BASE_DIR="/mnt/homeGPU/pbovera/results/full_training_reference_add"
OVERWRITE_FLAG="--overwrite"
ZERO_INIT="${ZERO_INIT:-0}"

COMMON_ARGS=(
    --models "${MODELS[@]}"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --lr "$LR"
    --reference-lr "$REFERENCE_LR"
    --output-dir "$RESULTS_BASE_DIR"
    --reference-dir "$REFERENCE_BASE_DIR"
    "$OVERWRITE_FLAG"
)

if [[ "$ZERO_INIT" == "1" ]]; then
    COMMON_ARGS+=(--zero-init)
fi

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
    echo "Running class-addition baseline | dataset=${dataset_name} | all configured classes | zero_init=${ZERO_INIT}"
    "$ENV_PY" "/mnt/homeGPU/pbovera/experiments/baseline_addition_after_class_introduction.py" \
        "${COMMON_ARGS[@]}" \
        --dataset "$dataset_name"
done

echo "Finished. Summary table:"
echo "  $RESULTS_BASE_DIR/all_experiments_summary.csv"
