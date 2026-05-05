#!/bin/bash
#SBATCH --job-name=CLASS_ELIM_FT_PORC
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

set -euo pipefail

cd /mnt/homeGPU/pbovera/

ENV_PY="/mnt/homeGPU/pbovera/envs/tfg/bin/python"
PORCS=(50 20 10)

"$ENV_PY" -c "import sys; print(sys.executable)"
"$ENV_PY" -V
"$ENV_PY" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

for porc in "${PORCS[@]}"; do
    echo "Running head-only fine-tuning with --porc ${porc}"
    "$ENV_PY" -m experiments.finetuning_retrain_after_class_removal \
        --all-datasets \
        --overwrite \
        --porc "$porc"
done

for porc in "${PORCS[@]}"; do
    echo "Running two-stage fine-tuning with --porc ${porc}"
    "$ENV_PY" -m experiments.finetuning_retrain_after_class_removal \
        --all-datasets \
        --overwrite \
        --porc "$porc" \
        --two-stage-finetuning
done

mail -s "tfg_train finalizado" pauboverfemenias@gmail.com <<< "finetuning_elimination.sh ha terminado"
