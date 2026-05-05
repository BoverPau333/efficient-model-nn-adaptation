#!/bin/bash
#SBATCH --job-name=CLASS_ELIM_FROZEN_HEAD
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

cd /mnt/homeGPU/pbovera/

ENV_PY="/mnt/homeGPU/pbovera/envs/tfg/bin/python"
DATASET_PORC="${DATASET_PORC:-100}"

"$ENV_PY" -c "import sys; print(sys.executable)"
"$ENV_PY" -V
"$ENV_PY" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

"$ENV_PY" -m experiments.finetuning_retrain_after_class_removal --all-datasets --overwrite --porc "$DATASET_PORC"

mail -s "tfg_train finalizado" pauboverfemenias@gmail.com <<< "finetuning_retrain_after_class_removal.py ha terminado"
