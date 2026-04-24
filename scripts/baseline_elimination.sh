#!/bin/bash
#SBATCH --job-name=CLASS_ELIM_BASELINE
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

cd /mnt/homeGPU/pbovera/

ENV_PY="/mnt/homeGPU/pbovera/envs/tfg/bin/python"

"$ENV_PY" -c "import sys; print(sys.executable)"
"$ENV_PY" -V
"$ENV_PY" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

"$ENV_PY" -m experiments.baseline_retrain_after_class_removal --all-datasets

mail -s "tfg_train finalizado" pauboverfemenias@gmail.com <<< "baseline_retrain_after_class_removal.py ha terminado"
