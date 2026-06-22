#!/bin/bash
#SBATCH --job-name=reduce_strategy_1
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

cd /mnt/homeGPU/pbovera/

ENV_PATH="/home/pbovera/.conda/envs/tfg"

conda run -p "$ENV_PATH" python -c "import sys; print(sys.executable)"
conda run -p "$ENV_PATH" python -V
conda run -p "$ENV_PATH" python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

conda run -p "$ENV_PATH" python -m experiments.reduce_and_evaluate

mail -s "tfg_train finalizado" pauboverfemenias@gmail.com <<< "reduce_and_evaluate.py ha terminado"
