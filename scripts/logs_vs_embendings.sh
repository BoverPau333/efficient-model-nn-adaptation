#!/bin/bash
#SBATCH --job-name=LogsVSEmbendigs
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.out

cd /mnt/homeGPU/pbovera/

ENV_PY="/mnt/homeGPU/pbovera/envs/tfg/bin/python"

"$ENV_PY" -c "import sys; print(sys.executable)"
"$ENV_PY" -V
"$ENV_PY" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

"$ENV_PY" -m  experiments.logits_vs_embeddings_distance_comparison

mail -s "tfg_train finalizado" pauboverfemenias@gmail.com <<< "embedding_distance_comparison.py ha terminado"
