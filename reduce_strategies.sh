#!/bin/bash
#SBATCH --job-name reduce_strategy_1
#SBATCH --partition dios 
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/mnt/homeGPU/pbovera/logs/train_%j.outs

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"


cd /mnt/homeGPU/pbovera/

conda activate tfg

python reduce_and_evaluate.py

mail -s "tfg_train finalizado" pauboverfemenias@gmail.com <<< "reduce_stetegies terminado"
