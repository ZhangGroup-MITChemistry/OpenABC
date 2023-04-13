#!/bin/bash
#SBATCH --job-name=NPT
#SBATCH --output=slurm-output/NPT-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=10GB
#SBATCH --time=12:00:00

module load cuda/10.0

python compress.py

echo "job done"

