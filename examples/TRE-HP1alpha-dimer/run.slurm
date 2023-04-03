#!/bin/bash
#SBATCH --job-name=TRE
#SBATCH --output=TRE-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive
#SBATCH --mem=10GB
#SBATCH --time=10-00:00:00

module load cuda/10.0

export WORLD_SIZE=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))
echo "WORLD_SIZE=${WORLD_SIZE}"

export MASTER_PORT=$(expr 30000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
echo "MASTER_PORT=${MASTER_PORT}"

master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_ADDR=${master_addr}
echo "MASTER_ADDR=${MASTER_ADDR}"

mkdir -p output-dcd

srun python run_TRE.py

echo "job done"

