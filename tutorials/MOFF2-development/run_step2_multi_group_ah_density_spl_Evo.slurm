#!/bin/bash
#SBATCH --job-name=step2
#SBATCH --array=1,2,3,4,5,6,7,8,9,10,12,14,16,18,19,20,21,22,23,24,25,26,30,31,33,34,38,41,42,46,47,48,51,52
#SBATCH --output=slurm-output/step2-multi-group-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40GB
#SBATCH --time=12:00:00
#SBATCH --partition=mit_normal

module purge
eval "$(conda shell.bash hook)"
conda activate idpcg


# For Evo protein dataset, the training set indices are 1-10, 12, 14, 16, 18-26, 30, 31, 33, 34, 38, 41, 42, 46, 47, 48, 51, 52
protein=Evo${SLURM_ARRAY_TASK_ID}

res_group_mapping=$1

echo "Running Step 2 for protein: ${protein} with res_group_mapping: ${res_group_mapping}"

python step2_multi_group_compute_basis_ah_density_spl.py --protein ${protein} --n0 50000 --n1 50000 --T1 300.0 --ionic_strength 150 --res_group_mapping ${res_group_mapping}

echo "job done"

