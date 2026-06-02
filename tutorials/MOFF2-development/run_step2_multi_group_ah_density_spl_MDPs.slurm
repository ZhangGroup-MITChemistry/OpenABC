#!/bin/bash
#SBATCH --job-name=step2
#SBATCH --array=0-12
#SBATCH --output=slurm-output/step2-multi-group-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=360GB
#SBATCH --time=12:00:00
#SBATCH --partition=mit_normal

module purge
eval "$(conda shell.bash hook)"
conda activate idpcg


all_MDPs=(THB-C2 Ub2 Ub3 Gal3 hnRNPA1-star FPs-GS8 FPs-GS16 SH4UD-SH3-SH2 TDP43_WtoA D12 D23 D34 SMAD4)
MDP=${all_MDPs[${SLURM_ARRAY_TASK_ID}]}

res_group_mapping=$1

echo "Running step 2 for MDP: ${MDP} with res_group_mapping: ${res_group_mapping}"

python step2_multi_group_compute_basis_MDP_ah_density_spl.py --protein ${MDP} --n0 10000 --n1 10000 --res_group_mapping ${res_group_mapping}

echo "job done"

