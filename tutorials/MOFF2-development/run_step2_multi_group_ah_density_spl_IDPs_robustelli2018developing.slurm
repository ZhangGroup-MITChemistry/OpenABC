#!/bin/bash
#SBATCH --job-name=step2
#SBATCH --array=0-7
#SBATCH --output=slurm-output/step2-multi-group-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=120GB
#SBATCH --time=12:00:00
#SBATCH --partition=mit_normal

module purge
eval "$(conda shell.bash hook)"
conda activate idpcg


all_proteins=(ACTR Abeta40 Ash1 NTail alpha-synuclein drkN-SH3 p15PAF sic1)
protein=${all_proteins[${SLURM_ARRAY_TASK_ID}]}
all_ionic_strengths=(150 50 150 100 100 50 50 150)
ionic_strength=${all_ionic_strengths[${SLURM_ARRAY_TASK_ID}]}

res_group_mapping=$1

echo "Running step 2 for protein: ${protein} with ionic strength: ${ionic_strength} mM and residue group mapping: ${res_group_mapping}"

python step2_multi_group_compute_basis_ah_density_spl.py --protein ${protein} --n0 50000 --n1 50000 --T1 300.0 --ionic_strength ${ionic_strength} --res_group_mapping ${res_group_mapping}

echo "job done"

