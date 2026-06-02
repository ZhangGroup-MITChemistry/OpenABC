#!/bin/bash
#SBATCH --job-name=step2
#SBATCH --array=0-17
#SBATCH --output=slurm-output/step2-multi-group-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=120GB
#SBATCH --time=6:00:00
#SBATCH --partition=mit_normal

module purge
eval "$(conda shell.bash hook)"
conda activate idpcg

all_proteins=(1soy_clean 1wla_clean 2ea9_clean 5tvz_clean AF-P0C232-F1-model_v4_w_H AF-P0CG98-F1-model_v4_w_H AF-P00251-F1-model_v4_w_H AF-P21149-F1-model_v4_w_H AF-P21318-F1-model_v4_w_H AF-P29669-F1-model_v4_w_H AF-P31960-F1-model_v4_w_H AF-P32729-F1-model_v4_w_H AF-P61734-F1-model_v4_w_H AF-P69995-F1-model_v4_w_H AF-P75202-F1-model_v4_w_H AF-P75459-F1-model_v4_w_H AF-P80353-F1-model_v4_w_H AF-P87285-F1-model_v4_w_H)
protein=${all_proteins[${SLURM_ARRAY_TASK_ID}]}
T1=300.0
ionic_strength=150.0

res_group_mapping=$1

echo "Running Step2 for protein: ${protein}, T1: ${T1}, ionic_strength: ${ionic_strength}, res_group_mapping: ${res_group_mapping}"

python step2_multi_group_compute_basis_ext_OPs_ah_density_spl.py --protein ${protein} --n0 50000 --n1 50000 --T1 ${T1} --ionic_strength ${ionic_strength} --res_group_mapping ${res_group_mapping}

echo "job done"

