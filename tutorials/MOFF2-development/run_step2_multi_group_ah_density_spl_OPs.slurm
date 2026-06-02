#!/bin/bash
#SBATCH --job-name=step2
#SBATCH --array=0-18
#SBATCH --output=slurm-output/step2-multi-group-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00
#SBATCH --partition=mit_normal

module purge
eval "$(conda shell.bash hook)"
conda activate idpcg


# alpha3d, bba, bbl, engrailed, gpw, lambda-repressor, NTL9, Protein-B all-atom simulations are TRE simulations, and we use replicas near 300 K as training data samples
ordered_proteins=(BPTI Chignolin GB3 Hen-Egg-White-Lysozyme Homeodomain NTL9 Protein-B Protein-G Trp-cage Ubiquitin Villin WW-domain alpha3d bba bbl calmodulin engrailed gpw lambda-repressor)
protein=${ordered_proteins[${SLURM_ARRAY_TASK_ID}]}
T1_array=(300 340 300 300 360 300 300 350 290 300 360 360 300 300 300 300 300 300 300)
T1=${T1_array[${SLURM_ARRAY_TASK_ID}]}
ionic_strength_array=(37.02 25.9 9.98 26.63 45 121.46 51.44 100 65 150 40 7.1 3.11 33.04 199.58 249.98 280.46 36.54 51.04)
ionic_strength=${ionic_strength_array[${SLURM_ARRAY_TASK_ID}]}

res_group_mapping=$1

echo "Running step 2 for protein: ${protein}, T1: ${T1}, Ionic Strength: ${ionic_strength}, Res Group Mapping: ${res_group_mapping}"

python step2_multi_group_compute_basis_ah_density_spl.py --protein ${protein} --n0 50000 --n1 50000 --T1 ${T1} --ionic_strength ${ionic_strength} --res_group_mapping ${res_group_mapping}

echo "job done"

