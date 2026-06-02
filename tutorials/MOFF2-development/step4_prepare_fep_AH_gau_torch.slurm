#!/bin/bash
#SBATCH --job-name=A1-LCD-NLS
#SBATCH --output=slurm-output/train-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=8GB

module purge
eval "$(conda shell.bash hook)"
conda activate idpcg

sim_path='/home/yumzhang/orcd/pool/work/4-idpcg/simulation_test_gau/'

protein=$1
results=$2
ana=$3
#for protein in "${PROTEINS[@]}"
for protein in $protein
do
        echo  $protein
        python s4_prepare_fep_AH_gau_torch.py  \
		--protein_dir ${sim_path}/${results}/${ana}_long/$protein \
	        --gauss_delta_mu_nm 0.25 \
	        --gauss_width_nm 0.1
done


