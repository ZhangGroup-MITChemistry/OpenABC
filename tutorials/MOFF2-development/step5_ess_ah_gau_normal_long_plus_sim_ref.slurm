#!/bin/bash
#SBATCH --job-name=50800
#SBATCH --output=slurm-output/train-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu

sim_path='/home/yumzhang/orcd/pool/work/4-idpcg/simulation_test_gau/'

results='group_default_IDP_w_0.5_OP_w_1.0_MDP_w_1.0_10_zeta_1.0_0.005_gauss_delta_mu_0.25_gauss_width_0.1'

eval "$(conda shell.bash hook)"
conda activate idpcg

alpha=$1
ess0=$2
nstep=$3

for alpha in $alpha
do
        for ess0 in $ess0
        do
		for nstep in $nstep
		do
                newdir=ess_optimize_ah_gau_idp_mdp_op_IDP_w_0.5_OP_w_1.0_MDP_w_1.0_10_zeta_1.0_0.005_plus_sim_ref_llps
                mkdir -p $newdir
python s5_ess_ah_gau_density_all_idp_mdp_op.py \
  --results_pkl ${sim_path}/results/${results}.pkl \
  --fep_input_root ${sim_path}/fep_AH_gau_${results}/ \
  --exp_csv exp_plus_sim_a1_ref.csv \
  --output_pkl ${newdir}/alpha_${alpha}_ess0_${ess0}_ah_idp_mdp_op_${nstep}/results.pkl \
  --output_dir ${newdir}/alpha_${alpha}_ess0_${ess0}_ah_idp_mdp_op_${nstep}/ \
  --alpha $alpha \
  --ess0 $ess0 \
  --lam_density 1e-2 \
  --device cuda \
  --n_epochs ${nstep}
done
done
done

