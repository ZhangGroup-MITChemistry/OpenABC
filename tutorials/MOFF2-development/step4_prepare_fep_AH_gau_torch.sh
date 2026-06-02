#!/bin/bash


sim_path='/home/yumzhang/orcd/pool/work/4-idpcg/simulation_test_gau/'
#results='group_default_IDP_w_0.5_OP_w_1.0_MDP_w_1.0_10_zeta_1.0_0.005_gauss_delta_mu_0.25_gauss_width_0.1'
results='group_default_IDP_w_0.5_OP_w_1.0_MDP_w_0.2_10_zeta_1.5_0.001_gauss_delta_mu_0.25_gauss_width_0.1'
ana='mdp_KL'
PROTEINS=($(ls ${sim_path}/${results}/${ana}_long/ | xargs -n 1 basename))

for protein in "${PROTEINS[@]}"
#for protein in A1-LCD-2K A1-LCD+7R+12D A1-LCD-10G+10S A1-LCD+7R+10D
do
        echo  $protein
        python s4_prepare_fep_AH_gau_torch.py  \
		--protein_dir ${sim_path}/${results}/${ana}_long/$protein \
	        --gauss_delta_mu_nm 0.25 \
	        --gauss_width_nm 0.1
done


