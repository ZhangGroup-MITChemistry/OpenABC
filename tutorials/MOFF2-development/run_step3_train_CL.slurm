#!/bin/bash
#SBATCH --job-name=1.0_1.0
#SBATCH --output=slurm-output/train-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --partition=mit_preemptable
#SBATCH --time=10:00:00

module purge
eval "$(conda shell.bash hook)"
conda activate idpcg

# AH potential training params
MJ_min=0.0
MJ_max=0.8
res_group_mapping="default"

echo "Training with MJ_min=${MJ_min}, MJ_max=${MJ_max}, res_group_mapping=${res_group_mapping}"

IDP_weights=(1.0)
MDP_weights=(0.4)
OP_weights=(1.0)

for zeta1 in 0.2; do
#  for zeta2 in 0.001 0.0005 0.0001; do
   for zeta2 in  0.002 0.004 0.006 0.01 0.008; do
    for IDP_weight in "${IDP_weights[@]}"; do
      for MDP_weight in "${MDP_weights[@]}"; do
        for OP_weight in "${OP_weights[@]}"; do

          # ----------------------------------------------
          # 1. Construct output folder name
          # ----------------------------------------------
          OUTDIR="results/group_${res_group_mapping}_IDP_w_${IDP_weight}_OP_w_${OP_weight}_MDP_w_${MDP_weight}_10_zeta_${zeta1}_${zeta2}_gauss_delta_mu_0.25_gauss_width_0.1"

          echo "Checking folder: ${OUTDIR}"

          # ----------------------------------------------
          # 2. Security check → skip if results.pkl exists
          # ----------------------------------------------
          if [ -f "${OUTDIR}/results.pkl" ]; then
              echo "⚠️  SKIP: ${OUTDIR}/results.pkl already exists."
              continue
          fi

          echo "🚀 Running training for:"
          echo "    IDP=${IDP_weight}, MDP=${MDP_weight}, OP=${OP_weight}"
          echo "    zeta1=${zeta1}, zeta2=${zeta2}"
          echo "    Output folder: ${OUTDIR}"
          echo ""

          # ----------------------------------------------
          # 3. Run training
          # ----------------------------------------------
          python -u train_model_gau.py \
            --n_epochs 10000 \
            --lr 0.5 \
            --MJ_min "${MJ_min}" \
            --MJ_max "${MJ_max}" \
            --zeta1 "${zeta1}" \
            --zeta2 "${zeta2}" \
            --res_group_mapping "${res_group_mapping}" \
            --IDP_weight "${IDP_weight}" \
            --MDP_weight "${MDP_weight}" \
            --OP_weight "${OP_weight}" 

          echo "----------------------------------------------"
          echo "Done: ${OUTDIR}"
          echo "----------------------------------------------"

        done
      done
    done
  done
done

echo "job done"

