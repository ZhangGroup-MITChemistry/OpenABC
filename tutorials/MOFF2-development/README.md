# MOFF2 Training Workflow

This folder contains the scripts used to train and locally refine the MOFF2
coarse-grained force field. The workflow has five stages:

1. Build mixed noise/data ensembles.
2. Compute reduced-energy basis terms for the MOFF2 energy function.
3. Train the global model by contrastive learning.
4. Prepare FEP/reweighting inputs from production or validation simulations.
5. Apply ESS-constrained FEP refinement against target observables.

The training scripts import MOFF2 through the OpenABC namespace:

```python
from openabc.forcefields.MOFF2.core import compute_PE
from openabc.forcefields.MOFF2.forcefields import HPSMOFFCosAngleTestModel
```

Before running scripts directly from this repository, make sure the repository
root is on `PYTHONPATH`, or install OpenABC in editable mode.

```bash
export PYTHONPATH=/path/to/openabc:$PYTHONPATH
```

## Requirements

The workflow assumes a Python environment with OpenABC and the scientific
simulation stack used by the scripts:

- `openabc`
- `openmm`
- `mdtraj`
- `numpy`
- `pandas`
- `torch`
- `FastMBAR`
- `scipy`
- `matplotlib`
- `seaborn`
- `tqdm`

GPU support is used in the Potential Contrasting training `step3_train_pc.py` and FEP/ESS fine-tuning stages `step5_ess_ah_gau_density_all_idp_mdp_op.py`.  currently
expects two GPUs on one node.

Required input files include:

reference CA trajectories;
noise simulations;
training-input/;
parameters/; # CSV files such as raw_MJ.csv and HPS_Urry_parameters.csv;
simulation outputs used for FEP refinement.

MOFF2 was trained on files below: 

- reference CA trajectories;
- noise simulations;
- `parameters/`; parameters used as starting point for MOFF2.
- simulation outputs used for FEP/ESS refinement.


## Stage 1: Build Mixed Noise/Data Ensembles

Script:

```text
step1_compute_noise_u0.py
```

Purpose:

This step combines noise samples and reference data samples for one protein,
computes their reduced energies under the mixed noise ensemble, and writes the
basic training input files.

Main inputs:

- reference CA trajectory and CA PDB for the selected protein;
- noise simulation trajectories from umbrella-biased HPS or HPS-SBM runs;
- the corresponding unbiased OpenMM system XML.

Example:

```bash
python step1_compute_noise_u0.py \
  --protein ACTR \
  --n0 50000 \
  --n1 50000 \
  --T0 300.0
```

Important arguments:

- `--protein`: protein name.
- `--n0`: target number of noise samples.
- `--n1`: target number of data/reference samples.
- `--T0`: noise simulation temperature in K.

Outputs:

```text
training-input/n0-<n0>-n1-<n1>/<protein>/
  <protein>_ca.pdb
  traj.dcd
  labels.npy
  u0.npy
```

`labels.npy` marks noise samples as `0` and data samples as `1`. `u0.npy`
contains the reduced energy of each sample under the mixed reference/noise
ensemble.

## Stage 2: Compute MOFF2 Basis Terms

Scripts:

```text
step2_multi_group_compute_basis_ah_density_spl.py
step2_multi_group_compute_basis_ext_OPs_ah_density_spl.py
step2_multi_group_compute_basis_MDP_ah_density_spl.py
```

Purpose:

This step converts the trajectories from Stage 1 into reduced-energy basis
terms for the MOFF2 energy function. The basis terms include:

- bonded baseline terms;
- Debye-Huckel electrostatics;
- AH pairwise contact basis;
- Gaussian first-solvation-shell basis;
- density-dependent B-spline basis.

For IDPs, OPs, and Evo proteins, use:

```bash
python step2_multi_group_compute_basis_ah_density_spl.py \
  --protein ACTR \
  --n0 50000 \
  --n1 50000 \
  --T1 300.0 \
  --ionic_strength 150 \
  --res_group_mapping default
```

For extended OPs, use:

```bash
python step2_multi_group_compute_basis_ext_OPs_ah_density_spl.py \
  --protein 1soy_clean \
  --n0 50000 \
  --n1 50000 \
  --T1 300.0 \
  --ionic_strength 150 \
  --res_group_mapping default
```

For MDPs, use:

```bash
python step2_multi_group_compute_basis_MDP_ah_density_spl.py \
  --protein Ub2 \
  --n0 10000 \
  --n1 10000 \
  --res_group_mapping default
```

Important arguments:

- `--protein`: protein name.
- `--n0`, `--n1`: sample counts matching Stage 1.
- `--T1`: reference/data temperature in K, for IDP/OP scripts.
- `--ionic_strength`: ionic strength in mM, for IDP/OP scripts.
- `--eta`, `--r0`: density switching parameters.
- `--rho_min`, `--rho_max`: density range for spline basis.
- `--n_internal_knots`: number of internal B-spline knots.
- `--res_group_mapping`: residue grouping used for density basis.

Outputs:

Each script writes a pickle file in the corresponding protein folder:

```text
training-input/n0-<n0>-n1-<n1>/<protein>/
  training_input_ah_density_spl_group_<group>_eta_<eta>_r0_<r0>_rho_range_<rho_min>_<rho_max>_n_internal_knots_<n>_gau.pkl
```

The pickle contains the arrays used by CL training, including:

- `labels`
- `u0`
- `u1_all_bonded`
- `u1_lj_excl`
- `u1_elec`
- `u1_ah_basis`
- `u1_gauss_basis`
- `u1_density_spl_basis`
- density spline metadata
- Gaussian parameters

## Stage 3: Potential-Contrastive Training

Script:

```text
step3_train_pc.py
```

Purpose:

This step trains the global MOFF2 energy model by potential contrasting training across
IDPs, OPs, and MDPs. The optimized parameters include:

- 210 AH pair coefficients;
- 210 Gaussian amplitudes;
- 240 residue/group-dependent density spline coefficients.

The model initializes the AH coefficients from a scaled Miyazawa-Jernigan
matrix and initializes Gaussian coefficients to zero.

Example:

```bash
python -u step3_train_pc.py \
  --n_epochs 10000 \
  --lr 0.5 \
  --MJ_min 0.0 \
  --MJ_max 0.8 \
  --zeta1 0.2 \
  --zeta2 0.004 \
  --res_group_mapping default \
  --IDP_weight 1.0 \
  --OP_weight 1.0 \
  --MDP_weight 0.4 \
  --gauss_delta_mu 0.25 \
  --gauss_width 0.1
```

Important arguments:

- `--MJ_min`, `--MJ_max`: range used to initialize AH coefficients.
- `--zeta1`: L2 regularization for AH and Gaussian coefficient updates.
- `--zeta2`: L2 regularization for density spline coefficients.
- `--IDP_weight`, `--OP_weight`, `--MDP_weight`: class weights in the CL loss.
- `--gauss_delta_mu`, `--gauss_width`: Gaussian shape metadata saved with the model.
- `--backend`: PyTorch distributed backend. Default is `nccl`.

Outputs:

```text
results/group_<group>_IDP_w_<...>_OP_w_<...>_MDP_w_<...>_<knots>_zeta_<zeta1>_<zeta2>_gauss_delta_mu_<...>_gauss_width_<...>/
  results.pkl
  loss.csv
  hydrophobic_scale.pdf
  density_spline.pdf
  pairwise_potentials.pdf
  tests/
```

`results.pkl` is the main potential-contrastive trained parameter file. It contains:

- `hydrophobic_scale`
- `gauss_coeffs`
- `gauss_height_map`
- `spl_coeffs`
- `spl_values`
- density and Gaussian hyperparameters

The script also performs a reweighting-based validation at the end and writes
the results into the `tests/` subdirectory.

## Stage 4: Prepare FEP/ESS fine-tuning Inputs

Script:

```text
step4_prepare_fep_AH_gau_torch.py
```

Purpose:

This step converts simulation trajectories generated with a pc-trained model
into compact FEP input pickles. These files are used by Stage 5 to refine the
parameters without rerunning simulations at every optimization step.

Each protein simulation directory should contain:

```text
<protein_dir>/
  input_parameters.json
  system.xml
  output.dcd
  <protein>_ca.pdb
```

`input_parameters.json` must include at least:

```json
{
  "protein": "A1-LCD+12E",
  "temperature": 298.0,
  "ionic_strength": 150.0,
  "results_pkl": "/path/to/pc/results.pkl"
}
```

Example:

```bash
python step4_prepare_fep_AH_gau_torch.py \
  --protein_dir /path/to/simulation/A1-LCD+12E \
  --output_pkl /path/to/fep_AH_gau/A1-LCD+12E/fep_AH_gau_input.pkl \
  --gauss_delta_mu_nm 0.25 \
  --gauss_width_nm 0.1
```

If `--output_pkl` is omitted, the script writes:

```text
<protein_dir>/fep_AH_input.pkl
```

Stage 5 searches for `fep_AH_gau_input.pkl` first and falls back to
`fep_AH_input.pkl`.

Output keys include:

- `u1_intercept`
- `u1_ah_basis`
- `u1_gauss_basis`
- `u1_density_spl_basis`
- `rg`
- `results_pkl_used`

## Stage 5: FEP/ESS-Constrained FEP Refinement

Script:

```text
step5_ess_ah_gau_density_all_idp_mdp_op.py
```

Purpose:

This step locally refines the pc-trained model using ensemble-averaged target
observables, currently radius of gyration. The optimization uses FEP weights
from the CL reference ensemble and an effective-sample-size penalty to prevent
updates that rely on too few configurations.

Example:

```bash
python step5_ess_ah_gau_density_all_idp_mdp_op.py \
  --results_pkl /path/to/pc/results.pkl \
  --fep_input_root /path/to/fep_AH_gau_inputs \
  --exp_csv exp_plus_sim_a1_ref.csv \
  --output_pkl ess_optimize/results.pkl \
  --output_dir ess_optimize/ \
  --alpha 100.0 \
  --ess0 750.0 \
  --lam_density 1e-2 \
  --lr 1e-3 \
  --n_epochs 500 \
  --device cuda
```

Expected FEP input layout:

```text
<fep_input_root>/
  <protein>/
    fep_AH_gau_input.pkl
```

Expected experimental/target CSV:

The script expects an input CSV containing protein names and target Rg values.
The loaded proteins are matched against the built-in A1-LCD, MDP, and OP lists
in `step5_ess_ah_gau_density_all_idp_mdp_op.py`.

Outputs:

```text
<output_dir>/
  training_log.csv
  loss_curve.pdf
  final_predictions.csv
  ...
<output_pkl>
```

The exported `results.pkl` preserves the original CL parameter structure and
adds the FEP-refined parameter values:

- updated `hydrophobic_scale`
- updated `gauss_coeffs`
- updated `gauss_height_map`
- updated `spl_coeffs`
- updated `spl_values`
- FEP metadata

The script also reruns the reweighting validation using the refined parameter
file and writes results under:

```text
<dirname(output_pkl)>/tests/
```

## Slurm Wrappers

This folder includes Slurm wrappers for the original cluster workflow:

- `run_step1_all.sh`
- `run_step2_multi_group_ah_density_spl_all.sh`
- `run_step2_multi_group_ah_density_spl_*.slurm`
- `run_step3_train_CL.slurm`
- `step4_prepare_fep_AH_gau_torch.slurm`
- `step5_ess_ah_gau_normal_long_plus_sim_ref.slurm`

Treat these as templates. Before submission, check that:

- the conda environment name is correct;
- paths to data and simulation folders are updated;
- the script names match the current canonical names.


## End-to-End Summary

A typical full training/refinement run is:

```bash
# 1. Build mixed noise/data ensemble
python step1_compute_noise_u0.py --protein ACTR --n0 50000 --n1 50000 --T0 300.0

# 2. Compute MOFF2 basis terms
python step2_multi_group_compute_basis_ah_density_spl.py \
  --protein ACTR --n0 50000 --n1 50000 \
  --T1 300.0 --ionic_strength 150 \
  --res_group_mapping default

# 3. Train global CL model
python -u step3_train_CL.py \
  --n_epochs 10000 --lr 0.5 \
  --MJ_min 0.0 --MJ_max 0.8 \
  --zeta1 0.2 --zeta2 0.004 \
  --res_group_mapping default \
  --IDP_weight 1.0 --OP_weight 1.0 --MDP_weight 0.4

# 4. Prepare FEP input from simulations generated by the CL model
python step4_prepare_fep_AH_gau_torch.py \
  --protein_dir /path/to/simulation/A1-LCD+12E \
  --output_pkl /path/to/fep_inputs/A1-LCD+12E/fep_AH_gau_input.pkl

# 5. Run ESS-constrained FEP refinement
python step5_ess_ah_gau_density_all_idp_mdp_op.py \
  --results_pkl /path/to/CL/results.pkl \
  --fep_input_root /path/to/fep_inputs \
  --exp_csv exp_plus_sim_a1_ref.csv \
  --output_pkl ess_results/results.pkl \
  --output_dir ess_results \
  --alpha 100.0 --ess0 750.0 \
  --lam_density 1e-2 \
  --device cuda
```

## Additional Notes 

Trajectories and intermediate training inputs are available upon request. 
