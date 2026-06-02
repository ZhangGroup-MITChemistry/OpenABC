import numpy as np
import pandas as pd
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import warnings
warnings.filterwarnings('ignore')
import pickle
from openabc.forcefields.parsers import HPSParser
from openabc.forcefields import HPSModel
import mdtraj
import argparse
import json
import sys
import os
from openabc.forcefields.MOFF2.core import compute_PE
from openabc.forcefields.MOFF2.utils import compute_distance_matrices
from openabc.forcefields.MOFF2.lib import GAS_CONST, VEP, kB, NA, EC, _amino_acids, _kcal_to_kj, _res_group_mappings
from openabc.forcefields.MOFF2.forcefields import compute_multi_group_density_spline_basis
from openabc.forcefields.MOFF2.forcefields import compute_density_switch_ashbaugh_hatch_gauss_basis

parser = argparse.ArgumentParser()
parser.add_argument('--protein', required=True, help='Protein name')
parser.add_argument('--n0', type=int, default=20000, help='The target number of noise samples used for training')
parser.add_argument('--n1', type=int, default=20000, help='The target number of data samples used for training')
parser.add_argument('--T1', type=float, required=True, help='Data trajectory simulation temperature in unit K')
parser.add_argument('--ionic_strength', type=float, required=True, help='Data trajectory ionic strength in unit mM')
parser.add_argument('--eta', type=float, default=10.0, help='eta parameter in unit 1 / nm')
parser.add_argument('--r0', type=float, default=0.7, help='r0 parameter in unit nm')
parser.add_argument('--rho_min', type=float, default=0.0, help='rho min')
parser.add_argument('--rho_max', type=float, default=15.0, help='rho max')
parser.add_argument('--n_internal_knots', type=int, default=10, help='Number of internal knots')
parser.add_argument('--res_group_mapping', type=str, default='default', help='Residue group mapping, can be "default", "hydrophobic_polar", and etc.')
args = parser.parse_args()

robustelli2018developing_IDPs = ['ACTR', 'Abeta40', 'Ash1', 'NTail', 'alpha-synuclein', 'drkN-SH3', 'p15PAF', 'sic1']
lindorff2011fast_OPs = ['Chignolin', 'Homeodomain', 'Trp-cage', 'Villin', 'WW-domain', 'Protein-G']
piana2020development_OPs = ['NTL9', 'Protein-B', 'alpha3d', 'bba', 'bbl', 'engrailed', 'gpw', 'lambda-repressor', 'BPTI', 'calmodulin']
robustelli2018developing_OPs = ['Ubiquitin', 'GB3', 'Hen-Egg-White-Lysozyme']
all_OPs = lindorff2011fast_OPs + piana2020development_OPs + robustelli2018developing_OPs

output_dict = {}

T1 = args.T1 * unit.kelvin
RT1_value = (GAS_CONST * T1).value_in_unit(unit.kilojoule_per_mole)
output_dict['T1'] = T1.value_in_unit(unit.kelvin)

# load trajectory, labels, and u0
p_path='/orcd/data/binz/001/congwang/TW-PCCG-develop/TW-PCCG-develop/train-ca-models/transferable-ca-models/'
main_working_dir = f'{p_path}/training-input/n0-{args.n0}-n1-{args.n1}/{args.protein}'
ca_pdb = f'{main_working_dir}/{args.protein}_ca.pdb'

traj = mdtraj.load_dcd(f'{main_working_dir}/traj.dcd', top=ca_pdb)
labels = np.load(f'{main_working_dir}/labels.npy')
u0 = np.load(f'{main_working_dir}/u0.npy')
output_dict['labels'] = labels
output_dict['u0'] = u0
output_dict['res_group_mapping'] = args.res_group_mapping

# compute bonded energy directly with openmm unbiased system
if args.protein in all_OPs:
    # compute energy
    noise_main_dir = '/orcd/data/binz/001/congwang/TW-PCCG-develop/TW-PCCG-develop/noise-cg-simulations/noise-HPS-Urry-sbm-T-5ldby-simulations'
    unbiased_system_xml = f'{noise_main_dir}/{args.protein}-HPS-Urry-sbm/unbiased-system/unbiased_system.xml'
    with open(unbiased_system_xml, 'r') as f:
        unbiased_system = mm.XmlSerializer.deserialize(f.read())
    u1_bond = compute_PE(traj, unbiased_system, platform_name='Reference', groups={1}) / RT1_value
    u1_angle = compute_PE(traj, unbiased_system, platform_name='Reference', groups={2}) / RT1_value
    u1_dihedral = compute_PE(traj, unbiased_system, platform_name='Reference', groups={3}) / RT1_value
    u1_native_pair = compute_PE(traj, unbiased_system, platform_name='Reference', groups={4}) / RT1_value
    u1_all_bonded = u1_bond + u1_angle + u1_dihedral + u1_native_pair
    df_exclusions = pd.read_csv(f'{noise_main_dir}/{args.protein}-HPS-Urry-sbm/unbiased-system/exclusions.csv')
    exclusions = df_exclusions[['a1', 'a2']].to_numpy().astype(int)
    # save to output_dict
    output_dict['u1_bond'] = u1_bond
    output_dict['u1_angle'] = u1_angle
    output_dict['u1_dihedral'] = u1_dihedral
    output_dict['u1_native_pair'] = u1_native_pair
    output_dict['u1_all_bonded'] = u1_all_bonded
    output_dict['exclusions'] = exclusions
else:
    # compute energy
    noise_main_dir = '/orcd/data/binz/001/congwang/TW-PCCG-develop/TW-PCCG-develop/noise-cg-simulations/noise-HPS-Urry-simulations'
    unbiased_system_xml = f'{noise_main_dir}/{args.protein}-HPS-Urry/unbiased-system/unbiased_system.xml'
    with open(unbiased_system_xml, 'r') as f:
        unbiased_system = mm.XmlSerializer.deserialize(f.read())
    u1_bond = compute_PE(traj, unbiased_system, platform_name='Reference', groups={1}) / RT1_value
    u1_all_bonded = u1_bond
    # get exclusions
    df_exclusions = pd.read_csv(f'{noise_main_dir}/{args.protein}-HPS-Urry/unbiased-system/exclusions.csv')
    exclusions = df_exclusions[['a1', 'a2']].to_numpy().astype(int)
    # save to output_dict
    output_dict['u1_bond'] = u1_bond
    output_dict['u1_all_bonded'] = u1_all_bonded
    output_dict['exclusions'] = exclusions

# build the openmm system with electrostatic cutoff as 5 * ldby
# just use HPSModel to compute electrostatic interactions
dielectric = 80.0
ionic_strength = args.ionic_strength * unit.millimolar
hps_model = HPSModel()
hps_model.append_mol(HPSParser(ca_pdb))
top = app.PDBFile(ca_pdb).getTopology()
hps_model.create_system(top=top, box_a=1000, box_b=1000, box_c=1000)
assert traj.n_chains == 1
charge = hps_model.atoms['charge'].to_numpy()
charge[0] += 1 # modify charge for N-terminal CA atom
charge[-1] -= 1 # modify charge for C-terminal CA atom
hps_model.atoms['charge'] = charge
ldby = (kB * T1 * VEP * dielectric / (2 * NA  * ionic_strength * EC**2))**0.5
elec_cutoff = 5 * ldby
hps_model.exclusions = df_exclusions # update exclusions
hps_model.add_dh_elec(ldby, dielectric, elec_cutoff, force_group=1)
u1_elec = compute_PE(traj, hps_model.system, platform_name='Reference', groups={1}) / RT1_value
output_dict['dielectric'] = dielectric
output_dict['ionic_strength'] = ionic_strength.value_in_unit(unit.millimolar)
output_dict['ldby'] = ldby.value_in_unit(unit.nanometer)
output_dict['elec_cutoff'] = elec_cutoff.value_in_unit(unit.nanometer)
output_dict['u1_elec'] = u1_elec
    
# compute pair basis
# collect pair distances, note do not consider PBC
distance_matrices = compute_distance_matrices(traj, use_pbc=False)
n_frames = traj.n_frames
n_atoms = traj.n_atoms
assert distance_matrices.shape == (n_frames, n_atoms, n_atoms)

# get atom types
df_atoms = hps_model.atoms.copy()
resnames = df_atoms['resname'].tolist()
atom_types = np.array([_amino_acids.index(x) for x in resnames])
output_dict['atom_types'] = atom_types

# get a mask for atom pairs
rows, cols = np.triu_indices(20, k=0)
Y = np.zeros((20, 20))
Y[rows, cols] = np.arange(210)
Y[cols, rows] = np.arange(210)
pair_type_matrix = np.zeros((n_atoms, n_atoms))
for i in range(n_atoms):
    for j in range(i, n_atoms):
        pair_type_matrix[i, j] = Y[atom_types[i], atom_types[j]]
        pair_type_matrix[j, i] = pair_type_matrix[i, j]
pair_type_matrix[exclusions[:, 0], exclusions[:, 1]] = -1 # set exclusions
pair_type_matrix[exclusions[:, 1], exclusions[:, 0]] = -1 # set exclusions
rows, cols = np.tril_indices(n_atoms, k=0)
pair_type_matrix[rows, cols] = -1 # exclude diagonal pairs and lower triangular pairs to avoid duplicates

# set sigma and epsilon
sigma = np.zeros((20, 20))
df_hps_urry_param = pd.read_csv('../parameters/HPS_Urry_parameters.csv')
for _, row in df_hps_urry_param.iterrows():
    a1 = _amino_acids.index(row['atom_type1'])
    a2 = _amino_acids.index(row['atom_type2'])
    sigma[a1, a2] = row['sigma']
    sigma[a2, a1] = row['sigma']
rows, cols = np.triu_indices(20, k=0)
sigma_1d = sigma[rows, cols]
epsilon = 0.2 * _kcal_to_kj # use the same epsilon as the HPS-Urry model

# express the reduced nonbonded pair energy as u1_lj_excl + np.dot(u1_ah_basis, ah_coeff)
# note here we expect ah_coeff to be the hydrophobic scale and unitless, so u1_ah_basis is of reduced energy unit
u1_lj_excl = np.zeros((n_frames, 210))
u1_ah_basis = np.zeros((n_frames, 210))
for i in range(210):
    flag = pair_type_matrix == i
    if np.any(flag):
        r = distance_matrices[:, flag] # shape = (n_frames, n_selected_pairs)
        r1 = 2**(1 / 6) * sigma_1d[i] # switch distance
        ah_cutoff = 4 * sigma_1d[i]
        s1 = np.heaviside(r1 - r, 0) # switch function for r <= r1 = 2**(1/6) * sigma
        s2 = np.heaviside(ah_cutoff - r, 0) * np.heaviside(r - r1, 0) # switch function for r1 < r <= ah_cutoff
        lj_at_cutoff = 4 * epsilon * ((sigma_1d[i] / ah_cutoff)**12 - (sigma_1d[i] / ah_cutoff)**6)
        lj = 4 * epsilon * ((sigma_1d[i] / r)**12 - (sigma_1d[i] / r)**6) # shape = (n_frames, n_selected_pairs)
        u1_lj_excl[:, i] = np.sum((lj + epsilon) * s1, axis=1) / RT1_value
        u1_ah_basis[:, i] = np.sum((-epsilon - lj_at_cutoff) * s1 + (lj - lj_at_cutoff) * s2, axis=1) / RT1_value
u1_lj_excl = np.sum(u1_lj_excl, axis=1) # shape = (n_frames,)
output_dict['epsilon'] = epsilon
output_dict['sigma'] = sigma
output_dict['exclusions'] = exclusions
output_dict['u1_lj_excl'] = u1_lj_excl # of reduced energy unit
output_dict['u1_ah_basis'] = u1_ah_basis # of reduced energy unit

gauss_delta_mu = 0.0      # (your chosen value)
gauss_width = 0.05        # (your chosen value)

gauss_basis_dict = compute_density_switch_ashbaugh_hatch_gauss_basis(
    traj=traj,
    temperature=T1,               # must be unit.Quantity (correct!)
    exclusions=exclusions,
    epsilon=epsilon,              # float, LJ epsilon (kJ/mol)
    sigma_ah=sigma,               # 20×20 sigma matrix
    eta=args.eta,
    r0=args.r0,
    mu=2.0,                       # default or your chosen value
    rho0=5.5,                     # default or your chosen value
    gauss_delta_mu=gauss_delta_mu,
    gauss_width=gauss_width,
)


# store Gaussian basis + parameters into training_input dict
output_dict['u1_gauss_basis'] = gauss_basis_dict['u1_gauss_basis']
output_dict['gauss_delta_mu'] = gauss_delta_mu
output_dict['gauss_width'] = gauss_width


# compute density spline basis
density_spl_basis_dict = compute_multi_group_density_spline_basis(traj, T1, _res_group_mappings[args.res_group_mapping],
                                                      args.eta, args.r0, 
                                                      args.rho_min, args.rho_max, 
                                                      args.n_internal_knots, intercept=True)
output_dict['eta'] = args.eta
output_dict['r0'] = args.r0
output_dict['rho_min'] = args.rho_min
output_dict['rho_max'] = args.rho_max
output_dict['n_internal_knots'] = args.n_internal_knots
output_dict['degree'] = density_spl_basis_dict['degree']
output_dict['u1_density_spl_basis'] = density_spl_basis_dict['u1_density_spl_basis']
output_dict['augmented_knots'] = density_spl_basis_dict['augmented_knots']
output_dict['basis_coeffs'] = density_spl_basis_dict['basis_coeffs']
output_dict['omega'] = density_spl_basis_dict['omega']

# save parameters
output_pkl = f'{main_working_dir}/training_input_ah_density_spl_group_{args.res_group_mapping}_eta_{args.eta}_r0_{args.r0}_rho_range_{args.rho_min}_{args.rho_max}_n_internal_knots_{args.n_internal_knots}_gau.pkl'
with open(output_pkl, 'wb') as f:
    pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

