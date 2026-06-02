#!/usr/bin/env python3
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
import mdtraj
import argparse
import sys
import os
import json
import glob
import math
import gc

# make sure your local library is on the path

from openabc.forcefields.MOFF2.core import compute_PE
from openabc.forcefields.MOFF2.utils import compute_CA_traj_radius_of_gyration
from openabc.forcefields.MOFF2.lib import (
    GAS_CONST, VEP, kB, NA, EC,
    _amino_acids, _kcal_to_kj, _res_group_mappings
)
from openabc.forcefields.parsers import HPSParser
from openabc.forcefields import HPSModel
from tqdm import tqdm

import torch
torch.set_default_dtype(torch.float64)

from scipy.interpolate import BSpline
from scipy.integrate import quad

# Protein class lists (same as before)
A1_LCDs = [
    'A1-LCD+12E','A1-LCD+7K+12D','A1-LCD-3R+3K','A1-LCD-10R+10K','A1-LCD+4D',
    'A1-LCD-4D','A1-LCD-9F+6Y','A1-LCD+2R','A1-LCD+8D','A1-LCD-12F+12Y',
    'A1-LCD-9F+3Y','A1-LCD+NLS','A1-LCD+12D','A1-LCD-10R','A1-LCD-6R',
    'A1-LCD-NLS','A1-LCD+7R','A1-LCD+7F-7Y','A1-LCD-8F+4Y','A1-LCD-6R+6K'
]

MDPs = [
    'GS32','GS48','TIA1','D14','GS24','hSUMO_hnRNPA1S','HeV_V','PCPE',
    'GS0','NiV_V','S4FL','ChiAM','Ubq4','H46'
]

OPs = [
'Chignolin',
'Homeodomain',
'Protein-G',
'Trp-cage',
'Villin',
'WW-domain',
'alpha3d',
'bba',
'bbl',
'engrailed',
'gpw',
'lambda-repressor',
'NTL9',
'Protein-B',
'BPTI',
'calmodulin',
'GB3',
'Hen-Egg-White-Lysozyme',
'1soy',
'1wla',
'2ea9',
'5tvz',
'Ubiquitin'
]

# B-spline helper (SciPy, identical to original)
def clamped_bspline_basis_1d(x, x_min, x_max, n_internal_knots,
                             degree=3, intercept=False, omega=False):
    """
    Clamped 1D B-spline basis (SciPy BSpline version).
    Copied from your original implementation to guarantee
    identical design_matrix / basis / omega.
    """
    assert x_min < x_max
    if (np.min(x) < x_min) or (np.max(x) > x_max):
        warnings.warn(f'Input values are clipped to [{x_min}, {x_max}].')
        x = np.clip(x, x_min, x_max)

    M = degree + 1
    left_boundary_knots = np.array([float(x_min)] * M)
    right_boundary_knots = np.array([float(x_max)] * M)
    internal_knots = np.linspace(x_min, x_max, num=n_internal_knots + 2)[1:-1]
    augmented_knots = np.concatenate(
        (left_boundary_knots, internal_knots, right_boundary_knots),
        axis=0
    )

    basis_coeffs = []
    design_matrix = []
    bspl_list = []
    for i in range(M + n_internal_knots):
        c = np.zeros(M + n_internal_knots)
        c[i] = 1.0
        basis_coeffs.append(c)
        f = BSpline(augmented_knots, c, degree, extrapolate=False)
        design_matrix.append(f(x))
        bspl_list.append(f)

    basis_coeffs = np.array(basis_coeffs)
    design_matrix = np.array(design_matrix)

    if not intercept:
        # drop a basis to remove intercept
        basis_coeffs = basis_coeffs[1:]
        design_matrix = design_matrix[1:]
        bspl_list = bspl_list[1:]

    design_matrix = np.moveaxis(design_matrix, 0, -1)  # basis last dim

    if omega:
        _omega = np.zeros((len(bspl_list), len(bspl_list)))
        for i in range(len(bspl_list)):
            for j in range(i, len(bspl_list)):
                d2_i = bspl_list[i].derivative(2)
                d2_j = bspl_list[j].derivative(2)
                _omega[i, j] = quad(
                    lambda a: d2_i(a) * d2_j(a),
                    x_min, x_max, limit=10000
                )[0]
                _omega[j, i] = _omega[i, j]  # symmetric
        assert not np.any(np.isnan(_omega))
    else:
        _omega = None

    return {
        'design_matrix': design_matrix,
        'degree': degree,
        'augmented_knots': augmented_knots,
        'basis_coeffs': basis_coeffs,
        'omega': _omega,
    }


def compute_multi_group_density_spline_basis(
    traj,
    temperature,
    res_group_map=None,
    eta=10.0,
    r0=0.7,
    rho_min=0.0,
    rho_max=15.0,
    n_internal_knots=10,
    intercept=True,
    batch_size=50,
):
    """
    GPU-hybrid density spline basis
    """


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Torch] density spline: using device {device}", flush=True)

    # atom types
    atoms, _ = traj.topology.to_dataframe()
    resnames = atoms['resName'].tolist()
    atom_types = np.array([_amino_acids.index(x) for x in resnames])
    n_atoms = traj.n_atoms
    n_frames = traj.n_frames

    # group mapping
    if res_group_map is None:
        res_group_map = {resname: 0 for resname in _amino_acids}

    group_atom_map = {}
    for _, group_name in res_group_map.items():
        group_atom_map.setdefault(group_name, [])
    for atom_index, atom_type_idx in enumerate(atom_types):
        resname = _amino_acids[atom_type_idx]
        if resname in res_group_map:
            group_name = res_group_map[resname]
            group_atom_map[group_name].append(atom_index)
        else:
            raise ValueError(f"Residue {resname} not found in res_group_map.")

    group_name_list = sorted(list(group_atom_map.keys()))
    n_groups = len(group_name_list)

    # allocate rho (CPU)
    rho = np.zeros((n_frames, n_atoms, n_groups), dtype=np.float64)

    # scalar constants
    cutoff = r0 + 10.0 / eta
    offset_switch_rho = 0.5 * (1.0 + math.tanh(eta * (r0 - cutoff)))

    # coords
    xyz = traj.xyz  # (n_frames, n_atoms, 3)

    # batched distance + rho on GPU
    for start in tqdm(
        range(0, n_frames, batch_size),
        desc="[Torch] density: distance + rho (batched)"
    ):
        end = min(start + batch_size, n_frames)

        coords_batch = torch.from_numpy(xyz[start:end]).to(
            device=device, dtype=torch.float64
        )  # (B, n_atoms, 3)

        dist_batch = torch.cdist(coords_batch, coords_batch)  # (B, Na, Na)

        eta_t = torch.tensor(float(eta), dtype=torch.float64, device=device)
        r0_t = torch.tensor(float(r0), dtype=torch.float64, device=device)
        cutoff_t = torch.tensor(float(cutoff), dtype=torch.float64, device=device)
        offset_t = torch.tensor(float(offset_switch_rho),
                                dtype=torch.float64, device=device)

        rho_3d = 0.5 * (1.0 + torch.tanh(eta_t * (r0_t - dist_batch)))
        rho_3d = rho_3d - offset_t
        rho_3d = torch.where(
            dist_batch > cutoff_t,
            torch.zeros_like(rho_3d),
            rho_3d
        )

        idx = torch.arange(n_atoms, device=device)
        rho_3d[:, idx, idx] = 0.0

        for group_index, group_name in enumerate(group_name_list):
            atom_indices = group_atom_map[group_name]
            if len(atom_indices) == 0:
                continue
            atom_idx_t = torch.tensor(atom_indices,
                                      dtype=torch.long,
                                      device=device)
            rho_group = rho_3d[:, :, atom_idx_t].sum(dim=2)  # (B, n_atoms)
            rho[start:end, :, group_index] = rho_group.cpu().numpy()

        del coords_batch, dist_batch, rho_3d, rho_group
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================
    # B-spline basis on CPU
    # ============================

    degree = 3
    spl_basis_dict = clamped_bspline_basis_1d(
        rho,
        rho_min,
        rho_max,
        n_internal_knots,
        degree,
        intercept=intercept,
        omega=True
    )

    design_matrix = spl_basis_dict['design_matrix']
    augmented_knots = spl_basis_dict['augmented_knots']
    basis_coeffs = spl_basis_dict['basis_coeffs']
    omega = spl_basis_dict['omega']

    n_frames = design_matrix.shape[0]
    d = design_matrix.shape[-1]

    u1_density_spl_basis = np.zeros(
        (n_frames, 20, n_groups, d),
        dtype=np.float64
    )

    RT_value = (GAS_CONST * temperature).value_in_unit(
        unit.kilojoule_per_mole
    )

    for i in range(20):
        flag = (atom_types == i)
        if np.any(flag):
            u1_density_spl_basis[:, i, :, :] = (
                np.sum(design_matrix[:, flag, :, :], axis=1) / RT_value
            )

    del design_matrix
    del rho
    gc.collect()

    output_dict = {
        'eta': eta,
        'r0': r0,
        'rho_min': rho_min,
        'rho_max': rho_max,
        'n_internal_knots': n_internal_knots,
        'degree': degree,
        'u1_density_spl_basis': u1_density_spl_basis,
        'augmented_knots': augmented_knots,
        'basis_coeffs': basis_coeffs,
        'omega': omega,
    }

    return output_dict




def get_custom_nb_exclusions(system):
    """
    Extract exclusions from the first CustomNonbondedForce.
    Returns np.array of shape [n_excl, 2] with sorted (i,j).
    """
    exclusions = []
    custom_nb_forces = [
        f for f in system.getForces()
        if isinstance(f, mm.CustomNonbondedForce)
    ]
    if len(custom_nb_forces) == 0:
        print('No custom nonbonded force found')
        return np.zeros((0, 2), dtype=int)
    force = custom_nb_forces[0]
    n_exclusions = force.getNumExclusions()
    if n_exclusions == 0:
        return np.zeros((0, 2), dtype=int)
    for i in range(n_exclusions):
        p1, p2 = force.getExclusionParticles(i)
        if p1 > p2:
            p1, p2 = p2, p1
        exclusions.append([p1, p2])
    return np.array(exclusions, dtype=int)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FEP features (u1_intercept, u1_ah_basis, "
                    "u1_gauss_basis, u1_density_spl_basis, rg) for one protein."
    )
    parser.add_argument(
        '--protein_dir', required=True,
        help='Directory containing system.xml, parameters.json, *_ca.pdb, output.dcd'
    )
    parser.add_argument(
        '--output_pkl', default=None,
        help='Path to output fep_input.pkl (default: <protein_dir>/fep_input.pkl)'
    )
    # Gaussian shape parameters (in Angstrom)
    parser.add_argument(
        '--gauss_delta_mu_nm', type=float, default=0.25,
        help='Offset beyond LJ minimum for Gaussian center mu (nm), '
             'i.e., mu = r_min + Δμ.'
    )
    parser.add_argument(
        '--gauss_width_nm', type=float, default=0.1,
        help='Gaussian width (standard deviation) in nm.'
    )
    args = parser.parse_args()

    protein_dir = os.path.abspath(args.protein_dir)
    if not os.path.isdir(protein_dir):
        raise FileNotFoundError(f"protein_dir not found: {protein_dir}")

    # -----------------------------
    # Read parameters.json
    # -----------------------------
    param_path = os.path.join(protein_dir, 'input_parameters.json')
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"parameters.json not found in {protein_dir}")
    with open(param_path, 'r') as f:
        params = json.load(f)

    protein = params['protein']
    temperature = float(params['temperature'])      # K
    ionic_strength_mM = float(params['ionic_strength'])  # mM
    res_group_mapping_key = params.get('res_group_mapping', 'default')

    # -----------------------------
    # Load learned parameters θ⁰ from results.pkl (used in MD)
    # -----------------------------
    results_pkl_path = params.get('results_pkl', None)
    if results_pkl_path is None:
        raise ValueError("parameters.json must contain 'results_pkl' used for MD.")

        results_dict = pickle.load(f)

    hydrophobic_scale = np.asarray(results_dict['hydrophobic_scale'])
    spl_values = np.asarray(results_dict['spl_values'])
    eta = results_dict['eta']
    r0 = results_dict['r0']
    rho_min = results_dict['rho_min']
    rho_max = results_dict['rho_max']

    # Gaussian amplitudes
    gauss_coeffs_1d = results_dict.get('gauss_coeffs', None)
    if gauss_coeffs_1d is None:
        gauss_coeffs_1d = np.zeros(210)
    else:
        gauss_coeffs_1d = np.asarray(gauss_coeffs_1d)

    rows20, cols20 = np.triu_indices(20, k=0)
    gauss_height_map = np.zeros((20, 20))
    gauss_height_map[rows20, cols20] = gauss_coeffs_1d
    gauss_height_map[cols20, rows20] = gauss_coeffs_1d


    print(f"Preparing FEP features for {protein}", flush=True)
    print(f"  temperature = {temperature} K", flush=True)
    print(f"  ionic_strength = {ionic_strength_mM} mM", flush=True)
    print(f"  res_group_mapping = {res_group_mapping_key}", flush=True)
    print(f"  gauss_delta_mu_nm = {args.gauss_delta_mu_nm}", flush=True)
    print(f"  gauss_width_nm    = {args.gauss_width_nm}", flush=True)

    delta_mu_nm = args.gauss_delta_mu_nm
    delta_gauss_nm = args.gauss_width_nm

    # -----------------------------
    # Paths inside protein_dir
    # -----------------------------
    system_xml = os.path.join(protein_dir, 'system.xml')
    dcd_path = os.path.join(protein_dir, 'output.dcd')

    if not os.path.exists(system_xml):
        raise FileNotFoundError(f"system.xml not found in {protein_dir}")
    if not os.path.exists(dcd_path):
        raise FileNotFoundError(f"output.dcd not found in {protein_dir}")

    # CA PDB: try "<protein>_ca.pdb" first, otherwise glob "*_ca.pdb"
    ca_pdb_guess = os.path.join(protein_dir, f"{protein}_ca.pdb")
    if os.path.exists(ca_pdb_guess):
        ca_pdb = ca_pdb_guess
    else:
        hits = glob.glob(os.path.join(protein_dir, "*_ca.pdb"))
        if len(hits) == 0:
            raise FileNotFoundError(f"No *_ca.pdb found in {protein_dir}")
        if len(hits) > 1:
            print("WARNING: multiple *_ca.pdb found, using the first one:", hits[0])
        ca_pdb = hits[0]

    print(f"  system_xml = {system_xml}", flush=True)
    print(f"  dcd_path   = {dcd_path}", flush=True)
    print(f"  ca_pdb     = {ca_pdb}", flush=True)

    # -----------------------------
    # Load trajectory
    # -----------------------------
    traj = mdtraj.load_dcd(dcd_path, top=ca_pdb)
    n_frames, n_atoms = traj.n_frames, traj.n_atoms
    print(f"Loaded traj with {n_frames} frames, {n_atoms} atoms", flush=True)

    # -----------------------------
    # Thermo / RT
    # -----------------------------
    T = temperature * unit.kelvin
    RT_value = (GAS_CONST * T).value_in_unit(unit.kilojoule_per_mole)

    # -----------------------------
    # Load unbiased system.xml
    # -----------------------------
    with open(system_xml, 'r') as f:
        unbiased_system = mm.XmlSerializer.deserialize(f.read())

    # ─────────────────────────────
    # Bonded terms: IDP vs OP/MDP
    # ─────────────────────────────
    if (protein in OPs) or (protein in MDPs):
        # OP + MDP: groups 1–4
        u1_bond = compute_PE(traj, unbiased_system,
                             platform_name='CUDA',
                             groups={1}) / RT_value
        u1_angle = compute_PE(traj, unbiased_system,
                              platform_name='CUDA',
                              groups={2}) / RT_value
        u1_dihedral = compute_PE(traj, unbiased_system,
                                 platform_name='CUDA',
                                 groups={3}) / RT_value
        u1_native_pair = compute_PE(traj, unbiased_system,
                                    platform_name='CUDA',
                                    groups={4}) / RT_value
        u1_all_bonded = u1_bond + u1_angle + u1_dihedral + u1_native_pair
    else:
        # IDP: only bond (group 1); no angle/dihedral/native
        u1_bond = compute_PE(traj, unbiased_system,
                             platform_name='CUDA',
                             groups={1}) / RT_value
        u1_angle = np.zeros_like(u1_bond)
        u1_dihedral = np.zeros_like(u1_bond)
        u1_native_pair = np.zeros_like(u1_bond)
        u1_all_bonded = u1_bond

    # Exclusions from the system
    exclusions = get_custom_nb_exclusions(unbiased_system)
    df_exclusions = pd.DataFrame(exclusions, columns=['a1', 'a2'])

    # -----------------------------
    # Electrostatics via HPSModel
    # -----------------------------
    dielectric = 80.0
    ionic_strength = ionic_strength_mM * unit.millimolar

    hps_model = HPSModel()
    hps_model.append_mol(HPSParser(ca_pdb))
    top = app.PDBFile(ca_pdb).getTopology()
    hps_model.create_system(top=top, box_a=1000, box_b=1000, box_c=1000)

    assert traj.n_chains == 1
    charge = hps_model.atoms['charge'].to_numpy()
    charge[0] += 1
    charge[-1] -= 1
    hps_model.atoms['charge'] = charge

    ldby = (kB * T * VEP * dielectric /
            (2 * NA * ionic_strength * EC**2))**0.5
    elec_cutoff = 5.0 * ldby
    hps_model.exclusions = df_exclusions
    hps_model.add_dh_elec(ldby, dielectric, elec_cutoff, force_group=1)

    u1_elec = compute_PE(traj, hps_model.system,
                         platform_name='CUDA', groups={1}) / RT_value

    # -----------------------------
    # Distance matrices and pair types
    # -----------------------------
    #distance_matrices = compute_distance_matrices_torch(traj)
    #assert distance_matrices.shape == (n_frames, n_atoms, n_atoms)

    df_atoms = hps_model.atoms.copy()
    resnames = df_atoms['resname'].tolist()
    atom_types = np.array([_amino_acids.index(x) for x in resnames])

    rows_aa, cols_aa = np.triu_indices(20, k=0)
    Y = np.zeros((20, 20), dtype=int)
    Y[rows_aa, cols_aa] = np.arange(210)
    Y[cols_aa, rows_aa] = np.arange(210)

    pair_type_matrix = np.zeros((n_atoms, n_atoms), dtype=int)
    for i in range(n_atoms):
        for j in range(i, n_atoms):
            pair_type_matrix[i, j] = Y[atom_types[i], atom_types[j]]
            pair_type_matrix[j, i] = pair_type_matrix[i, j]

    # apply exclusions
    if exclusions.size > 0:
        pair_type_matrix[exclusions[:, 0], exclusions[:, 1]] = -1
        pair_type_matrix[exclusions[:, 1], exclusions[:, 0]] = -1
    rows_lt, cols_lt = np.tril_indices(n_atoms, k=0)
    pair_type_matrix[rows_lt, cols_lt] = -1

    # -----------------------------
    # HPS sigma and epsilon (Urry)
    # -----------------------------
    sigma = np.zeros((20, 20))
    df_hps_urry_param = pd.read_csv(p_path + '/parameters/HPS_Urry_parameters.csv')
    for _, row in df_hps_urry_param.iterrows():
        a1 = _amino_acids.index(row['atom_type1'])
        a2 = _amino_acids.index(row['atom_type2'])
        sigma[a1, a2] = row['sigma']
        sigma[a2, a1] = row['sigma']

    sigma_1d = sigma[rows_aa, cols_aa]
    epsilon = 0.2 * _kcal_to_kj

    # -------------------------------------------
    # GPU-safe STREAMING LJ + AH + Gaussian basis
    # -------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for LJ/AH/Gauss: {device}", flush=True)

    sigma_1d_t = torch.from_numpy(sigma_1d).to(device)
    epsilon_t = torch.tensor(float(epsilon), device=device)
    RT_t = torch.tensor(float(RT_value), device=device)

    pair_type_t = torch.from_numpy(pair_type_matrix).to(device)

    u1_lj_excl = np.zeros((n_frames, 210), dtype=np.float64)
    u1_ah_basis = np.zeros((n_frames, 210), dtype=np.float64)
    u1_gauss_basis = np.zeros((n_frames, 210), dtype=np.float64)

    batch_size = 50
    c_2_16 = 2.0 ** (1.0 / 6.0)

    delta_gauss_t = torch.tensor(float(delta_gauss_nm), device=device)

    xyz = traj.xyz  # (n_frames, n_atoms, 3) numpy

    delta_mu_t = torch.tensor(float(args.gauss_delta_mu_nm), device=device, dtype=torch.float64)
    delta_gauss_t = torch.tensor(float(delta_gauss_nm), device=device, dtype=torch.float64)

    for start in tqdm(
        range(0, n_frames, batch_size),
        desc="Streaming LJ/AH/Gaussian (Torch)"
    ):
        end = min(start + batch_size, n_frames)

        coords_t = torch.from_numpy(xyz[start:end]).to(device=device, dtype=torch.float64)
        dist_batch = torch.cdist(coords_t, coords_t)  # (B, Na, Na)

        # --- critical safety: kill diagonal singularities ---
        idx = torch.arange(n_atoms, device=device)
        dist_batch[:, idx, idx] = torch.inf

        for k in range(210):
            mask_k = (pair_type_t == k)
            if not mask_k.any():
                continue

            r = dist_batch[:, mask_k]  # (B, n_pairs_k)
   
            sig_k = sigma_1d_t[k]
            r1 = c_2_16 * sig_k
            ah_cutoff = 4.0 * sig_k

            lj_at_cutoff = 4.0 * epsilon_t * (
                (sig_k / ah_cutoff) ** 12 - (sig_k / ah_cutoff) ** 6
            )

            s1 = (r < r1)
            s2 = (r >= r1) & (r < ah_cutoff)

            # Avoid any division by ~0 even beyond diagonal masking
            # (optional but very safe)
            r_safe = torch.clamp(r, min=1e-6)
    
            lj = 4.0 * epsilon_t * ((sig_k / r_safe) ** 12 - (sig_k / r_safe) ** 6)

            # excluded LJ
            u1_lj_excl[start:end, k] = (
                torch.sum((lj + epsilon_t) * s1, dim=1) / RT_t
            ).detach().cpu().numpy()

            # AH basis
            ah_base = (-epsilon_t - lj_at_cutoff) * s1 + (lj - lj_at_cutoff) * s2
            u1_ah_basis[start:end, k] = (
                torch.sum(ah_base, dim=1) / RT_t
            ).detach().cpu().numpy()

            # Gaussian basis (height-linear)
            mu_k_t = r1 + delta_mu_t
            gauss = torch.exp(-0.5 * ((r_safe - mu_k_t) / delta_gauss_t) ** 2) * s2
            u1_gauss_basis[start:end, k] = (
                torch.sum(gauss, dim=1) / RT_t
            ).detach().cpu().numpy()

            # delete per-k tensors (safe)
            del r, r_safe, s1, s2, lj, ah_base, gauss, mu_k_t

        # delete per-batch tensors
        del coords_t, dist_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



    # Sum excluded LJ over all pair types
    u1_lj_excl = u1_lj_excl.sum(axis=1)

    print("LJ/AH/Gaussian done. Proceeding to density basis...", flush=True)



    # -----------------------------
    # Density spline basis (GPU-hybrid)
    # -----------------------------
    res_group_mapping = _res_group_mappings[res_group_mapping_key]
    density_spl_basis_dict = compute_multi_group_density_spline_basis(
        traj, T, res_group_mapping,
        eta=eta,
        r0=r0,
        rho_min=rho_min,
        rho_max=rho_max,
        n_internal_knots=10,
#        n_internal_knots=spl_values.shape[-1] - 1,
        intercept=True,
        batch_size=50
    )

    u1_density_spl_basis = density_spl_basis_dict['u1_density_spl_basis']
    degree = density_spl_basis_dict['degree']
    augmented_knots = density_spl_basis_dict['augmented_knots']
    basis_coeffs = density_spl_basis_dict['basis_coeffs']
    omega = density_spl_basis_dict['omega']

    # -----------------------------
    # Rg
    # -----------------------------
    rg = compute_CA_traj_radius_of_gyration(traj)

    # -----------------------------
    # Intercept = bonded + elec + LJ_excl
    # -----------------------------
    u1_intercept = u1_all_bonded + u1_elec + u1_lj_excl

    def _check(name, arr):
        arr = np.asarray(arr)
        nbad = np.sum(~np.isfinite(arr))
        if nbad > 0:
            bad = np.where(~np.isfinite(arr))[0][:10]
            raise RuntimeError(f"{name} has {nbad} NaN/Inf. First bad frames: {bad}")

    _check("u1_lj_excl", u1_lj_excl)
    _check("u1_ah_basis", u1_ah_basis)
    _check("u1_gauss_basis", u1_gauss_basis)
    _check("u1_intercept", u1_intercept)


    # -----------------------------
    # Build output dict
    # -----------------------------
    output_dict = {}
    output_dict['protein'] = protein
    output_dict['temperature'] = temperature
    output_dict['ionic_strength_mM'] = ionic_strength_mM
    output_dict['dielectric'] = dielectric
    output_dict['ldby'] = ldby.value_in_unit(unit.nanometer)
    output_dict['elec_cutoff'] = elec_cutoff.value_in_unit(unit.nanometer)



    output_dict['atom_types'] = atom_types
    output_dict['exclusions'] = exclusions
    output_dict['sigma'] = sigma
    output_dict['epsilon'] = epsilon

    output_dict['u1_bond'] = u1_bond
    output_dict['u1_angle'] = u1_angle
    output_dict['u1_dihedral'] = u1_dihedral
    output_dict['u1_native_pair'] = u1_native_pair
    output_dict['u1_all_bonded'] = u1_all_bonded
    output_dict['u1_lj_excl'] = u1_lj_excl
    output_dict['u1_elec'] = u1_elec


    output_dict['u1_intercept'] = u1_intercept
    output_dict['u1_ah_basis'] = u1_ah_basis
    output_dict['u1_gauss_basis'] = u1_gauss_basis  # NEW
    output_dict['u1_density_spl_basis'] = u1_density_spl_basis
    output_dict['degree'] = degree
    output_dict['augmented_knots'] = augmented_knots
    output_dict['basis_coeffs'] = basis_coeffs
    output_dict['omega'] = omega

    output_dict['res_group_mapping'] = res_group_mapping_key
    output_dict['rg'] = rg

    # also record Gaussian shape params used
    output_dict['gauss_delta_mu_nm'] = args.gauss_delta_mu_nm
    output_dict['gauss_width_nm'] = args.gauss_width_nm
    # -----------------------------
    # Store baseline θ⁰ (exact MD force field)
    # -----------------------------
    output_dict['hydrophobic_scale'] = hydrophobic_scale
    output_dict['gauss_height_map'] = gauss_height_map
    output_dict['spl_values'] = spl_values
    output_dict['eta'] = eta
    output_dict['r0'] = r0
    output_dict['rho_min'] = rho_min
    output_dict['rho_max'] = rho_max

    output_dict['results_pkl_used'] = results_pkl_path


    # -----------------------------
    # Save fep_input.pkl
    # -----------------------------
    if args.output_pkl is None:
        output_pkl = os.path.join(protein_dir, 'fep_AH_input.pkl')
    else:
        output_pkl = args.output_pkl

    outdir = os.path.dirname(output_pkl)
    if outdir != '':
        os.makedirs(outdir, exist_ok=True)

    with open(output_pkl, 'wb') as f:
        pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved FEP input to {output_pkl}", flush=True)


if __name__ == '__main__':
    main()

