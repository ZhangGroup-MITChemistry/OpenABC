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
import mdtraj
import math
from openabc.forcefields.MOFF2.lib import _amino_acids, GAS_CONST
from openabc.forcefields.MOFF2.utils import compute_distance_matrices, dual_boundary_tanh_step, clamped_bspline_basis_1d

"""
Compute some basis as numpy arrays so that the energy can be expressed as a function of the basis. 
"""


def compute_direct_water_burial_basis(traj, temperature, exclusions, eta_direct=10.0, r1_min=0.0, r1_max=0.7, 
                                      eta_water=10.0, r2_min=0.7, r2_max=1.1, rho0=4.6, rho_use_exclusions=False, 
                                      rho_lim=np.array([[2, 5], [5, 8], [8, 11]])):
    """
    Compute the direct, water-mediated, and burial basis. The output basis is in reduced energy unit. 
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory.
    
    temperature : unit.Quantity
        The temperature.
    
    exclusions : None or 2d array_like or pd.DataFrame
        Nonbonded exclusions. 
        If None, no exclusions. 
        If 2d array_like, shape = (n_exclusions, 2). 
        If pd.DataFrame, excluded pairs are specificed by columns 'a1' and 'a2'.
    
    eta_direct : float or int
        Direct contact eta_direct parameter in unit 1 / nm.
    
    r1_min : float or int
        Direct contact range lower bound in unit nm.
    
    r1_max : float or int
        Direct contact range upper bound in unit nm.
    
    eta_water : float or int
        Water-mediated contact eta_water parameter in unit 1 / nm.
    
    r2_min : float or int
        Water-mediated contact range lower bound in unit nm.
    
    r2_max : float or int
        Water-mediated contact range upper bound in unit nm.
    
    rho0 : float or int
        Water-mediated contact density threshold value.
    
    rho_use_exclusions : bool
        Whether to use exclusions when computing rho.
        If rho_use_exclusions is False, normally you may want to increase rho0 and rho_lim correspondingly.
    
    rho_lim : 2d array-like, shape = (n_rho_ranges, 2)
        rho value ranges when computing burial potential.
        rho[i] is 1d array-like of shape (2,). 
        rho[i][0] is the lower bound of the i-th rho range.
        rho[i][1] is the upper bound of the i-th rho range.
    
    Returns
    -------
    output_dict : dict
        Dictionary including direct, water-mediated, and burial basis.
        output_dict['u1_direct_basis'] : np.ndarray, shape = (n_frames, 210)
            The direct contact basis.
        output_dict['u1_water_basis'] : np.ndarray, shape = (n_frames, 210)
            The water-mediated contact basis.
        output_dict['u1_protein_basis'] : np.ndarray, shape = (n_frames, 210)
            The protein-mediated contact basis.
        output_dict['u1_burial_basis'] : np.ndarray, shape = (n_frames, n_rho_ranges, 20)
            The burial basis.
    
    """
    # get atom types and pair types
    atoms, _ = traj.topology.to_dataframe()
    resnames = atoms['resName'].tolist()
    atom_types = np.array([_amino_acids.index(x) for x in resnames])
    n_atoms = traj.n_atoms
    rows, cols = np.triu_indices(20, k=0)
    pair_type_2d_map = np.zeros((20, 20))
    pair_type_2d_map[rows, cols] = np.arange(210)
    pair_type_2d_map[cols, rows] = np.arange(210)
    pair_type_matrix = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i, n_atoms):
            pair_type_matrix[i, j] = pair_type_2d_map[atom_types[i], atom_types[j]]
            pair_type_matrix[j, i] = pair_type_matrix[i, j]
    
    # set exclusions
    if exclusions is not None:
        if isinstance(exclusions, pd.DataFrame):
            exclusions = exclusions[['a1', 'a2']].to_numpy()
        if not isinstance(exclusions, np.ndarray):
            exclusions = np.array(exclusions)
        pair_type_matrix[exclusions[:, 0], exclusions[:, 1]] = -1
        pair_type_matrix[exclusions[:, 1], exclusions[:, 0]] = -1
    
    # exclude diagonal pairs and lower triangular pairs to avoid redundant interactions
    rows, cols = np.tril_indices(n_atoms, k=0)
    pair_type_matrix[rows, cols] = -1
    
    # compute distance matrices
    distance_matrices = compute_distance_matrices(traj, use_pbc=False) # do not use PBC
    
    # compute direct contact basis
    # the contact potential is np.dot(u1_direct_basis, gamma_direct)
    # gamma_direct > 0 means attraction
    cutoff_direct = r1_max + 10 / eta_direct
    offset_direct = dual_boundary_tanh_step(np.array([cutoff_direct]), r1_min, r1_max, eta_direct)[0]
    n_frames = traj.n_frames
    u1_direct_basis = np.zeros((n_frames, 210))
    RT_value = (GAS_CONST * temperature).value_in_unit(unit.kilojoule_per_mole)
    for i in range(210):
        flag = pair_type_matrix == i
        if np.any(flag):
            distances_i = distance_matrices[:, flag]
            switch_i = dual_boundary_tanh_step(distances_i, r1_min, r1_max, eta_direct)
            switch_i -= offset_direct
            switch_i[distances_i > cutoff_direct] = 0.0
            u1_direct_basis[:, i] = -1 * np.sum(switch_i, axis=1) / RT_value
    
    # compute rho
    rho_3d = dual_boundary_tanh_step(distance_matrices, r1_min, r1_max, eta_direct)
    rho_3d -= offset_direct
    rho_3d[distance_matrices > cutoff_direct] = 0.0
    rho_3d[:, np.arange(n_atoms), np.arange(n_atoms)] = 0.0 # remove the contribution by the atom itself
    if rho_use_exclusions and (exclusions is not None):
        rho_3d[:, exclusions[:, 0], exclusions[:, 1]] = 0.0
        rho_3d[:, exclusions[:, 1], exclusions[:, 0]] = 0.0
    rho = np.sum(rho_3d, axis=2) # rho.shape = (n_frames, n_atoms)
    
    # compute water-mediated basis
    # the water-mediated potential is np.dot(u1_water_basis, gamma_water) + np.dot(u1_protein_basis, gamma_protein)
    # positive gamma_water and gamma_protein mean attraction
    cutoff_water = r2_max + 10 / eta_water
    offset_water = dual_boundary_tanh_step(np.array([cutoff_water]), r2_min, r2_max, eta_water)[0]
    eta_rho = 7.0
    u1_water_basis = np.zeros((n_frames, 210))
    u1_protein_basis = np.zeros((n_frames, 210))
    nu_water_3d = (1 + np.tanh(eta_rho * (rho0 - rho[:, :, None]))) * (1 + np.tanh(eta_rho * (rho0 - rho[:, None, :]))) / 4
    for i in range(210):
        flag = pair_type_matrix == i
        if np.any(flag):
            distances_i = distance_matrices[:, flag]
            nu_water = nu_water_3d[:, flag]
            nu_protein = 1 - nu_water
            switch_i = dual_boundary_tanh_step(distances_i, r2_min, r2_max, eta_water)
            switch_i -= offset_water
            switch_i[distances_i > cutoff_water] = 0.0
            u1_water_basis[:, i] = -1 * np.sum(nu_water * switch_i, axis=1) / RT_value
            u1_protein_basis[:, i] = -1 * np.sum(nu_protein * switch_i, axis=1) / RT_value
    
    # compute burial basis
    # the burial potential is np.dot(u1_burial_basis.reshape(n_frames, -1), gamma_burial)
    # positive gamma_burial means burial is favorable
    eta_burial = 4.0
    n_rho_ranges = len(rho_lim)
    u1_burial_basis = np.zeros((n_frames, n_rho_ranges, 20))
    for i in range(n_rho_ranges):
        rho_min = rho_lim[i][0]
        rho_max = rho_lim[i][1]
        assert rho_min <= rho_max
        for j in range(20):
            flag = atom_types == j
            if np.any(flag):
                rho_j = rho[:, flag] # rho_j.shape = (n_frames, n_selected_atoms)
                switch_j = (np.tanh(eta_burial * (rho_j - rho_min)) + np.tanh(eta_burial * (rho_max - rho_j))) / 2
                u1_burial_basis[:, i, j] = -1 * np.sum(switch_j, axis=1) / RT_value
    
    # set output
    output_dict = {'u1_direct_basis': u1_direct_basis, 
                   'u1_water_basis': u1_water_basis, 
                   'u1_protein_basis': u1_protein_basis,
                   'u1_burial_basis': u1_burial_basis}
    return output_dict
    
    
def compute_density_switch_ashbaugh_hatch_basis(traj, temperature, exclusions, epsilon, sigma_ah,
                                                eta=10, r0=0.7, mu=2.0, rho0=5.5):
    """
    Compute the density based switch hydrophobic scale Ashbaugh-Hatch basis. 
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory.
    
    temperature : unit.Quantity
        The temperature.
    
    exclusions : None or 2d array_like or pd.DataFrame
        Nonbonded exclusions. 
        If None, no exclusions. 
        If 2d array_like, shape = (n_exclusions, 2). 
        If pd.DataFrame, excluded pairs are specificed by columns 'a1' and 'a2'.
    
    epsilon : float or int
        Parameter epsilon in unit kJ/mol.
    
    sigma_ah : array-like, shape = (20, 20)
        The Ashbaugh-Hatch sigma matrix.
    
    eta : float or int
        Parameter eta in unit 1 / nm.
    
    r0 : float or int
        Parameter r0 in unit nm. 
    
    mu : float or int
        Parameter mu. 
    
    rho0 : float or int
        Parameter rho0.
    
    Returns
    -------
    output_dict : dict
        Dictionary including direct, water-mediated, and burial basis.
        output_dict['u1_lj_excl'] : np.ndarray, shape = (n_frames,)
            The LJ excluded volume potential.
        output_dict['u1_dilute_ah_basis'] : np.ndarray, shape = (n_frames, 210)
            The Ashbaugh-Hatch basis of the dilute regime.
        output_dict['u1_dense_ah_basis'] : np.ndarray, shape = (n_frames, 210)
            The Ashbaugh-Hatch basis of the dense regime.
    
    """
    # get atom types and pair types
    atoms, _ = traj.topology.to_dataframe()
    resnames = atoms['resName'].tolist()
    atom_types = np.array([_amino_acids.index(x) for x in resnames])
    n_atoms = traj.n_atoms
    rows, cols = np.triu_indices(20, k=0)
    pair_type_2d_map = np.zeros((20, 20))
    pair_type_2d_map[rows, cols] = np.arange(210)
    pair_type_2d_map[cols, rows] = np.arange(210)
    pair_type_matrix = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i, n_atoms):
            pair_type_matrix[i, j] = pair_type_2d_map[atom_types[i], atom_types[j]]
            pair_type_matrix[j, i] = pair_type_matrix[i, j]
    
    # set exclusions
    if exclusions is not None:
        if isinstance(exclusions, pd.DataFrame):
            exclusions = exclusions[['a1', 'a2']].to_numpy()
        if not isinstance(exclusions, np.ndarray):
            exclusions = np.array(exclusions)
        pair_type_matrix[exclusions[:, 0], exclusions[:, 1]] = -1
        pair_type_matrix[exclusions[:, 1], exclusions[:, 0]] = -1
    
    # exclude diagonal pairs and lower triangular pairs to avoid redundant interactions
    rows, cols = np.tril_indices(n_atoms, k=0)
    pair_type_matrix[rows, cols] = -1
    
    # compute distance matrices
    distance_matrices = compute_distance_matrices(traj, use_pbc=False) # do not use PBC
    
    # compute rho
    cutoff_rho = r0 + 10 / eta # cutoff distance for computing rho
    offset_switch_rho = 0.5 * (1 + math.tanh(eta * (r0 - cutoff_rho)))
    rho_3d = 0.5 * (1 + np.tanh(eta * (r0 - distance_matrices)))
    rho_3d -= offset_switch_rho
    rho_3d[distance_matrices > cutoff_rho] = 0.0
    rho_3d[:, np.arange(n_atoms), np.arange(n_atoms)] = 0.0 # remove the contribution by the atom itself
    rho = np.sum(rho_3d, axis=2) # rho.shape = (n_frames, n_atoms)
    
    # compute nu
    nu = (1 + np.tanh(mu * (rho0 - rho[:, :, None]))) * (1 + np.tanh(mu * (rho0 - rho[:, None, :]))) / 4
    
    # compute LJ excluded volume potential and AH basis
    n_frames = traj.n_frames
    u1_lj_excl = np.zeros((n_frames, 210))
    u1_dilute_ah_basis = np.zeros((n_frames, 210))
    u1_dense_ah_basis = np.zeros((n_frames, 210))
    rows, cols = np.triu_indices(20, k=0)
    sigma_1d = sigma_ah[rows, cols]
    RT_value = (GAS_CONST * temperature).value_in_unit(unit.kilojoule_per_mole)
    for i in range(210):
        flag = pair_type_matrix == i
        if np.any(flag):
            r = distance_matrices[:, flag] # r.shape = (n_frames, n_selected_pairs)
            r1 = 2**(1 / 6) * sigma_1d[i] # switch distance
            ah_cutoff = 4 * sigma_1d[i]
            lj_at_cutoff = 4 * epsilon * ((sigma_1d[i] / ah_cutoff)**12 - (sigma_1d[i] / ah_cutoff)**6)
            lj = 4 * epsilon * ((sigma_1d[i] / r)**12 - (sigma_1d[i] / r)**6) # shape = (n_frames, n_selected_pairs)
            s1 = np.heaviside(r1 - r, 0)
            s2 = np.heaviside(ah_cutoff - r, 0) * np.heaviside(r - r1, 0)
            u1_lj_excl[:, i] = np.sum((lj + epsilon) * s1, axis=1) / RT_value
            selected_nu = nu[:, flag] # shape = (n_frames, n_selected_pairs)
            u1_ah_basis_i = (-epsilon - lj_at_cutoff) * s1 + (lj - lj_at_cutoff) * s2 # shape = (n_frames, n_selected_pairs)
            u1_dilute_ah_basis[:, i] = np.sum(u1_ah_basis_i * selected_nu, axis=1) / RT_value
            u1_dense_ah_basis[:, i] = np.sum(u1_ah_basis_i * (1 - selected_nu), axis=1) / RT_value
    u1_lj_excl = np.sum(u1_lj_excl, axis=1)
    output_dict = {'u1_lj_excl': u1_lj_excl, 
                   'u1_dilute_ah_basis': u1_dilute_ah_basis, 
                   'u1_dense_ah_basis': u1_dense_ah_basis}
    return output_dict


def compute_density_switch_ashbaugh_hatch_gauss_basis(
    traj,
    temperature,
    exclusions,
    epsilon,
    sigma_ah,
    eta=10.0,
    r0=0.7,
    mu=2.0,
    rho0=5.5,
    gauss_delta_mu=0.25,
    gauss_width=0.10):
    """
    Compute the density-based Ashbaugh–Hatch basis AND an extra Gaussian
    basis per pair type. All outputs are in reduced energy units (kBT).

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory.

    temperature : unit.Quantity
        The temperature.

    exclusions : None or 2d array_like or pd.DataFrame
        Nonbonded exclusions. If None, no exclusions.

    epsilon : float
        LJ epsilon (kJ/mol).

    sigma_ah : 2d array-like, shape = (20, 20)
        Sigma for AH/LJ (nm).

    eta, r0, mu, rho0 : float
        Same as in compute_density_switch_ashbaugh_hatch_basis.

    gauss_delta_mu : float
        Offset beyond LJ minimum r_min (nm) for Gaussian center:
        mu_k = r_min_k + gauss_delta_mu.

    gauss_width : float
        Gaussian width (standard deviation, nm).

    Returns
    -------
    output_dict : dict
        Keys:
          'u1_lj_excl'          : (n_frames,)
          'u1_dilute_ah_basis'  : (n_frames, 210)
          'u1_dense_ah_basis'   : (n_frames, 210)
          'u1_gauss_basis'      : (n_frames, 210)  # NEW
          'gauss_delta_mu'      : float (nm)
          'gauss_width'         : float (nm)
    """
    # --- atom & pair-type bookkeeping (copy of AH function logic) ---
    atoms, _ = traj.topology.to_dataframe()
    resnames = atoms['resName'].tolist()
    atom_types = np.array([_amino_acids.index(x) for x in resnames])
    n_atoms = traj.n_atoms

    rows, cols = np.triu_indices(20, k=0)
    pair_type_2d_map = np.zeros((20, 20))
    pair_type_2d_map[rows, cols] = np.arange(210)
    pair_type_2d_map[cols, rows] = np.arange(210)

    pair_type_matrix = np.zeros((n_atoms, n_atoms), dtype=int)
    for i in range(n_atoms):
        for j in range(i, n_atoms):
            pt = pair_type_2d_map[atom_types[i], atom_types[j]]
            pair_type_matrix[i, j] = pt
            pair_type_matrix[j, i] = pt

    # apply exclusions
    if exclusions is not None:
        if isinstance(exclusions, pd.DataFrame):
            exclusions = exclusions[['a1', 'a2']].to_numpy()
        if not isinstance(exclusions, np.ndarray):
            exclusions = np.array(exclusions)
        pair_type_matrix[exclusions[:, 0], exclusions[:, 1]] = -1
        pair_type_matrix[exclusions[:, 1], exclusions[:, 0]] = -1

    # remove diagonal/lower-triangular (no double counting)
    tril_rows, tril_cols = np.tril_indices(n_atoms, k=0)
    pair_type_matrix[tril_rows, tril_cols] = -1

    # --- distances ---
    distance_matrices = compute_distance_matrices(traj, use_pbc=False)
    n_frames = traj.n_frames

    # --- density rho, same as in AH function ---
    cutoff_rho = r0 + 10.0 / eta
    offset_switch_rho = 0.5 * (1.0 + math.tanh(eta * (r0 - cutoff_rho)))
    rho_3d = 0.5 * (1.0 + np.tanh(eta * (r0 - distance_matrices)))
    rho_3d -= offset_switch_rho
    rho_3d[distance_matrices > cutoff_rho] = 0.0
    rho_3d[:, np.arange(n_atoms), np.arange(n_atoms)] = 0.0
    rho = np.sum(rho_3d, axis=2)

    # nu(rho_i, rho_j)
    nu = (
        (1.0 + np.tanh(mu * (rho0 - rho[:, :, None])))
        * (1.0 + np.tanh(mu * (rho0 - rho[:, None, :])))
        / 4.0
    )

    # --- LJ + AH parts (same structure as existing function) ---
    u1_lj_excl = np.zeros((n_frames, 210))
    u1_dilute_ah_basis = np.zeros((n_frames, 210))
    u1_dense_ah_basis = np.zeros((n_frames, 210))
    # NEW: Gaussian basis
    u1_gauss_basis = np.zeros((n_frames, 210))

    rows, cols = np.triu_indices(20, k=0)
    sigma_1d = sigma_ah[rows, cols]

    RT_value = (GAS_CONST * temperature).value_in_unit(
        unit.kilojoule_per_mole
    )

    for k in range(210):
        flag = (pair_type_matrix == k)
        if not np.any(flag):
            continue

        r = distance_matrices[:, flag]  # (n_frames, n_pairs_k)
        # AH geometry
        r1 = 2.0 ** (1.0 / 6.0) * sigma_1d[k]   # LJ minimum
        ah_cutoff = 4.0 * sigma_1d[k]          # AH cutoff

        lj_at_cutoff = 4.0 * epsilon * (
            (sigma_1d[k] / ah_cutoff) ** 12
            - (sigma_1d[k] / ah_cutoff) ** 6
        )
        lj = 4.0 * epsilon * (
            (sigma_1d[k] / r) ** 12
            - (sigma_1d[k] / r) ** 6
        )

        s1 = (r <= r1).astype(float)
        s2 = ((r > r1) & (r <= ah_cutoff)).astype(float)

        # LJ excluded volume (same as original)
        u1_lj_excl[:, k] = np.sum((lj + epsilon) * s1, axis=1) / RT_value

        # AH basis per unit hydrophobic scale (same as original)
        selected_nu = nu[:, flag]
        ah_base = (-epsilon - lj_at_cutoff) * s1 + (lj - lj_at_cutoff) * s2

        u1_dilute_ah_basis[:, k] = np.sum(
            ah_base * selected_nu, axis=1
        ) / RT_value
        u1_dense_ah_basis[:, k] = np.sum(
            ah_base * (1.0 - selected_nu), axis=1
        ) / RT_value

        # --- NEW: Gaussian basis per unit amplitude ---
        # mu_k = r_min_k + gauss_delta_mu
        mu_k = r1 + gauss_delta_mu
        gauss = np.exp(
            -0.5 * ((r - mu_k) / gauss_width) ** 2
        ) * s2  # confined to AH region

        u1_gauss_basis[:, k] = np.sum(gauss, axis=1) / RT_value

    # sum LJ over all pair types -> one vector
    u1_lj_excl = np.sum(u1_lj_excl, axis=1)

    return {
        'u1_lj_excl': u1_lj_excl,
        'u1_dilute_ah_basis': u1_dilute_ah_basis,
        'u1_dense_ah_basis': u1_dense_ah_basis,
        'u1_gauss_basis': u1_gauss_basis,
        'gauss_delta_mu': gauss_delta_mu,
        'gauss_width': gauss_width,
    }





def compute_density_spline_basis(traj, temperature, eta=10.0, r0=0.7, rho_min=0.0, 
                                 rho_max=15.0, n_internal_knots=10, intercept=True):
    """
    Compute the basis of the local density term as a spline function with degree = 3.
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory.
    
    temperature : unit.Quantity
        The temperature.
    
    eta : float or int
        Parameter eta in unit 1 / nm.
    
    r0 : float or int
        Parameter r0 in unit nm.
    
    rho_min : float or int
        Parameter rho_min.
    
    rho_max : float or int
        Parameter rho_max.
    
    n_internal_knots : int
        The number of internal knots.
    
    intercept : bool
        Whether to include the intercept in the basis.
    
    Returns
    -------
    output_dict : dict
        Output dictionary.
    
    Here are the details of output_dict: 
    
    output_dict['eta'] : float or int
        Parameter eta.
    
    output_dict['r0'] : float or int
        Parameter r0.
    
    output_dict['rho_min'] : float or int
        Parameter rho_min.
    
    output_dict['rho_max'] : float or int
        Parameter rho_max.
    
    output_dict['n_internal_knots'] : int
        The number of internal knots.
    
    output_dict['degree'] : int
        The degree of the B-spline basis.
    
    output_dict['u1_density_spl_basis'] : np.ndarray, shape = (n_frames, 20, d)
        The local density B-spline basis.
        d is the number of bases.
    
    output_dict['augmented_knots'] : np.ndarray, shape = (n_internal_knots + 2 * degree + 2,)
        The augmented knots.
    
    output_dict['basis_coeffs'] : np.ndarray, shape = (d, degree + n_internal_knots + 1)
        The basis coefficients.
    
    output_dict['omega'] : np.ndarray, shape = (d, d)
        The omega matrix.
    
    """
    # get atom types and pair types
    atoms, _ = traj.topology.to_dataframe()
    resnames = atoms['resName'].tolist()
    atom_types = np.array([_amino_acids.index(x) for x in resnames])
    n_atoms = traj.n_atoms
    
    # compute distance matrices
    distance_matrices = compute_distance_matrices(traj, use_pbc=False) # do not use PBC
    
    # compute rho
    cutoff = r0 + 10 / eta # cutoff distance for computing rho
    offset_switch_rho = 0.5 * (1 + math.tanh(eta * (r0 - cutoff)))
    rho_3d = 0.5 * (1 + np.tanh(eta * (r0 - distance_matrices)))
    rho_3d -= offset_switch_rho
    rho_3d[distance_matrices > cutoff] = 0.0
    rho_3d[:, np.arange(n_atoms), np.arange(n_atoms)] = 0.0 # remove the contribution by the atom itself
    rho = np.sum(rho_3d, axis=2) # rho.shape = (n_frames, n_atoms)
    
    # compute B-spline basis
    degree = 3
    spl_basis_dict = clamped_bspline_basis_1d(rho, rho_min, rho_max, n_internal_knots, 
                                              degree, intercept=intercept, omega=True)
    design_matrix = spl_basis_dict['design_matrix']
    augmented_knots = spl_basis_dict['augmented_knots']
    basis_coeffs = spl_basis_dict['basis_coeffs']
    omega = spl_basis_dict['omega']
    
    # rearrange the basis
    assert design_matrix.ndim == 3
    n_frames = design_matrix.shape[0]
    assert n_frames == traj.n_frames
    u1_density_spl_basis = np.zeros((n_frames, 20, design_matrix.shape[-1]))
    RT_value = (GAS_CONST * temperature).value_in_unit(unit.kilojoule_per_mole)
    for i in range(20):
        flag = atom_types == i
        if np.any(flag):
            u1_density_spl_basis[:, i, :] = np.sum(design_matrix[:, flag, :], axis=1) / RT_value
    output_dict = {'eta': eta,
                   'r0': r0, 
                   'rho_min': rho_min, 
                   'rho_max': rho_max, 
                   'n_internal_knots': n_internal_knots, 
                   'degree': degree, 
                   'u1_density_spl_basis': u1_density_spl_basis, 
                   'augmented_knots': augmented_knots, 
                   'basis_coeffs': basis_coeffs, 
                   'omega': omega}
    return output_dict


def compute_multi_group_density_spline_basis(traj, temperature, res_group_map=None, eta=10.0, r0=0.7, rho_min=0.0, 
                                 rho_max=15.0, n_internal_knots=10, intercept=True):
    """
    Compute the basis of the local density term as a spline function with degree = 3.
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory.
    
    temperature : unit.Quantity
        The temperature.

    res_group_map : None or dict
        A dictionary mapping residue names to group indices.
    
    eta : float or int
        Parameter eta in unit 1 / nm.
    
    r0 : float or int
        Parameter r0 in unit nm.
    
    rho_min : float or int
        Parameter rho_min.
    
    rho_max : float or int
        Parameter rho_max.
    
    n_internal_knots : int
        The number of internal knots.
    
    intercept : bool
        Whether to include the intercept in the basis.
    
    Returns
    -------
    output_dict : dict
        Output dictionary.
    
    Here are the details of output_dict: 
    
    output_dict['eta'] : float or int
        Parameter eta.
    
    output_dict['r0'] : float or int
        Parameter r0.
    
    output_dict['rho_min'] : float or int
        Parameter rho_min.
    
    output_dict['rho_max'] : float or int
        Parameter rho_max.
    
    output_dict['n_internal_knots'] : int
        The number of internal knots.
    
    output_dict['degree'] : int
        The degree of the B-spline basis.
    
    output_dict['u1_density_spl_basis'] : np.ndarray, shape = (n_frames, 20, n_groups, d)
        The local density B-spline basis.
        n_groups is the number of groups defined by res_group_map.
        d is the number of bases.
    
    output_dict['augmented_knots'] : np.ndarray, shape = (n_internal_knots + 2 * degree + 2,)
        The augmented knots.
    
    output_dict['basis_coeffs'] : np.ndarray, shape = (d, degree + n_internal_knots + 1)
        The basis coefficients.
    
    output_dict['omega'] : np.ndarray, shape = (d, d)
        The omega matrix.
    
    """
    # get atom types and pair types
    atoms, _ = traj.topology.to_dataframe()
    resnames = atoms['resName'].tolist()
    atom_types = np.array([_amino_acids.index(x) for x in resnames])
    n_atoms = traj.n_atoms
    
    # compute distance matrices
    distance_matrices = compute_distance_matrices(traj, use_pbc=False) # do not use PBC, shape: (n_frames, n_atoms, n_atoms)

    # get group_index -> [atom_indices] mapping
    if res_group_map is None:
        res_group_map = {resname: 0 for resname in _amino_acids}
    group_atom_map = {}
    for _, group_name in res_group_map.items():
        group_atom_map[group_name] = []
    for atom_index, atom_type in enumerate(atom_types):
        resname = _amino_acids[atom_type]
        if resname in res_group_map:
            group_name = res_group_map[resname]
            group_atom_map[group_name].append(atom_index)
        else:
            raise ValueError(f"Residue {resname} not found in res_group_map.")
    n_groups = len(group_atom_map)
    group_name_list = sorted(list(group_atom_map.keys()))
    
    # compute rho
    cutoff = r0 + 10 / eta # cutoff distance for computing rho
    offset_switch_rho = 0.5 * (1 + math.tanh(eta * (r0 - cutoff)))
    rho_3d = 0.5 * (1 + np.tanh(eta * (r0 - distance_matrices)))
    rho_3d -= offset_switch_rho
    rho_3d[distance_matrices > cutoff] = 0.0
    rho_3d[:, np.arange(n_atoms), np.arange(n_atoms)] = 0.0 # remove the contribution by the atom itself
    #rho = np.sum(rho_3d, axis=2) # rho.shape = (n_frames, n_atoms)
    rho = np.zeros((traj.n_frames, n_atoms, n_groups))
    for group_index, group_name in enumerate(group_name_list):
        atom_indices = group_atom_map[group_name]
        if len(atom_indices) == 0:
            continue
        atom_indices = np.array(atom_indices)
        rho_group = np.sum(rho_3d[:, :, atom_indices], axis=2) # rho_group.shape = (n_frames, n_atoms)
        rho[:, :, group_index] = rho_group
    
    
    # compute B-spline basis
    degree = 3
    spl_basis_dict = clamped_bspline_basis_1d(rho, rho_min, rho_max, n_internal_knots, 
                                              degree, intercept=intercept, omega=True)
    design_matrix = spl_basis_dict['design_matrix'] # shape = (n_frames, n_atoms, n_groups, d)
    augmented_knots = spl_basis_dict['augmented_knots']
    basis_coeffs = spl_basis_dict['basis_coeffs']
    omega = spl_basis_dict['omega']
    

    assert design_matrix.ndim == rho.ndim + 1
    n_frames = design_matrix.shape[0]
    assert n_frames == traj.n_frames
    u1_density_spl_basis = np.zeros((n_frames, 20, n_groups, design_matrix.shape[-1])) 
    RT_value = (GAS_CONST * temperature).value_in_unit(unit.kilojoule_per_mole)
    for i in range(20):
        flag = atom_types == i
        if np.any(flag):
            u1_density_spl_basis[:, i, :, :] = np.sum(design_matrix[:, flag, :, :], axis=1) / RT_value
        
    output_dict = {'eta': eta,
                   'r0': r0, 
                   'rho_min': rho_min, 
                   'rho_max': rho_max, 
                   'n_internal_knots': n_internal_knots, 
                   'degree': degree, 
                   'u1_density_spl_basis': u1_density_spl_basis, 
                   'augmented_knots': augmented_knots, 
                   'basis_coeffs': basis_coeffs, 
                   'omega': omega}
    return output_dict
