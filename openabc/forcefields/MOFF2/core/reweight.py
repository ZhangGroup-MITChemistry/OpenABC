import numpy as np
import pandas as pd
import torch
import mdtraj
from FastMBAR import FastMBAR
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
from openabc.forcefields.MOFF2.lib import GAS_CONST
import sys
import os
from typing import Union
import pickle

def compute_PE(traj, system, platform_name='Reference', properties={'Precision': 'double'}, groups=-1):
    """
    Compute the potential energy of samples in the trajectory with the given openmm system.
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory including all the samples.
    
    system : openmm.System
        The openmm system.
    
    platform_name : str
        The platform name for running openmm to evaulate energy.
    
    properties : dict
        Properties for running openmm. Only used when platform_name is CUDA or OpenCL.
    
    groups : int or set
        The force groups included when computing energy.
        See openmm context.getState for details about this parameter.
    
    Returns
    -------
    energy : np.ndarray, shape = (traj.n_frames,)
        The energy of each sample in the trajectory in unit kJ/mol.
    
    """
    top = traj.topology.to_openmm()
    # use any integrator with any parameter
    timestep = 1.0 * unit.femtosecond
    integrator = mm.VerletIntegrator(timestep)
    platform = mm.Platform.getPlatformByName(platform_name)
    if platform_name in ['CUDA', 'OpenCL']:
        simulation = app.Simulation(top, system, integrator, platform, properties)
    else:
        simulation = app.Simulation(top, system, integrator, platform)
    use_pbc = system.usesPeriodicBoundaryConditions()
    energy = []
    for i in range(traj.n_frames):
        simulation.context.setPositions(traj.xyz[i])
        state = simulation.context.getState(getEnergy=True, enforcePeriodicBox=use_pbc, groups=groups)
        energy.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    energy = np.array(energy)
    return energy


def compute_reduced_PE_matrix(traj, systems, temperatures, platform_name='Reference', 
                              properties={'Precision': 'double'}, groups=-1):
    """
    Compute the reduced potential energy matrix of each sample in the trajectory with all the input thermodynamic states.
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory including all the samples.
    
    systems : array-like, shape = (n_systems,)
        A collection of openmm systems. 
        The number of systems should be equal to the number of thermodynamic states.
    
    temperatures : array-like, shape = (n_systems,)
        The temperatures of different thermodynamic states.
        Each value in temperatures should be a Quantity with temperature unit.
        The length of temperatures should be equal to the number of thermodynamic states.
    
    platform_name : str
        The platform name for running openmm to evaulate energy.
    
    properties : dict
        Properties for running openmm. Only used when platform_name is CUDA or OpenCL.
    
    groups : int or set
        The force groups included when computing energy.
        See openmm context.getState for details about this parameter.
    
    Returns
    -------
    reduced_energy_matrix : np.ndarray, shape = (n_systems, traj.n_frames)
        The reduced energy matrix.
    
    """
    assert len(systems) == len(temperatures)
    reduced_energy_matrix = np.zeros((len(systems), traj.n_frames))
    for i in range(len(systems)):
        RT = GAS_CONST * temperatures[i]
        reduced_energy_matrix[i] = compute_PE(traj, systems[i], platform_name, properties, groups) / RT
    return reduced_energy_matrix


def compute_mixed_u0_from_reduced_PE_matrices(reduced_energy_matrix0, n_samples0, reduced_energy_matrix1, fastmbar_cuda=True, verbose=False):
    """
    Compute the reduced potential energy of samples in the mixed ensemble 0 and thermodynamic state 1.
    
    Parameters
    ----------
    reduced_energy_matrix0 : 2d array-like, shape = (n_systems0, np.sum(n_samples0))
        The reduced energy matrix of the mixed ensemble 0.
    
    n_samples0 : 2d array-like, shape = (n_systems0,)
        The number of samples from each therymodynamic state in the mixed ensemble 0.
    
    reduced_energy_matrix1 : 2d array-like, shape = (n_systems0, n_samples1)
        The reduced energy matrix of thermodynamic state 1 evaluated under the mixed ensemble 0.
        n_samples1 is an integer and indicates the number of samples from thermodynamic state 1.
    
    fastmbar_cuda : bool
        Whether to use cuda for FastMBAR.
    
    verbose : bool
        Whether to print FastMBAR outputs.
    
    Returns
    -------
    u0_traj0 : np.ndarray
        The reduced potential energy of each sample in the mixed ensemble 0 evaluated under the mixed ensemble 0.
    
    u0_traj1 : np.ndarray
        The reduced potential energy of each sample in thermodynamic state 1 evaluated under the mixed ensemble 0.
    
    fastmbar : FastMBAR
        FastMBAR object for computing the mixed ensemble.
    
    """
    if not isinstance(reduced_energy_matrix0, np.ndarray):
        reduced_energy_matrix0 = np.array(reduced_energy_matrix0)
    if not isinstance(n_samples0, np.ndarray):
        n_samples0 = np.array(n_samples0)
    if not isinstance(reduced_energy_matrix1, np.ndarray):
        reduced_energy_matrix1 = np.array(reduced_energy_matrix1)
    assert reduced_energy_matrix0.ndim == 2
    assert reduced_energy_matrix1.ndim == 2
    assert reduced_energy_matrix0.shape[0] == len(n_samples0)
    assert reduced_energy_matrix0.shape[1] == np.sum(n_samples0)
    assert reduced_energy_matrix1.shape[0] == len(n_samples0)
    fastmbar = FastMBAR(energy=reduced_energy_matrix0, num_conf=n_samples0, cuda=fastmbar_cuda, verbose=verbose, bootstrap=False)
    u0_traj0 = -fastmbar.log_prob_mix
    b = -torch.tensor(fastmbar.F, dtype=torch.float64) - torch.log(fastmbar.num_conf.to(torch.float64))
    u0_traj1 = -torch.logsumexp(-(torch.tensor(reduced_energy_matrix1, dtype=torch.float64) + b[:, None]), dim=0)
    u0_traj1 = u0_traj1.cpu().numpy()
    return u0_traj0, u0_traj1, fastmbar


def compute_mixed_u0(traj0, systems0, temperatures0, n_samples0, traj1, platform_name='CUDA', properties={'Precision': 'double'}, 
                     fastmbar_cuda=True, verbose=False):
    """
    Compute the reduced potential energy of samples in the mixed ensemble 0. 
    The samples come from traj0 and traj1, but evaluated under the mixed ensemble composed of traj0, systems0, temperatures0, and n_samples0.
    In the context of contrastive learning, mixed ensemble 0 is the noise, while traj1 is the data.
    
    Parameters
    ----------
    traj0 : mdtraj.Trajectory
        The trajectory including all the samples from thermodynamic states defined by systems0 and temperatures0.
    
    systems0 : array-like, shape = (n_systems0,)
        The openmm system of different therymodynamic states that define the mixed ensemble 0.
        The length of systems0 should be equal to the number of therymodynamic states in the mixed ensemble 0.
    
    temperatures0 : array-like, shape = (n_systems0,)
        The temperatures of different therymodynamic states that define the mixed ensemble 0.
        Each value in temperatures should be a Quantity with temperature unit.
    
    n_samples0 : array-like, shape = (n_systems0,)
        The number of samples from each therymodynamic state in the mixed ensemble 0.
        The sum of n_samples0 should be equal to traj0.n_frames.
    
    traj1 : mdtraj.Trajectory
        The trajectory including all the samples from thermodynamic state 1.
    
    platform_name : str
        The platform name for running openmm to evaulate energy.
    
    properties : dict
        Properties for running openmm. Only used when platform_name is CUDA or OpenCL.
    
    fastmbar_cuda : bool
        Whether to use cuda for FastMBAR.
    
    verbose : bool
        Whether to print FastMBAR outputs.
    
    Returns
    -------
    mixed_u0_traj0 : np.ndarray
        The reduced potential energy of each sample in traj0 evaluated under the mixed ensemble 0.
    
    mixed_u0_traj1 : np.ndarray
        The reduced potential energy of each sample in traj1 evaluated under the mixed ensemble 0.
    
    fastmbar : FastMBAR
        FastMBAR object for computing the mixed ensemble.
    
    """
    assert len(systems0) == len(temperatures0)
    assert len(systems0) == len(n_samples0)
    assert traj0.n_frames == np.sum(n_samples0)
    reduced_energy_matrix0 = compute_reduced_PE_matrix(traj0, systems0, temperatures0, platform_name, properties, groups=-1)
    reduced_energy_matrix1 = compute_reduced_PE_matrix(traj1, systems0, temperatures0, platform_name, properties, groups=-1)
    u0_traj0, u0_traj1, fastmbar = compute_mixed_u0_from_reduced_PE_matrices(reduced_energy_matrix0, n_samples0, reduced_energy_matrix1, fastmbar_cuda, verbose)
    return u0_traj0, u0_traj1, fastmbar




# torch_energy.py (for example)

import numpy as np
import torch
import pickle
from typing import Union


def _flatten_20x20_to_210(mat: np.ndarray) -> np.ndarray:
    """
    Convert a 20x20 symmetric matrix into a (210,) vector
    using the upper triangle (i <= j), consistent with
    how u1_ah_basis and u1_gauss_basis are constructed.
    """
    if mat.ndim != 2 or mat.shape != (20, 20):
        raise ValueError(f"Expected (20,20) matrix, got shape {mat.shape}")
    rows, cols = np.triu_indices(20, k=0)
    return mat[rows, cols]


def compute_PE_torch_from_fep(
    fep_pkl: str,
    results_pkl: str,
    device: Union[str, torch.device] = "cuda",
    return_numpy: bool = True,
):
    """
    Compute βU1 (dimensionless U1 / (k_B T)) for each frame using
    precomputed Torch FEP bases (AH, Gaussian, density spline).

    This uses EXACTLY the same formulas as in your latest
    AH + Gaussian + density implementation in
    s4_prepare_fep_AH_gau_torch.py.

    Parameters
    ----------
    fep_pkl : str
        Path to fep_input.pkl generated by s4_prepare_fep_AH_gau_torch.py.
        Must contain:
            'u1_intercept'          : (F,)
            'u1_ah_basis'           : (F, 210)
            'u1_gauss_basis'        : (F, 210)
            'u1_density_spl_basis'  : (F, 20, n_groups, d)
            'temperature'           : float
            'res_group_mapping'     : key string (for consistency check)
    results_pkl : str
        Path to results.pkl from training. Must contain:
            'hydrophobic_scale' : (20,20) or (210,)
            'spl_values'        : (d, n_groups, 20)
            'res_group_mapping' : same key as in fep_pkl
            optionally:
            'gauss_coeffs'      : (210,)
    device : 'cpu' or 'cuda' or torch.device
        Where to do the linear algebra.
    return_numpy : bool
        If True, return np.ndarray. If False, return torch.Tensor on CPU.

    Returns
    -------
    beta_u1 : np.ndarray or torch.Tensor, shape = (F,)
        βU1 = U1 / (k_B T) for each frame, dimensionless.
    """

    # -----------------------------------------
    # Load FEP input + training results
    # -----------------------------------------
    with open(fep_pkl, "rb") as f:
        fep = pickle.load(f)

    with open(results_pkl, "rb") as f:
        res = pickle.load(f)

    # ---- sanity checks on res_group_mapping ----
    fep_rgm = fep.get("res_group_mapping", None)
    res_rgm = res.get("res_group_mapping", None)
    if (fep_rgm is not None) and (res_rgm is not None):
        if fep_rgm != res_rgm:
            raise ValueError(
                f"res_group_mapping mismatch: fep_input={fep_rgm}, "
                f"results={res_rgm}"
            )

    # -----------------------------------------
    # Extract basis terms from fep_input.pkl
    # (All of these are ALREADY divided by RT)
    # -----------------------------------------
    u1_intercept = np.asarray(fep["u1_intercept"])          # (F,)
    u1_ah_basis = np.asarray(fep["u1_ah_basis"])            # (F, 210)
    u1_gauss_basis = np.asarray(fep["u1_gauss_basis"])      # (F, 210)
    u1_density_spl_basis = np.asarray(fep["u1_density_spl_basis"])
    # shape = (F, 20, n_groups, d)

    if u1_density_spl_basis.ndim != 4:
        raise ValueError(
            f"Expected u1_density_spl_basis to have ndim=4, got "
            f"{u1_density_spl_basis.ndim}"
        )

    F, n_aa_basis, n_groups_basis, d_basis = u1_density_spl_basis.shape

    # -----------------------------------------
    # Extract parameters from results.pkl
    # -----------------------------------------
    hydrophobic_scale = np.asarray(res["hydrophobic_scale"])
    # Accept either (20,20) or (210,)
    if hydrophobic_scale.ndim == 2 and hydrophobic_scale.shape == (20, 20):
        h_1d = _flatten_20x20_to_210(hydrophobic_scale)
    elif hydrophobic_scale.ndim == 1 and hydrophobic_scale.shape[0] == 210:
        h_1d = hydrophobic_scale
    else:
        raise ValueError(
            f"hydrophobic_scale must be (20,20) or (210,), got shape "
            f"{hydrophobic_scale.shape}"
        )

    # Gaussian heights: may or may not exist
    gauss_coeffs = res.get("gauss_coeffs", None)
    if gauss_coeffs is None:
        g_1d = np.zeros(210, dtype=float)
    else:
        g_1d = np.asarray(gauss_coeffs)
        if g_1d.shape[0] != 210:
            raise ValueError(
                f"gauss_coeffs must have length 210, got {g_1d.shape}"
            )

    # Density spline coefficients:
    # Expected shape: (d, n_groups, 20)
    #spl_values = np.asarray(res["spl_values"])
    #if "spl_coeffs" in res:
    #    spl_values = np.asarray(res["spl_coeffs"])
    #else:
    #    spl_values = np.asarray(res["spl_values"])
    if "spl_coeffs" in res:
    # convert (20, n_grp, d) → (d, n_grp, 20)
        spl_values = np.asarray(res["spl_coeffs"]).transpose(2,1,0)
    else:
        # Warning: spl_values has shape (20, n_grp, 500)
        # This is NOT suitable for torch reweighting but fallback allowed
        spl_values = np.asarray(res["spl_values"]).transpose(2,1,0)


    if spl_values.ndim != 3:
        raise ValueError(
            f"spl_values must have ndim=3, got shape {spl_values.shape}"
        )
    d_spl, n_groups_spl, n_aa_spl = spl_values.shape

    if (d_spl != d_basis) or (n_groups_spl != n_groups_basis) or (n_aa_spl != n_aa_basis):
        raise ValueError(
            "Shape mismatch between u1_density_spl_basis (F, 20, n_groups, d) and "
            f"spl_values (d, n_groups, 20). Got basis={u1_density_spl_basis.shape}, "
            f"spl_values={spl_values.shape}."
        )

    # -----------------------------------------
    # Move everything to Torch / device
    # -----------------------------------------
    device = torch.device(device)
    dtype = torch.float64  # keep high precision like OpenMM double

    u1_intercept_t = torch.as_tensor(u1_intercept, device=device, dtype=dtype)      # (F,)
    u1_ah_basis_t = torch.as_tensor(u1_ah_basis, device=device, dtype=dtype)        # (F, 210)
    u1_gauss_basis_t = torch.as_tensor(u1_gauss_basis, device=device, dtype=dtype)  # (F, 210)
    u1_density_spl_basis_t = torch.as_tensor(
        u1_density_spl_basis, device=device, dtype=dtype
    )  # (F, 20, n_groups, d)

    h_1d_t = torch.as_tensor(h_1d, device=device, dtype=dtype)      # (210,)
    g_1d_t = torch.as_tensor(g_1d, device=device, dtype=dtype)      # (210,)
    spl_values_t = torch.as_tensor(spl_values, device=device, dtype=dtype)
    # (d, n_groups, 20)

    # -----------------------------------------
    # Linear combination:
    #   βU1 = intercept
    #         + Σ_k h_k * basis_AH[:, k]
    #         + Σ_k g_k * basis_Gauss[:, k]
    #         + Σ_{aa,grp,b} spl[b,grp,aa] * basis_density[:, aa, grp, b]
    # -----------------------------------------
    # AH + Gaussian part: (F,210) @ (210,) → (F,)
    beta_u_ah = u1_ah_basis_t @ h_1d_t
    beta_u_gauss = u1_gauss_basis_t @ g_1d_t

    # Density spline part via einsum:
    # basis: (F, 20, n_groups, d)
    # spl  : (d, n_groups, 20)
    # → βU_density: (F,)
    beta_u_density = torch.einsum(
        "fagd,dga->f", u1_density_spl_basis_t, spl_values_t
    )

    beta_u_total = u1_intercept_t + beta_u_ah + beta_u_gauss + beta_u_density

    if return_numpy:
        return beta_u_total.detach().cpu().numpy()
    else:
        # Return on CPU for safety (can keep on GPU if you prefer)
        return beta_u_total.detach().cpu()

