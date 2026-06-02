import numpy as np
import mdtraj
from openabc.forcefields.MOFF2.lib import _amino_acid_mass_dict

"""
This script includes functions that can be applied to structures of any topology types.
"""

def compute_distance_matrices(traj, use_pbc=True):
    """
    Compute the distance matrices for a trajectory. 
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory object.
    
    use_pbc : bool
        Whether to consider periodic boundary condition when computing distances.
    
    Returns
    -------
    distances : 3d np.ndarray, shape = (n_frames, n_atoms, n_atoms)
        All the distances.
    
    """ 
    n_frames = traj.n_frames
    n_atoms = traj.n_atoms
    rows, cols = np.triu_indices(n_atoms, k=0)
    pairs = np.stack([rows, cols], axis=-1)
    distances = np.zeros((n_frames, n_atoms, n_atoms))
    distances[:, pairs[:, 0], pairs[:, 1]] = mdtraj.compute_distances(traj, pairs, periodic=use_pbc)
    distances[:, pairs[:, 1], pairs[:, 0]] = distances[:, pairs[:, 0], pairs[:, 1]] # symmetric
    return distances


def compute_radius_of_gyration(coords, masses):
    """
    Compute radius of gyration.
    Note this method does not consider PBC.
    
    Parameters
    ----------
    coords : 3d array-like, shape = (n_frames, n_atoms, 3)
        The coordinates.
    
    masses : 1d array-like, shape = (n_atoms,)
        The masses.
    
    Returns
    -------
    rg : 1d array-like, shape = (n_frames,)
        The radius of gyration.
    
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    assert coords.ndim == 3
    n_atoms = coords.shape[1]
    assert n_atoms == len(masses)
    if not isinstance(masses, np.ndarray):
        masses = np.array(masses)
    r_COM = np.average(coords, axis=1, weights=masses, keepdims=True)
    delta_r = coords - r_COM
    rg = np.sqrt(np.average(np.sum(delta_r**2, axis=2), axis=1, weights=masses))
    return rg


def compute_CA_traj_radius_of_gyration(traj):
    """
    Compute radius of gyration for the CA model.
    Note this method does not consider PBC.
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory object.
    
    Returns
    -------
    rg : 1d array-like, shape = (n_frames,)
        The radius of gyration.
    
    """
    atoms, _ = traj.topology.to_dataframe()
    assert (atoms['name'] == 'CA').all()
    masses = np.array([_amino_acid_mass_dict[x] for x in atoms['resName'].tolist()])
    rg = compute_radius_of_gyration(traj.xyz, masses)
    return rg

