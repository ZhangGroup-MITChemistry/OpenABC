import numpy as np
import mdtraj

"""
The script includes functions related to topology of linear chains without any branches. 
"""

def compute_linear_chain_internal_coords(traj, compute_bonds=True, compute_angles=True, compute_dihedrals=True, use_pbc=True):
    """
    Compute the internal coordinates for a linear chain. 
    For atoms on each chain, the atoms should be connected sequentially.
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        Input trajectory. 
    
    compute_bonds : bool
        Whether to compute bond lengths.
        If True, bond lengths are computed, else bond lengths are not computed.
    
    compute_angles : bool
        Whether to compute angles. 
        If True, angles are computed, else angles are not computed.
    
    compute_dihedrals : bool
        Whether to compute dihedrals.
        If True, dihedrals are computed, else dihedrals are not computed.
    
    use_pbc : bool
        Whether to consider periodic boundary conditions.
    
    Returns
    -------
    n_chains : int
        The number of chains.
    
    internal_coord_atoms : dict
        A dictionary with atom indices involved in internal coordinates.
        The keys in internal_coord_atoms are 0, 1, ..., n_chains - 1. 
        internal_coord_atoms[i] is also a dictionary with keys as 'bonds', 'angles', and 'dihedrals'. 
        internal_coord_atoms[i][j] is a 2d numpy array of shape (n_features, n_atoms), where n_features is the number of bonds or angles or dihedrals in the i-th chain, and n_atoms = 2, 3, or 4.
    
    internal_coords : dict
        A dictionary with internal coordinate values.
        The keys in internal_coords are 0, 1, ..., n_chains - 1. 
        internal_coords[i] is also a dictionary with keys as 'bonds', 'angles', and 'dihedrals'. 
        internal_coords[i][j] is a 2d numpy array of shape (n_frames, n_features).
    
    """
    # in mdtraj, chainIDs are renamed as serial numbers starting from 0 to n_chains - 1
    n_chains = traj.n_chains
    atoms, _ = traj.topology.to_dataframe()
    atoms.index = list(range(len(atoms.index))) # ensure index starts from 0
    atoms['chainID'] = atoms['chainID'].astype(int) # ensure chainIDs are of int type
    internal_coord_atoms = {}
    internal_coords = {}
    for i in range(n_chains):
        chain_i_atom_indices = atoms[atoms['chainID'] == i].index.tolist()
        chain_i_atom_indices = np.array(sorted([int(x) for x in chain_i_atom_indices]))
        if len(chain_i_atom_indices) > 1:
            for j in range(len(chain_i_atom_indices) - 1):
                # for a linear chain, atom indices should be consecutive
                assert chain_i_atom_indices[j] + 1 == chain_i_atom_indices[j + 1]
        internal_coord_atoms[i] = {}
        internal_coords[i] = {}
        if compute_bonds and (len(chain_i_atom_indices) >= 2):
            chain_i_bonds = np.stack((chain_i_atom_indices[:-1], 
                                      chain_i_atom_indices[1:]), axis=-1)
            internal_coord_atoms[i]['bonds'] = chain_i_bonds
            internal_coords[i]['bonds'] = mdtraj.compute_distances(traj, chain_i_bonds, periodic=use_pbc)
        if compute_angles and (len(chain_i_atom_indices) >= 3):
            chain_i_angles = np.stack((chain_i_atom_indices[:-2], 
                                       chain_i_atom_indices[1:-1], 
                                       chain_i_atom_indices[2:]), axis=-1)
            internal_coord_atoms[i]['angles'] = chain_i_angles
            angles = mdtraj.compute_angles(traj, chain_i_angles, periodic=use_pbc)
            # ensure angles are within range [0, pi)
            angles = np.clip(angles, 0, np.pi)
            angles[angles == np.pi] = 0
            internal_coords[i]['angles'] = angles
        if compute_dihedrals and (len(chain_i_atom_indices) >= 4):
            chain_i_dihedrals = np.stack((chain_i_atom_indices[:-3], 
                                          chain_i_atom_indices[1:-2], 
                                          chain_i_atom_indices[2:-1], 
                                          chain_i_atom_indices[3:]), axis=-1)
            internal_coord_atoms[i]['dihedrals'] = chain_i_dihedrals
            dihedrals = mdtraj.compute_dihedrals(traj, chain_i_dihedrals, periodic=use_pbc)
            # ensure angles are within range [-pi, pi)
            dihedrals = np.clip(dihedrals, -np.pi, np.pi)
            dihedrals[dihedrals == np.pi] = -np.pi
            internal_coords[i]['dihedrals'] = dihedrals
    return n_chains, internal_coord_atoms, internal_coords



