import numpy as np
import pandas as pd
import mdtraj
from openabc.forcefields.MOFF2.lib import _amino_acids

def get_traj_of_selected_atoms(traj, names, inplace=False):
    """
    Get the trajectory of selected atoms with given names. 
    The atom indices order is kept in the output trajectory.
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        Input trajectory. 
    
    names : array-like
        The names of atoms that will be kept. Each element in names is an atom name. 
    
    inplace : bool
        Whether to change the trajectory in place.
        If True, change in place. Else, return a new mdtraj.Trajectory object.
    
    Returns
    -------
    selected_atom_traj : mdtraj.Trajectory
        The trajectory includes only the selected atoms. 
    
    """
    top = traj.topology
    selected_atom_indices = []
    for each in names:
        selected_atom_indices += top.select(f'name {each}').tolist()
    selected_atom_indices = np.array(sorted([int(x) for x in selected_atom_indices]))
    selected_atom_traj = traj.atom_slice(selected_atom_indices, inplace=inplace)
    return selected_atom_traj


def get_heavy_atom_COM_ca_atoms(atoms):
    """
    Get the CA atom structure with CA at the center of mass (COM) of heavy atoms for each residue. 
    The code only works for canonical amino acids. 
    
    Parameters
    ----------
    atoms : pd.DataFrame
        Input all-atom structure. 
    
    Returns
    -------
    ca_atoms : pd.DataFrame
        The CA atom structure with CA at the COM of heavy atoms for each residue. 
    
    """
    assert (atoms['resname'].isin(_amino_acids)).all()
    atom_names = atoms['name'].to_list()
    atoms['element'] = [x[0] for x in atom_names]
    heavy_atoms = atoms[atoms['element'].isin(['C', 'N', 'O', 'S'])].copy()
    mass_dict = {'C': 12.011, 
                 'N': 14.007, 
                 'O': 15.999, 
                 'S': 32.07}
    ca_atoms = heavy_atoms[heavy_atoms['name'] == 'CA'].copy()
    resSeqs = ca_atoms['resSeq'].to_numpy()
    ca_atoms = ca_atoms.set_index('resSeq')
    for r in resSeqs:
        heavy_res_atoms = heavy_atoms[heavy_atoms['resSeq'] == r]
        coords = heavy_res_atoms[['x', 'y', 'z']].to_numpy()
        masses = heavy_res_atoms['element'].map(mass_dict).to_numpy()
        w = masses / np.sum(masses)
        COM = np.sum(coords * w[:, None], axis=0)
        ca_atoms.loc[r, ['x', 'y', 'z']] = COM
    ca_atoms['serial'] = 1 + np.arange(len(ca_atoms.index))
    ca_atoms = ca_atoms.reset_index()
    return ca_atoms
    
