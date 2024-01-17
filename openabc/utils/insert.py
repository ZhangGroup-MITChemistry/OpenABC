import numpy as np
import pandas as pd
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.distances import distance_array
from scipy.spatial.transform import Rotation as R
from openabc.utils import parse_pdb, write_pdb
import math

__author__ = 'Andrew Latham'

__modified_by__ = 'Shuming Liu'

"""
A python script for inserting molecules, similar to gmx insert-molecules. 
"""

def insert_molecules_dataframe(new_atoms, n_copies, radius=0.5, existing_atoms=None, max_n_attempts=10000, 
                               box=[100, 100, 100, 90.0, 90.0, 90.0], reset_serial=True):
    """
    Insert multiple copies of given atoms into a box with existing atoms or empty and ensure no non-physical overlap with FastNS method under PBC condition. 
    Note all the input length parameter units are nm, though coordinates in pdb are in unit angstroms.
    
    Parameters
    ----------
    new_atoms : pd.DataFrame
        New atoms of the molecule to be inserted. 
    
    n_copies : int
        Number of copies of the new atoms to be inserted. 
    
    radius : float or int
        Radius for each atom in unit nm. Insertion ensures the distance between any two atoms are larger than 2*radius under periodic boundary condition. 
    
    existing_atoms : None or pd.DataFrame
        Existing atoms. 
        If None, the molecules will be inserted into a new empty box, otherwise, inserted into a copy of existing_atoms. 
    
    max_n_attempts : int
        Maximal number of attempts to insert. Ensure this value is >= n_copies.
    
    box : 1d-array like, shape = (6,)
        Box shape array as [a, b, c, alpha, beta, gamma]. 
        Box lengths are a, b, and c in unit nm.
        Box angles are alpha, beta, and gamma in unit degree.
        Be careful with the unit of lengths and angles. 
    
    reset_serial : bool
        Whether to reset serial to 0, 1, ..., N - 1. 
        If True, the serial in the final pdb is reset as 0, 1, ..., N - 1. 
        If False, the serial remains unchanged. 
        If True and atom number > 1,000,000, the serial remains unchanged since the largest atom serial number allowed in pdb is 999,999. 
    
    Returns
    -------
    atoms : pd.DataFrame
        Output atoms.
    
    """
    assert max_n_attempts >= n_copies
    if existing_atoms is None:
        atoms = pd.DataFrame()
    else:
        atoms = existing_atoms.copy()
    new_coords = new_atoms[['x', 'y', 'z']].to_numpy() # in unit angstrom
    new_coords -= np.mean(new_coords, axis=0) # move to origin to facilitate further operations
    count_n_copies = 0
    count_n_attempts = 0
    cutoff = float(2 * 10 * radius) # convert nm to angstrom
    assert len(box) == 6
    if not isinstance(box, np.ndarray):
        box = np.array(box)
        box[:3] *= 10 # convert nm to angstrom
        box = box.astype(np.float32)
        if len(atoms.index) > 0:
            coords = atoms[['x', 'y', 'z']].to_numpy().astype(np.float32)
            grid_search = FastNS(cutoff, coords, box, pbc=True)
    a = float(box[0]) # in unit angstrom
    b = float(box[1]) # in unit angstrom
    c = float(box[2]) # in unit angstrom
    alpha = float(box[3]) * np.pi / 180 # in unit radian
    beta = float(box[4]) * np.pi / 180 # in unit radian
    gamma = float(box[5]) * np.pi / 180 # in unit radian
    v1 = np.array([a, 0, 0]) # primitive cell vector 1
    v2 = np.array([b * math.cos(gamma), b * math.sin(gamma), 0]) # primitive cell vector 2
    v3_x = c * math.cos(beta)
    v3_y = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    v3_z = (c**2 - v3_x**2 - v3_y**2)**0.5
    v3 = np.array([v3_x, v3_y, v3_z]) # primitive cell vector 3
    v = np.stack([v1, v2, v3], axis=0) # primitive cell vectors
    while (count_n_copies < n_copies) and (count_n_attempts < max_n_attempts):
        rotate = R.random()
        new_coords_i = rotate.apply(new_coords)
        translate = np.dot(v.T, np.random.uniform(0, 1, 3)) # in unit angstrom
        new_coords_i += translate # in unit angstrom
        if len(atoms.index) == 0:
            flag = True
        else:
            flag = False
            results = grid_search.search(new_coords_i.astype(np.float32))
            if len(results.get_pair_distances()) == 0:
                flag = True # no overlap
        if flag:
            new_atoms_i = new_atoms.copy()
            new_atoms_i[['x', 'y', 'z']] = new_coords_i
            atoms = pd.concat([atoms, new_atoms_i], ignore_index=True)
            count_n_copies += 1
            coords = atoms[['x', 'y', 'z']].to_numpy().astype(np.float32) # in unit angstrom
            grid_search = FastNS(cutoff, coords, box, pbc=True)
        count_n_attempts += 1
    
    # determine if n_copies of new atoms were successfully added
    if count_n_copies == n_copies:
        print(f'Successfully inserted {n_copies} molecules.')
    else:
        print(f'Failed to insert {n_copies} molecules. Only {count_n_copies} molecules were inserted.')
        print('You may need to increase the box size or the number of attempts and retry.')
    if reset_serial:
        n_atoms = len(atoms.index)
        if n_atoms > 1000000:
            print(f'Too many atoms. Cannot reset serial as 0, 1, ..., N - 1. Serial remains unchanged.')
        else:
            atoms['serial'] = np.arange(n_atoms)
    return atoms
    

def insert_molecules(new_pdb, output_pdb, n_copies, radius=0.5, existing_pdb=None, max_n_attempts=10000, 
                     box=[100, 100, 100, 90.0, 90.0, 90.0], reset_serial=True):
    """
    Insert multiple copies of given PDB into a box with existing PDB or empty and ensure no non-physical overlap with FastNS method under PBC condition. 
    Note all the input length parameter units are nm, though coordinates in pdb are in unit angstroms.
    
    Parameters
    ----------
    new_pdb : str
        Path of the new PDB file of the molecule to be inserted. 
    
    output_pdb : str
        Path of the output PDB file with inserted molecules.
    
    n_copies : int
        Number of copies of the new atoms to be inserted. 
    
    radius : float or int
        Radius for each atom in unit nm. Insertion ensures the distance between any two atoms are larger than 2*radius under periodic boundary condition. 
    
    existing_pdb : str or None
        Existing PDB of the molecules in the box. 
        If None, the molecules will be inserted into a new empty box, otherwise, inserted into a box with atoms from existing_pdb.
    
    max_n_attempts : int
        Maximal number of attempts to insert. Ensure this value is >= n_copies.
    
    box : 1d-array like, shape = (6,)
        Box shape array as [a, b, c, alpha, beta, gamma]. 
        Box lengths are a, b, and c in unit nm.
        Box angles are alpha, beta, and gamma in unit degree.
        Be careful with the unit of lengths and angles. 
    
    reset_serial : bool
        Whether to reset serial to 0, 1, ..., N - 1. 
        If True, the serial in the final pdb is reset as 0, 1, ..., N - 1. 
        If False, the serial remains unchanged. 
        If True and atom number > 1,000,000, the serial remains unchanged since the largest atom serial number allowed in pdb is 999,999. 
    
    """
    new_atoms = parse_pdb(new_pdb)
    if existing_pdb is None:
        existing_atoms = None
    else:
        existing_atoms = parse_pdb(existing_pdb)
    atoms = insert_molecules_dataframe(new_atoms, n_copies, radius, existing_atoms, max_n_attempts, box, 
                                       reset_serial)
    print(f'Write atoms to {output_pdb}')
    write_pdb(atoms, output_pdb)
    



