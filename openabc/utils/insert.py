import numpy as np
import pandas as pd
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.distances import distance_array
from scipy.spatial.transform import Rotation as R
from openabc.utils import parse_pdb, write_pdb

__author__ = 'Andrew Latham'

__modified_by__ = 'Shuming Liu'

"""
A python script for inserting molecules, similar to gmx insert-molecules. 
"""

def insert_molecules_dataframe(new_atoms, n_mol, radius=0.5, existing_atoms=None, max_n_attempts=10000, 
                               box=[100, 100, 100], method='FastNS', reset_serial=True):
    """
    Insert multiple copies of given molecule as dataframes into an existing dataframe or a new empty box. 
    Currently this function only supports using orthogonal box. 
    
    Parameters
    ----------
    new_atoms : pd.DataFrame
        New atoms of the molecule to be inserted. 
    
    n_mol : int
        Number of copies of the new molecule to be inserted. 
    
    radius : float or int
        Radius for each atom in unit nm. 
        Insertion ensures the distance between any two atoms are larger than 2*radius under periodic boundary condition. 
    
    existing_atoms : None or pd.DataFrame
        Existing atoms.
        If None, the molecules will be inserted into a new empty box, otherwise, inserted into a copy of existing_atoms. 
    
    max_n_attempts : int
        Maximal number of attempts to insert. Ensure this value no smaller than n_mol. 
    
    box : list or numpy array, shape = (3,)
        New box size in unit nm. 
    
    method : str
        Which method to use for detecting if the inserted molecule has contact with existing atoms. 
        Currently only 'FastNS' is supported. Any string will lead to method as 'FastNS'. 
    
    reset_serial : bool
        Whether to reset serial to 0, 1, ..., N - 1. 
        If True, the serial in the output dataframe is reset as 0, 1, ..., N - 1. 
        If False, the serial remains unchanged. 
        If True and atom number > 1,000,000, the serial remains unchanged since the largest atom serial number allowed in pdb is 999,999. 
    
    Returns
    -------
    atoms : pd.DataFrame
        Output atoms.
    
    """
    new_coord = new_atoms[['x', 'y', 'z']].to_numpy()
    new_coord -= np.mean(new_coord, axis=0)
    if existing_atoms is None:
        atoms = pd.DataFrame()
    else:
        atoms = existing_atoms.copy()
    if method != 'FastNS':
        print('Currently only FastNS is supported. Use FastNS method to check contacts.')
        method = 'FastNS'
    assert method == 'FastNS'
    box_a, box_b, box_c = 10 * box[0], 10 * box[1], 10 * box[2] # convert nm to angstrom
    dim = np.array([box_a, box_b, box_c, 90.0, 90.0, 90.0]).astype(np.float32)
    count_n_mol = 0
    count_n_attempts = 0
    cutoff = float(2 * 10 * radius) # convert nm to angstrom
    # to save time, we only run FastNS once for a given configuration of atoms
    # that is to say, we may try multiple times to insert a molecule with only running FastNS once
    if len(atoms.index) > 0:
        # run FastNS to prepare for inserting the first molecule
        coord = atoms[['x', 'y', 'z']].to_numpy().astype(np.float32)
        grid_search = FastNS(cutoff, coord, dim, pbc=True)
    while (count_n_mol < n_mol) and (count_n_attempts < max_n_attempts):
        # get a random rotation
        rotate = R.random()
        new_coord_i = rotate.apply(new_coord) # create a new array
        # get a random translation
        translation = np.random.uniform(0, 1, 3) * np.array([box_a, box_b, box_c])
        new_coord_i += translation
        flag = False
        if len(atoms.index) == 0:
            # the box is empty, we can insert the new molecule
            flag = True
        else:
            # check if we can insert the new molecule
            results = grid_search.search(new_coord_i.astype(np.float32))
            if len(results.get_pair_distances()) == 0:
                flag = True # no overlap
        if flag:
            # insert the new molecule
            new_atoms_i = new_atoms.copy()
            new_atoms_i[['x', 'y', 'z']] = new_coord_i
            atoms = pd.concat([atoms, new_atoms_i], ignore_index=True)
            count_n_mol += 1
            # run FastNS to prepare for inserting the next molecule
            coord = atoms[['x', 'y', 'z']].to_numpy().astype(np.float32)
            grid_search = FastNS(cutoff, coord, dim, pbc=True)
        count_n_attempts += 1
    
    # determine if the number of molecules were successfully added:
    if count_n_mol == n_mol:
        print(f'Successfully inserted {n_mol} molecules.')
    else:
        print(f'Could not successfully insert {n_mol} molecules in {count_n_attempts} attempts.')
        print(f'Only added {count_n_mol} molecules. Try increasing the box size or the max number of attempts.')
    
    if reset_serial:
        # reset serial
        n_atoms = len(atoms.index)
        if n_atoms > 1000000:
            print(f'Too many atoms. Cannot reset serial as 0, 1, ..., N - 1. Serial remains unchanged.')
        else:
            atoms['serial'] = list(range(len(atoms.index)))
    return atoms
    

def insert_molecules(new_pdb, output_pdb, n_mol, radius=0.5, existing_pdb=None, max_n_attempts=10000, 
                     box=[100, 100, 100], method='FastNS', reset_serial=True):
    """
    Insert multiple copies of given molecule into an existing pdb or a new empty box. 
    Currently this function only supports using orthogonal box. 
    
    Parameters
    ----------
    new_pdb : str
        Path of the new pdb file to be inserted. 
    
    output_pdb : str
        Path of the output pdb file. 
    
    n_mol : int
        Number of copies of the new molecule to be inserted. 
    
    radius : float or int
        Radius for each atom in unit nm. 
        Insertion ensures the distance between any two atoms are larger than 2*radius under periodic boundary condition. 
    
    existing_pdb : None or str
        Path of an existing pdb file. 
        If None, the molecules will be inserted into a new empty box, otherwise, inserted into a copy of existing_pdb. 
    
    max_n_attempts : int
        Maximal number of attempts to insert. Ensure this value no smaller than n_mol. 
    
    box : list or numpy array, shape = (3,)
        New box size in unit nm. 
    
    method : str
        Which method to use for detecting if the inserted molecule has contact with existing atoms. 
        Currently only 'FastNS' is supported. Any string will lead to method as 'FastNS'.  
    
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
    atoms = insert_molecules_dataframe(new_atoms, n_mol, radius, existing_atoms, max_n_attempts, 
                                       box, method, reset_serial)
    # write the final pdb
    write_pdb(atoms, output_pdb)



