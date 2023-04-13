import numpy as np
import pandas as pd
import sys
import os
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.transformations import rotate
import random

__author__ = 'Andrew Latham'

__modified_by__ = 'Shuming Liu'

'''
A python script for inserting molecules, similar to gmx insert-molecules. 
'''

def insert_molecules(new_pdb, output_pdb='system.pdb', n_mol=100, radius=0.5, existing_pdb=None, 
                     max_n_attempts=10000, reset_box=True, box=[100, 100, 100]):
    '''
    Insert multiple copies of given molecule into an existing pdb or a new empty box. Note all the length parameter units are nm, though coordinates in pdb are in unit angstroms. Currently this function only supports using orthogonal box. 
    
    Parameters
    ----------
    new_pdb : str
        Path of the new pdb file to be inserted. 
    
    output_pdb : str
        Path of the output pdb file. 
    
    n_mol : int
        Number of copies of the new molecule to be inserted. 
    
    radius : float or int
        Radius for each atom in unit nm. Insertion ensures the distance between any two atoms are larger than 2*radius. 
    
    existing_pdb : None or str
        Path of an existing pdb file. If None, the molecules will be inserted into a new empty box, otherwise, inserted into a copy of existing_pdb. 
    
    max_n_attempts : int
        Maximal number of attempts to insert. Ensure this value no smaller than n_mol. 
    
    reset_box : bool
        Whether to reset the box lengths. 
    
    box : list or numpy array, shape = (3,)
        New box size in unit nm. This is effective if reset_box is True. 
    
    '''
    # determine if adding to a new or existing pdb_file:
    # if not adding to a pdb file, create a blank universe
    if existing_pdb is None:
        u1 = mda.Universe.empty(0, trajectory=True) # necessary for adding coordinates
    # if adding to an existing pdb_file, load the existing universe
    else:
        u1 = mda.Universe(existing_pdb, existing_pdb)
    
    # load molecule to insert to the box
    to_insert = mda.Universe(new_pdb, new_pdb)
    # read in the molecule and center it in the old simulation box
    molecule = to_insert.select_atoms('all')
    cog = molecule.atoms.center_of_geometry()
    molecule.atoms.positions -= cog
    
    # if reset_box, change box to the desired box.
    if reset_box:
        # adjust nm to angstrom
        box = [box[0]*10, box[1]*10, box[2]*10]
        u1.dimensions = [box[0], box[1], box[2], 90, 90, 90]
    else:
        box = u1.dimensions[0:3]
    print(f'Box size is {box[0]}*{box[1]}*{box[2]} angstrom^3. ')
    
    assert max_n_attempts >= n_mol
    count_n_mol = 0
    count_n_attempts = 0
    while (count_n_mol < n_mol) and (count_n_attempts < max_n_attempts):
        # determine a random translation and rotation in the simulation box
        translate = [u1.dimensions[0]*random.uniform(0, 1), u1.dimensions[1]*random.uniform(0, 1), 
                     u1.dimensions[2]*random.uniform(0, 1)]
        # get random angle
        angle = random.uniform(0, 360)
        # get random unit vector
        vec = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
        vec = vec/(np.linalg.norm(vec))
        # copy universe of added atom
        copy_to_insert = to_insert.copy()
        # apply translation / rotation
        copy_to_insert.atoms.translate(translate)
        copy_to_insert.atoms.rotateby(angle, vec)
        if len(u1.atoms) == 0:
            # merge universe to add atoms, count number of atoms added
            u1 = mda.Merge(copy_to_insert.atoms)
            count_n_mol += 1
            # dimensions of box are reset by merging. Reset box dimensions
            u1.dimensions = [box[0], box[1], box[2], 90, 90, 90]
        else:
            # only add atoms if their distance >= 2*radius. Adjust radius from nm to angstrom
            dist = distance_array(u1.atoms.positions, copy_to_insert.atoms.positions, u1.dimensions)
            min_dist = np.amin(dist)
            if min_dist >= 2*10*radius:
                # merge universe to add atoms, count number of atoms added
                u1 = mda.Merge(u1.atoms, copy_to_insert.atoms)
                count_n_mol += 1
                # dimensions of box are reset by merging. Reset box dimensions
                u1.dimensions = [box[0], box[1], box[2], 90, 90, 90]
        count_n_attempts += 1
        
    # determine if the number of molecules were successfully added:
    if count_n_mol == n_mol:
        print(f'Successfully inserted {n_mol} molecules.')
    else:
        print(f'Could not successfully insert {n_mol} molecules in {count_n_attempts} attempts.')
        print(f'Only added {count_n_mol} molecules. Try increasing the box size or number of attempts to add more molecules.')
    # write the final pdb
    u1.atoms.write(output_pdb)



