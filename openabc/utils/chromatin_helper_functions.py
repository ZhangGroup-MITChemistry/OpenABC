import numpy as np
import pandas as pd
import sys
import os

"""
Tools for setting up nucleosome related simulations. 
For example, we need functions to remove dihedrals of disordered tails. 
"""

# histone tails (atom index starts from 1): 1-43, 136-159, 238-257, 353-400, 488-530, 623-646, 725-744, 840-887
# in openmm, we use index starting from 0
_histone_tail_start_atoms = np.array([1, 136, 238, 353, 488, 623, 725, 840]) - 1
_histone_tail_end_atoms = np.array([43, 159, 257, 400, 530, 646, 744, 887]) - 1
_histone_tail_atoms = []
for i in range(8):
    _histone_tail_atoms += list(range(_histone_tail_start_atoms[i], _histone_tail_end_atoms[i] + 1))
_histone_tail_atoms = np.array(_histone_tail_atoms)

_n_CA_atoms_per_histone = 974

_histone_core_atoms = np.array([x for x in range(_n_CA_atoms_per_histone) if x not in _histone_tail_atoms])

_n_bp_per_nucl = 147


def remove_histone_tail_dihedrals(df_dihedrals):
    """
    Remove histone tail dihedral potentials from a single histone. 
    
    A dihedral potential is removed if at least one atom involved is within histone tail. 
    
    Parameters
    ----------
    df_dihedrals : pd.DataFrame
        Dihedral potential. This should only include histones. 
        Each histone includes 974 CA atoms, and there should be 974*n CG atoms in all (n is the number of histones). 
    
    Returns
    -------
    new_df_dihedrals : pd.DataFrame
        New dihedral potential.
    
    """
    new_df_dihedrals = pd.DataFrame(columns=df_dihedrals.columns)
    for i, row in df_dihedrals.iterrows():
        a1 = int(row['a1']) % _n_CA_atoms_per_histone
        a2 = int(row['a2']) % _n_CA_atoms_per_histone
        a3 = int(row['a3']) % _n_CA_atoms_per_histone
        a4 = int(row['a4']) % _n_CA_atoms_per_histone
        if not any(x in _histone_tail_atoms for x in [a1, a2, a3, a4]):
            new_df_dihedrals.loc[len(new_df_dihedrals.index)] = row
    return new_df_dihedrals


def remove_histone_tail_native_pairs_and_exclusions(df_native_pairs, df_exclusions):
    """
    Remove histone tail native pair potentials and corresponding exclusions from a single histone. 
    A native pair potential is removed if at least one atom involved is within histone tail. 
    The corresponding exclusions should also be removed if the native pair is removed.
    Also remove native pairs between atoms from different histones. 
    
    Parameters
    ----------
    df_native_pairs : pd.DataFrame
        Native pair potential. This should only include histones. 
        Each histone includes 974 CA atoms, and there should be 974*n CG atoms in all (n is the number of histones). 
    
    df_exclusions : pd.DataFrame
        Nonbonded exclusions.
    
    Returns
    -------
    new_df_native_pairs : pd.DataFrame
        New native pair potential.
    
    new_df_exclusions : pd.DataFrame
        New nonbonded exclusions. 
    
    """
    df1 = df_native_pairs.copy()
    for i, row in df1.iterrows():
        a1 = int(row['a1']) % _n_CA_atoms_per_histone
        a2 = int(row['a2']) % _n_CA_atoms_per_histone
        if any(x in _histone_tail_atoms for x in [a1, a2]):
            df1.loc[i, 'state'] = 'removed'
        elif (int(row['a1']) // _n_CA_atoms_per_histone) != (int(row['a2']) // _n_CA_atoms_per_histone):
            df1.loc[i, 'state'] = 'removed'
        else:
            df1.loc[i, 'state'] = 'kept'
    new_df_native_pairs = df1.loc[df1['state'] == 'kept'].copy().reset_index(drop=True)
    df2 = df1.loc[df1['state'] == 'removed', ['a1', 'a2']].copy().reset_index(drop=True)
    df3 = pd.merge(df2, df_exclusions, how='outer', on=['a1', 'a2'], indicator=True)
    new_df_exclusions = df3.loc[df3['_merge'] == 'right_only', ['a1', 'a2']].copy().reset_index(drop=True)
    return new_df_native_pairs, new_df_exclusions


def get_chromatin_rigid_bodies(n_nucl, nrl, n_rigid_bp_per_nucl=73):
    """
    Get chromatin rigid bodies. 
    The chromatin should possess uniform linker length without additional linkers on both ends. 
    
    Parameters
    ----------
    n_nucl : int
        Nucleosome number. 
    
    nrl : int
        Nucleosome repeat length. 
    
    n_flexible_bp_per_nucl : int
        The number of flexible nucleosomal base pairs for each nucleosome. 
    
    Returns
    -------
    rigid_bodies : list
        List of rigid bodies. 
    
    """
    n_bp = nrl*(n_nucl - 1) + _n_bp_per_nucl
    assert n_rigid_bp_per_nucl > 0
    n_CA_atoms = n_nucl*_n_CA_atoms_per_histone
    n_dna_atoms = 6*n_bp - 2
    n_atoms = n_CA_atoms + n_dna_atoms
    bp_id_to_atom_id_dict = {}
    for i in range(n_bp):
        bp_id_to_atom_id_dict[i] = []
        # first ssDNA chain
        if i == 0:
            bp_id_to_atom_id_dict[i] += (np.arange(2) + n_CA_atoms).tolist()
        else:
            bp_id_to_atom_id_dict[i] += (np.arange(3) + n_CA_atoms + 3*i - 1).tolist()
        if i == (n_bp - 1):
            bp_id_to_atom_id_dict[i] += (np.arange(2) + n_atoms - 3*n_bp + 1).tolist()
        else:
            bp_id_to_atom_id_dict[i] += (np.arange(3) + n_atoms - 3*i - 3).tolist()
    rigid_bodies = []
    for i in range(n_nucl):
        rigid_bodies.append([])
        rigid_bodies[i] += (_histone_core_atoms + i*_n_CA_atoms_per_histone).tolist()
        start_bp_id = int((_n_bp_per_nucl - n_rigid_bp_per_nucl)/2) + i*nrl
        end_bp_id = start_bp_id + n_rigid_bp_per_nucl - 1
        for j in range(start_bp_id, end_bp_id + 1):
            rigid_bodies[i] += bp_id_to_atom_id_dict[j]
        rigid_bodies[i] = sorted(rigid_bodies[i])
    return rigid_bodies
        
        
        

