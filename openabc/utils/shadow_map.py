import numpy as np
import pandas as pd
import mdtraj
from MDAnalysis.lib.nsgrid import FastNS
import networkx as nx
import math
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

__author__ = 'Shuming Liu'


'''
A python implementation of shadow map algorithm. The related reference is: 

Noel, Jeffrey K., Paul C. Whitford, and José N. Onuchic. "The shadow map: a general contact definition for capturing the dynamics of biomolecular folding and function." The journal of physical chemistry B 116.29 (2012): 8692-8702.

This algorithm is applied to find contacts between residues (a residue can be an amino acid or a nucleotide)

This script also includes some useful functions. 
'''

_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL']


def get_neighbor_pairs_and_distances(coord, cutoff=0.6, box=None, use_pbc=False):
    '''
    Find spatially neighboring atom pairs within cutoff range based on MDAnalysis. 
    
    Parameters
    ----------
    coord : np.ndarray, shape = (n_atoms, 3)
        Input atom coordinates. 
    
    cutoff : float or int (default = 0.6)
        Cutoff distance for searching neighboring atom pairs. 
        Note `coord` and `cutoff` share the same length unit. 
    
    box : None or np.ndarray
        Specify box shape. 
        Note `use_pbc` = False is only compatible with orthogonal box. 
        If `use_pbc` = False, then set `box` as None. 
        If `use_pbc` = True, then `box` = np.array([lx, ly, lz, alpha1, alpha2, alpha3]). 
    
    use_pbc : bool
        Whether to use periodic boundary condition (PBC). 
    
    Returns
    -------
    neighbor_pairs : np.ndarray, shape = (n_pairs, 2)
        Neighboring atom pairs.
    
    neighbor_pair_distances: np.ndarray, shape = (N,)
        Neighboring atom pair distances. 
    
    
    References
    ----------
    https://docs.mdanalysis.org/1.1.1/documentation_pages/lib/nsgrid.html
    
    '''
    if use_pbc:
        grid_search = FastNS(cutoff, coord.astype(np.float32), box.astype(np.float32), use_pbc)
    else:
        x_min, x_max = np.amin(coord[:, 0]), np.amax(coord[:, 0])
        y_min, y_max = np.amin(coord[:, 1]), np.amax(coord[:, 1])
        z_min, z_max = np.amin(coord[:, 2]), np.amax(coord[:, 2])
        shifted_coord = coord.copy() - np.array([x_min, y_min, z_min])
        shifted_coord = shifted_coord.astype(np.float32)
        lx = max(1.1*(x_max - x_min), 2.1*cutoff)
        ly = max(1.1*(y_max - y_min), 2.1*cutoff)
        lz = max(1.1*(z_max - z_min), 2.1*cutoff)
        pseudo_box = np.array([lx, ly, lz, 90, 90, 90]).astype(np.float32) # build an orthogonal pseudo box
        grid_search = FastNS(cutoff, shifted_coord, pseudo_box, use_pbc)
    results = grid_search.self_search()
    neighbor_pairs = results.get_pairs()
    neighbor_pair_distances = results.get_pair_distances()
    return neighbor_pairs, neighbor_pair_distances


def get_bonded_neighbor_dict(atomistic_pdb):
    '''
    Find atoms that are directly connected by bonds based on MDTraj.
    
    Parameters
    ----------
    atomistic_pdb : str
        Path for the atomistic pdb file. 
    
    Returns
    -------
    bonded_neighbor_dict : dict
        A dictionary that the value is a list including all the atoms bonded to the atom of index key. 
    
    References
    ----------
    https://mdtraj.org/1.9.4/api/generated/mdtraj.Topology.html
    
    '''
    traj = mdtraj.load_pdb(atomistic_pdb)
    top = traj.topology
    bond_graph = top.to_bondgraph()
    bonded_neighbor_dict = {}
    for a1 in list(bond_graph.nodes):
        bonded_neighbor_dict[a1.index] = []
        for a2 in bond_graph.neighbors(a1):
            bonded_neighbor_dict[a1.index].append(a2.index)
    return bonded_neighbor_dict


def light_is_blocked(d12, d13, d23, r2, r3):
    '''
    Check if the light from atom1 to atom2 is blocked by atom3. 
    
    Parameters
    ----------
    d12: float or int, positive
        The distance between atom1 and atom2. 
    
    d13: float or int, positive
        The distance between atom1 and atom3. 
        
    d23: float or int, positive
        The distance between atom2 and atom3. 
    
    r2: float or int, non-negative
        The radius of atom2. 
        Require r2 <= d12. 
    
    r3: float or int, non-negative
        The radius of atom3. 
        Require r3 <= d13. 
    
    Returns
    -------
    flag : bool
        Whether the light is blocked. 
        
    '''
    assert d12 > 0
    assert d13 > 0
    assert d23 > 0
    assert r2 >= 0
    assert r3 >= 0
    assert r2 <= d12
    assert r3 <= d13
    angle213 = math.acos((d12**2 + d13**2 - d23**2)/(2*d12*d13)) # law of cosines
    theta12 = math.asin(r2/d12)
    theta13 = math.asin(r3/d13)
    if theta12 + theta13 >= angle213:
        flag = True
    else:
        flag = False
    return flag


def find_res_pairs_from_atomistic_pdb(atomistic_pdb, frame=0, radius=0.1, bonded_radius=0.05, cutoff=0.6, box=None, 
                                      use_pbc=False):
    '''
    Find native pairs between residues following the shadow algorithm.
    They algorithm only searches native pairs between residues that do not have 1-2, 1-3, or 1-4 interactions. 
    If two heavy atoms from different residues are in contact, then the two residues are in contact.
    
    Parameters
    ----------
    atomistic_pdb : str
        Path for the atomistic pdb file. 
    
    frame : int
        PDB frame index used for searching pairs. 
    
    radius : float or int
        Shadow radius for candidate pair atoms and blocking atoms if not bonded to any pair atom.
    
    bonded_radius : float or int
        Shadow radius if the candidate blocking atom is directly bonded to any candidate pair atom. 
    
    cutoff: float or int
        Cutoff distance for searching neighboring atom pairs. 

    box : None or np.ndarray
        Specify box shape. 
        Note `use_pbc` = False is only compatible with orthogonal box. 
        If `use_pbc` = False, then set `box` as None. 
        If `use_pbc` = True, then `box` = np.array([lx, ly, lz, alpha1, alpha2, alpha3]). 
    
    use_pbc : bool
        Whether to use periodic boundary condition (PBC). 

    Returns
    -------
    res_pairs : np.ndarray, shape = (n_pairs, 2)
        Residue pairs found by the shadow algorithm.
    
    df_atoms : pd.DataFrame
        Pandas DataFrame including atom information from file `atomistic_pdb`. 
    
    '''
    traj = mdtraj.load_pdb(atomistic_pdb)
    top = traj.topology
    n_atoms = top.n_atoms
    df_atoms, _bonds = top.to_dataframe()
    # in df_atoms, each value in columns 'serial', 'resSeq', and 'chainID' is an integer
    df_atoms.index = list(range(len(df_atoms.index)))
    # set unique_resSeq, which starts from 0
    unique_resSeq = 0
    for i, row in df_atoms.iterrows():
        if i >= 1:
            if (row['resSeq'] != df_atoms.loc[i - 1, 'resSeq']) or (row['chainID'] != df_atoms.loc[i - 1, 'chainID']):
                unique_resSeq += 1
        df_atoms.loc[i, 'unique_resSeq'] = unique_resSeq
    df_atoms['unique_resSeq'] = df_atoms['unique_resSeq'].astype('int64')
    neighbors_no_hyd_dict = {}
    for i in range(n_atoms):
        if df_atoms.loc[i, 'element'] != 'H':
            neighbors_no_hyd_dict[i] = []
    coord = traj.xyz[frame]
    neighbor_atom_pairs, neighbor_atom_pair_distances = get_neighbor_pairs_and_distances(coord, cutoff, box, use_pbc)
    dist_matrix = np.zeros((n_atoms, n_atoms))
    dist_matrix[:, :] = -1 # initialize distance matrix elements as -1
    beyond_1_4_neighbors_no_hyd_atom_pairs = []
    for i in range(len(neighbor_atom_pairs)):
        a1, a2 = int(neighbor_atom_pairs[i, 0]), int(neighbor_atom_pairs[i, 1])
        if a1 > a2:
            a1, a2 = a2, a1
        if (df_atoms.loc[a1, 'element'] != 'H') and (df_atoms.loc[a2, 'element'] != 'H'):
            # save the distances of all the heavy atom neighboring pairs into dist_matrix
            dist_matrix[a1, a2] = neighbor_atom_pair_distances[i]
            dist_matrix[a2, a1] = neighbor_atom_pair_distances[i]
            neighbors_no_hyd_dict[a1].append(a2)
            neighbors_no_hyd_dict[a2].append(a1)
            chain1, chain2 = df_atoms.loc[a1, 'chainID'], df_atoms.loc[a2, 'chainID']
            resSeq1, resSeq2 = df_atoms.loc[a1, 'resSeq'], df_atoms.loc[a2, 'resSeq']
            if (chain1 != chain2) or abs(resSeq1 - resSeq2) > 3:
                # two atoms should be from different residues that are not in 1-2, 1-3, or 1-4 interactions
                beyond_1_4_neighbors_no_hyd_atom_pairs.append([a1, a2])
    res_pairs = []
    bonded_neighbor_dict = get_bonded_neighbor_dict(atomistic_pdb)
    for each in beyond_1_4_neighbors_no_hyd_atom_pairs:
        # only check heavy atom pairs that are from residues beyond 1-4 interactions
        a1, a2 = each[0], each[1]
        unique_resSeq1, unique_resSeq2 = df_atoms.loc[a1, 'unique_resSeq'], df_atoms.loc[a2, 'unique_resSeq']
        # since a1 < a2 and their residues are beyond 1-4 interactions, thus unique_resSeq1 < unique_resSeq2
        if [unique_resSeq1, unique_resSeq2] in res_pairs:
            continue
        d12 = dist_matrix[a1, a2]
        assert d12 > 0
        if d12 < radius:
            print(f'Distance between atom {a1} and {a2} is {d12} nm, which is smaller than the radius ({radius} nm), so we ignore this atom pair. This means maybe the radius is too large or atoms {a1} and {a2} are too close.')
            continue
        r1, r2 = radius, radius
        flag = True
        # test if atom a3 blocks the contact between atom a1 and a2
        # only test atom a3 if d13 < d12 and d23 < d12
        # a1 and a2 have contact, d12 < cutoff, so d13 < cutoff and d23 < cutoff
        # so atom a3 has to be the neighbor of both atom a1 and a2
        block_candidates = [a3 for a3 in neighbors_no_hyd_dict[a1] if a3 in neighbors_no_hyd_dict[a2]]
        for a3 in block_candidates:
            # check if a3 blocks the light from a1 to a2 or the light from a2 to a1
            # (a1, a3) and (a2, a3) are both neighboring pairs, so d13 and d23 can be directly read from dist_matrix
            d13, d23 = dist_matrix[a1, a3], dist_matrix[a2, a3]
            assert d13 > 0
            assert d23 > 0
            if (d12 <= d13) or (d12 <= d23):
                continue
            # if a3 is bonded to a1 or a2, then set r3 as bonded_radius
            if (a3 in bonded_neighbor_dict[a1]) or (a3 in bonded_neighbor_dict[a2]):
                r3 = bonded_radius
            else:
                r3 = radius
            # if r3 is larger than d13 or d23, then recognize that a3 blocks the contact between a1 and a2
            if (r3 > d13) or (r3 > d23):
                flag = False
                break
            elif light_is_blocked(d12, d13, d23, r2, r3) or light_is_blocked(d12, d23, d13, r1, r3):
                flag = False
                break
        if flag:
            res_pairs.append([unique_resSeq1, unique_resSeq2])
    res_pairs = np.array(sorted(res_pairs))
    return res_pairs, df_atoms


def find_ca_pairs_from_atomistic_pdb(atomistic_pdb, frame=0, radius=0.1, bonded_radius=0.05, cutoff=0.6, box=None, 
                                     use_pbc=False):
    '''
    Find protein CA atom pairs whose residues are in contact.
    
    Parameters
    ----------
    atomistic_pdb : str
        Path for the atomistic pdb file. 
    
    frame : int
        PDB frame index used for searching pairs. 
    
    radius : float or int
        Shadow radius for candidate pair atoms and blocking atoms if not bonded to any pair atom.
    
    bonded_radius : float or int
        Shadow radius if the candidate blocking atom is directly bonded to any candidate pair atom. 
    
    cutoff: float or int
        Cutoff distance for searching neighboring atom pairs. 

    box : None or np.ndarray
        Specify box shape. 
        Note `use_pbc` = False is only compatible with orthogonal box. 
        If `use_pbc` = False, then set `box` as None. 
        If `use_pbc` = True, then `box` = np.array([lx, ly, lz, alpha1, alpha2, alpha3]). 
    
    use_pbc : bool
        Whether to use periodic boundary condition (PBC). 
    
    Returns
    -------
    df_ca_pairs : pd.DataFrame
        Pandas DataFrame including CA atom pair indices and distances. 
    
    '''
    res_pairs, df_atoms = find_res_pairs_from_atomistic_pdb(atomistic_pdb, frame, radius, bonded_radius, cutoff, box, 
                                                            use_pbc)
    # pick out CA atoms for each residue
    dict_res_CA = {}
    for i, row in df_atoms.iterrows():
        if (row['resName'] in _amino_acids) and (row['name'] == 'CA'):
            unique_resSeq = df_atoms.loc[i, 'unique_resSeq']
            dict_res_CA[unique_resSeq] = i
    df_ca_pairs = pd.DataFrame(columns=['a1', 'a2', 'mu'])
    ca_atom_pairs = []
    for each in res_pairs:
        a1, a2 = int(each[0]), int(each[1])
        if (a1 in dict_res_CA) and (a2 in dict_res_CA):
            df_ca_pairs.loc[len(df_ca_pairs.index)] = [a1, a2, None]
            ca_atom_pairs.append([dict_res_CA[a1], dict_res_CA[a2]])
    ca_atom_pairs = np.array(ca_atom_pairs)
    traj = mdtraj.load_pdb(atomistic_pdb)
    df_ca_pairs['mu'] = mdtraj.compute_distances(traj, ca_atom_pairs, use_pbc)[frame]
    return df_ca_pairs


def load_ca_pairs_from_gmx_top(top_file, ca_pdb, frame=0, use_pbc=False):
    '''
    Load native pairs from GROMACS topology file. 
    Note in GROMACS, atom indices start from 1, while in OpenMM, atom indices start from 0. 
    
    Parameters
    ----------
    top_file : str
        Path for the GROMACS topology file. 
    
    ca_pdb : str
        Path for the CA model PDB file. 
    
    frame: int
        PDB frame index used for computing pair distances. 
    
    use_pbc : bool
        Whether to use periodic boundary condition (PBC). 
    
    Returns
    -------
    df_ca_pairs : pd.DataFrame
        Pandas DataFrame including CA atom pair indices and distances. 
    
    '''
    with open(top_file, 'r') as input_reader:
        top_file_lines = input_reader.readlines()
    flag = False
    for i in range(len(top_file_lines)):
        if '[ pairs ]' in top_file_lines[i]:
            start_index = i + 2
            flag = True
            continue
        if flag and ('[' in top_file_lines[i]):
            end_index = i - 1
            break
    ca_pairs = []
    for i in range(start_index, end_index + 1):
        elements = top_file_lines[i].split()
        if len(elements) >= 1:
            a1 = int(elements[0]) - 1
            a2 = int(elements[1]) - 1
            if a1 > a2: 
                a1, a2 = a2, a1
            ca_pairs.append([a1, a2])
    ca_pairs = np.array(sorted(ca_pairs))
    df_ca_pairs = pd.DataFrame(ca_pairs, columns=['a1', 'a2'])
    traj = mdtraj.load_pdb(ca_pdb)
    df_ca_pairs['mu'] = mdtraj.compute_distances(traj, ca_pairs, use_pbc)[frame]
    return df_ca_pairs



