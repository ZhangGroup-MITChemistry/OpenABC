import numpy as np
import pandas as pd
import PeptideBuilder
from PeptideBuilder import Geometry
from Bio import PDB

def rotate_align(v1, v2):
    """
    Rotate 3d vector v1 to align to v2.
    v1 and v2 are two points in 3d space. 
    The goal is to compute the rotation along an axis that goes through the origin to align v1 and v2 on the same line.
    
    Parameters
    ----------
    v1 : np.ndarray, shape = (3,)
        The 3d vector to be rotated. 
    
    v2 : np.ndarray, shape = (3,)
        The 3d vector as the reference.
    
    Returns
    -------
    R : np.ndarray, shape = (3, 3)
        The rotation matrix.
    
    """
    assert isinstance(v1, np.ndarray)
    assert isinstance(v2, np.ndarray)
    assert v1.shape == (3,)
    assert v2.shape == (3,)
    rot_axis = np.cross(v1, v2)
    if np.all(rot_axis == 0):
        # x1 and x2 are already aligned
        rot_axis = np.identity(3)
    rot_axis = rot_axis / np.linalg.norm(rot_axis) # normalize
    kx = rot_axis[0]
    ky = rot_axis[1]
    kz = rot_axis[2]
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    # use the matrix form of Rodrigues' rotation formula
    # k is the rotation axis
    # K is the matrix such that k \times v = Kv and k \times (k \times v) = K(Kv)
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    R = np.identity(3) + np.sin(theta) * K + (1 - cos_theta) * np.matmul(K, K)
    return R


def concat_peptide_chains_df(chain1, chain2, chainID='A', r0=1.32, phi=114*np.pi/180, 
                             psi=123*np.pi/180, theta=np.pi):
    """
    Concatenate peptide chains as dataframe.
    The input chain dataframes (i.e. chain1 and chain2) will not be modified by the function. 
    CA1 and C1 represent the last backbone CA and C atoms in chain 1, respectively.
    CA2 and N2 represent the first backbone CA and N atoms in chain 2, respectively.
    
    Parameters
    ----------
    chain1 : pd.DataFrame
        The first chain.
    
    chain2 : pd.DataFrame
        The second chain.
    
    chainID : str
        The chain ID for the concatenated chain.
    
    r0 : float
        The peptide bond (i.e. C1-N2) length in Angstrom. 
    
    phi : float
        The backbone CA1-C1-N2 angle in radians at the junction. 
        phi should be in the range of [0, pi], and it will be clipped to this range. 
    
    psi : float
        The backbone C1-N2-CA2 angle in radians at the junction.
        psi should be in the range of [0, pi], and it will be clipped to this range.
    
    theta : float
        The backbone CA1-C1-N2-CA2 dihedral in radians at the junction. 
    
    Returns
    -------
    c : pd.DataFrame
        The concatenated chain. 
    
    """
    assert len(chain1['chainID'].unique()) == 1
    assert len(chain2['chainID'].unique()) == 1
    # make copies for chain1 and chain2
    c1 = chain1.copy()
    c2 = chain2.copy()
    c1['resSeq'] = c1['resSeq'] - c1['resSeq'].min() + 1
    # remove OXT in c1
    flag = c1['name'] != 'OXT' 
    c1 = c1.loc[flag]
    c1['chainID'] = 1
    c2['resSeq'] = c2['resSeq'] - c2['resSeq'].min() + 1
    c2['chainID'] = 2
    # concatenate two chains
    # first compute the rotation matrix to align the chains
    n_res1 = c1['resSeq'].nunique()
    n_res2 = c2['resSeq'].nunique()
    assert n_res1 == c1['resSeq'].max()
    assert n_res2 == c2['resSeq'].max()
    c1_last_CA = c1.loc[(c1['name'] == 'CA') & (c1['resSeq'] == n_res1)]
    c1_last_C = c1.loc[(c1['name'] == 'C') & (c1['resSeq'] == n_res1)]
    c2_first_CA = c2.loc[(c2['name'] == 'CA') & (c2['resSeq'] == 1)]
    c2_first_N = c2.loc[(c2['name'] == 'N') & (c2['resSeq'] == 1)]
    # peptide bond (i.e. backbone C-N bond) length is about 1.32 Angstrom
    # chain 1 last CA and C are CA1 and C1
    # chain 2 first CA and N are CA2 and N2
    # backbone CA1-C1-N2 angle is phi
    # backbone C1-N2-CA2 angle is psi
    # backbone CA1-C1-N2-CA2 dihedral is theta
    CA1 = c1_last_CA[['x', 'y', 'z']].to_numpy().flatten().copy()
    C1 = c1_last_C[['x', 'y', 'z']].to_numpy().flatten().copy()
    N2 = c2_first_N[['x', 'y', 'z']].to_numpy().flatten().copy()
    CA2 = c2_first_CA[['x', 'y', 'z']].to_numpy().flatten().copy()
    c1_coords = c1[['x', 'y', 'z']].to_numpy().copy()
    c2_coords = c2[['x', 'y', 'z']].to_numpy().copy()
    # translate chain 1 so that C1 is at the origin
    t = (-C1).copy()
    c1_coords += t
    C1 += t
    CA1 += t
    # rotate chain 1 so that CA1 is at the negative x-axis
    R = rotate_align(CA1, np.array([-1, 0, 0]))
    c1_coords = (np.matmul(R, c1_coords.T)).T
    C1 = np.dot(R, C1)
    CA1 = np.dot(R, CA1)
    # check phi and psi range
    assert 0 <= phi <= np.pi
    assert 0 <= psi <= np.pi
    # translate chain 2 so that N2 is at (-r0*cos(phi), r0*sin(phi), 0)
    t = (r0 * np.array([-np.cos(phi), np.sin(phi), 0]) - N2).copy()
    c2_coords += t
    N2 += t
    CA2 += t
    # translate both chains together so that N2 is at the origin
    t = (-N2).copy()
    c1_coords += t
    c2_coords += t
    C1 += t
    CA1 += t
    N2 += t
    CA2 += t
    # rotate chain 2 so that CA2 is aligned to (x1, y1, z1)
    # x1 = cos(psi) * cos(phi) - sin(psi) * cos(theta) * sin(phi)
    # y1 = -cos(psi) * sin(phi) - sin(psi) * cos(theta) * cos(phi)
    # z1 = sin(psi) * sin(theta)
    x1 = np.cos(psi) * np.cos(phi) - np.sin(psi) * np.cos(theta) * np.sin(phi)
    y1 = -np.cos(psi) * np.sin(phi) - np.sin(psi) * np.cos(theta) * np.cos(phi)
    z1 = np.sin(psi) * np.sin(theta)
    v = np.array([x1, y1, z1])
    R = rotate_align(CA2, v)
    c2_coords = (np.matmul(R, c2_coords.T)).T
    N2 = np.dot(R, N2)
    CA2 = np.dot(R, CA2)
    # concatenate
    c = pd.concat([c1, c2], ignore_index=True)
    c[['x', 'y', 'z']] = np.concatenate((c1_coords, c2_coords), axis=0)
    # update resSeq
    r = 1
    new_resSeq = []
    for i in range(len(c.index)):
        if i == 0:
            new_resSeq.append(r)
        else:
            flag1 = c.loc[i, 'chainID'] != c.loc[i - 1, 'chainID']
            flag2 = c.loc[i, 'resSeq'] != c.loc[i - 1, 'resSeq']
            if flag1 or flag2:
                r += 1
            new_resSeq.append(r)
    c['resSeq'] = new_resSeq
    # update serial
    c['serial'] = 1 + np.arange(len(c.index))
    # update chainID
    c.loc[:, 'chainID'] = chainID
    return c


def compute_angle(coord1, coord2, coord3):
    assert isinstance(coord1, np.ndarray)
    assert isinstance(coord2, np.ndarray)
    assert isinstance(coord3, np.ndarray)
    assert coord1.shape == (3,)
    assert coord2.shape == (3,)
    assert coord3.shape == (3,)
    v1 = coord1 - coord2
    v2 = coord3 - coord2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    return theta


def compute_dihedral(coord1, coord2, coord3, coord4):
    assert isinstance(coord1, np.ndarray)
    assert isinstance(coord2, np.ndarray)
    assert isinstance(coord3, np.ndarray)
    assert isinstance(coord4, np.ndarray)
    assert coord1.shape == (3,)
    assert coord2.shape == (3,)
    assert coord3.shape == (3,)
    assert coord4.shape == (3,)
    v1 = coord1 - coord2
    v2 = coord3 - coord2
    v3 = coord4 - coord3
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    cos_omega = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    omega = np.arccos(np.clip(cos_omega, -1, 1))
    theta = np.pi - omega
    # determine the sign of the dihedral angle
    if np.dot(np.cross(n1, n2), v2) < 0:
        sign = 1
    else:
        sign = -1
    theta *= sign
    return theta


def build_chain(seq, phi, psi_im1, output_pdb):
    """
    Build peptide chain with given phi and psi_im1 angles. 
    
    Parameters
    ----------
    seq : str or array-like
        The sequence of amino acids.
    
    phi : float or array-like
        The phi angle for each amino acid.
        If phi is a scalar, then all amino acids will have the same phi angle.
        If phi is an array-like, then it should have the same length as seq.
    
    psi_im1 : float or array-like
        The psi_im1 angle for each amino acid.
        If psi_im1 is a scalar, then all amino acids will have the same psi_im1 angle.
        If psi_im1 is an array-like, then it should have the same length as seq.
    
    output_pdb : str
        The output PDB file name.
    
    """
    if isinstance(phi, (int, float)):
        phi = phi * np.ones(len(seq))
    if isinstance(psi_im1, (int, float)):
        psi_im1 = psi_im1 * np.ones(len(seq))
    assert len(seq) == len(phi)
    assert len(seq) == len(psi_im1)
    for i, each in enumerate(seq):
        geo = Geometry.geometry(each)
        geo.phi = phi[i]
        geo.psi_im1 = psi_im1[i]
        if i == 0:
            structure = PeptideBuilder.initialize_res(geo)
        else:
            PeptideBuilder.add_residue(structure, geo)
    PeptideBuilder.add_terminal_OXT(structure)
    output = PDB.PDBIO()
    output.set_structure(structure)
    output.save(output_pdb)


