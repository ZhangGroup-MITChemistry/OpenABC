import numpy as np
import pandas as pd

"""
Use this script to produce the .csv files for HPS model nonbonded interaction parameters. 
There are two sets of hydrophobicity scales (KR scale and Urry scale). 
"""

_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL']

_normalized_KR_hydropathy_scale = dict(ALA=0.730, ARG=0.000, ASN=0.432, ASP=0.378, 
                                       CYS=0.595, GLN=0.514, GLU=0.459, GLY=0.649, 
                                       HIS=0.514, ILE=0.973, LEU=0.973, LYS=0.514, 
                                       MET=0.838, PHE=1.000, PRO=1.000, SER=0.595, 
                                       THR=0.676, TRP=0.946, TYR=0.865, VAL=0.892)

_normalized_Urry_hydropathy_scale = dict(ALA=0.602942, ARG=0.558824, ASN=0.588236, ASP=0.294119, 
                                         CYS=0.64706, GLN=0.558824, GLU=0.0, GLY=0.57353, 
                                         HIS=0.764707, ILE=0.705883, LEU=0.720589, LYS=0.382354, 
                                         MET=0.676471, PHE=0.82353, PRO=0.758824, SER=0.588236, 
                                         THR=0.588236, TRP=1.0, TYR=0.897059, VAL=0.664707)

_HPS_amino_acid_radius_dict = dict(ALA=0.504, ARG=0.656, ASN=0.568, ASP=0.558, CYS=0.548, 
                                   GLN=0.602, GLU=0.592, GLY=0.450, HIS=0.608, ILE=0.618,
                                   LEU=0.618, LYS=0.636, MET=0.618, PHE=0.636, PRO=0.556,
                                   SER=0.518, THR=0.562, TRP=0.678, TYR=0.646, VAL=0.586)

df_KR_scale = pd.DataFrame(columns=['atom_type1', 'atom_type2', 'sigma', 'lambda'])
for i in range(len(_amino_acids)):
    for j in range(i, len(_amino_acids)):
        atom_type_i = _amino_acids[i]
        atom_type_j = _amino_acids[j]
        sigma_i = _HPS_amino_acid_radius_dict[atom_type_i]
        sigma_j = _HPS_amino_acid_radius_dict[atom_type_j]
        sigma_ij = 0.5*(sigma_i + sigma_j)
        lambda_i = _normalized_KR_hydropathy_scale[atom_type_i]
        lambda_j = _normalized_KR_hydropathy_scale[atom_type_j]
        lambda_ij = 0.5*(lambda_i + lambda_j)
        df_KR_scale.loc[len(df_KR_scale.index)] = [atom_type_i, atom_type_j, sigma_ij, lambda_ij]
df_KR_scale['sigma'] = df_KR_scale['sigma'].round(4)
df_KR_scale['lambda'] = df_KR_scale['lambda'].round(7)
df_KR_scale.to_csv('HPS_KR_parameters.csv', index=False)

df_Urry_scale = pd.DataFrame(columns=['atom_type1', 'atom_type2', 'sigma', 'lambda'])
for i in range(len(_amino_acids)):
    for j in range(i, len(_amino_acids)):
        atom_type_i = _amino_acids[i]
        atom_type_j = _amino_acids[j]
        sigma_i = _HPS_amino_acid_radius_dict[atom_type_i]
        sigma_j = _HPS_amino_acid_radius_dict[atom_type_j]
        sigma_ij = 0.5*(sigma_i + sigma_j)
        lambda_i = _normalized_Urry_hydropathy_scale[atom_type_i]
        lambda_j = _normalized_Urry_hydropathy_scale[atom_type_j]
        lambda_ij = 0.5*(lambda_i + lambda_j)
        df_Urry_scale.loc[len(df_Urry_scale.index)] = [atom_type_i, atom_type_j, sigma_ij, lambda_ij]
df_Urry_scale['sigma'] = df_Urry_scale['sigma'].round(4)
df_Urry_scale['lambda'] = df_Urry_scale['lambda'].round(7)
df_Urry_scale.to_csv('HPS_Urry_parameters.csv', index=False)



