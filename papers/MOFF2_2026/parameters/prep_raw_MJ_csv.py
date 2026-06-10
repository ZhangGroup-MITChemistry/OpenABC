import numpy as np
import pandas as pd
from openabc.lib import _amino_acids
import sys
import os

"""
From a scaled MJ potential file, we try to recover the original MJ file. 
We also need to check if the output is consistent as the original MJ parameters reported in the paper.
Reference: Miyazawa, Sanzo, and Robert L. Jernigan. "Residue–residue potentials with a favorable contact pair term and an unfavorable high packing density term, for simulation and threading." Journal of molecular biology 256.3 (1996): 623-644.
"""

scaled_MJ_csv = 'pp_MJ.csv'
scaled_MJ = pd.read_csv(scaled_MJ_csv)
scaled_MJ_new_index = scaled_MJ.set_index(['atom_type1', 'atom_type2'])
raw_CYS_CYS_MJ_epsilon = -5.44 # in RT unit
scaled_CYS_CYS_MJ_epsilon = scaled_MJ_new_index.loc[('CYS', 'CYS'), 'epsilon (kj/mol)']
scale_factor = raw_CYS_CYS_MJ_epsilon / scaled_CYS_CYS_MJ_epsilon
raw_MJ = pd.DataFrame(columns=['amino acid1', 'amino acid2', 'epsilon (RT)'])
raw_MJ['amino acid1'] = scaled_MJ['atom_type1']
raw_MJ['amino acid2'] = scaled_MJ['atom_type2']
raw_MJ['epsilon (RT)'] = scaled_MJ['epsilon (kj/mol)'] * scale_factor
raw_MJ['epsilon (RT)'] = raw_MJ['epsilon (RT)'].round(2)
raw_MJ.to_csv('raw_MJ.csv', index=False)

# also reformat raw_MJ so we can directly compare with the table shown in the reference
ref_amino_acids = ['CYS', 'MET', 'PHE', 'ILE', 'LEU', 
                   'VAL', 'TRP', 'TYR', 'ALA', 'GLY', 
                   'THR', 'SER', 'ASN', 'GLN', 'ASP', 
                   'GLU', 'HIS', 'ARG', 'LYS', 'PRO']
columns = ['amino acids'] + ref_amino_acids
raw_MJ_ref_table_RT = pd.DataFrame(index=ref_amino_acids, columns=columns)
raw_MJ_ref_table_RT['amino acids'] = ref_amino_acids

for _, row in raw_MJ.iterrows():
    a1 = row['amino acid1']
    a2 = row['amino acid2']
    epsilon = row['epsilon (RT)']
    raw_MJ_ref_table_RT.loc[a1, a2] = epsilon
    raw_MJ_ref_table_RT.loc[a2, a1] = epsilon

# raw_MJ_ref_table_RT can be directly compared with the table shown in the reference
raw_MJ_ref_table_RT.to_csv('raw_MJ_ref_table_RT.csv', index=False)


