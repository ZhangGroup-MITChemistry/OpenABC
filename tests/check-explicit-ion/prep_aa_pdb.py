import numpy as np
import sys
import os
sys.path.append('../..')
from openabc.utils.helper_functions import parse_pdb, write_pdb

"""
Prepare all-atom PDB files.
"""

df_atoms = parse_pdb('1apl.pdb')

dna_atoms = df_atoms[df_atoms['chainID'].isin(['A', 'B'])].copy()
dna_atoms['serial'] = np.arange(len(dna_atoms.index)) + 1
write_pdb(dna_atoms, 'dna.pdb')

single_chain_protein_atoms = df_atoms[df_atoms['chainID'] == 'C'].copy()
single_chain_protein_atoms['serial'] = np.arange(len(single_chain_protein_atoms.index)) + 1
write_pdb(single_chain_protein_atoms, 'protein.pdb')


