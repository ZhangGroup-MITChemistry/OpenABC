import numpy as np
import pandas as pd
import sys
import os

sys.path.append('/home/gridsan/sliu/Projects/smog-3spn2-openmm')
import OpenSMOG3SPN2.utils.helper_functions as helper_functions

amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
               'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO',
               'SER', 'THR', 'TRP', 'TYR', 'VAL']

nucl = helper_functions.parse_pdb('1kx5.pdb')
nucl = nucl.loc[nucl['recname'] == 'ATOM'].copy()

dna = nucl.loc[nucl['resname'].isin(['DA', 'DT', 'DC', 'DG'])].copy()
protein = nucl.loc[nucl['resname'].isin(amino_acids)].copy()

dna['serial'] = list(range(len(dna.index)))
protein['serial'] = list(range(len(protein.index)))

helper_functions.write_pdb(dna, 'dna.pdb')
helper_functions.write_pdb(protein, 'histone.pdb')



