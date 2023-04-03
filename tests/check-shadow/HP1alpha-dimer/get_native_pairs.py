import numpy as np
import pandas as pd
import mdtraj
import sys
import os

sys.path.append('../../..')
from openabc.forcefields.parsers import MOFFParser
import openabc.utils.legacy_shadow_map as legacy_shadow_map

# get native pairs with our code
protein = MOFFParser.from_atomistic_pdb('hp1a.pdb', 'hp1alpha_dimer_CA.pdb')
protein.native_pairs[['a1', 'a2']].to_csv('native_pairs.csv', index=False)

# get legacy native pairs with our code
legacy_native_pairs = legacy_shadow_map.legacy_find_ca_pairs_from_atomistic_pdb('hp1a.pdb')
legacy_native_pairs[['a1', 'a2']].to_csv('legacy_native_pairs.csv', index=False)

# get native pairs from SMOG online server output
with open('hp1a_dimer.29065.pdb.sb/hp1a_dimer.29065.pdb.top', 'r') as f:
    top_lines = f.readlines()

flag = False
for i in range(len(top_lines)):
    if '[ pairs ]' in top_lines[i]:
        start_line_id = i + 2
        flag = True
        continue
    if flag and ('[' in top_lines[i]):
        end_line_id = i - 1
        break

smog_pairs = []
for i in range(start_line_id, end_line_id + 1):
    row = top_lines[i].split()
    if len(row) >= 1:
        a1, a2 = int(row[0]) - 1, int(row[1]) - 1
        smog_pairs.append([a1, a2])
df_smog_pairs = pd.DataFrame(np.array(smog_pairs), columns=['a1', 'a2'])
df_smog_pairs.to_csv('smog_native_pairs.csv', index=False)


