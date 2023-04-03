import numpy as np
import pandas as pd
import mdtraj
import sys
import os

sys.path.append('../../..')
from openabc.forcefields.parsers import MOFFParser
import openabc.utils.legacy_shadow_map as legacy_shadow_map

# get native pairs with our code
#histone_aa_pdb = 'smog/histone.pdb'
histone_aa_pdb = 'histone.29068.pdb.sb/histone.29068.pdb'
#histone_aa_pdb = 'smog-2.4.5-output/histone.pdb'
#smog_top = 'smog/smog.top'
smog_top = 'histone.29068.pdb.sb/histone.29068.pdb.top'
#smog_top = 'smog-2.4.5-output/smog.top'
protein = MOFFParser.from_atomistic_pdb(histone_aa_pdb, 'histone_CA.pdb')
protein.native_pairs[['a1', 'a2']].to_csv('native_pairs.csv', index=False)

# get legacy native pairs with our code
legacy_native_pairs = legacy_shadow_map.legacy_find_ca_pairs_from_atomistic_pdb(histone_aa_pdb)
legacy_native_pairs[['a1', 'a2']].to_csv('legacy_native_pairs.csv', index=False)

# get native pairs from SMOG output
with open(smog_top, 'r') as f:
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


