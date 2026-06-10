import numpy as np
import pandas as pd
import sys
import os
sys.path.append('/home/gridsan/sliu/Projects/TW-PCCG-develop')
from clfftools.lib import _amino_acids


"""
Do some analysis of MJ parameters. 
"""

# load MJ potential parameters and initialize ah_coeff as MJ parameters
df_MJ = pd.read_csv('raw_MJ.csv')
MJ_map = np.zeros((20, 20))
for _, row in df_MJ.iterrows():
    i = _amino_acids.index(row['amino acid1'])
    j = _amino_acids.index(row['amino acid2'])
    MJ_map[i, j] = row['epsilon (RT)']
    MJ_map[j, i] = row['epsilon (RT)']
MJ_map *= -1.0 # important, change to positive values indicating attraction, consistent with hydrophobicity definition
MJ_map = (MJ_map - np.min(MJ_map)) / (np.max(MJ_map) - np.min(MJ_map)) # scale and shift to [0, 1]
MJ_min = -0.5
MJ_max = 0.5
MJ_map = MJ_map * (MJ_max - MJ_min) + MJ_min # scale and shift to [args.MJ_min, args.MJ_max]
rows, cols = np.triu_indices(20, 0)
ah_coeff0 = MJ_map[rows, cols]

print(f'mean ah_coeff0 = {np.mean(ah_coeff0)}')
print(f'std ah_coeff0 = {np.std(ah_coeff0)}')

