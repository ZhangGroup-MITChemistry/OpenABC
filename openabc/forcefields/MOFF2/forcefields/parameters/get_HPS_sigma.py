import numpy as np
import pandas as pd
from openabc.forcefields.MOFF2.lib import _amino_acids
import sys
import os

"""
Get the HPS model sigma parameter as a numpy array.
"""

__location__ = os.path.dirname(os.path.abspath(__file__))

df_HPS_Urry_param = pd.read_csv(f'{__location__}/HPS_Urry_parameters.csv')

HPS_sigma = np.zeros((20, 20))

for _, row in df_HPS_Urry_param.iterrows():
    a1 = _amino_acids.index(row['atom_type1'])
    a2 = _amino_acids.index(row['atom_type2'])
    HPS_sigma[a1, a2] = row['sigma']
    HPS_sigma[a2, a1] = row['sigma']




