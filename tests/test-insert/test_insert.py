import numpy as np
import pandas as pd
import sys
import os
import time

sys.path.append('../..')
from openabc.utils.insert import insert_molecules

n_mol = 100
n_repeat = 3
box = [75, 75, 75]

# test speed
time1 = time.time()
for i in range(n_repeat):
    insert_molecules('hp1alpha_dimer_CA.pdb', 'start1.pdb', n_mol=n_mol, box=box, method='FastNS')
time2 = time.time()
print(f'On average, FastNS method takes {(time2 - time1)/n_repeat} seconds.')

time3 = time.time()
for i in range(n_repeat):
    insert_molecules('hp1alpha_dimer_CA.pdb', 'start2.pdb', n_mol=n_mol, box=box, method='distance_array')
time4 = time.time()
print(f'On average, distance_array method takes {(time4 - time3)/n_repeat} seconds.')

# test with existing pdb
insert_molecules('hp1alpha_dimer_CA.pdb', 'start3.pdb', n_mol=2, box=box, existing_pdb='start1.pdb', method='FastNS')
insert_molecules('hp1alpha_dimer_CA.pdb', 'start4.pdb', n_mol=2, box=box, existing_pdb='start2.pdb', method='distance_array')
