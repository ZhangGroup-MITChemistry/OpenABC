import numpy as np
import pandas as pd
import sys
import os
import time

sys.path.insert(0, '../..') # ensure use the specific openabc we aim to test
from openabc.utils.insert import insert_molecules

# test orthogonal box
box1 = [75, 75, 75, 90.0, 90.0, 90.0]
n_copies = 100
time1 = time.time()
insert_molecules('hp1alpha_dimer_CA.pdb', n_copies, 'start1.pdb', box=box1)
time2 = time.time()
print(f'insert_molecules takes {time2 - time1} seconds.')

# test with existing pdb
n_copies = 10
time1 = time.time()
insert_molecules('hp1alpha_dimer_CA.pdb', n_copies, 'start2.pdb', existing_pdb='start1.pdb', box=box1)
time2 = time.time()
print(f'insert_molecules takes {time2 - time1} seconds.')

# test with non-orthogonal box
box1 = [75, 75, 75, 60.0, 60.0, 60.0]
n_copies = 100
time1 = time.time()
insert_molecules('hp1alpha_dimer_CA.pdb', n_copies, 'start3.pdb', box=box1)
time2 = time.time()
print(f'insert_molecules takes {time2 - time1} seconds.')

