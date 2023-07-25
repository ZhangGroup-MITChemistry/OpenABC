import numpy as np
import pandas as pd
import sys
import os

with open('log.lammps.0', 'r') as f:
    log_lines = f.readlines()

n_lines = len(log_lines)

for i in range(n_lines):
    line_i = log_lines[i]
    if len(line_i) >= 4 and line_i[:4] == 'Step':
        # drop step 0 result, as lammps gives strange values
        start_line_id = i + 2
    if len(line_i) >= 4 and line_i[:4] == 'Loop':
        end_line_id = i - 1

columns = ['dna base pair', 'dna cross stacking', 'dna vdwl', 'dna elec', 'dna nbp', 'dna bond', 
           'protein bond', 'dna stacking', 'dna angle', 'protein angle', 'dna dihedral', 'protein dihedral', 
           'native pair', 'PD and PP nonbonded', 'PD and PP vdwl', 'PD and PP elec', 'all vdwl', 'all elec']
data = []
for i in range(start_line_id, end_line_id + 1):
    values = log_lines[i].split()[2:]
    values = [float(x) for x in values]
    data += [values]
data = pd.DataFrame(np.array(data), columns=columns)
new_columns = ['protein bond', 'protein angle', 'protein dihedral', 'native pair', 'dna bond',
               'dna angle', 'dna stacking', 'dna dihedral', 'dna base pair', 'dna cross stacking',
               'all vdwl', 'all elec']
data = data[new_columns].copy()
data.round(2).to_csv('lammps_energy_kcal.csv', index=False)

data.loc[:, data.columns] *= 4.184
data.round(2).to_csv('lammps_energy_kj.csv', index=False)

