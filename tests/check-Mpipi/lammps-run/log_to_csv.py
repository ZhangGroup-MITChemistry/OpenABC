import numpy as np
import pandas as pd

_kcal_to_kj = 4.184

with open('log.lammps', 'r') as f:
    log_lammps_lines = f.readlines()

for i in range(len(log_lammps_lines)):
    line_i = log_lammps_lines[i]
    if 'Step PotEng' in line_i:
        start_line_id = i + 1
    if 'Loop time' in line_i:
        end_line_id = i - 1

data = pd.DataFrame(columns=['sum', 'bond', 'contact', 'elec'])
for i in range(start_line_id, end_line_id + 1):
    line_i = log_lammps_lines[i]
    row = line_i.split()
    row = [_kcal_to_kj*float(x) for x in row][1:]
    data.loc[len(data.index)] = row

data = data[['bond', 'contact', 'elec']].copy()
data.round(2).to_csv('lammps_energy.csv', index=False)


