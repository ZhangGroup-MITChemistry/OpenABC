import numpy as np
import pandas as pd
import sys
import os

'''
Convert HOOMD-Blue output energy log file to csv. 
'''

with open('energy_KR.log', 'r') as f1:
    energy_KR_lines = f1.readlines()[1:]

with open('energy_Urry.log', 'r') as f2:
    energy_Urry_lines = f2.readlines()[1:]

data = []
tolerance = 1e-6
for i in range(len(energy_KR_lines)):
    row1 = energy_KR_lines[i].split()
    row1 = [float(x) for x in row1]
    row2 = energy_Urry_lines[i].split()
    row2 = [float(x) for x in row2]
    if (len(row1) > 0) and (len(row2) > 0):
        assert abs(row1[1] - row2[1]) < tolerance # ensure bond energies are the same
        assert abs(row1[2] - row2[2]) < tolerance # ensure electrostatic energies are the same
        row = row1[1:4] + [row2[3]]
        data.append(row)
data = np.array(data)
df_energy = pd.DataFrame(data, columns=['bond', 'elec', 'contact (KR)', 'contact (Urry)'])
df_energy = df_energy[['bond', 'contact (Urry)', 'contact (KR)', 'elec']]
df_energy.round(10).to_csv('DDX4_hoomd_energy.csv', index=False)


