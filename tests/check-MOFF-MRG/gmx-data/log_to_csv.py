import numpy as np
import pandas as pd

'''
Convert the energies in GROMACS log file to csv file. 
'''

with open('rerun.log', 'r') as f:
    rerun_log_lines = f.readlines()

energy_line_indices = []
for i in range(len(rerun_log_lines)):
    line_i = rerun_log_lines[i]
    if 'Energies (kJ/mol)' in line_i:
        energy_line_indices.append([i + 2, i + 4, i + 6])

df_energy = pd.DataFrame(columns=['protein bond', 'protein angle', 'protein dihedral', 'native pair', 'dna bond and fan bond', 'dna angle', 'contact', 'elec switch', 'sum'])
n_snapshots = len(energy_line_indices)
for i in range(n_snapshots):
    row = []
    elements1 = rerun_log_lines[energy_line_indices[i][0]].split()
    elements1 = [float(x) for x in elements1]
    elements2 = rerun_log_lines[energy_line_indices[i][1]].split()
    elements2 = [float(x) for x in elements2]
    elements3 = rerun_log_lines[energy_line_indices[i][2]].split()
    elements3 = [float(x) for x in elements3]
    row += [elements1[0], elements1[3], elements2[0], elements2[1]]
    row += [elements1[1] + elements1[2], elements1[4]]
    row += [elements2[3], elements2[2] + elements2[4]]
    row += [elements3[0]]
    df_energy.loc[len(df_energy.index)] = row

df_energy = df_energy.round(6)
df_energy.to_csv('gmx_energies.csv', index=False)

