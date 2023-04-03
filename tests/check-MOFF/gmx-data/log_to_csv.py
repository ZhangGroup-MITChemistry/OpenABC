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
        energy_line_indices.append([i + 2, i + 4])

df_energy = pd.DataFrame(columns=['bond', 'angle', 'dihedral', 'native pair', 'contact', 'elec switch', 'sum'])
n_snapshots = len(energy_line_indices)
for i in range(n_snapshots):
    elements = rerun_log_lines[energy_line_indices[i][0]].split()
    elements = [float(x) for x in elements]
    bond_energy = elements[0]
    angle_energy = elements[1]
    dihedral_energy = elements[2]
    native_pair_energy = elements[3]
    native_pair_elec_energy = elements[4] # involve electrostatic interactions between native pairs
    elements = rerun_log_lines[energy_line_indices[i][1]].split()
    elements = [float(x) for x in elements]
    contact_energy = elements[0]
    elec_energy = elements[1] + native_pair_elec_energy
    row = [bond_energy, angle_energy, dihedral_energy, native_pair_energy, contact_energy, elec_energy]
    sum_energy = np.sum(np.array(row))
    row.append(sum_energy)
    df_energy.loc[len(df_energy.index)] = row
df_energy = df_energy.round(6)
df_energy.to_csv('gmx_energies.csv', index=False)

