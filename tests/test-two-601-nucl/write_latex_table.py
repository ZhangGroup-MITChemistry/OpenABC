import numpy as np
import pandas as pd

columns = ['protein bond', 'protein angle', 'dna bond', 'dna angle', 'dna stacking', 'dna dihedral', 
           'dna base pair', 'dna cross stacking', 'all vdwl', 'all elec']

df1 = pd.read_csv('lammps-rerun/lammps_energy_kcal.csv')
df2 = pd.read_csv('openmm-rerun/openmm_energy_kcal.csv')
df3 = pd.read_csv('openmm-rerun-old-geometry/openmm_energy_kcal.csv')

c = ['protein bond', 'protein angle', 'dna stacking', 'dna base pair', 'dna cross stacking', 'all vdwl', 'all elec']
df3.loc[:, c] = ''

for i in range(len(df1.index)):
    line1 = f'    {i + 1} & LAMMPS & ' + ' & '.join(['%.2f' % x for x in df1.loc[i, columns].tolist()]) + r' \\'
    line2 = f'    {i + 1} & OpenMM & ' + ' & '.join(['%.2f' % x for x in df2.loc[i, columns].tolist()]) + r' \\'
    print(line1)
    print(line2)
    print('    \hline')
    

