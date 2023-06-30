import numpy as np
import pandas as pd
import sys
import os

"""
Use this script to get Mpipi parameters saved as .csv files. 
The original LAMMPS parameters are saved in lammps_Mpipi_RNA.in. 
"""

_kcal_to_kj = 4.184
_angstrom_to_nm = 0.1

# the Mpipi protein and RNA nonbonded parameters are listed in lammps_Mpipi_RNA.in
_mpipi_name_to_type_dict = dict(MET=1, GLY=2, LYS=3, THR=4, ARG=5, 
                                ALA=6, ASP=7, GLU=8, TYR=9, VAL=10, 
                                LEU=11, GLN=12, TRP=13, PHE=14, SER=15, 
                                HIS=16, ASN=17, PRO=18, CYS=19, ILE=20, 
                                RNA_A=41, RNA_C=42, RNA_G=43, RNA_U=44)
_mpipi_type_to_name_dict = {int(v): k for k, v in _mpipi_name_to_type_dict.items()}

with open('lammps_Mpipi_RNA.in', 'r') as f:
    lines = f.readlines()

df_Mpipi_parameters = pd.DataFrame(columns=['atom_type1', 'atom_type2', 'epsilon', 'sigma', 'mu', 'nu'])
n_lines = len(lines)
for i in range(n_lines):
    line_i = lines[i]
    if ('pair_coeff' in line_i) and ('wf/cut' in line_i):
        elements = line_i.split()
        atom_type1 = int(elements[1])
        atom_type2 = int(elements[2])
        if (atom_type1 in _mpipi_type_to_name_dict) and (atom_type2 in _mpipi_type_to_name_dict):
            atom_name1 = _mpipi_type_to_name_dict[atom_type1]
            atom_name2 = _mpipi_type_to_name_dict[atom_type2]
            epsilon = float(elements[4])*_kcal_to_kj
            sigma = float(elements[5])*_angstrom_to_nm
            if (atom_name1 == 'ILE') and (atom_name2 == 'ILE'):
                mu = 11
            elif set([atom_name1, atom_name2]) == {'VAL', 'ILE'}:
                mu = 4
            elif ('RNA' in atom_name1) or ('RNA' in atom_name2):
                mu = 3
            else:
                mu = 2
            nu = 1
            row = [atom_name1, atom_name2, epsilon, sigma, mu, nu]
            df_Mpipi_parameters.loc[len(df_Mpipi_parameters.index)] = row
df_Mpipi_parameters[['epsilon', 'sigma']] = df_Mpipi_parameters[['epsilon', 'sigma']].round(6)
df_Mpipi_parameters.to_csv('Mpipi_parameters.csv', index=False)


