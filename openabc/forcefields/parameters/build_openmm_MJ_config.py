import numpy as np
import pandas as pd

aa_resname_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

# read the lammps input MJ potential ff parameter file and convert it to openmm format
# proteinDna_pairCoeff.in includes lammps MJ potential ff parameters
# load protein-protein pair-wise interaction parameters
with open('proteinDna_pairCoeff.in', 'r') as proteinDna_pairCoeff:
    proteinDna_pairCoeff_lines = proteinDna_pairCoeff.readlines()

pp_pairCoeff_lines = []
flag = False
for each_line in proteinDna_pairCoeff_lines:
    if '# protein protein pair-wise interaction' in each_line:
        flag = True
    if flag and ('pair_coeff' in each_line):
        pp_pairCoeff_lines.append(each_line)

# write protein-protein pair-wise interaction parameters into one file with lammps format
with open('pp_pairCoeff.in', 'w') as pp_pairCoeff:
    for each_line in pp_pairCoeff_lines:
        pp_pairCoeff.write(each_line)

# buil openmm protein-protein pair-wise interaction parameter file
# load pp_pairCoeff.in
# cutoff1 is for LJ cutoff, and cutoff2 is for Coulombic cutoff
col_names = ['pair_coeff', 'atom_type1', 'atom_type2', 'pair_style', 'epsilon (kj/mol)', 
             'sigma (nm)', 'cutoff_LJ (nm)', 'cutoff_coul (nm)']
df_pp_pairCoeff = pd.read_csv('pp_pairCoeff.in', header=None, names=col_names, delim_whitespace=True)
# convert from lammps units (kcal/mol, A) to openmm units (kj/mol, nm)
df_pp_pairCoeff['epsilon (kj/mol)'] *= 4.184 # convert kcal/mol to kj/mol
df_pp_pairCoeff['sigma (nm)'] /= 10 # convert A to nm
df_pp_pairCoeff['cutoff_LJ (nm)'] /= 10 # convert A to nm
df_pp_pairCoeff['cutoff_coul (nm)'] /= 10 # convert A to nm
# replace atom_type1 and atom_type2 with amino acid names
for i in range(len(df_pp_pairCoeff.index)):
    df_pp_pairCoeff.loc[i, 'atom_type1'] = aa_resname_list[int(df_pp_pairCoeff.loc[i, 'atom_type1']) - 15]
    df_pp_pairCoeff.loc[i, 'atom_type2'] = aa_resname_list[int(df_pp_pairCoeff.loc[i, 'atom_type2']) - 15]
    df_pp_pairCoeff.loc[i, 'epsilon (kj/mol)'] = round(float(df_pp_pairCoeff.loc[i, 'epsilon (kj/mol)']), 8)

# drop some columns that are not useful here
df_pp_pairCoeff.drop(columns=['pair_coeff', 'pair_style', 'cutoff_coul (nm)'], inplace=True)
df_pp_pairCoeff.to_csv('pp_MJ.csv', index=False)


