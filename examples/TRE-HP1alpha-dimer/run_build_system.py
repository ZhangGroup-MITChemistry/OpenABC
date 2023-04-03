import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import sys
import os

sys.path.append('../..')
from openabc.forcefields import MOFFMRGModel
from openabc.forcefields.parsers import MOFFParser

'''
Build HP1alpha dimer system for further simulations. 
'''

# parse HP1alpha dimer
hp1alpha_dimer_parser = MOFFParser.from_atomistic_pdb('hp1a.pdb', 'hp1alpha_dimer_CA.pdb')
old_native_pairs = hp1alpha_dimer_parser.native_pairs.copy()
new_native_pairs = pd.DataFrame(columns=old_native_pairs.columns)
cd1 = np.arange(16, 72)
csd1 = np.arange(114, 176)
n_atoms_per_hp1alpha_dimer = len(hp1alpha_dimer_parser.atoms.index)
print(f'{n_atoms_per_hp1alpha_dimer} atoms in each hp1alpha dimer')
print(f'{int(n_atoms_per_hp1alpha_dimer/2)} atoms in each hp1alpha monomer')
cd2 = cd1 + int(n_atoms_per_hp1alpha_dimer/2)
csd2 = csd1 + int(n_atoms_per_hp1alpha_dimer/2)
for i, row in old_native_pairs.iterrows():
    a1, a2 = int(row['a1']), int(row['a2'])
    if a1 > a2:
        a1, a2 = a2, a1
    flag1 = ((a1 in cd1) and (a2 in cd1)) or ((a1 in csd1) and (a2 in csd1))
    flag2 = ((a1 in cd2) and (a2 in cd2)) or ((a1 in csd2) and (a2 in csd2))
    flag3 = ((a1 in csd1) and (a2 in csd2))
    if flag1 or flag2 or flag3:
        new_native_pairs.loc[len(new_native_pairs.index)] = row
hp1alpha_dimer_parser.native_pairs = new_native_pairs
hp1alpha_dimer_parser.parse_exclusions() # update exclusions based on the new native pairs

# prepare system
protein = MOFFMRGModel()
protein.append_mol(hp1alpha_dimer_parser)
top = app.PDBFile('hp1alpha_dimer_CA.pdb').getTopology()
protein.create_system(top)
salt_concentration = 82*unit.millimolar
protein.add_protein_bonds(force_group=1)
protein.add_protein_angles(force_group=2)
protein.add_protein_dihedrals(force_group=3)
protein.add_native_pairs(epsilon=6.0, force_group=4)
protein.add_contacts(force_group=5)
protein.add_elec_switch(salt_concentration, 300*unit.kelvin, force_group=6)
protein.atoms.to_csv('hp1alpha_dimer_CA.csv', index=False)
protein.save_system('hp1alpha_dimer_system.xml')


