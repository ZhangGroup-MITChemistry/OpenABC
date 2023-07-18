import numpy as np
import pandas as pd
import sys
import os
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit

sys.path.append('../..')
from openabc.forcefields.parsers import MOFFParser
from openabc.forcefields import MOFFMRGModel
from openabc.utils.insert import insert_molecules

hp1alpha_dimer_parser = MOFFParser.from_atomistic_pdb('hp1a.pdb', 'hp1alpha_dimer_CA.pdb')

cd1 = np.arange(16, 72)
csd1 = np.arange(114, 176)
n_atoms_per_hp1alpha_dimer = len(hp1alpha_dimer_parser.atoms.index)
print(f'There are {n_atoms_per_hp1alpha_dimer} CA atoms in each HP1alpha dimer.')
cd2 = cd1 + int(n_atoms_per_hp1alpha_dimer/2)
csd2 = csd1 + int(n_atoms_per_hp1alpha_dimer/2)

# remove redundant native pairs
old_native_pairs = hp1alpha_dimer_parser.native_pairs.copy()
new_native_pairs = pd.DataFrame(columns=old_native_pairs.columns)
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

# insert molecules into a simulation box randomly
n_mol = 100
box_length = 75
box = [box_length, box_length, box_length]
if not os.path.exists('start.pdb'):
    insert_molecules('hp1alpha_dimer_CA.pdb', 'start.pdb', n_mol=n_mol, box=box)

hp1alpha_dimers = MOFFMRGModel()
for i in range(n_mol):
    hp1alpha_dimers.append_mol(hp1alpha_dimer_parser)
hp1alpha_dimers.native_pairs.loc[:, 'epsilon'] = 6.0
top = app.PDBFile('start.pdb').getTopology()
hp1alpha_dimers.create_system(top, box_a=box_length, box_b=box_length, box_c=box_length)
salt_concentration = 82*unit.millimolar
temperature = 300*unit.kelvin
hp1alpha_dimers.add_protein_bonds(force_group=1)
hp1alpha_dimers.add_protein_angles(force_group=2)
hp1alpha_dimers.add_protein_dihedrals(force_group=3)
hp1alpha_dimers.add_native_pairs(force_group=4)
hp1alpha_dimers.add_contacts(force_group=5)
hp1alpha_dimers.add_elec_switch(salt_concentration, temperature, force_group=6)
hp1alpha_dimers.save_system('system.xml')

