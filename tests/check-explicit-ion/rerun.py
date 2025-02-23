import numpy as np
import pandas as pd
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import mdtraj
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.insert(0, '../..') # ensure use the specific openabc we aim to test
from openabc.forcefields.parsers import SMOGParser, DNA3SPN2Parser
from openabc.forcefields import SMOG3SPN2ExplicitIonModel
from openabc.utils import parse_pdb, write_pdb, insert_molecules, insert_molecules_dataframe, compute_PE
from openabc.lib import _amino_acids, _dna_nucleotides, df_sodium_ion, df_chloride_ion

"""
Rerun the protein-DNA system with explicit ion model.
"""

# separate the original pdb file into protein and dna pdb files
# build the system consistent with lammps system
# CG protein is chain C of 1apl
df_atoms = parse_pdb('1apl.pdb')
protein_atoms = df_atoms[df_atoms['chainID'] == 'C'].copy()
protein_atoms['serial'] = np.arange(len(protein_atoms.index)) + 1
write_pdb(protein_atoms, 'protein.pdb')
dna_atoms = df_atoms[df_atoms['resname'].isin(_dna_nucleotides)].copy()
dna_atoms['serial'] = np.arange(len(dna_atoms.index)) + 1
write_pdb(dna_atoms, 'dna.pdb')

# parse protein, dna, and ions
protein_parser = SMOGParser.from_atomistic_pdb('protein.pdb', 'cg_protein.pdb')
dna_parser = DNA3SPN2Parser.from_atomistic_pdb('dna.pdb', 'cg_dna.pdb')
model = SMOG3SPN2ExplicitIonModel()
model.append_mol(protein_parser)
model.append_mol(dna_parser)
n_NA_ions = 32
#n_CL_ions = 0
model.append_ions('NA', n_NA_ions)
#model.append_ions('CL', n_CL_ions)
output_atoms = model.atoms.copy()
output_atoms.loc[:, 'charge'] = '' # remove charge to avoid format error
write_pdb(output_atoms, 'start.pdb')


"""
# prepare the pdb file to get topology
box = [20, 20, 20]
insert_molecules('cg_dna.pdb', 'cg_protein_dna.pdb', 1, existing_pdb='cg_protein.pdb', box=box)
df_atoms = parse_pdb('cg_protein_dna.pdb')
df_atoms = insert_molecules_dataframe(df_sodium_ion, n_NA_ions, 
                                      existing_atoms=df_atoms, box=box)
#df_atoms = insert_molecules_dataframe(df_chloride_ion, n_CL_ions, existing_atoms=df_atoms, box=box)
is_NA = df_atoms['name'] == 'NA'
is_CL = df_atoms['name'] == 'CL'
df_atoms.loc[is_NA, 'resSeq'] = 1 + np.arange(n_NA_ions)
#df_atoms.loc[is_CL, 'resSeq'] = 1 + np.arange(n_CL_ions)
df_atoms.loc[:, 'charge'] = '' # remove charge information here to avoid pdb format error
#df_atoms.to_csv('cg_atoms.csv', index=False)
write_pdb(df_atoms, 'start.pdb', write_TER=True) # start.pdb includes CG protein, DNA, and ions
top = app.PDBFile('start.pdb').getTopology()
#print(f'{top.getNumAtoms()} atoms')
"""

# create openmm system and compute energies
top = app.PDBFile('start.pdb').getTopology()
model.create_system(top, box_a=20, box_b=20, box_c=20)
model.add_protein_bonds(force_group=1)
model.add_protein_angles(force_group=2)
model.add_protein_dihedrals(force_group=3)
model.add_native_pairs(force_group=4)
model.add_dna_bonds(force_group=5)
model.add_dna_angles(force_group=6)
model.add_dna_stackings(force_group=7)
model.add_dna_dihedrals(force_group=8)
model.add_dna_base_pairs(force_group=9)
model.add_dna_cross_stackings(force_group=10)
model.parse_all_exclusions() # set exclusions before adding nonbonded forces
model.add_all_vdwl_elec(force_group_sr=11, force_group_PME=12)
system = model.system
#print(f'{system.getNumParticles()} atoms')
dcd = 'lammps-output/DUMP_Extract_temp300.dcd'
traj = mdtraj.load_dcd(dcd, top='start.pdb')
print(f'{dcd} has {traj.n_frames} frames')
energy = compute_PE(traj, system, 'CUDA', properties={'Precision': 'double'})
print(energy)



