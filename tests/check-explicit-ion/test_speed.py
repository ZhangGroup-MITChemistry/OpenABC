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
Rerun a protein-DNA system with explicit ion model.
"""

# separate the original pdb file into protein and dna pdb files
df_atoms = parse_pdb('1apl.pdb')
protein_atoms = df_atoms[df_atoms['resname'].isin(_amino_acids)].copy()
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
n_NA_ions = 514
n_CL_ions = 482
model.append_ions('NA', n_NA_ions)
model.append_ions('CL', n_CL_ions)

# prepare the pdb file to get topology
box = [20, 20, 20]
insert_molecules('cg_dna.pdb', 'cg_protein_dna.pdb', 1, existing_pdb='cg_protein.pdb', box=box)
df_atoms = parse_pdb('cg_protein_dna.pdb')
df_atoms = insert_molecules_dataframe(df_sodium_ion, n_NA_ions, existing_atoms=df_atoms, box=box)
df_atoms = insert_molecules_dataframe(df_chloride_ion, n_CL_ions, existing_atoms=df_atoms, box=box)
df_atoms.loc[:, 'charge'] = '' # remove charge information here to avoid pdb format error
write_pdb(df_atoms, 'start.pdb') # start.pdb includes CG protein, DNA, and ions
#top = app.PDBFile('start.pdb').getTopology()
#print(top.getNumAtoms()) # incorrect atom number, app.PDBFile cannot load the topology correctly
traj = mdtraj.load_pdb('start.pdb')
top = traj.topology.to_openmm()
#print(top.getNumAtoms()) # correct atom number

# create openmm system and compute energies
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
model.add_all_vdwl(force_group=11)
model.add_all_elec(force_group=12)
#model.save_system('system.xml')

# run simulation on GPU
T = 300 * unit.kelvin
friction_coeff = 1 / unit.picosecond
timestep = 10 * unit.femtosecond
integrator = mm.LangevinMiddleIntegrator(T, friction_coeff, timestep)
platform_name = 'CUDA'
#init_coord = app.PDBFile('start.pdb').getPositions(asNumpy=True)
#print(init_coord.shape) # app.PDBFile cannot load the coordinates correctly
init_coord = traj.xyz[0] * unit.nanometer
model.set_simulation(integrator, platform_name, init_coord=init_coord)
model.simulation.minimizeEnergy()
output_interval = 1000
output_dcd = 'output.dcd'
model.add_reporters(output_interval, output_dcd)
model.simulation.context.setVelocitiesToTemperature(T)
model.simulation.step(100000)


