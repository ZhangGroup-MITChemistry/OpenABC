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
import sys
import os

sys.path.append('../..')
from openabc.forcefields.parsers import MpipiProteinParser, MpipiRNAParser
from openabc.forcefields.mpipi_zero_offset_model import MpipiZeroOffsetModel
import openabc.utils.helper_functions as helper_functions

"""
Compare the energies computed by OpenMM and LAMMPS. 
"""
polyR_atoms = helper_functions.build_straight_CA_chain('RRRRRRRRRR')
polyR_atoms.loc[:, 'chainID'] = 'A'
helper_functions.write_pdb(polyR_atoms, 'polyR_CA.pdb')
polyK_atoms = helper_functions.build_straight_CA_chain('KKKKKKKKKK')
polyK_atoms.loc[:, 'chainID'] = 'B'
helper_functions.write_pdb(polyK_atoms, 'polyK_CA.pdb')
polyU_atoms = helper_functions.build_straight_chain(n_atoms=10, chainID='C', r0=0.5)
polyU_atoms.loc[:, 'name'] = 'RN'
polyU_atoms.loc[:, 'resname'] = 'U'
helper_functions.write_pdb(polyU_atoms, 'polyU_CG.pdb')

polyR = MpipiProteinParser('polyR_CA.pdb')
polyK = MpipiProteinParser('polyK_CA.pdb')
polyU = MpipiRNAParser('polyU_CG.pdb')
pdb_path = 'cg_protein_rna.pdb'
top = app.PDBFile(pdb_path).getTopology()

protein_rna = MpipiZeroOffsetModel()
protein_rna.append_mol(polyR)
protein_rna.append_mol(polyK)
for i in range(2):
    protein_rna.append_mol(polyU)
protein_rna.create_system(top, box_a=10, box_b=10, box_c=10)
protein_rna.add_protein_bonds(force_group=1)
protein_rna.add_rna_bonds(force_group=2)
protein_rna.add_contacts(force_group=3)
protein_rna.add_dh_elec(ldby=(1/1.26)*unit.nanometer, force_group=4)
temperature = 300*unit.kelvin
friction_coeff = 1/unit.picosecond
timestep = 10*unit.femtosecond
integrator = mm.LangevinMiddleIntegrator(temperature, friction_coeff, timestep)
protein_rna.set_simulation(integrator, platform_name='CPU')

t = mdtraj.load_dcd('lammps-run/DUMP_FILE.dcd', pdb_path)
n_frames = t.xyz.shape[0]
df_openmm_energy = pd.DataFrame(columns=['protein bond', 'RNA bond', 'contact', 'elec'])
for i in range(n_frames):
    row = []
    protein_rna.simulation.context.setPositions(t.xyz[i])
    for j in range(1, 5):
        state = protein_rna.simulation.context.getState(getEnergy=True, groups={j})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        row.append(energy)
    df_openmm_energy.loc[len(df_openmm_energy.index)] = row
df_openmm_energy['bond'] = df_openmm_energy['protein bond'] + df_openmm_energy['RNA bond']
df_openmm_energy = df_openmm_energy[['bond', 'contact', 'elec']].copy()
df_openmm_energy.round(2).to_csv('openmm_energy.csv', index=False)

df_lammps_energy = pd.read_csv('lammps-run/lammps_energy.csv')
for i in ['bond', 'contact', 'elec']:
    diff = np.absolute(df_openmm_energy[i].to_numpy() - df_lammps_energy[i].to_numpy())
    assert np.amax(diff) <= 0.01

