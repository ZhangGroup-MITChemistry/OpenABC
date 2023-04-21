import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import mdtraj
import MDAnalysis
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{__location__}/../..')
from openabc.forcefields.hps_zero_offset_model import HPSZeroOffsetModel
from openabc.forcefields.parsers import HPSParser

'''
Compare energies of configurations computed with our code and HOOMD-Blue. 
Run tests with DDX4.
Add both Urry scale and KR scale nonbonded force so we can compute them in one rerun. 
'''

ca_pdb = 'init_DDX4_CA.pdb'
protein_parser = HPSParser(ca_pdb)
protein = HPSZeroOffsetModel() # use zero offset model
protein.append_mol(protein_parser)
top = app.PDBFile(ca_pdb).getTopology()
protein.create_system(top)
protein.add_protein_bonds(force_group=1)
protein.add_contacts('Urry', mu=1, delta=0.08, force_group=2)
protein.add_contacts('KR', mu=1, delta=0, force_group=3)
protein.add_dh_elec(force_group=4)
integrator = mm.NoseHooverIntegrator(300*unit.kelvin, 1/unit.picosecond, 10*unit.femtosecond)
protein.set_simulation(integrator)
simulation = protein.simulation

t = mdtraj.load_dcd('data/DDX4/output.dcd', top=ca_pdb)
n_frames = t.xyz.shape[0]
df_openmm_energy = pd.DataFrame(columns=['bond', 'contact (Urry)', 'contact (KR)', 'elec'])
for i in range(n_frames):
    row = []
    simulation.context.setPositions(t.xyz[i])
    for j in range(1, 5):
        state = simulation.context.getState(getEnergy=True, groups={j})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        row.append(energy)
    df_openmm_energy.loc[len(df_openmm_energy.index)] = row
df_openmm_energy.round(2).to_csv('DDX4_openmm_energy_zero_offset.csv', index=False)



