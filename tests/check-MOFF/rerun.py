import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import mdtraj
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{__location__}/../..')
from openabc.forcefields import MOFFMRGModel
from openabc.forcefields.parsers import MOFFParser
from openabc.utils.shadow_map import load_ca_pairs_from_gmx_top

'''
Compare energy with GROMACS output. 
Note the native pairs in GROMACS topology file are produced by SMOG, which may be slightly different from the native pairs found by our shadow map algorithm code. 
To keep consistency, we directly load native pairs from GROMACS topology file. 
'''

hp1alpha_dimer_parser = MOFFParser.from_atomistic_pdb('hp1a.pdb', 'hp1alpha_dimer_CA.pdb', default_parse=False)
hp1alpha_dimer_parser.parse_mol(get_native_pairs=False) # do not get native pairs with our shadow algorithm code
hp1alpha_dimer_parser.native_pairs = load_ca_pairs_from_gmx_top('hp1a.itp', 'hp1alpha_dimer_CA.pdb')
hp1alpha_dimer_parser.parse_exclusions()
hp1alpha_dimer = MOFFMRGModel()
hp1alpha_dimer.append_mol(hp1alpha_dimer_parser)
hp1alpha_dimer.native_pairs.loc[:, 'epsilon'] = 6.0
top = app.PDBFile('hp1alpha_dimer_CA.pdb').getTopology()
hp1alpha_dimer.create_system(top, box_a=200, box_b=200, box_c=200)
salt_conc = 82*unit.millimolar
temperature = 300*unit.kelvin
hp1alpha_dimer.add_protein_bonds(force_group=1)
hp1alpha_dimer.add_protein_angles(force_group=2)
hp1alpha_dimer.add_protein_dihedrals(force_group=3)
hp1alpha_dimer.add_native_pairs(force_group=4)
hp1alpha_dimer.add_contacts(force_group=5)
hp1alpha_dimer.add_elec_switch(salt_conc, temperature, force_group=6)
hp1alpha_dimer.save_system('system.xml')
collision = 1/unit.picosecond
timestep = 10*unit.femtosecond
integrator = mm.NoseHooverIntegrator(temperature, collision, timestep)
platform_name = 'CPU'
hp1alpha_dimer.set_simulation(integrator, platform_name, init_coord=None)
simulation = hp1alpha_dimer.simulation

traj = mdtraj.load_xtc('gmx-data/md.xtc', top='hp1alpha_dimer_CA.pdb')
n_frames = traj.xyz.shape[0]
openmm_energies = []
for i in range(n_frames):
    row = []
    simulation.context.setPositions(traj.xyz[i])
    for j in range(1, 7):
        state = hp1alpha_dimer.simulation.context.getState(getEnergy=True, groups={j})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        row.append(energy)
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    row.append(energy)
    openmm_energies.append(row)

openmm_energies = np.array(openmm_energies)
columns = ['bond', 'angle', 'dihedral', 'native pair', 'contact', 'elec switch', 'sum']
df_openmm_energies = pd.DataFrame(openmm_energies, columns=columns).round(6)
df_openmm_energies.round(2).to_csv('openmm_energies.csv', index=False)

# compare with GROMACS output energies
df_gmx_energies = pd.read_csv('gmx-data/gmx_energies.csv')
n_snapshots = len(df_openmm_energies.index)
assert n_snapshots == len(df_gmx_energies)
for i in range(n_snapshots):
    row1 = df_openmm_energies.loc[i]
    row2 = df_gmx_energies.loc[i]
    for j in columns:
        if (j not in ['elec switch', 'sum']) and (abs(row1[j] - row2[j]) > 0.01):
            print(f'Warning: snapshot {i} energy {j} do not match!')
        elif (j in ['elec switch', 'sum']) and (abs(row1[j] - row2[j]) > 0.02):
            # we assign a slightly larger tolerance for electrostatic interaction and overall energy
            print(f'Warning: snapshot {i} energy {j} do not match!')


