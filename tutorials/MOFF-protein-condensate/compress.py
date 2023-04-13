import numpy as np
import pandas as pd
import sys
import os
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit

sys.path.append('../../')
from openabc.forcefields.parsers import MOFFParser
from openabc.forcefields import MOFFMRGModel

'''
Use this script to compress molecules. 
'''

if not os.path.exists('NPT-output-files'):
    os.makedirs('NPT-output-files')

platform_name = 'CUDA'
hp1alpha_dimer_parser = MOFFParser.from_atomistic_pdb('input-pdb/hp1a.pdb', 'hp1alpha_dimer_CA.pdb')

# remove redundant native pairs
old_native_pairs = hp1alpha_dimer_parser.native_pairs.copy()
new_native_pairs = pd.DataFrame(columns=old_native_pairs.columns)
cd1 = np.arange(16, 72)
csd1 = np.arange(114, 176)
n_atoms_per_hp1alpha_dimer = len(hp1alpha_dimer_parser.atoms.index)
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
hp1alpha_dimer_parser.parse_exclusions()

# use gmx insert-molecules to put molecules into the simulation box randomly
n_mol = 20
cmd = f'gmx insert-molecules -ci hp1alpha_dimer_CA.pdb -nmol {n_mol} -box 50 50 50 -radius 1.0 -scale 2.0 -o start.pdb'
if not os.path.exists('start.pdb'):
    os.system(cmd)
init_coord = app.PDBFile('start.pdb').getPositions()

multi_dimers = MOFFMRGModel()
for i in range(n_mol):
    multi_dimers.append_mol(hp1alpha_dimer_parser)
multi_dimers.native_pairs.loc[:, 'epsilon'] = 6.0
top = app.PDBFile('start.pdb').getTopology()
box_a, box_b, box_c = 50, 50, 50
multi_dimers.create_system(top, box_a=box_a, box_b=box_b, box_c=box_c)

salt_conc = 82*unit.millimolar
temperature = 150*unit.kelvin
multi_dimers.add_protein_bonds(force_group=1)
multi_dimers.add_protein_angles(force_group=2)
multi_dimers.add_protein_dihedrals(force_group=3)
multi_dimers.add_native_pairs(force_group=4)
multi_dimers.add_contacts(force_group=5)
multi_dimers.add_elec_switch(salt_conc, 300*unit.kelvin, force_group=6)
pressure = 1*unit.bar
multi_dimers.system.addForce(mm.MonteCarloBarostat(pressure, temperature))

friction_coeff = 1/unit.picosecond
timestep = 10*unit.femtosecond
integrator = mm.LangevinMiddleIntegrator(temperature, friction_coeff, timestep)
multi_dimers.set_simulation(integrator, platform_name, init_coord=init_coord)
simulation = multi_dimers.simulation
simulation.minimizeEnergy()
output_interval = 100000
output_dcd = 'NPT-output-files/NPT_compress.dcd'
multi_dimers.add_reporters(output_interval, output_dcd)
simulation.context.setVelocitiesToTemperature(temperature)

# run NPT compression
print('Start NPT compression.')
simulation.step(2000000)
print('NPT compression is finished.')

# print final box vectors
state = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True,
                                    getParameters=True, enforcePeriodicBox=True)
box_vec = state.getPeriodicBoxVectors(asNumpy=True)
print('Final box vectors:')
print(box_vec)

# save the final state
with open('NPT-output-files/NPT_compressed_state.xml', 'w') as f:
    f.write(mm.XmlSerializer.serialize(state))


