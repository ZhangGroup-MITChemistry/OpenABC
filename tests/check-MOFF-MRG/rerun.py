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
from openabc.forcefields.parsers import MOFFParser, MRGdsDNAParser
from openabc.utils.shadow_map import load_ca_pairs_from_gmx_top
from openabc.utils.helper_functions import write_pdb

'''
Compare energy with GROMACS output. 
Note the native pairs in GROMACS topology file are produced by SMOG, which may be slightly different from the native pairs found by our shadow map algorithm code. 
To keep consistency, we directly load native pairs from GROMACS topology file. 
'''

hp1alpha_dimer_parser = MOFFParser.from_atomistic_pdb('hp1a.pdb', 'hp1alpha_dimer_CA.pdb')
hp1alpha_dimer_parser.native_pairs = load_ca_pairs_from_gmx_top('hp1a.itp', 'hp1alpha_dimer_CA.pdb') # update native pairs
hp1alpha_dimer_parser.parse_exclusions()

dsDNA_parser = MRGdsDNAParser.from_atomistic_pdb('all_atom_200bpDNA.pdb', 'MRG_dsDNA.pdb')

protein_dna = MOFFMRGModel()
protein_dna.append_mol(hp1alpha_dimer_parser)
protein_dna.append_mol(dsDNA_parser)
protein_dna.native_pairs.loc[:, 'epsilon'] = 6.0

# we need to write all the atoms to pdb file
protein_dna.atoms_to_pdb('cg_protein_dna.pdb')

top = app.PDBFile('cg_protein_dna.pdb').getTopology()
protein_dna.create_system(top, box_a=100, box_b=100, box_c=100)
salt_conc = 82*unit.millimolar
temperature = 300*unit.kelvin
protein_dna.add_protein_bonds(force_group=1)
protein_dna.add_protein_angles(force_group=2)
protein_dna.add_protein_dihedrals(force_group=3)
protein_dna.add_native_pairs(force_group=4)
protein_dna.add_dna_bonds(force_group=5)
protein_dna.add_dna_angles(force_group=6)
protein_dna.add_dna_fan_bonds(force_group=7)
protein_dna.add_contacts(force_group=8)
protein_dna.add_elec_switch(salt_conc, temperature, force_group=9)
protein_dna.save_system('system.xml')
collision = 1/unit.picosecond
timestep = 10*unit.femtosecond
integrator = mm.NoseHooverIntegrator(temperature, collision, timestep)
platform_name = 'CPU'
protein_dna.set_simulation(integrator, platform_name, init_coord=None)
simulation = protein_dna.simulation

traj = mdtraj.load_xtc('gmx-data/md.xtc', top='cg_protein_dna.pdb')
n_frames = traj.xyz.shape[0]
openmm_energies = []
for i in range(n_frames):
    row = []
    simulation.context.setPositions(traj.xyz[i])
    for j in range(1, 10):
        state = protein_dna.simulation.context.getState(getEnergy=True, groups={j})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        row.append(energy)
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    row.append(energy)
    openmm_energies.append(row)

openmm_energies = np.array(openmm_energies)
columns = ['protein bond', 'protein angle', 'protein dihedral', 'native pair', 'dna bond', 'dna angle', 'dna fan bond', 'contact', 'elec switch', 'sum']
df_openmm_energies = pd.DataFrame(openmm_energies, columns=columns).round(6)
df_openmm_energies.round(2).to_csv('openmm_energies.csv', index=False)



