import numpy as np
import pandas as pd
import sys
import os
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import mdtraj

sys.path.append('../../..')

from OpenSMOG3SPN2.forcefields.parsers import SMOGParser, DNA3SPN2Parser
from OpenSMOG3SPN2.forcefields import SMOG3SPN2Model
from OpenSMOG3SPN2.utils.helper_functions import get_WC_paired_seq
from OpenSMOG3SPN2.utils.chromatin_helper_functions import remove_histone_tail_dihedrals, remove_histone_tail_native_pairs_and_exclusions
from OpenSMOG3SPN2.utils.insert import insert_molecules

n_nucl = 2
platform_name = sys.argv[1]

# load single nucleosome
single_nucl = SMOG3SPN2Model()
histone = SMOGParser.from_atomistic_pdb('../single-nucl-pdb-files/histone.pdb', 'histone_CA.pdb', 
                                        default_parse=False)
histone.parse_mol(get_native_pairs=False)
histone.protein_dihedrals = remove_histone_tail_dihedrals(histone.protein_dihedrals)
df_native_pairs = pd.read_csv('../pairs.dat', delim_whitespace=True, skiprows=1, 
                              names=['a1', 'a2', 'type', 'epsilon_G', 'mu', 'sigma_G', 'alpha_G'])
df_native_pairs = df_native_pairs.drop(['type'], axis=1)
df_native_pairs[['a1', 'a2']] -= 1
df_native_pairs[['epsilon_G', 'alpha_G']] *= 2.5
df_native_pairs, _ = remove_histone_tail_native_pairs_and_exclusions(df_native_pairs, histone.protein_exclusions)
histone.native_pairs = df_native_pairs
single_nucl.append_mol(histone)

with open('dna_seq.txt', 'r') as f:
    seq1 = f.readlines()[0].strip()
seq2 = get_WC_paired_seq(seq1)
target_seq = seq1 + seq2
dna = DNA3SPN2Parser.from_atomistic_pdb('../single-nucl-pdb-files/dna.pdb', 'cg_dna.pdb', new_sequence=target_seq, 
                                        default_parse=False)
# use old geometry parameters
dna.parse_config_file()
bs_geometry = dna.base_step_geometry.copy()
columns = ['twist', 'roll', 'tilt', 'shift', 'slide', 'rise']
stea_is_T = (bs_geometry['stea'] == 'T')
steb_is_T = (bs_geometry['steb'] == 'T')
bs_geometry.loc[stea_is_T & steb_is_T, columns] = [35.31, 0.91, 1.84, 0.05, -0.21, 3.27]
stea_is_C = (bs_geometry['stea'] == 'C')
steb_is_G = (bs_geometry['steb'] == 'G')
bs_geometry.loc[stea_is_C & steb_is_G, columns] = [34.38, 4.29, 0.00, 0.00, 0.57, 3.49]
dna.base_step_geometry = bs_geometry.copy()
dna.parse_mol()

single_nucl.append_mol(dna)
single_nucl.atoms_to_pdb('cg_nucl.pdb') # write pdb of single nucleosome

# prepare the system composed of two nucleosomes
two_nucl = SMOG3SPN2Model()
box_a, box_b, box_c = 200, 200, 200
for i in range(n_nucl):
    two_nucl.append_mol(histone)
    two_nucl.append_mol(dna)
two_nucl.base_step_geometry = bs_geometry.copy()
insert_molecules('cg_nucl.pdb', 'two_cg_nucl.pdb', n_mol=n_nucl, box=[box_a, box_b, box_c])

top = app.PDBFile('two_cg_nucl.pdb').getTopology()
init_coord = app.PDBFile('two_cg_nucl.pdb').getPositions()
two_nucl.create_system(top, box_a=box_a, box_b=box_b, box_c=box_c)
two_nucl.add_protein_bonds(force_group=1)
two_nucl.add_protein_angles(force_group=2)
two_nucl.add_protein_dihedrals(force_group=3)
two_nucl.add_native_pairs(force_group=4)
two_nucl.add_dna_bonds(force_group=5)
two_nucl.add_dna_angles(force_group=6)
two_nucl.add_dna_stackings(force_group=7)
two_nucl.add_dna_dihedrals(force_group=8)
two_nucl.add_dna_base_pairs(force_group=9)
two_nucl.add_dna_cross_stackings(force_group=10)
two_nucl.parse_all_exclusions()
two_nucl.add_all_vdwl(force_group=11)
two_nucl.add_all_elec(force_group=12)

temperature = 300*unit.kelvin
friction_coeff = 0.01/unit.picosecond
timestep = 10*unit.femtosecond
integrator = mm.LangevinMiddleIntegrator(temperature, friction_coeff, timestep)
two_nucl.set_simulation(integrator, platform_name=platform_name, init_coord=init_coord)
simulation = two_nucl.simulation

dcd = '../lammps-rerun/traj.dcd'
traj = mdtraj.load_dcd(dcd, top='two_cg_nucl.pdb')
n_frames = traj.n_frames
columns = ['protein bond', 'protein angle', 'protein dihedral', 'native pair', 'dna bond', 
           'dna angle', 'dna stacking', 'dna dihedral', 'dna base pair', 'dna cross stacking', 
           'all vdwl', 'all elec']
df_energies_kj = pd.DataFrame(columns=columns)
df_energies_kcal = pd.DataFrame(columns=columns)
for i in range(1, n_frames):
    # since lammps gives strange results for the 0th snapshot, we do not compute energy for that one
    simulation.context.setPositions(traj.xyz[i])
    row_kj, row_kcal = [], []
    for j in range(1, 13):
        state = simulation.context.getState(getEnergy=True, groups={j})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        row_kj.append(energy)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        #print(f'Group {j} energy is {energy} kcal/mol')
        row_kcal.append(energy)
    df_energies_kj.loc[len(df_energies_kj.index)] = row_kj
    df_energies_kcal.loc[len(df_energies_kcal.index)] = row_kcal

#df_energies_kj.round(6).to_csv(f'openmm_energy_kj_{platform_name}.csv', index=False)
#df_energies_kcal.round(6).to_csv(f'openmm_energy_kcal_{platform_name}.csv', index=False)
df_energies_kj.round(6).to_csv(f'openmm_energy_kj.csv', index=False)
df_energies_kcal.round(6).to_csv(f'openmm_energy_kcal.csv', index=False)

