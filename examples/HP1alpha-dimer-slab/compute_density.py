import numpy as np
import mdtraj
import sys
import os
import pandas as pd
import simtk.unit as unit
import simtk.openmm as mm
import simtk.openmm.app as app

COM_traj_npy = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])
boundary = float(sys.argv[4])
output_density_csv = sys.argv[5]

print(f'Use frames {start_frame}-{end_frame} when computing density')

COM_traj = np.load(COM_traj_npy)
n_frames = COM_traj.shape[0]
print(f'The trajecotry has {n_frames} frames')
n_monomers = COM_traj.shape[1]
print(f'The system has {n_monomers} monomers.')
box_a, box_b, box_c = 25, 25, 400
# double check if all the COM are within the main box
assert np.amin(COM_traj) >= 0
assert ((np.amax(COM_traj[:, :, 0]) <= box_a) and (np.amax(COM_traj[:, :, 1]) <= box_b) and (np.amax(COM_traj[:, :, 2]) <= box_c))

# load mass
system_xml = 'system.xml'
with open(system_xml, 'r') as f:
    system = mm.XmlSerializer.deserialize(f.read())
monomer_mass = []
n_atoms = system.getNumParticles()
n_atoms_per_monomer = int(n_atoms/n_monomers)
for i in range(n_atoms_per_monomer):
    monomer_mass.append(system.getParticleMass(i).value_in_unit(unit.dalton))
monomer_mass = np.sum(np.array(monomer_mass))

bin_width = 5
n_bins = int(box_c/bin_width)
bins = np.linspace(0, box_c, n_bins + 1)
bin_width = bins[1] - bins[0] # reset bin_width
z = 0.5*(bins[1:] + bins[:-1])
rho_M, rho_g_per_L = [], []
NA = 6.02214076e+23
for i in range(start_frame, end_frame + 1):
    count_i, _ = np.histogram(COM_traj[i, :, 2], bins=bins)
    rho_M_i = count_i/(NA*box_a*box_b*bin_width*10**-27*10**3) # unit mol/L
    rho_g_per_L_i = rho_M_i*monomer_mass # unit g/L
    rho_M.append(rho_M_i)
    rho_g_per_L.append(rho_g_per_L_i)
rho_M = np.mean(np.array(rho_M), axis=0)
rho_g_per_L = np.mean(np.array(rho_g_per_L), axis=0)
z_shifted = z - np.mean(z)
df_density = pd.DataFrame(columns=['z (nm)', 'rho (M)', 'rho (g/L)'])
df_density['z (nm)'] = z_shifted
df_density['rho (M)'] = rho_M
df_density['rho (g/L)'] = rho_g_per_L
df_density.to_csv(output_density_csv, index=False)

# compute the density of two phases
print(f'Dense phase regime is {-1*boundary} <= z <= {boundary}')
rho_dilute_phase = df_density.loc[(df_density['z (nm)'] < -50) | (df_density['z (nm)'] > 50), 'rho (g/L)'].mean()
rho_concentrated_phase = df_density.loc[(df_density['z (nm)'] >= -1*boundary) & (df_density['z (nm)'] <= boundary), 'rho (g/L)'].mean()
print(f'Dilute phase concentration is {rho_dilute_phase:.2f} g/L (mg/mL), and concentrated phase concentration is {rho_concentrated_phase:.2f} g/L (mg/mL)')

