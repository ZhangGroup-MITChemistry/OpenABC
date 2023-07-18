import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.unit as unit
import mdtraj
from mdtraj.formats import DCDTrajectoryFile
from MDAnalysis.lib.nsgrid import FastNS
import networkx as nx
import sys
import os

sys.path.append('../..')
from openabc.utils.helper_functions import make_mol_whole, move_atoms_to_closest_pbc_image

input_dcd = sys.argv[1]
output_dcd = sys.argv[2]
output_COM_traj_npy = sys.argv[3]
pdb = 'start.pdb'

traj = mdtraj.load_dcd(input_dcd, pdb)
n_frames = traj.xyz.shape[0]
n_atoms = traj.xyz.shape[1]
n_monomers = 200 # monomer number
n_atoms_per_monomer = int(n_atoms/n_monomers)
print(f'{n_atoms_per_monomer} atoms in each monomer')

box_a, box_b, box_c = 25, 25, 400

# load mass
system_xml = 'system.xml'
with open(system_xml, 'r') as f:
    system = mm.XmlSerializer.deserialize(f.read())
monomer_mass = []
for i in range(n_atoms_per_monomer):
    monomer_mass.append(system.getParticleMass(i).value_in_unit(unit.dalton))
monomer_mass = np.array(monomer_mass)

# first make sure each monomer is intact
# we align monomers as during the simulation the dimer may dissociate into monomers
intact_coord = np.zeros((n_frames, n_atoms, 3))
for i in range(n_frames):
    for j in range(n_monomers):
        coord_i_j = traj.xyz[i, j*n_atoms_per_monomer:(j + 1)*n_atoms_per_monomer]
        intact_coord_i_j = make_mol_whole(coord_i_j.copy(), box_a, box_b, box_c)
        intact_coord[i, j*n_atoms_per_monomer:(j + 1)*n_atoms_per_monomer] = intact_coord_i_j

# then compute COM coordinate
intact_COM_coord = np.zeros((n_frames, n_monomers, 3))
for i in range(n_frames):
    for j in range(n_monomers):
        weights = monomer_mass/np.sum(monomer_mass)
        intact_COM_coord[i, j] = np.average(intact_coord[i, j*n_atoms_per_monomer:(j + 1)*n_atoms_per_monomer], axis=0, weights=weights)
intact_COM_coord = intact_COM_coord.astype(np.float32)

# find the largest cluster
cutoff = 5.0
pseudo_box = np.array([box_a, box_b, box_c, 90.0, 90.0, 90.0]).astype(np.float32)
pbc = True
for i in range(n_frames):
    grid_search = FastNS(cutoff, intact_COM_coord[i], pseudo_box, pbc)
    results = grid_search.self_search()
    pairs = results.get_pairs()
    pairs = [(int(x[0]), int(x[1])) for x in pairs]
    # create a graph to find the largest cluster
    G = nx.Graph()
    G.add_nodes_from(list(range(n_monomers)))
    G.add_edges_from(pairs)
    largest_cluster = sorted(list(max(nx.connected_components(G), key=len)))
    #print(f'Frame {i}, the largest cluster size is {len(largest_cluster)}')
    
    # then make sure the COMs in the largest cluster are within the same periodic image
    r0 = intact_COM_coord[i, largest_cluster[0]]
    r1 = intact_COM_coord[i, largest_cluster].copy()
    intact_COM_coord[i, largest_cluster] = move_atoms_to_closest_pbc_image(r1.copy(), r0, box_a, box_b, box_c)
    # update atom coordinates
    for j in range(len(largest_cluster)):
        k = largest_cluster[j]
        delta_r = intact_COM_coord[i, k] - r1[j]
        intact_coord[i, k*n_atoms_per_monomer:(k + 1)*n_atoms_per_monomer] += delta_r
    
    # translate all the atoms so the largest clsuter is at box center
    box_center = 0.5*np.array([box_a, box_b, box_c])
    delta_r = box_center - np.mean(intact_COM_coord[i, largest_cluster], axis=0)
    intact_coord[i, :] += delta_r
    intact_COM_coord[i, :] += delta_r
    
    # move all the monomers back to the main box
    r1 = intact_COM_coord[i].copy()
    intact_COM_coord[i] = move_atoms_to_closest_pbc_image(r1.copy(), box_center, box_a, box_b, box_c)
    for j in range(n_monomers):
        delta_r = intact_COM_coord[i, j] - r1[j]
        intact_coord[i, j*n_atoms_per_monomer:(j + 1)*n_atoms_per_monomer] += delta_r

dcd_coord = 10*intact_coord # convert nm to A
dcd_coord = dcd_coord.astype(np.float32)
stride = 10
with DCDTrajectoryFile(output_dcd, 'w') as f:
    f.write(xyz=dcd_coord[::stride])

np.save(output_COM_traj_npy, intact_COM_coord)


