import numpy as np
import pandas as pd
import torch
try:
    import openmm as mm
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.unit as unit
import warnings
warnings.filterwarnings('ignore')
import argparse
import mdtraj
import shutil
import glob
import json
import time
import sys
import os
from openabc.forcefields.MOFF2.core import compute_mixed_u0_from_reduced_PE_matrices, compute_PE
from openabc.forcefields.MOFF2.lib import GAS_CONST
from openabc.forcefields.MOFF2.utils import compute_CA_traj_radius_of_gyration

"""
Compute the reduced energy of data and noise samples under the mixed noise ensemble. 
"""

parser = argparse.ArgumentParser()
parser.add_argument('--protein', required=True, help='Protein name')
parser.add_argument('--n0', type=int, default=50000, help='The target number of noise samples used for training')
parser.add_argument('--n1', type=int, default=50000, help='The target number of data samples used for training')
parser.add_argument('--T0', type=float, required=True, help='Noise trajectory simulation temperature')
args = parser.parse_args()

output_dir = f'training-input/n0-{args.n0}-n1-{args.n1}/{args.protein}'
os.makedirs(output_dir, exist_ok=True)

# load data samples
robustelli2018developing_IDPs = ['ACTR', 'Abeta40', 'Ash1', 'NTail', 'alpha-synuclein', 'drkN-SH3', 'p15PAF', 'sic1']
lindorff2011fast_OPs = ['Chignolin', 'Homeodomain', 'Trp-cage', 'Villin', 'WW-domain', 'Protein-G']
piana2020development_OPs = ['NTL9', 'Protein-B', 'alpha3d', 'bba', 'bbl', 'engrailed', 'gpw', 'lambda-repressor', 'BPTI', 'calmodulin']
robustelli2018developing_OPs = ['Ubiquitin', 'GB3', 'Hen-Egg-White-Lysozyme']
all_OPs = lindorff2011fast_OPs + piana2020development_OPs + robustelli2018developing_OPs
robustelli2018developing_proteins = robustelli2018developing_IDPs + robustelli2018developing_OPs

HPS_Urry_IDP_noise_main_dir = '/home/gridsan/sliu/Projects/TW-PCCG-develop/noise-cg-simulations/noise-HPS-Urry-simulations'
HPS_Urry_sbm_OP_noise_main_dir = '/home/gridsan/sliu/Projects/TW-PCCG-develop/noise-cg-simulations/noise-HPS-Urry-sbm-T-5ldby-simulations'

if args.protein in robustelli2018developing_proteins:
    ca_data_dir = f'/home/gridsan/sliu/Projects/TW-PCCG-develop/train-ca-models/ca-data-trajs/robustelli2018developing-ca-trajs/{args.protein}'
    ca_pdb = f'{ca_data_dir}/{args.protein}_ca.pdb'
    ca_dcd = f'{ca_data_dir}/{args.protein}_ca.dcd'
    traj1 = mdtraj.load_dcd(ca_dcd, top=ca_pdb)
elif args.protein in lindorff2011fast_OPs:
    ca_data_dir = f'/home/gridsan/sliu/Projects/TW-PCCG-develop/train-ca-models/ca-data-trajs/lindorff2011fast-ca-trajs/{args.protein}'
    ca_dcd = f'{ca_data_dir}/{args.protein}_ca.dcd'
    # use the pdb file for noise simulations in case there are amino acids not within 20 standard amino acids
    ca_pdb = f'{HPS_Urry_sbm_OP_noise_main_dir}/{args.protein}-HPS-Urry-sbm/unbiased-system/{args.protein}_ca.pdb'
    traj1 = mdtraj.load_dcd(ca_dcd, top=ca_pdb)
elif args.protein in piana2020development_OPs:
    ca_data_main_dir = '/home/gridsan/sliu/Projects/TW-PCCG-develop/train-ca-models/ca-data-trajs/piana2020development-ca-trajs'
    if args.protein in ['BPTI', 'calmodulin']:
        # proteins with NVT simulation
        ca_data_dir_dict = {'BPTI': f'{ca_data_main_dir}/BPTI-ch0.9-20us', 
                            'calmodulin': f'{ca_data_main_dir}/calmodulin-ch0.9'}
        ca_data_dir = ca_data_dir_dict[args.protein]
        ca_dcd = f'{ca_data_dir}/{args.protein}.dcd'
        # use the pdb file for noise simulations in case there are amino acids not within 20 standard amino acids
        ca_pdb = f'{HPS_Urry_sbm_OP_noise_main_dir}/{args.protein}-HPS-Urry-sbm/unbiased-system/{args.protein}_ca.pdb'
        traj1 = mdtraj.load_dcd(ca_dcd, top=ca_pdb)
    else:
        # proteins with replica exchange simulations
        name_dict = {'NTL9': 'ntl9', 'lambda-repressor': 'lambda', 'Protein-B': 'prb', 'alpha3d': 'a3d'}
        if args.protein in name_dict:
            name = name_dict[args.protein]
        else:
            name = args.protein
        ca_data_dir = f'{ca_data_main_dir}/rungs_FigS4-{name}'
        # use data samples produced near 300 K
        ca_dcd_list = [f'{ca_data_dir}/rung019_temp_299.299.dcd', f'{ca_data_dir}/rung020_temp_300.464.dcd']
        # use the pdb file for noise simulations in case there are amino acids not within 20 standard amino acids
        ca_pdb = f'{HPS_Urry_sbm_OP_noise_main_dir}/{args.protein}-HPS-Urry-sbm/unbiased-system/{args.protein}_ca.pdb'
        traj1 = mdtraj.join([mdtraj.load_dcd(ca_dcd, top=ca_pdb) for ca_dcd in ca_dcd_list])
elif (len(args.protein) >= 3) and (args.protein[:3] == 'Evo'):
    i = int(args.protein[3:])
    ca_data_dir = f'/home/gridsan/sliu/Projects/TW-PCCG-develop/train-ca-models/ca-data-trajs/Evo-ca-trajs-10.5us/cluster_{i}'
    ca_pdb = f'{ca_data_dir}/CA.pdb'
    ca_dcd = f'{ca_data_dir}/CA.dcd'
    traj1 = mdtraj.load_dcd(ca_dcd, top=ca_pdb)
    t_traj_us = 10.5
    t_relax_traj_us = 0.5
    traj1 = traj1[int(round(t_relax_traj_us * traj1.n_frames / t_traj_us)):]
else:
    sys.exit(f'{args.protein} is an invalid input protein name')
shutil.copyfile(ca_pdb, f'{output_dir}/{args.protein}_ca.pdb') # keep a copy of the CA pdb file
n_frames1 = traj1.n_frames
if n_frames1 > args.n1:
    selection = np.sort(np.random.choice(n_frames1, args.n1, replace=False))
    traj1 = traj1[selection] # subsample
traj1 = mdtraj.Trajectory(traj1.xyz, traj1.topology) # only keep coordinates and topology

# load noise samples
if args.protein in all_OPs:
    noise_simulation_dirs = sorted(glob.glob(f'{HPS_Urry_sbm_OP_noise_main_dir}/{args.protein}-HPS-Urry-sbm/rmsd-biased-simulations/kappa*center*'))
    unbiased_system_xml = f'{HPS_Urry_sbm_OP_noise_main_dir}/{args.protein}-HPS-Urry-sbm/unbiased-system/unbiased_system.xml'
else:
    noise_simulation_dirs = sorted(glob.glob(f'{HPS_Urry_IDP_noise_main_dir}/{args.protein}-HPS-Urry/rg-biased-simulations/kappa*center*'))
    unbiased_system_xml = f'{HPS_Urry_IDP_noise_main_dir}/{args.protein}-HPS-Urry/unbiased-system/unbiased_system.xml'
traj0_list = []
kappa_list = []
center_list = []
n_noise_trajs = len(noise_simulation_dirs)
target_n_frames_per_noise_traj = int(round(args.n0 / n_noise_trajs))
for each_dir in noise_simulation_dirs:
    each_traj = mdtraj.load_dcd(f'{each_dir}/output.dcd', top=ca_pdb)
    if each_traj.n_frames > target_n_frames_per_noise_traj:
        selection = np.sort(np.random.choice(each_traj.n_frames, target_n_frames_per_noise_traj, replace=False))
        each_traj = each_traj[selection]
    traj0_list.append(each_traj)
    with open(f'{each_dir}/input_parameters.json') as f:
        parameters = json.load(f)
    kappa = parameters['kappa'] # umbrella bias force constant
    center = parameters['center'] # umbrella bias center
    kappa_list.append(kappa)
    center_list.append(center)
    if args.protein not in all_OPs:
        # ensure unbiased system is used as the input system for the noise simulation
        system_xml = parameters['system_xml']
        assert system_xml == f'{args.protein}-HPS-Urry/unbiased-system/unbiased_system.xml'
n_frames0_array = np.array([x.n_frames for x in traj0_list])
traj0 = mdtraj.join(traj0_list)
traj0 = mdtraj.Trajectory(traj0.xyz, traj0.topology) # only keep coordinates and topology

# compute the unbiased energy
with open(unbiased_system_xml, 'r') as f:
    unbiased_system = mm.XmlSerializer.deserialize(f.read())
unbiased_U0 = compute_PE(traj0, unbiased_system, platform_name='Reference')
unbiased_U1 = compute_PE(traj1, unbiased_system, platform_name='Reference')

# compute bias and the reduced energy matrix including bias
T0 = args.T0 * unit.kelvin
RT0_value = (GAS_CONST * T0).value_in_unit(unit.kilojoule_per_mole)
kappa = np.array(kappa_list)[:, None]
center = np.array(center_list)[:, None]
if args.protein in all_OPs:
    # for ordered proteins, bias is applied to RMSD
    ref_ca_pdb = f'{HPS_Urry_sbm_OP_noise_main_dir}/{args.protein}-HPS-Urry-sbm/unbiased-system/{args.protein}_ca.pdb'
    ref_ca_traj = mdtraj.load_pdb(ref_ca_pdb)
    traj0_xyz = traj0.xyz.copy()
    traj1_xyz = traj1.xyz.copy()
    ref_ca_traj_xyz = ref_ca_traj.xyz.copy()
    cv0 = mdtraj.rmsd(traj0[:], ref_ca_traj[0])
    cv1 = mdtraj.rmsd(traj1[:], ref_ca_traj[0])
    assert np.all(traj0.xyz - traj0_xyz == 0)
    assert np.all(traj1.xyz - traj1_xyz == 0)
    assert np.all(ref_ca_traj.xyz - ref_ca_traj_xyz == 0)
else:
    # for disordered proteins, bias is applied to Rg
    cv0 = compute_CA_traj_radius_of_gyration(traj0)
    cv1 = compute_CA_traj_radius_of_gyration(traj1)
A0 = (unbiased_U0 + 0.5 * kappa * (cv0 - center)**2) / RT0_value
A1 = (unbiased_U1 + 0.5 * kappa * (cv1 - center)**2) / RT0_value

# compute the reduced energy of the mixed ensemble 0
fastmbar_cuda = torch.cuda.is_available()
time0 = time.time()
u0_traj0, u0_traj1, fastmbar = compute_mixed_u0_from_reduced_PE_matrices(A0, n_frames0_array, A1, 
                                                                         fastmbar_cuda=fastmbar_cuda)
time1 = time.time()
if fastmbar_cuda:
    print(f'Compute mixed u0 on GPU takes {(time1 - time0):.2f} seconds')
else:
    print(f'Compute mixed u0 on CPU takes {(time1 - time0):.2f} seconds')

# save output
traj = mdtraj.join([traj0, traj1])
traj.save_dcd(f'{output_dir}/traj.dcd')
labels = np.concatenate((np.zeros(len(traj0)), np.ones(len(traj1))), axis=0)
u0 = np.concatenate((u0_traj0, u0_traj1), axis=0)
np.save(f'{output_dir}/labels.npy', labels)
np.save(f'{output_dir}/u0.npy', u0)

