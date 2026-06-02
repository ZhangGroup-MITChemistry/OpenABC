import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.distributed as distributed
import torch.multiprocessing as mp
import socket
import pickle
import argparse
import time
import json
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})
from openabc.forcefields.MOFF2.lib import _amino_acids, _amino_acid_3_letters_to_1_letter_dict, _res_group_mappings
from openabc.forcefields.MOFF2.core import DistributedCLBase
from openabc.forcefields.MOFF2.utils import clamped_bspline_basis_1d
from openabc.forcefields.MOFF2.core.compute_rg_reweight_gau import run_full_reweighting # for reweighting

"""
Train the model with contrastive learning.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--backend', type=str, default='nccl', help='The backend for torch distributed')
parser.add_argument('--MJ_min', type=float, default=0.0, help='The minimal value of scaled MJ parameters')
parser.add_argument('--MJ_max', type=float, default=1.0, help='The maximal value of scaled MJ parameters')
parser.add_argument('--eta', type=float, default=10.0, help='eta parameter in unit 1 / nm')
parser.add_argument('--r0', type=float, default=0.7, help='r0 parameter in unit nm')
parser.add_argument('--rho_min', type=float, default=0.0, help='rho min')
parser.add_argument('--rho_max', type=float, default=15.0, help='rho max')
parser.add_argument('--n_internal_knots', type=int, default=10, help='Number of internal knots')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
parser.add_argument('--n_epochs', type=int, default=1000, help='The number of epochs for training')
parser.add_argument('--zeta1', type=float, default=0.0, help='Regularization strength coefficient zeta1')
parser.add_argument('--zeta2', type=float, default=0.0, help='Regularization strength coefficient zeta2')
parser.add_argument('--res_group_mapping', type=str, default='default', help='Residue group mapping, can be "default", "hydrophobic_polar", and etc.')
parser.add_argument(
    '--gauss_delta_mu',
    type=float,
    default=0.25,
    help='Gaussian center offset Δμ (nm) beyond r_min used in basis generation'
)
parser.add_argument(
    '--gauss_width',
    type=float,
    default=0.10,
    help='Gaussian width σ_G (nm) used in basis generation'
)
parser.add_argument('--IDP_weight', type=float, default=1.0, help='Weight for IDPs in the loss function')
parser.add_argument('--OP_weight', type=float, default=1.0, help='Weight for OPs in the loss function')
parser.add_argument('--MDP_weight', type=float, default=1.0, help='Weight for MDPs in the loss function')
args = parser.parse_args()

for key, value in vars(args).items():
    print(f'{key} = {value}', flush=True)

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

# set world_size and n_gpus_per_node
n_gpus_per_node = torch.cuda.device_count()
assert n_gpus_per_node == 2 # there are 2 GPUs per node

# ACTR and NTail data samples have some very short CA-CA bonds
# alpha-synuclein is also tricky as it has 2 peaks
robustelli2018developing_training_IDPs = ['ACTR', 'Abeta40', 'Ash1', 'NTail', 'drkN-SH3', 'p15PAF', 'sic1']
Evo_training_IDP_indices = list(range(1, 11)) + [12, 14, 16] + list(range(18, 27)) + [30, 31, 33, 34, 38, 41, 42, 46, 47, 48, 51, 52]
Evo_training_IDPs = [f'Evo{i}' for i in Evo_training_IDP_indices]
ordered_proteins = ['BPTI', 'Chignolin', 'GB3', 'Hen-Egg-White-Lysozyme', 'Homeodomain', 
                    'NTL9', 'Protein-B', 'Protein-G', 'Trp-cage', 'Ubiquitin', 'Villin', 
                    'WW-domain', 'alpha3d', 'bba', 'bbl', 'calmodulin', 'engrailed', 'gpw', 
                    'lambda-repressor', '1soy_clean', '1wla_clean', '2ea9_clean', '5tvz_clean', 
                    'AF-P0C232-F1-model_v4_w_H', 'AF-P0CG98-F1-model_v4_w_H', 
                    'AF-P00251-F1-model_v4_w_H', 'AF-P21149-F1-model_v4_w_H', 
                    'AF-P21318-F1-model_v4_w_H', 'AF-P29669-F1-model_v4_w_H', 
                    'AF-P31960-F1-model_v4_w_H', 'AF-P32729-F1-model_v4_w_H', 
                    'AF-P61734-F1-model_v4_w_H', 'AF-P69995-F1-model_v4_w_H', 
                    'AF-P75202-F1-model_v4_w_H', 'AF-P75459-F1-model_v4_w_H', 
                    'AF-P80353-F1-model_v4_w_H', 'AF-P87285-F1-model_v4_w_H']
MDPs = ['THB-C2', 'Ub2', 'Ub3', 'Gal3', 'hnRNPA1-star', 'FPs-GS8', 'FPs-GS16', 
        'SH4UD-SH3-SH2', 'TDP43_WtoA', 'D12', 'D23', 'D34', 'SMAD4']
all_training_proteins = robustelli2018developing_training_IDPs + Evo_training_IDPs + ordered_proteins + MDPs
n_mols = len(all_training_proteins)
n_OPs = len(ordered_proteins)
n_MDPs = len(MDPs)
n_IDPs = n_mols - n_OPs - n_MDPs
print(f'Train with {n_mols} proteins in all, including {n_IDPs} IDPs, {n_OPs} OPs, and {n_MDPs} MDPs.', flush=True)
n_groups = len(set(_res_group_mappings[args.res_group_mapping].values()))

# set mol_weights
mol_weights = []
for i in range(n_mols):
    if all_training_proteins[i] in ordered_proteins:
        mol_weights.append(args.OP_weight / n_OPs)
        # mol_weights.append(1 / n_OPs)
    elif all_training_proteins[i] in MDPs:
        mol_weights.append(args.MDP_weight / n_MDPs)
        # mol_weights.append(1 / n_MDPs)
    else:
        mol_weights.append(args.IDP_weight / n_IDPs)
        # mol_weights.append(1 / n_IDPs)

#IDP_OP_training_input_main_dir = f'../training-input/n0-50000-n1-50000'
train_path='/orcd/data/binz/001/congwang/TW-PCCG-develop/TW-PCCG-develop/train-ca-models/transferable-ca-models/'
IDP_OP_training_input_main_dir = train_path + f'/training-input/n0-50000-n1-50000'

# read some global parameters
pkl_path = f'{IDP_OP_training_input_main_dir}/ACTR/training_input_ah_density_spl_group_{args.res_group_mapping}_eta_{args.eta}_r0_{args.r0}_rho_range_{args.rho_min}_{args.rho_max}_n_internal_knots_{args.n_internal_knots}_gau.pkl'
with open(pkl_path, 'rb') as f:
    p = pickle.load(f)
u1_density_spl_basis = p['u1_density_spl_basis']
assert u1_density_spl_basis.ndim == 4
n_bases_per_residue = u1_density_spl_basis.shape[-1]
n_internal_knots = p['n_internal_knots']
degree = p['degree']
assert degree == 3
assert n_bases_per_residue == n_internal_knots + degree + 1 # ensure all bases are kept



def load_training_input(protein, device):
    info_dict = {}
    if protein in MDPs:
        training_input_main_dir = train_path + f'/training-input/n0-10000-n1-10000'
    else:
        training_input_main_dir = train_path + f'/training-input/n0-50000-n1-50000'
    pkl_path = f'{training_input_main_dir}/{protein}/training_input_ah_density_spl_group_{args.res_group_mapping}_eta_{args.eta}_r0_{args.r0}_rho_range_{args.rho_min}_{args.rho_max}_n_internal_knots_{args.n_internal_knots}_gau.pkl'
    with open(pkl_path, 'rb') as f:
        p = pickle.load(f)
    for each in ['labels', 'u0', 'u1_ah_basis']:
        info_dict[each] = torch.tensor(p[each], dtype=default_dtype, device=device)
    info_dict['u1_density_spl_basis'] = torch.tensor(p['u1_density_spl_basis'][:, :, :, :-(degree - 1)], 
                                                     dtype=default_dtype, device=device)

    # ─────────────────────────────────────────────
    # NEW: Gaussian basis per pair type (n_samples, 210)
    # ─────────────────────────────────────────────
    if 'u1_gauss_basis' in p:
        info_dict['u1_gauss_basis'] = torch.tensor(
            p['u1_gauss_basis'],
            dtype=default_dtype,
            device=device,
        )
        # Optional sanity check if gauss metadata is stored
        if 'gauss_delta_mu' in p:
            if abs(p['gauss_delta_mu'] - args.gauss_delta_mu) > 1e-6:
                print(f"[WARN] pkl gauss_delta_mu={p['gauss_delta_mu']} "
                      f"!= args.gauss_delta_mu={args.gauss_delta_mu}", flush=True)
        if 'gauss_width' in p:
            if abs(p['gauss_width'] - args.gauss_width) > 1e-6:
                print(f"[WARN] pkl gauss_width={p['gauss_width']} "
                      f"!= args.gauss_width={args.gauss_width}", flush=True)
    else:
        # If for some reason Gaussian basis is missing, fall back to zeros
        info_dict['u1_gauss_basis'] = torch.zeros_like(
            info_dict['u1_ah_basis'],
            dtype=default_dtype,
            device=device,
        )


    labels = info_dict['labels']
    n_data_samples = torch.sum(labels).item()
    n_noise_samples = len(labels) - n_data_samples
    print(f'{protein} has {n_noise_samples} noise samples and {n_data_samples} data samples', flush=True)
    u1_all_bonded = p['u1_all_bonded']
    u1_lj_excl = p['u1_lj_excl']
    u1_elec = p['u1_elec']
    u1_intercept = torch.tensor(u1_all_bonded + u1_lj_excl + u1_elec, dtype=default_dtype, device=device)
    info_dict['u1_intercept'] = u1_intercept
    return info_dict

# load MJ potential parameters and initialize ah_coeff as MJ parameters
df_MJ = pd.read_csv(train_path+f'/parameters/raw_MJ.csv')
MJ_map = np.zeros((20, 20))
for _, row in df_MJ.iterrows():
    i = _amino_acids.index(row['amino acid1'])
    j = _amino_acids.index(row['amino acid2'])
    MJ_map[i, j] = row['epsilon (RT)']
    MJ_map[j, i] = row['epsilon (RT)']
MJ_map *= -1.0 # important, change to positive values indicating attraction, consistent with hydrophobicity definition
MJ_map = (MJ_map - np.min(MJ_map)) / (np.max(MJ_map) - np.min(MJ_map)) # scale and shift to [0, 1]
MJ_map = MJ_map * (args.MJ_max - args.MJ_min) + args.MJ_min # scale and shift to [args.MJ_min, args.MJ_max]
rows, cols = np.triu_indices(20, 0)
ah_coeff0 = MJ_map[rows, cols]

# ─────────────────────────────────────────────
# NEW: baseline Gaussian coefficients A_ij
# Start at 0 (no Gaussian bump), to be learned
# ─────────────────────────────────────────────
gauss_coeff0 = np.zeros_like(ah_coeff0, dtype=float)


#class CLModel(DistributedCLBase):
#    def __init__(self, n_mols, local_mol_ids, n_groups, dtype=default_dtype):
#        super().__init__(n_mols, local_mol_ids, dtype)
#        self.delta_ah_coeff = nn.Parameter(torch.zeros(210, dtype=dtype))
#        self.ah_coeff0 = torch.tensor(ah_coeff0, dtype=dtype)
#        self.spl_coeffs = nn.Parameter(torch.zeros(20, n_groups, n_bases_per_residue - (degree - 1), dtype=dtype))
class CLModel(DistributedCLBase):
    def __init__(self, n_mols, local_mol_ids, n_groups, dtype=default_dtype):
        super().__init__(n_mols, local_mol_ids, dtype)

        # AH coefficients (eps_ij)
        self.delta_ah_coeff = nn.Parameter(torch.zeros(210, dtype=dtype))
        self.ah_coeff0 = torch.tensor(ah_coeff0, dtype=dtype)

        # NEW: Gaussian coefficients (A_ij)
        self.delta_gauss_coeff = nn.Parameter(torch.zeros(210, dtype=dtype))
        self.gauss_coeff0 = torch.tensor(gauss_coeff0, dtype=dtype)

        # Density spline
        self.spl_coeffs = nn.Parameter(
            torch.zeros(
                20,
                n_groups,
                n_bases_per_residue - (degree - 1),
                dtype=dtype,
            )
        )
    
    def set_training_input(self):
        for mol_id in self.local_mol_ids:
            # load training input to self.device
            self.training_input[mol_id] = load_training_input(all_training_proteins[mol_id], device=self.device)
    
    def u0(self, mol_id):
        _u0 = self.training_input[mol_id]['u0']
        return _u0
    
#   def u1(self, mol_id):
#        u1_intercept = self.training_input[mol_id]['u1_intercept']
#        u1_ah_basis = self.training_input[mol_id]['u1_ah_basis']
#        if self.ah_coeff0.device != self.device:
#            self.ah_coeff0 = self.ah_coeff0.to(self.device)
#        ah_coeff = self.ah_coeff0 + self.delta_ah_coeff
#        u1_density_spl_basis = self.training_input[mol_id]['u1_density_spl_basis']
#        u1_ah = torch.mv(u1_ah_basis, ah_coeff)
#        u1_density_spl = torch.sum(u1_density_spl_basis * self.spl_coeffs, dim=(1, 2, 3))
#        _u1 = u1_intercept + u1_ah + u1_density_spl
#        return _u1
    def u1(self, mol_id):
        u1_intercept = self.training_input[mol_id]['u1_intercept']
        u1_ah_basis = self.training_input[mol_id]['u1_ah_basis']
        u1_gauss_basis = self.training_input[mol_id]['u1_gauss_basis']
        u1_density_spl_basis = self.training_input[mol_id]['u1_density_spl_basis']

        # AH coeffs
        if self.ah_coeff0.device != self.device:
            self.ah_coeff0 = self.ah_coeff0.to(self.device)
        ah_coeff = self.ah_coeff0 + self.delta_ah_coeff
        u1_ah = torch.mv(u1_ah_basis, ah_coeff)

        # Gaussian coeffs
        if self.gauss_coeff0.device != self.device:
            self.gauss_coeff0 = self.gauss_coeff0.to(self.device)
        gauss_coeff = self.gauss_coeff0 + self.delta_gauss_coeff
        u1_gauss = torch.mv(u1_gauss_basis, gauss_coeff)

        # Density spline
        u1_density_spl = torch.sum(
            u1_density_spl_basis * self.spl_coeffs,
            dim=(1, 2, 3)
        )

        _u1 = u1_intercept + u1_ah + u1_gauss + u1_density_spl
        return _u1


def ddp_setup(rank: int, world_size: int, backend: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)

# set the function to do cleanup
def cleanup():
    distributed.destroy_process_group()


# set the main function
def train_model(rank, world_size, args):
    ddp_setup(rank, world_size, args.backend)
    local_rank = rank
    hostname = socket.gethostname()
    print(f'rank = {rank}, world_size = {world_size}, hostname = {hostname}, local_rank = {local_rank}', flush=True)
    i1 = int(round(rank * len(all_training_proteins) / world_size))
    i2 = int(round((rank + 1) * len(all_training_proteins) / world_size))
    local_mol_ids = list(range(i1, i2))
    local_mol_ids_string = ', '.join([str(x) for x in local_mol_ids])
    print(f'rank = {rank}, local_mol_ids = {local_mol_ids_string}', flush=True)
    model = CLModel(n_mols, local_mol_ids, n_groups, dtype=default_dtype)
    model = model.to(local_rank)
    model.check_local_mol_ids() # check if each protein is covered by one and only one rank
    model.set_training_input() # set training input after moving to target device, thus the tensors are also loaded to the target device
    model.compute_delta_fs(update=True) # initialize model.delta_fs
    ddp_model = DDP(model, device_ids=[local_rank], output_device=None)
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    df_loss = pd.DataFrame(columns=['step', 'overall loss'])
    time0 = time.time()
    for i in range(args.n_epochs):
        optimizer.zero_grad()
    #    reg1 = 0.5 * args.zeta1 * torch.mean(model.delta_ah_coeff**2)
        reg1 = 0.5 * args.zeta1 * (
            torch.mean(model.delta_ah_coeff**2) +
            torch.mean(model.delta_gauss_coeff**2)
        )

        reg2 = args.zeta2 * torch.mean(model.spl_coeffs**2)
        loss = ddp_model(reduction='mean', mol_weights=mol_weights) # balance OPs and IDPs
        loss += (reg1 + reg2) / world_size # divide by world_size as we will finally add up loss across all ranks
        loss.backward()
        optimizer.step()
        scheduler.step()
        _loss = loss.detach().clone()
        distributed.all_reduce(_loss, op=distributed.ReduceOp.SUM)
        _loss = _loss.item()
        if (rank == 0) and (i % 50 == 0):
            df_loss.loc[len(df_loss.index)] = [i, _loss]
            print(f'Step {i}, overall loss = {_loss}', flush=True)
    time1 = time.time()
    cleanup()
    if rank == 0:
        print(f'Optimization takes {(time1 - time0):.2f} seconds', flush=True)
        ah_coeff = (model.ah_coeff0 + model.delta_ah_coeff).detach().cpu().numpy()
        hydrophobic_scale = np.zeros((20, 20))
        rows, cols = np.triu_indices(20, 0)
        hydrophobic_scale[rows, cols] = ah_coeff
        hydrophobic_scale[cols, rows] = ah_coeff
        # NEW: Gaussian coefficients / height map
        gauss_coeffs = (model.gauss_coeff0 + model.delta_gauss_coeff).detach().cpu().numpy()
        gauss_height_map = np.zeros((20, 20))
        gauss_height_map[rows, cols] = gauss_coeffs
        gauss_height_map[cols, rows] = gauss_coeffs
        spl_coeffs = model.spl_coeffs.detach().cpu().numpy()
        # compute the density spline values
        rho = np.linspace(args.rho_min, args.rho_max, 500)
        spl_basis_dict = clamped_bspline_basis_1d(rho, args.rho_min, args.rho_max, 
                                                  args.n_internal_knots, degree=3, 
                                                  intercept=True, omega=False)
        design_matrix = spl_basis_dict['design_matrix']
        spl_values = np.sum(spl_coeffs[:, :, None, :] * design_matrix[:, :-(degree - 1)], 
                            axis=3) # spl_values in unit kJ/mol
        # output_dir = f'results/group_{args.res_group_mapping}_MJ_{args.MJ_min}_{args.MJ_max}_eta_{args.eta}_r0_{args.r0}_rho_{args.rho_min}_{args.rho_max}_n_internal_knots_{args.n_internal_knots}_zeta_{args.zeta1}_{args.zeta2}_lr_{args.lr}_n_epochs_{args.n_epochs}'
        output_dir = (
            f'results/group_{args.res_group_mapping}_'
            f'IDP_w_{args.IDP_weight}_OP_w_{args.OP_weight}_MDP_w_{args.MDP_weight}_'
            f'{args.n_internal_knots}_zeta_{args.zeta1}_{args.zeta2}_'
            f'gauss_delta_mu_{args.gauss_delta_mu}_gauss_width_{args.gauss_width}')
        os.makedirs(output_dir, exist_ok=True)
        df_loss.to_csv(f'{output_dir}/loss.csv', index=False)
        results_dict = {
            'hydrophobic_scale': hydrophobic_scale,
            'spl_coeffs': spl_coeffs,
            'spl_values': spl_values,
            'res_group_mapping': args.res_group_mapping,
            'MJ_min': args.MJ_min,
            'MJ_max': args.MJ_max,
            'eta': args.eta,
            'r0': args.r0,
            'rho_min': args.rho_min,
            'rho_max': args.rho_max,
            # NEW:
            'gauss_coeffs': gauss_coeffs,
            'gauss_height_map': gauss_height_map,
            'gauss_delta_mu': args.gauss_delta_mu,
            'gauss_width': args.gauss_width,
        }

        with open(f'{output_dir}/results.pkl', 'wb') as f:
            pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # draw the plot of hydrophobic scale
        aa_labels = [_amino_acid_3_letters_to_1_letter_dict[i] for i in _amino_acids]
        plt.imshow(hydrophobic_scale, cmap='coolwarm', origin='upper')
        plt.colorbar()
        xticks = np.arange(20)
        yticks = np.arange(20)
        plt.xticks(xticks, labels=aa_labels)
        plt.yticks(yticks, labels=aa_labels)
        plt.title(f'Optimized hydrophobic scale')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/hydrophobic_scale.pdf')
        plt.close()
        
        # draw the plot of density spline
        with PdfPages(f'{output_dir}/density_spline.pdf') as pdf:
            for i in range(20):
                for j in range(spl_values.shape[1]):
                    plt.plot(rho, spl_values[i, j, :], label=f"Group {j+1}")
                plt.legend()
                plt.xlabel('rho')
                plt.ylabel('Potential (kJ/mol)')
                plt.title(f'Density potential spline for {_amino_acids[i]}')
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                 # ─────────────────────────────────────────────
        # New: pairwise AH + Gaussian potentials
        #   → saves: pairwise_potentials.pdf
        # ─────────────────────────────────────────────

        print("Generating pairwise AH + Gaussian potential plots...", flush=True)

        # === Gaussian parameters (must match FEP scripts) ===
        with open(f'{output_dir}/results.pkl', 'rb') as f:
    	    res = pickle.load(f)

        H = res["hydrophobic_scale"]
        G = res["gauss_height_map"]
        delta = res["gauss_width"]
        extra = res["gauss_delta_mu"]

        c216 = 2**(1.0/6.0)

        # r-range for potential plots
        r = np.linspace(0.25, 2.0, 400)

        # === epsilon for AH ===
        _kcal_to_kj = 4.184
        epsilon = 0.2 * _kcal_to_kj   # same scale as your training

        # -----------------------------------------------------------------
        # Build sigma matrix from HPS Urry parameters
        # -----------------------------------------------------------------
        param_csv = "../parameters/HPS_Urry_parameters.csv"
        df_param = pd.read_csv(param_csv)
        sigma = np.zeros((20, 20))
        for _, row in df_param.iterrows():
            i = _amino_acids.index(row["atom_type1"])
            j = _amino_acids.index(row["atom_type2"])
            sigma[i, j] = sigma[j, i] = row["sigma"]

        # -----------------------------------------------------------------
        # AH piecewise potential (same definition used in FEP prep)
        # -----------------------------------------------------------------
        def AH_MOFF_piecewise(r_vals, eps, sig, lam):
            r_vals = np.asarray(r_vals)
            U = np.zeros_like(r_vals)

            r_min = c216 * sig
            r_cut = 4.0 * sig

            LJ = 4.0 * eps * ((sig / r_vals)**12 - (sig / r_vals)**6)
            LJ_cut = 4.0 * eps * ((sig / r_cut)**12 - (sig / r_cut)**6)

            s1 = (r_vals < r_min)
            s2 = (r_vals >= r_min) & (r_vals < r_cut)

            U_excl = np.zeros_like(r_vals)
            U_excl[s1] = LJ[s1] + eps

            AH_base = np.zeros_like(r_vals)
            AH_base[s1] = -eps - LJ_cut
            AH_base[s2] = LJ[s2] - LJ_cut

            return U_excl + lam * AH_base

        # -----------------------------------------------------------------
        # Gaussian tail (learned)
        # -----------------------------------------------------------------
        def Gaussian_tail(r_vals, sig, theta):
            r_vals = np.asarray(r_vals)
            r_min = c216 * sig
            r_cut = 4.0 * sig
            mu = r_min + extra

            g = np.zeros_like(r_vals)
            mask = (r_vals < r_cut)
            g[mask] = theta * np.exp(- (r_vals[mask] - mu)**2 / (2 * delta**2))
            return g

        # -----------------------------------------------------------------
        # Create 20-page PDF: each page = one AA vs 20 AA
        # -----------------------------------------------------------------
        pair_pdf_path = f"{output_dir}/pairwise_potentials.pdf"
        with PdfPages(pair_pdf_path) as pdf:

            for center_idx, aa3_center in enumerate(_amino_acids):
                fig, axes = plt.subplots(4, 5, figsize=(18, 14))
                axes = axes.flatten()

                for partner_idx, aa3_partner in enumerate(_amino_acids):
                    ax = axes[partner_idx]

                    i = center_idx
                    j = partner_idx

                    sig_ij = sigma[i, j]
                    lam_ij = H[i, j]
                    theta_ij = G[i, j]

                    # Compute potentials
                    V_AH = AH_MOFF_piecewise(r, epsilon, sig_ij, lam_ij)
                    V_G  = Gaussian_tail(r, sig_ij, theta_ij)
                    V_tot = V_AH + V_G

                    # Plot
                    ax.plot(r, V_AH, label="AH", lw=1.0, ls="--")
                    ax.plot(r, V_G,  label="Gaussian", lw=1.0, ls=":")
                    ax.plot(r, V_tot, label="Total", lw=1.5)

                    ax.axhline(0, color="black", lw=0.6)
                    ax.set_title(f"{aa3_center}-{aa3_partner}", fontsize=10)
                    ax.set_xlim(0.25, 2.0)
                    ax.set_ylim(-1.2, 0.5)

                    if partner_idx == 0:
                        ax.legend(fontsize=8)

                plt.suptitle(
                    f"Pair potentials centered at {aa3_center}",
                    fontsize=16
                )
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved pairwise potentials to {pair_pdf_path}", flush=True)

#        run_full_reweighting(
#            results_pkl=f"{output_dir}/results.pkl",
#            output_dir=f"{output_dir}/tests",
#            res_group_mapping=args.res_group_mapping,
#        )

        run_full_reweighting(
            results_pkl=f"{output_dir}/results.pkl",
            output_dir=f"{output_dir}/tests",
            res_group_mapping=args.res_group_mapping,
        )





if __name__=="__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_model, args=(world_size, args), nprocs=world_size, join=True)

