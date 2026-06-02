#!/usr/bin/env python3
import os
import sys
import pickle
import argparse

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# If needed, adjust this to your repo root
from openabc.forcefields.MOFF2.lib import _amino_acids
from openabc.forcefields.MOFF2.core.compute_rg_reweight_gau import run_full_reweighting # for reweighting


# Protein lists (same style as your other scripts)
A1_LCDs = [
    'A1-LCD+12E','A1-LCD+7K+12D','A1-LCD-3R+3K','A1-LCD-10R+10K','A1-LCD+4D',
    'A1-LCD-4D','A1-LCD-9F+6Y','A1-LCD+2R','A1-LCD+8D','A1-LCD-12F+12Y',
    'A1-LCD-9F+3Y','A1-LCD+NLS','A1-LCD+12D','A1-LCD-10R','A1-LCD-6R',
    'A1-LCD-NLS','A1-LCD+7R','A1-LCD+7F-7Y','A1-LCD-8F+4Y','A1-LCD-6R+6K',
    'A1-LCD-10G+10S',  'A1-LCD+7R+12D',  'A1-LCD+7R+10D',  'A1-LCD-2K'
]

MDPs = [
    'GS32','GS48','TIA1','D14','GS24','HeV_V','PCPE',
    'GS0','NiV_V','S4FL','ChiAM','Ubq4','H46'
]

OPs = [
'Chignolin',
'Homeodomain',
'Protein-G',
'Trp-cage',
'alpha3d',
'bba',
#'bbl',
'engrailed',
'gpw',
'lambda-repressor',
'NTL9',
'Protein-B',
'BPTI',
'calmodulin',
'GB3',
'Hen-Egg-White-Lysozyme',
'Villin',
'WW-domain',
'1soy',
'1wla',
'2ea9',
'5tvz',
'Ubiquitin'
]

# 'hSUMO_hnRNPA1S'
ALL_PROTEINS = A1_LCDs + MDPs + OPs


# ─────────────────────────────────────────────────────────────
# Utility: ESS and ESS regularization
# ─────────────────────────────────────────────────────────────
def compute_ess(u1, u0):
    """
    ESS under reweighting from U0 to U1 using ΔU = U1-U0 (dimensionless).
    """
    du = u1 - u0
    w = torch.exp(-du)          # unnormalized
    ess = (w.sum() ** 2) / (w.pow(2).sum())
    return ess


def ess_reg(ess, alpha, ess0):
    """
    Quadratic penalty if ESS < ess0, else 0.
    """
    if ess >= ess0:
        return torch.zeros((), dtype=ess.dtype, device=ess.device)
    return alpha * (ess - ess0) ** 2


# ─────────────────────────────────────────────────────────────
# FEP reweighting model: AH + Gaussian + Density spline
# ─────────────────────────────────────────────────────────────
class FEP_AH_Gauss_Density_Model(nn.Module):
    def __init__(self,
                 results_pkl_path,
                 fep_input_root,
                 protein_list,
                 exp_rg_dict,
                 alpha,
                 ess0,
                 lam_ah=0.0,
                 lam_gauss=0.0,
                 lam_density=0.0,
                 device='cuda'):
        super().__init__()
        self.device = torch.device(device)

        # Load baseline results (θ⁰) from CL
        with open(results_pkl_path, 'rb') as f:
            res = pickle.load(f)

        # ---- AH baseline: (20,20) -> 210 upper-tri coefficients ----
        hydrophobic_scale0 = np.asarray(res['hydrophobic_scale'], dtype=float)
        rows, cols = np.triu_indices(20, 0)
        ah_coeff0_np = hydrophobic_scale0[rows, cols]      # (210,)

        # ---- Gaussian baseline: 210 ----
        if 'gauss_coeffs' in res:
            gauss_coeff0_np = np.asarray(res['gauss_coeffs'], dtype=float)
        else:
            gauss_coeff0_np = np.zeros_like(ah_coeff0_np, dtype=float)

        # ---- Density spline baseline: spl_coeffs (nbasis,) ----
        if 'spl_coeffs' not in res:
            raise KeyError("results.pkl is missing 'spl_coeffs'. "
                           "If CL trained density spline, it should be present.")
        spl_coeff0_np = np.asarray(res['spl_coeffs'], dtype=float)  # (nbasis,)

        # register fixed baselines as buffers
        self.register_buffer('ah_coeff0',
                             torch.tensor(ah_coeff0_np, dtype=torch.float32))
        self.register_buffer('gauss_coeff0',
                             torch.tensor(gauss_coeff0_np, dtype=torch.float32))
        self.register_buffer('spl_coeff0',
                             torch.tensor(spl_coeff0_np, dtype=torch.float32))

        # learnable deltas (start from CL, not blank)
        self.delta_ah_coeff = nn.Parameter(torch.zeros_like(self.ah_coeff0))
        self.delta_gauss_coeff = nn.Parameter(torch.zeros_like(self.gauss_coeff0))
        self.delta_spl_coeff = nn.Parameter(torch.zeros_like(self.spl_coeff0))

        self.alpha = float(alpha)
        self.ess0 = float(ess0)
        self.lam_ah = float(lam_ah)
        self.lam_gauss = float(lam_gauss)
        self.lam_density = float(lam_density)

        self.fep_input_root = fep_input_root
        self.protein_list = protein_list
        self.exp_rg_dict = {k: float(v) for k, v in exp_rg_dict.items()
                            if k in protein_list}

        self.protein_data = {}
        self._load_all_proteins()

        self.to(self.device)

    # ─────────────────────────────────────────
    # FEP input loading
    # ─────────────────────────────────────────
    def _load_one_protein(self, protein):
        """
        Expect fep_input.pkl with keys:
          - 'u1_intercept':          (N,)
          - 'u1_ah_basis':           (N, 210)
          - 'u1_gauss_basis':        (N, 210)
          - 'u1_density_spl_basis':  (N, nbasis)
          - 'rg':                    (N,)
        """
        # try newer file name first, then fallback
        cand_paths = [
            os.path.join(self.fep_input_root, protein, 'fep_AH_gau_input.pkl'),
            os.path.join(self.fep_input_root, protein, 'fep_AH_input.pkl'),
        ]
        pkl_path = None
        for cp in cand_paths:
            if os.path.exists(cp):
                pkl_path = cp
                break
        if pkl_path is None:
            raise FileNotFoundError(
                f"FEP input not found for {protein}. Tried:\n  " +
                "\n  ".join(cand_paths)
            )

        with open(pkl_path, 'rb') as f:
            p = pickle.load(f)

        # required keys
        needed = ['u1_intercept', 'u1_ah_basis', 'u1_gauss_basis',
                  'u1_density_spl_basis', 'rg']
        for k in needed:
            if k not in p:
                raise KeyError(f"{protein}: missing key '{k}' in {pkl_path}")

        u1_intercept = torch.tensor(p['u1_intercept'],
                                    dtype=torch.float32,
                                    device=self.device)
        u1_ah_basis = torch.tensor(p['u1_ah_basis'],
                                   dtype=torch.float32,
                                   device=self.device)
        u1_gauss_basis = torch.tensor(p['u1_gauss_basis'],
                                      dtype=torch.float32,
                                      device=self.device)
        u1_density_spl_basis = torch.tensor(p['u1_density_spl_basis'],
                                            dtype=torch.float32,
                                            device=self.device)
        rg = torch.tensor(p['rg'],
                          dtype=torch.float32,
                          device=self.device)

        # sanity checks
        if u1_ah_basis.shape[1] != self.ah_coeff0.shape[0]:
            raise ValueError(f"{protein}: AH basis dim {u1_ah_basis.shape[1]} "
                             f"!= ah_coeff dim {self.ah_coeff0.shape[0]}")
        if u1_gauss_basis.shape[1] != self.gauss_coeff0.shape[0]:
            raise ValueError(f"{protein}: Gauss basis dim {u1_gauss_basis.shape[1]} "
                             f"!= gauss_coeff dim {self.gauss_coeff0.shape[0]}")
        if u1_density_spl_basis.shape[1] != self.spl_coeff0.shape[0]:
            raise ValueError(f"{protein}: Density basis dim {u1_density_spl_basis.shape[1]} "
                             f"!= spl_coeff dim {self.spl_coeff0.shape[0]}")

        return {
            'u1_intercept': u1_intercept,
            'u1_ah_basis': u1_ah_basis,
            'u1_gauss_basis': u1_gauss_basis,
            'u1_density_spl_basis': u1_density_spl_basis,
            'rg': rg,
            'pkl_path': pkl_path,
        }

    def _load_all_proteins(self):
        for prot in self.protein_list:
            if prot not in self.exp_rg_dict:
                continue
            data = self._load_one_protein(prot)
            self.protein_data[prot] = data
            print(f"Loaded FEP input for {prot} with {data['rg'].shape[0]} frames "
                  f"from {os.path.basename(data['pkl_path'])}",
                  flush=True)

    # ─────────────────────────────────────────
    # Energies
    # ─────────────────────────────────────────
    def energy_ref(self, protein):
        """
        Reference energy U0(x) under baseline θ⁰ (CL):
          U0 = intercept
             + basis_AH·θ_AH⁰
             + basis_G ·θ_G⁰
             + basis_D ·θ_D⁰
        """
        d = self.protein_data[protein]
        u_int = d['u1_intercept']                  # (N,)
        basis_ah = d['u1_ah_basis']                # (N,210)
        basis_gauss = d['u1_gauss_basis']          # (N,210)

        u_ah0 = torch.mv(basis_ah, self.ah_coeff0)
        u_g0 = torch.mv(basis_gauss, self.gauss_coeff0)

        basis_den = d['u1_density_spl_basis']  # (N,20,G,B_total)

        B_total = basis_den.shape[-1]
        B_eff = self.spl_coeff0.shape[-1]
        if B_eff < B_total:
            basis_den_eff = basis_den[..., :B_eff]
        else:
            basis_den_eff = basis_den

        u_d0 = torch.sum(basis_den_eff * self.spl_coeff0, dim=(1, 2, 3))


        return u_int + u_ah0 + u_g0 + u_d0

    def energy_new(self, protein):
        """
        New energy U1(x;θ) under updated parameters:
          θ_AH = θ_AH⁰ + Δθ_AH
          θ_G  = θ_G⁰  + Δθ_G
          θ_D  = θ_D⁰  + Δθ_D
          U1   = intercept + basis_AH·θ_AH + basis_G·θ_G + basis_D·θ_D
        """
        d = self.protein_data[protein]
        u_int = d['u1_intercept']
        basis_ah = d['u1_ah_basis']
        basis_gauss = d['u1_gauss_basis']

        ah_coeff = self.ah_coeff0 + self.delta_ah_coeff
        g_coeff = self.gauss_coeff0 + self.delta_gauss_coeff


        u_ah = torch.mv(basis_ah, ah_coeff)
        u_g = torch.mv(basis_gauss, g_coeff)

        basis_den = d['u1_density_spl_basis']  # (N,20,G,B_total)
        spl_coeff = self.spl_coeff0 + self.delta_spl_coeff

        B_total = basis_den.shape[-1]
        B_eff = spl_coeff.shape[-1]
        if B_eff < B_total:
            basis_den_eff = basis_den[..., :B_eff]
        else:
            basis_den_eff = basis_den

        u_d = torch.sum(basis_den_eff * spl_coeff, dim=(1, 2, 3))


        return u_int + u_ah + u_g + u_d

    # ─────────────────────────────────────────
    # Loss computation
    # ─────────────────────────────────────────
    def fep_loss_for_protein(self, protein):
        """
        For one protein:
          - compute Rg FEP reweighted under U1
          - MSE vs experimental Rg
          - ESS penalty
        """
        d = self.protein_data[protein]
        rg = d['rg']
        rg_exp = torch.tensor(self.exp_rg_dict[protein],
                              dtype=torch.float32,
                              device=self.device)

        u0 = self.energy_ref(protein)
        u1 = self.energy_new(protein)
        du = u1 - u0

        # FEP weights
        w_unnorm = torch.exp(-du)
        w = w_unnorm / w_unnorm.sum()

        rg_mean = (w * rg).sum()
        obs_loss = (rg_mean - rg_exp) ** 2

        ess_val = compute_ess(u1, u0)
        ess_loss = ess_reg(ess_val, self.alpha, self.ess0)

        total_loss = obs_loss + ess_loss
        return total_loss, obs_loss.detach(), ess_val.detach(), rg_mean.detach()

    def total_loss(self):
        """
        Average over all proteins:
          L = mean_p [ L_FEP(p) ] + L2 regs
        """
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        n_used = 0
        stats = {}

        for prot in self.protein_list:
            if prot not in self.protein_data:
                continue
            loss_p, obs_p, ess_p, rg_mean_p = self.fep_loss_for_protein(prot)
            total = total + loss_p
            n_used += 1
            stats[prot] = {
                'obs_loss': float(obs_p.cpu().item()),
                'ESS': float(ess_p.cpu().item()),
                'Rg_mean': float(rg_mean_p.cpu().item()),
                'Rg_exp': float(self.exp_rg_dict[prot]),
            }

        if n_used > 0:
            total = total / n_used

        # L2 regularization on deltas (strongly recommended esp. density)
        if self.lam_ah > 0.0:
            total = total + self.lam_ah * torch.sum(self.delta_ah_coeff ** 2)
        if self.lam_gauss > 0.0:
            total = total + self.lam_gauss * torch.sum(self.delta_gauss_coeff ** 2)
        if self.lam_density > 0.0:
            total = total + self.lam_density * torch.sum(self.delta_spl_coeff ** 2)

        return total, stats

    # ─────────────────────────────────────────
    # Export updated results.pkl
    # ─────────────────────────────────────────
    def export_results(self, original_results_pkl, output_path):
        """
        Write out a new results.pkl containing:
          - updated hydrophobic_scale (20x20)
          - updated gauss_coeffs (210-dim)
          - updated spl_coeffs (nbasis-dim)
        Other keys copied from original results.
        """
        with open(original_results_pkl, 'rb') as f:
            old = pickle.load(f)

        # final AH coefficients (210) -> hydrophobic_scale 20x20
        ah_coeff_final = (self.ah_coeff0 + self.delta_ah_coeff).detach().cpu().numpy()
        hydrophobic_scale = np.zeros((20, 20), float)
        rows, cols = np.triu_indices(20, 0)
        hydrophobic_scale[rows, cols] = ah_coeff_final
        hydrophobic_scale[cols, rows] = ah_coeff_final

        # final Gaussian coeffs (210)
        gauss_coeff_final = (self.gauss_coeff0 + self.delta_gauss_coeff).detach().cpu().numpy()

        # final density spline coeffs (nbasis)
        spl_coeff_final = (self.spl_coeff0 + self.delta_spl_coeff).detach().cpu().numpy()

        new = dict(old)
        new['hydrophobic_scale'] = hydrophobic_scale
        new['gauss_coeffs'] = gauss_coeff_final
        new['spl_coeffs'] = spl_coeff_final

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(new, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\nSaved updated parameters to {output_path}", flush=True)


# ─────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_pkl', required=True)
    parser.add_argument('--fep_input_root', required=True)
    parser.add_argument('--exp_csv', required=True)
    parser.add_argument('--output_pkl', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--alpha', type=float, default=100.0)
    parser.add_argument('--ess0', type=float, default=750.0)
    parser.add_argument('--lam_ah', type=float, default=0.0)
    parser.add_argument('--lam_gauss', type=float, default=0.0)
    parser.add_argument('--lam_density', type=float, default=1e-2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--res_group_mapping', type=str, default='default')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load experimental Rg
    df_exp = pd.read_csv(args.exp_csv)
    if not {'protein', 'ref_rg'}.issubset(df_exp.columns):
        raise ValueError("Experimental CSV must contain 'protein' and 'ref_rg'.")
    exp_rg_dict = dict(zip(df_exp['protein'], df_exp['ref_rg']))

    # Select proteins
    protein_list = [p for p in ALL_PROTEINS if p in exp_rg_dict]
    print("\nWill use proteins:", protein_list, flush=True)

    # Build model
    model = FEP_AH_Gauss_Density_Model(
        results_pkl_path=args.results_pkl,
        fep_input_root=args.fep_input_root,
        protein_list=protein_list,
        exp_rg_dict=exp_rg_dict,
        alpha=args.alpha,
        ess0=args.ess0,
        lam_ah=args.lam_ah,
        lam_gauss=args.lam_gauss,
        lam_density=args.lam_density,
        device=args.device
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare loss logging
    outdir = os.path.dirname(args.output_pkl)
    os.makedirs(outdir, exist_ok=True)
    loss_csv_path = os.path.join(
        outdir,
        f"alpha_{args.alpha}_ess0_{args.ess0}_AHGaussDensity_loss.csv"
    )
    with open(loss_csv_path, "w") as f:
        f.write("epoch,total_loss\n")

    # Track whether training failed
    training_failed = False

    # ─────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        loss, stats = model.total_loss()

        # NaN/Inf detection
        if not torch.isfinite(loss):
            print(f"\n!!! STOPPING EARLY: loss became NaN/Inf at epoch {epoch} !!!",
                  flush=True)
            with open(loss_csv_path, "a") as f:
                f.write(f"{epoch},NaN\n")
            training_failed = True
            break

        loss.backward()
        optimizer.step()

        with open(loss_csv_path, "a") as f:
            f.write(f"{epoch},{loss.item():.6g}\n")

        if (epoch % 20 == 0) or (epoch == args.n_epochs - 1):
            print(f"\nEpoch {epoch:5d}, total loss = {loss.item():.6e}", flush=True)
            for prot, st in stats.items():
                print(f"  {prot}: obs_loss={st['obs_loss']:.4e}, "
                      f"ESS={st['ESS']:.1f}, "
                      f"Rg_mean={st['Rg_mean']:.3f}, "
                      f"Rg_exp={st['Rg_exp']:.3f}",
                      flush=True)

    # ─────────────────────────────────────────
    # Abort export + reweight if training failed
    # ─────────────────────────────────────────
    if training_failed:
        print("\n⚠️ Training ended with NaN — NOT exporting results.pkl "
              "and NOT running reweighting.", flush=True)
        print("Please adjust lr or regularization and rerun.", flush=True)
        return

    # ─────────────────────────────────────────
    # Export updated results.pkl
    # ─────────────────────────────────────────
    model.export_results(args.results_pkl, args.output_pkl)

    print("\nRunning full reweighting with optimized parameters...", flush=True)

    reweight_outdir = os.path.join(os.path.dirname(args.output_pkl), "tests")
    os.makedirs(reweight_outdir, exist_ok=True)

    run_full_reweighting(
        results_pkl=args.output_pkl,
        output_dir=reweight_outdir,
        res_group_mapping=args.res_group_mapping,
    )

    print("Reweighting finished.", flush=True)


if __name__ == '__main__':
    main()

