import numpy as np
import pandas as pd
import torch
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit

from FastMBAR import FastMBAR
from openabc.forcefields.parsers import MOFFParser, HPSParser
import warnings
warnings.filterwarnings('ignore')
import time
import mdtraj
import pickle
import json
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})

from openabc.forcefields.MOFF2.core import compute_PE, compute_PE_torch_from_fep
from openabc.forcefields.MOFF2.lib import GAS_CONST, _amino_acids, _kcal_to_kj, kB, VEP, EC, NA, _res_group_mappings
from openabc.forcefields.MOFF2.utils import compute_CA_traj_radius_of_gyration, select_native_pairs_all_ss_dssp, compute_distance_matrices
from openabc.forcefields.MOFF2.forcefields import MOFF2Model, density_spline_term_group_i

# =====================================================================
#  Protein lists (UNCHANGED)
# =====================================================================
robustelli2018developing_IDPs = ['ACTR', 'Abeta40', 'Ash1', 'NTail', 'alpha-synuclein', 'drkN-SH3', 'p15PAF', 'sic1']
lindorff2011fast_OPs = ['Chignolin', 'Homeodomain', 'Trp-cage', 'Villin', 'WW-domain', 'Protein-G']
piana2020development_OPs = ['NTL9', 'Protein-B', 'alpha3d', 'bba', 'bbl', 'engrailed', 'gpw', 'lambda-repressor', 'BPTI', 'calmodulin']
robustelli2018developing_OPs = ['Ubiquitin', 'GB3', 'Hen-Egg-White-Lysozyme']
extended_OPs = [
    '1soy_clean', '1wla_clean', '2ea9_clean', '5tvz_clean',
    'AF-P0C232-F1-model_v4_w_H', 'AF-P0CG98-F1-model_v4_w_H',
    'AF-P00251-F1-model_v4_w_H', 'AF-P21149-F1-model_v4_w_H',
    'AF-P21318-F1-model_v4_w_H', 'AF-P29669-F1-model_v4_w_H',
    'AF-P31960-F1-model_v4_w_H', 'AF-P32729-F1-model_v4_w_H',
    'AF-P61734-F1-model_v4_w_H', 'AF-P69995-F1-model_v4_w_H',
    'AF-P75202-F1-model_v4_w_H', 'AF-P75459-F1-model_v4_w_H',
    'AF-P80353-F1-model_v4_w_H', 'AF-P87285-F1-model_v4_w_H'
]

ordered_proteins = lindorff2011fast_OPs + piana2020development_OPs + robustelli2018developing_OPs + extended_OPs

MDPs = [
    'THB-C2', 'Ub2', 'Ub3', 'Gal3', 'hnRNPA1-star', 'FPs-GS8', 'FPs-GS16',
    'SH4UD-SH3-SH2', 'TDP43_WtoA', 'D12', 'D23', 'D34', 'SMAD4'
]


# =====================================================================
#  FULL protein_params dictionary (UNCHANGED)
# =====================================================================
protein_params = {
    # --- OPs ---
    "BPTI": {"T0": 300, "T1": 300, "ionic": 37.02},
    "Chignolin": {"T0": 300, "T1": 340, "ionic": 25.9},
    "GB3": {"T0": 300, "T1": 300, "ionic": 9.98},
    "Hen-Egg-White-Lysozyme": {"T0": 300, "T1": 300, "ionic": 26.63},
    "Homeodomain": {"T0": 300, "T1": 360, "ionic": 45},
    "NTL9": {"T0": 300, "T1": 300, "ionic": 121.46},
    "Protein-B": {"T0": 300, "T1": 300, "ionic": 51.44},
    "Protein-G": {"T0": 300, "T1": 350, "ionic": 100},
    "Trp-cage": {"T0": 300, "T1": 290, "ionic": 65},
    "Ubiquitin": {"T0": 300, "T1": 300, "ionic": 150},
    "Villin": {"T0": 300, "T1": 360, "ionic": 40},
    "WW-domain": {"T0": 300, "T1": 360, "ionic": 7.1},
    "alpha3d": {"T0": 300, "T1": 300, "ionic": 3.11},
    "bba": {"T0": 300, "T1": 300, "ionic": 33.04},
    "bbl": {"T0": 300, "T1": 300, "ionic": 199.58},
    "calmodulin": {"T0": 300, "T1": 300, "ionic": 249.98},
    "engrailed": {"T0": 300, "T1": 300, "ionic": 280.46},
    "gpw": {"T0": 300, "T1": 300, "ionic": 36.54},
    "lambda-repressor": {"T0": 300, "T1": 300, "ionic": 51.04},

    # --- IDPs ---
    "ACTR": {"T0": 300, "T1": 300, "ionic": 150},
    "Abeta40": {"T0": 300, "T1": 300, "ionic": 50},
    "Ash1": {"T0": 300, "T1": 300, "ionic": 150},
    "NTail": {"T0": 300, "T1": 300, "ionic": 100},
    "alpha-synuclein": {"T0": 300, "T1": 300, "ionic": 100},
    "drkN-SH3": {"T0": 300, "T1": 300, "ionic": 50},
    "p15PAF": {"T0": 300, "T1": 300, "ionic": 50},
    "sic1": {"T0": 300, "T1": 300, "ionic": 150},

    # --- MDPs ---
    "THB-C2": {"T0": 295.15, "T1": 295.15, "ionic": 146},
    "Ub2": {"T0": 293.0, "T1": 293.0, "ionic": 332},
    "Ub3": {"T0": 293.0, "T1": 293.0, "ionic": 332},
    "Gal3": {"T0": 303.0, "T1": 303.0, "ionic": 337},
    "hnRNPA1-star": {"T0": 293.15, "T1": 293.15, "ionic": 150},
    "FPs-GS8": {"T0": 293.15, "T1": 293.15, "ionic": 149},
    "FPs-GS16": {"T0": 293.15, "T1": 293.15, "ionic": 149},
    "SH4UD-SH3-SH2": {"T0": 293.15, "T1": 293.15, "ionic": 216},
    "TDP43_WtoA": {"T0": 293.15, "T1": 293.15, "ionic": 312},
    "D12": {"T0": 283.15, "T1": 283.15, "ionic": 156},
    "D23": {"T0": 283.15, "T1": 283.15, "ionic": 156},
    "D34": {"T0": 283.15, "T1": 283.15, "ionic": 156},
    "SMAD4": {"T0": 283.15, "T1": 283.15, "ionic": 188},

    # --- NMR IDPs ---
    "IBB": {"T0": 300, "T1": 300, "ionic": 168},
    "N49": {"T0": 300, "T1": 300, "ionic": 168},
    "NUS": {"T0": 300, "T1": 300, "ionic": 168},
    "NUL": {"T0": 300, "T1": 300, "ionic": 168},
    "NLS": {"T0": 300, "T1": 300, "ionic": 159},
    "K18": {"T0": 300, "T1": 288, "ionic": 168},
    "K19": {"T0": 300, "T1": 288, "ionic": 168},
    "K25": {"T0": 300, "T1": 288, "ionic": 168},
    "Hst5": {"T0": 300, "T1": 293, "ionic": 150},
    "SH4UD": {"T0": 300, "T1": 293, "ionic": 216},
    "A1_no_NLS": {"T0": 300, "T1": 298, "ionic": 153},
    "A1_12F_to_12Y": {"T0": 300, "T1": 298, "ionic": 153},
    "A1_7Y_to_7F": {"T0": 300, "T1": 298, "ionic": 153},
    "A1_add_7R": {"T0": 300, "T1": 298, "ionic": 153},
    "A1_6R_to_6K": {"T0": 300, "T1": 298, "ionic": 153},
    "A1_10R_to_10K": {"T0": 300, "T1": 298, "ionic": 153},
    "A1_add_12D": {"T0": 300, "T1": 298, "ionic": 153},
    "A1_add_12E": {"T0": 300, "T1": 298, "ionic": 153},
    "A1_add_7K_12D": {"T0": 300, "T1": 298, "ionic": 153},
    "Hst5_2": {"T0": 300, "T1": 298, "ionic": 168},
    "p53_NTD": {"T0": 300, "T1": 277, "ionic": 99},

    # --- EVO ---
    **{
        f"Evo{i}": {"T0": 300, "T1": 300, "ionic": 150}
        for i in list(range(1, 11)) + [12, 14, 16, 18, 19, 20, 21, 22,
                                       23, 24, 25, 26, 30, 31, 33, 34,
                                       38, 41, 42, 46, 47, 48, 51, 52]
    },

    # --- AlphaFold OPs ---
    "1soy_clean": {"T0": 300, "T1": 300, "ionic": 150},
    "1wla_clean": {"T0": 300, "T1": 300, "ionic": 150},
    "2ea9_clean": {"T0": 300, "T1": 300, "ionic": 150},
    "5tvz_clean": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P0C232-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P0CG98-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P00251-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P21149-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P21318-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P29669-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P31960-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P32729-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P61734-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P69995-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P75202-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P75459-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P80353-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
    "AF-P87285-F1-model_v4_w_H": {"T0": 300, "T1": 300, "ionic": 150},
}


# =====================================================================
#  MAIN FUNCTION (EXACT COPY WRAPPED)
# =====================================================================

def run_one_protein(
    protein,
    results_pkl,
    output_dir,
    res_group_mapping="default",
):

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------
    # CONDITIONS FIX
    # ---------------------
    if protein not in protein_params:
        raise KeyError(f"Protein {protein} not found in condition table!")

    T0 = protein_params[protein]["T0"]
    T1 = protein_params[protein]["T1"]
    ionic_strength = protein_params[protein]["ionic"]

    # ---- from here on, everything is your EXACT SCRIPT, with args.* replaced ----

    # load noise samples
    if protein in ordered_proteins:
        noise_main_dir = '/orcd/data/binz/001/congwang/TW-PCCG-develop/TW-PCCG-develop/noise-cg-simulations/noise-HPS-Urry-sbm-T-5ldby-simulations'
        noise_simulation_dirs = sorted(glob.glob(f'{noise_main_dir}/{protein}-HPS-Urry-sbm/rmsd-biased-simulations/kappa*center*'))
        unbiased_system_xml = f'{noise_main_dir}/{protein}-HPS-Urry-sbm/unbiased-system/unbiased_system.xml'
        ca_pdb = f'{noise_main_dir}/{protein}-HPS-Urry-sbm/unbiased-system/{protein}_ca.pdb'

    elif protein in MDPs:
        noise_main_dir = '/orcd/data/binz/001/congwang/TW-PCCG-develop/TW-PCCG-develop/noise-cg-simulations/noise-TWPCCG-multi-v1-MDP'
        noise_simulation_dirs = sorted(glob.glob(f'{noise_main_dir}/{protein}-TWPCCG-multi-v1-MDP/rg-biased-simulations/kappa*center*'))
        unbiased_system_xml = f'{noise_main_dir}/{protein}-TWPCCG-multi-v1-MDP/unbiased-system/system.xml'
        ca_pdb = f'{noise_main_dir}/{protein}-TWPCCG-multi-v1-MDP/unbiased-system/{protein}_ca.pdb'

    else:
        noise_main_dir = '/orcd/data/binz/001/congwang/TW-PCCG-develop/TW-PCCG-develop/noise-cg-simulations/noise-HPS-Urry-simulations'
        noise_simulation_dirs = sorted(glob.glob(f'{noise_main_dir}/{protein}-HPS-Urry/rg-biased-simulations/kappa*center*'))
        unbiased_system_xml = f'{noise_main_dir}/{protein}-HPS-Urry/unbiased-system/unbiased_system.xml'
        ca_pdb = f'{noise_main_dir}/{protein}-HPS-Urry/unbiased-system/{protein}_ca.pdb'

    assert os.path.exists(unbiased_system_xml)
    assert os.path.exists(ca_pdb)

    traj0_list = []
    kappa_list = []
    center_list = []

    for each_dir in noise_simulation_dirs:
        each_traj = mdtraj.load_dcd(f'{each_dir}/output.dcd', top=ca_pdb)
#        if each_traj.n_frames > 5000:
#            selected_indices = np.sort(np.random.choice(each_traj.n_frames, 5000, replace=False))
#            each_traj = each_traj[selected_indices]
        traj0_list.append(each_traj)
        with open(f'{each_dir}/input_parameters.json') as f:
            parameters = json.load(f)
        kappa = parameters['kappa']
        center = parameters['center']
        kappa_list.append(kappa)
        center_list.append(center)

    n_frames0_array = np.array([x.n_frames for x in traj0_list])
    traj0 = mdtraj.join(traj0_list)

    # compute unbiased energy U0
    with open(unbiased_system_xml, 'r') as f:
        unbiased_system = mm.XmlSerializer.deserialize(f.read())

    unbiased_U0 = compute_PE(traj0, unbiased_system, platform_name='CUDA')

    # compute the biased energy and reduced A matrix
    T0_unit = T0 * unit.kelvin
    RT0_value = (GAS_CONST * T0_unit).value_in_unit(unit.kilojoule_per_mole)
    T1_unit = T1 * unit.kelvin
    RT1_value = (GAS_CONST * T1_unit).value_in_unit(unit.kilojoule_per_mole)

    kappa_arr = np.array(kappa_list)[:, None]
    center_arr = np.array(center_list)[:, None]

    if protein in ordered_proteins:
        noise_main_dir2 = '/orcd/data/binz/001/congwang/TW-PCCG-develop/TW-PCCG-develop/noise-cg-simulations/noise-HPS-Urry-sbm-T-5ldby-simulations'
        ref_ca_pdb = f'{noise_main_dir2}/{protein}-HPS-Urry-sbm/unbiased-system/{protein}_ca.pdb'
        ref_ca_traj = mdtraj.load_pdb(ref_ca_pdb)
        cv0 = mdtraj.rmsd(traj0[:], ref_ca_traj[0])
    else:
        cv0 = compute_CA_traj_radius_of_gyration(traj0)

    A = (unbiased_U0 + 0.5 * kappa_arr * (cv0 - center_arr)**2) / RT0_value


    # ------------------------------------------------
    # U1 from Torch FEP (βU1 = U1 / (R T1))
    # ------------------------------------------------
    # Find the torch-FEP pkl for this protein
    fep_root = "/orcd/data/binz/001/congwang/TW-PCCG-develop/TW-PCCG-develop/train-ca-models/transferable-ca-models/training-input/n0-50000-n1-50000"
    fep_dir = os.path.join(fep_root, protein)
    fep_candidates = glob.glob(os.path.join(fep_dir, "*_gau_torch.pkl"))
    if len(fep_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one *_gau_torch.pkl in {fep_dir}, "
            f"found {len(fep_candidates)}"
        )
    fep_pkl = fep_candidates[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unbiased_U1_over_RT = compute_PE_torch_from_fep(
        fep_pkl=fep_pkl,
        results_pkl=results_pkl,
        device=device,
        return_numpy=True,
    )

    # -------------------------------
    # reweighting
    # -------------------------------
    rg = compute_CA_traj_radius_of_gyration(traj0)
    rg_min = np.min(rg)
    rg_max = np.max(rg)
    n_bins = 50
    bins = np.linspace(rg_min - 1e-6, rg_max + 1e-6, n_bins + 1)
    B = np.zeros((n_bins, traj0.n_frames))

    for i in range(n_bins):
        # unbiased_U1_over_RT is already U1 / (R T1), no extra division
        B[i] = unbiased_U1_over_RT.copy()
        left = bins[i]
        right = bins[i + 1]
        B[i, rg < left] = np.inf
        B[i, rg >= right] = np.inf






    time0 = time.time()
    cuda = torch.cuda.is_available()

    fastmbar = FastMBAR(energy=A, num_conf=n_frames0_array, cuda=cuda, verbose=True)
    fastmbar_results = fastmbar.calculate_free_energies_of_perturbed_states(B)
    time1 = time.time()

    if cuda:
        print(f'FastMBAR on CUDA took {(time1-time0):.2f} seconds')
    else:
        print(f'FastMBAR on CPU took {(time1-time0):.2f} seconds')

    f_reweight = fastmbar_results['F']
    f_reweight -= np.min(f_reweight)
    w = np.exp(-f_reweight)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rg_mean = np.average(centers, weights=w)

    out_json = {
        "protein": protein,
        "T0": T0,
        "T1": T1,
        "ionic_strength": ionic_strength,
        "rg_mean": rg_mean,
    }

    with open(f"{output_dir}/reweight_rg.json", "w") as f:
        json.dump(out_json, f, indent=4)

    # ---- draw PMF ----
    plt.plot(centers, f_reweight, label='reweight')

    test_dcd = f"{output_dir}/output.dcd"
    if os.path.exists(test_dcd):
        test_traj = mdtraj.load_dcd(test_dcd, top=ca_pdb)
        rg_test = compute_CA_traj_radius_of_gyration(test_traj)
        hist_test, _ = np.histogram(rg_test, bins=bins)
        f_test = -np.log(hist_test + 1e-6)
        f_test -= np.min(f_test)
        plt.plot(centers, f_test, label='test simulation')

    plt.xlabel('Rg (nm)')
    plt.ylabel('Free energy (kT)')
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 5)
    plt.savefig(f"{output_dir}/rg_pmf.pdf")
    plt.close()


# ----------------------------------------------------------------------
# Small helper: build sigma matrix
# ----------------------------------------------------------------------
def _build_sigma():
    p_path = '/orcd/data/binz/001/congwang/TW-PCCG-develop/TW-PCCG-develop/train-ca-models/transferable-ca-models/'
    df = pd.read_csv(p_path + 'parameters/HPS_Urry_parameters.csv')
    sigma = np.zeros((20,20))
    for _, row in df.iterrows():
        i = _amino_acids.index(row['atom_type1'])
        j = _amino_acids.index(row['atom_type2'])
        sigma[i,j] = sigma[j,i] = row['sigma']
    return sigma


# =====================================================================
# RUN ALL PROTEINS
# =====================================================================
def run_full_reweighting(
    results_pkl,
    output_dir,
    res_group_mapping="default",
):
    os.makedirs(output_dir, exist_ok=True)
    for protein in protein_params.keys():
        print(f"\n========= Reweighting {protein} =========\n", flush=True)
        out_p = os.path.join(output_dir, protein)
        os.makedirs(out_p, exist_ok=True)

        run_one_protein(
            protein=protein,
            results_pkl=results_pkl,
            output_dir=out_p,
            res_group_mapping=res_group_mapping,
        )
