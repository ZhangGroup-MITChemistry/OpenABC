import numpy as np
import pandas as pd
from openabc.lib import _kcal_to_kj, NA, EC, VEP
try:
    import openmm.unit as unit
except ImportError:
    import simtk.unit as unit

"""
Prepare the parameter file for the explicit ion model.
"""

# parameters between two charged species
# vdwl parameters
ion_charged_atom_epsilon_dict = {('P', 'P'): 0.18379 * _kcal_to_kj, 
                                 ('NA', 'P'): 0.02510 * _kcal_to_kj, 
                                 ('NA', 'AA+'): 0.239 * _kcal_to_kj, 
                                 ('NA', 'AA-'): 0.239 * _kcal_to_kj, 
                                 ('MG', 'P'): 0.1195 * _kcal_to_kj, 
                                 ('MG', 'AA+'): 0.239 * _kcal_to_kj, 
                                 ('MG', 'AA-'): 0.239 * _kcal_to_kj, 
                                 ('CL', 'P'): 0.08121 * _kcal_to_kj, 
                                 ('CL', 'AA+'): 0.239 * _kcal_to_kj, 
                                 ('CL', 'AA-'): 0.239 * _kcal_to_kj, 
                                 ('NA', 'NA'): 0.01121 * _kcal_to_kj, 
                                 ('NA', 'MG'): 0.04971 * _kcal_to_kj, 
                                 ('NA', 'CL'): 0.08387 * _kcal_to_kj, 
                                 ('MG', 'MG'): 0.89460 * _kcal_to_kj, 
                                 ('MG', 'CL'): 0.49737 * _kcal_to_kj, 
                                 ('CL', 'CL'): 0.03585 * _kcal_to_kj}
ion_charged_atom_sigma_dict = {('P', 'P'): 0.686, 
                               ('NA', 'P'): 0.414, 
                               ('NA', 'AA+'): 0.4065, 
                               ('NA', 'AA-'): 0.4065,
                               ('MG', 'P'): 0.487, 
                               ('MG', 'AA+'): 0.3556, 
                               ('MG', 'AA-'): 0.3556, 
                               ('CL', 'P'): 0.55425, 
                               ('CL', 'AA+'): 0.48725, 
                               ('CL', 'AA-'): 0.48725, 
                               ('NA', 'NA'): 0.243, 
                               ('NA', 'MG'): 0.237, 
                               ('NA', 'CL'): 0.31352, 
                               ('MG', 'MG'): 0.1412, 
                               ('MG', 'CL'): 0.474, 
                               ('CL', 'CL'): 0.4045}

# hydration parameters
H1_dict = {('NA', 'P'): 3.15488 * _kcal_to_kj, 
           ('NA', 'AA+'): 3.15488 * _kcal_to_kj, 
           ('NA', 'AA-'): 3.15488 * _kcal_to_kj, 
           ('MG', 'P'): 1.29063 * _kcal_to_kj, 
           ('MG', 'AA+'): 1.29063 * _kcal_to_kj, 
           ('MG', 'AA-'): 1.29063 * _kcal_to_kj, 
           ('CL', 'P'): 0.83652 * _kcal_to_kj, 
           ('CL', 'AA+'): 0.83652 * _kcal_to_kj, 
           ('CL', 'AA-'): 0.83652 * _kcal_to_kj, 
           ('NA', 'NA'): 0.17925 * _kcal_to_kj, 
           ('NA', 'CL'): 5.49713 * _kcal_to_kj, 
           ('MG', 'CL'): 1.09943 * _kcal_to_kj, 
           ('CL', 'CL'): 0.23901 * _kcal_to_kj}
H2_dict = {('NA', 'P'): 0.47801 * _kcal_to_kj, 
           ('NA', 'AA-'): 0.47801 * _kcal_to_kj, 
           ('MG', 'P'): 0.97992 * _kcal_to_kj, 
           ('MG', 'AA-'): 0.97992 * _kcal_to_kj, 
           ('CL', 'AA+'): 0.47801 * _kcal_to_kj, 
           ('NA', 'CL'): 0.47801 * _kcal_to_kj, 
           ('MG', 'CL'): 0.05975 * _kcal_to_kj}
mu1_dict = {('NA', 'P'): 0.41, ('NA', 'AA+'): 0.41, 
            ('NA', 'AA-'): 0.41, ('MG', 'P'): 0.61, 
            ('MG', 'AA+'): 0.61, ('MG', 'AA-'): 0.61, 
            ('CL', 'P'): 0.67, ('CL', 'AA+'): 0.67, 
            ('CL', 'AA-'): 0.67, ('NA', 'NA'): 0.58, 
            ('NA', 'CL'): 0.33, ('MG', 'CL'): 0.548, 
            ('CL', 'CL'): 0.62}
mu2_dict = {('NA', 'P'): 0.65, ('NA', 'AA-'): 0.65, 
            ('MG', 'P'): 0.83, ('MG', 'AA-'): 0.83, 
            ('CL', 'AA+'): 0.56, ('NA', 'CL'): 0.56, 
            ('MG', 'CL'): 0.816}
eta1_dict = {('NA', 'P'): 0.057, ('NA', 'AA+'): 0.057, 
             ('NA', 'AA-'): 0.057, ('MG', 'P'): 0.05, 
             ('MG', 'AA+'): 0.05, ('MG', 'AA-'): 0.05, 
             ('CL', 'P'): 0.15, ('CL', 'AA+'): 0.15, 
             ('CL', 'AA-'): 0.15, ('NA', 'NA'): 0.057, 
             ('NA', 'CL'): 0.057, ('MG', 'CL'): 0.044, 
             ('CL', 'CL'): 0.05}
eta2_dict = {('NA', 'P'): 0.04, ('NA', 'AA-'): 0.04, 
             ('MG', 'P'): 0.12, ('MG', 'AA-'): 0.12, 
             ('CL', 'AA+'): 0.04, ('NA', 'CL'): 0.04, 
             ('MG', 'CL'): 0.035}

# distance dependent dielectric parameters
r_D_dict = {('P', 'P'): 0.686, 
            ('NA', 'P'): 0.344, 
            ('NA', 'AA+'): 0.344, 
            ('NA', 'AA-'): 0.344, 
            ('MG', 'P'): 0.375, 
            ('MG', 'AA+'): 0.375, 
            ('MG', 'AA-'): 0.375, 
            ('CL', 'P'): 0.42, 
            ('CL', 'AA+'): 0.42, 
            ('CL', 'AA-'): 0.42, 
            ('NA', 'NA'): 0.27, 
            ('NA', 'MG'): 0.237, 
            ('NA', 'CL'): 0.39, 
            ('MG', 'MG'): 0.1412, 
            ('MG', 'CL'): 0.448, 
            ('CL', 'CL'): 0.42}
zeta_dict = {('P', 'P'): 0.05, 
             ('NA', 'P'): 0.125, 
             ('NA', 'AA+'): 0.125, 
             ('NA', 'AA-'): 0.125, 
             ('MG', 'P'): 0.1, 
             ('MG', 'AA+'): 0.1, 
             ('MG', 'AA-'): 0.1, 
             ('CL', 'P'): 0.05, 
             ('CL', 'AA+'): 0.05, 
             ('CL', 'AA-'): 0.05, 
             ('NA', 'NA'): 0.057, 
             ('NA', 'MG'): 0.05, 
             ('NA', 'CL'): 0.206, 
             ('MG', 'MG'): 0.05, 
             ('MG', 'CL'): 0.057, 
             ('CL', 'CL'): 0.056}

# save parameters as dataframe
# units are in OpenMM standard unit (kJ/mol, nm)
df_param = pd.DataFrame(columns=['atom_type1', 'atom_type2', 
                                 'epsilon', 'sigma', 'cutoff_lj', 
                                 'gamma1', 'mu1', 'eta1', 
                                 'gamma2', 'mu2', 'eta2', 'cutoff_hydr',
                                 'r_D', 'zeta'])
all_pairs = list(ion_charged_atom_epsilon_dict.keys())
_ions = ['NA', 'MG', 'CL']
charge_dict = {'P': -1, 'AA+': 1, 'AA-': -1, 'NA': 1, 'MG': 2, 'CL': -1}
for p in all_pairs:
    assert (p[0] in _ions) or (p[0] == 'P')
    epsilon = ion_charged_atom_epsilon_dict[p]
    sigma = ion_charged_atom_sigma_dict[p]
    cutoff_lj = 1.2
    if p in H1_dict:
        H1 = H1_dict[p]
        mu1 = mu1_dict[p]
        eta1 = eta1_dict[p]
        gamma1 = H1 / (eta1 * (2 * np.pi)**0.5)
    else:
        H1 = np.nan
        mu1 = np.nan
        eta1 = np.nan
        gamma1 = np.nan
    if p in H2_dict:
        H2 = H2_dict[p]
        mu2 = mu2_dict[p]
        eta2 = eta2_dict[p]
        gamma2 = H2 / (eta2 * (2 * np.pi)**0.5)
    else:
        H2 = np.nan
        mu2 = np.nan
        eta2 = np.nan
        gamma2 = np.nan
    if np.isnan(H1) and np.isnan(H2):
        cutoff_hydr = np.nan
    else:
        cutoff_hydr = 1.2
    q1 = charge_dict[p[0]]
    if p[1] in charge_dict:
        q2 = charge_dict[p[1]]
    else:
        q2 = 0
    r_D = r_D_dict[p]
    zeta = zeta_dict[p]
    row = [p[0], p[1], epsilon, sigma, cutoff_lj, 
           gamma1, mu1, eta1, gamma2, mu2, eta2, cutoff_hydr, r_D, zeta]
    df_param.loc[len(df_param)] = row

# parameters between ion and neutral atoms
ion_neutral_atom_sigma_dict = {('NA', 'S'): 0.4315, 
                               ('NA', 'A'): 0.3915, 
                               ('NA', 'T'): 0.4765, 
                               ('NA', 'G'): 0.3665, 
                               ('NA', 'C'): 0.4415, 
                               ('NA', 'AA'): 0.4065, 
                               ('MG', 'S'): 0.3806, 
                               ('MG', 'A'): 0.3406, 
                               ('MG', 'T'): 0.4256, 
                               ('MG', 'G'): 0.3156, 
                               ('MG', 'C'): 0.3906, 
                               ('MG', 'AA'): 0.3556, 
                               ('CL', 'S'): 0.51225, 
                               ('CL', 'A'): 0.47225, 
                               ('CL', 'T'): 0.55725, 
                               ('CL', 'G'): 0.44725, 
                               ('CL', 'C'): 0.52225, 
                               ('CL', 'AA'): 0.48725}

for p in ion_neutral_atom_sigma_dict:
    assert p[0] in _ions
    epsilon = 0.239 * _kcal_to_kj # epsilon value for all the ion-neutral atom pairs
    sigma = ion_neutral_atom_sigma_dict[p]
    cutoff_lj = 2**(1 / 6) * sigma
    H1 = np.nan
    mu1 = np.nan
    eta1 = np.nan
    gamma1 = np.nan
    H2 = np.nan
    mu2 = np.nan
    eta2 = np.nan
    gamma2 = np.nan
    cutoff_hydr = np.nan
    q1 = charge_dict[p[0]]
    if p[1] in charge_dict:
        q2 = charge_dict[p[1]]
    else:
        q2 = 0
    r_D = np.nan
    zeta = np.nan
    row = [p[0], p[1], epsilon, sigma, cutoff_lj, 
           gamma1, mu1, eta1, gamma2, mu2, eta2, cutoff_hydr, r_D, zeta]
    df_param.loc[len(df_param)] = row

#print(df_param)
df_param = df_param.fillna('N/A')
df_param.to_csv('smog_3spn2_explicit_ion_parameters.csv', index=False)


