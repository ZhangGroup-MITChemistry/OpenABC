import numpy as np
import pandas as pd
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import math
from openabc.forcefields.MOFF2.utils import dual_boundary_tanh_step


def direct_water_burial_term(atom_types, exclusions, use_pbc, gamma_direct, gamma_water, gamma_protein, 
                             gamma_burial, eta_direct=10.0, r1_min=0.0, r1_max=0.7, eta_water=10.0, 
                             r2_min=0.7, r2_max=1.1, rho0=4.6, rho_use_exclusions=False, 
                             rho_lim=np.array([[2, 5], [5, 8], [8, 11]]), direct_on=True, water_on=True, 
                             burial_on=True, force_group=4):
    """
    Direct contact, water-mediated contact, and burial terms.
    
    Parameters
    ----------
    atom_types : 1d array_like, shape=(n_atoms,)
        Atom types.
    
    exclusions : None or 2d array_like or pd.DataFrame
        Nonbonded exclusions. 
        If None, no exclusions. 
        If 2d array_like, shape = (n_exclusions, 2). 
        If pd.DataFrame, excluded pairs are specificed by columns 'a1' and 'a2'.
    
    use_pbc : bool
        Whether to use periodic boundary conditions.
    
    gamma_direct : 2d array-like, shape = (20, 20)
        Direct contact interaction strength.
    
    gamma_water : 2d array-like, shape = (20, 20)
        Water-mediated contact interaction strength.
    
    gamma_protein : 2d array-like, shape = (20, 20)
        Protein-mediated contact interaction strength.
    
    gamma_burial : 2d array-like, shape = (20, len(rho_lim))
        Burial interaction strength.
    
    eta_direct : float or int
        Direct contact eta_direct parameter in unit 1 / nm.
    
    r1_min : float or int
        Direct contact range lower bound in unit nm.
    
    r1_max : float or int
        Direct contact range upper bound in unit nm.
    
    eta_water : float or int
        Water-mediated contact eta_water parameter in unit 1 / nm.
    
    r2_min : float or int
        Water-mediated contact range lower bound in unit nm.
    
    r2_max : float or int
        Water-mediated contact range upper bound in unit nm.
    
    rho0 : float or int
        Water-mediated contact density threshold value.
    
    rho_use_exclusions : bool
        Whether to use exclusions when computing rho.
        If rho_use_exclusions is False, normally you may want to increase rho0 and rho_lim correspondingly.
    
    rho_lim : 2d array-like, shape = (n_rho_ranges, 2)
        rho value ranges when computing burial potential.
        rho[i] is 1d array-like of shape (2,). 
        rho[i][0] is the lower bound of the i-th rho range.
        rho[i][1] is the upper bound of the i-th rho range.
    
    direct_on : bool
        Whether to turn on direct contact interactions.
    
    water_on : bool
        Whether to turn on water-mediated contact interactions.
    
    burial_on : bool
        Whether to turn on burial interactions.
    
    force_group : int
        Force group.
    
    Returns
    -------
    gb : openmm.CustomGBForce
        The CustomGBForce object.
    
    """
    
    assert r1_max >= r1_min
    assert r2_max >= r2_min
    
    # set CustomGBForce
    gb = mm.CustomGBForce()
    
    # set direct contact interactions
    cutoff_direct = r1_max + 10 / eta_direct
    offset_direct = dual_boundary_tanh_step(np.array([cutoff_direct]), r1_min, r1_max, eta_direct)[0]
    if direct_on:
        if not isinstance(gamma_direct, np.ndarray):
            gamma_direct = np.array(gamma_direct)
        gamma_direct_map = mm.Discrete2DFunction(20, 20, gamma_direct.flatten(order='F'))
        gb.addTabulatedFunction('gamma_direct_map', gamma_direct_map)
        gb.addEnergyTerm(f'''-gamma_direct*(switch_direct-{offset_direct})*step({cutoff_direct}-r);
                         switch_direct=(1+tanh({eta_direct}*(r-{r1_min})))*(1+tanh({eta_direct}*({r1_max}-r)))/4;
                         gamma_direct=gamma_direct_map(atom_type1, atom_type2);
                         ''', 
                         mm.CustomGBForce.ParticlePair)
    
    # set rho
    if rho_use_exclusions:
        gb.addComputedValue('rho', 
                            f'''(switch_rho-{offset_direct})*step({cutoff_direct}-r);
                            switch_rho=(1+tanh({eta_direct}*(r-{r1_min})))*(1+tanh({eta_direct}*({r1_max}-r)))/4; 
                            ''', 
                            mm.CustomGBForce.ParticlePair)
    else:
        gb.addComputedValue('rho', 
                            f'''(switch_rho-{offset_direct})*step({cutoff_direct}-r);
                            switch_rho=(1+tanh({eta_direct}*(r-{r1_min})))*(1+tanh({eta_direct}*({r1_max}-r)))/4; 
                            ''', 
                            mm.CustomGBForce.ParticlePairNoExclusions)
    
    # set water-mediated interactions
    cutoff_water = r2_max + 10 / eta_water
    offset_water = dual_boundary_tanh_step(np.array([cutoff_water]), r2_min, r2_max, eta_water)[0]
    eta_rho = 7.0
    if water_on:
        if not isinstance(gamma_water, np.ndarray):
            gamma_water = np.array(gamma_water)
        if not isinstance(gamma_protein, np.ndarray):
            gamma_protein = np.array(gamma_protein)
        gamma_water_map = mm.Discrete2DFunction(20, 20, gamma_water.flatten(order='F'))
        gb.addTabulatedFunction('gamma_water_map', gamma_water_map)
        gamma_protein_map = mm.Discrete2DFunction(20, 20, gamma_protein.flatten(order='F'))
        gb.addTabulatedFunction('gamma_protein_map', gamma_protein_map)
        gb.addEnergyTerm(f'''-(nu_water*gamma_water+nu_protein*gamma_protein)*effective_switch_water;
                         effective_switch_water=(switch_water-{offset_water})*step({cutoff_water}-r);
                         switch_water=(1+tanh({eta_water}*(r-{r2_min})))*(1+tanh({eta_water}*({r2_max}-r)))/4;
                         nu_protein=1-nu_water;
                         nu_water=(1+tanh({eta_rho}*({rho0}-rho1)))*(1+tanh({eta_rho}*({rho0}-rho2)))/4;
                         gamma_water=gamma_water_map(atom_type1, atom_type2);
                         gamma_protein=gamma_protein_map(atom_type1, atom_type2);
                         ''', 
                         mm.CustomGBForce.ParticlePair)
    
    # set burial interactions
    eta_burial = 4.0
    if burial_on:
        if not isinstance(gamma_burial, np.ndarray):
            gamma_burial = np.array(gamma_burial)
        n_rho_ranges = len(rho_lim)
        gamma_burial_map = mm.Discrete2DFunction(n_rho_ranges, 20, gamma_burial.flatten(order='F'))
        gb.addTabulatedFunction('gamma_burial_map', gamma_burial_map)
        for i in range(n_rho_ranges):
            rho_min = rho_lim[i][0]
            rho_max = rho_lim[i][1]
            assert rho_min <= rho_max
            gb.addEnergyTerm(f'''
                             -gamma_burial_map({i}, atom_type)*switch_burial;
                             switch_burial=(tanh({eta_burial}*(rho-{rho_min}))+tanh({eta_burial}*({rho_max}-rho)))/2;
                             ''', 
                             mm.CustomGBForce.SingleParticle)
    
    # set particles
    gb.addPerParticleParameter('atom_type')
    for each in atom_types:
        gb.addParticle([each])
    
    # set exclusions
    if exclusions is not None:
        if isinstance(exclusions, pd.DataFrame):
            exclusions = exclusions[['a1', 'a2']].to_numpy()
        for each in exclusions:
            gb.addExclusion(int(each[0]), int(each[1]))
    
    # set PBC, cutoff, and force group
    if use_pbc:
        gb.setNonbondedMethod(gb.CutoffPeriodic)
    else:
        gb.setNonbondedMethod(gb.CutoffNonPeriodic)
    cutoff = max(cutoff_direct, cutoff_water)
    gb.setCutoffDistance(cutoff)
    gb.setForceGroup(force_group)
    return gb


def density_switch_ashbaugh_hatch_term(atom_types, df_exclusions, use_pbc, epsilon, sigma_ah, 
                                       lambda_dilute_ah, lambda_dense_ah, eta=10, r0=0.7, 
                                       mu=2.0, rho0=5.5, force_group=2):
    """
    Ashbaugh-hatch term with density based switch hydrophobic scales. 
    
    """
    # set CustomGBForce
    gb = mm.CustomGBForce()
    
    # compute local density rho
    cutoff_rho = r0 + 10 / eta # cutoff distance for computing rho
    offset_switch_rho = 0.5 * (1 + math.tanh(eta * (r0 - cutoff_rho)))
    gb.addComputedValue('rho', 
                        f'''(switch_rho-{offset_switch_rho})*step({cutoff_rho}-r);
                        switch_rho=0.5*(1+tanh({eta}*({r0}-r))); 
                        ''', 
                        mm.CustomGBForce.ParticlePairNoExclusions)
    
    # set ashbaugh-hatch term
    if not isinstance(sigma_ah, np.ndarray):
        sigma_ah = np.array(sigma_ah)
    if not isinstance(lambda_dilute_ah, np.ndarray):
        lambda_dilute_ah = np.array(lambda_dilute_ah)
    if not isinstance(lambda_dense_ah, np.ndarray):
        lambda_dense_ah = np.array(lambda_dense_ah)
    sigma_ah_map = mm.Discrete2DFunction(20, 20, sigma_ah.flatten(order='F'))
    lambda_dilute_ah_map = mm.Discrete2DFunction(20, 20, lambda_dilute_ah.flatten(order='F'))
    lambda_dense_ah_map = mm.Discrete2DFunction(20, 20, lambda_dense_ah.flatten(order='F'))
    gb.addTabulatedFunction('sigma_ah_map', sigma_ah_map)
    gb.addTabulatedFunction('lambda_dilute_ah_map', lambda_dilute_ah_map)
    gb.addTabulatedFunction('lambda_dense_ah_map', lambda_dense_ah_map)
    lj_at_cutoff = 4 * epsilon * ((1 / 4)**12 - (1 / 4)**6)
    gb.addEnergyTerm(f'''energy;
                     energy=(f1+f2-offset)*step(4*sigma_ah-r);
                     offset=lambda_ah*{lj_at_cutoff};
                     f1=(lj+(1-lambda_ah)*{epsilon})*step(2^(1/6)*sigma_ah-r);
                     f2=lambda_ah*lj*step(r-2^(1/6)*sigma_ah);
                     lj=4*{epsilon}*((sigma_ah/r)^12-(sigma_ah/r)^6);
                     lambda_ah=nu*lambda_dilute_ah+(1-nu)*lambda_dense_ah;
                     nu=(1+tanh({mu}*({rho0}-rho1)))*(1+tanh({mu}*({rho0}-rho2)))/4;
                     lambda_dilute_ah=lambda_dilute_ah_map(atom_type1, atom_type2);
                     lambda_dense_ah=lambda_dense_ah_map(atom_type1, atom_type2);
                     sigma_ah=sigma_ah_map(atom_type1, atom_type2);
                     ''', 
                     mm.CustomGBForce.ParticlePair)
    
    # set particles
    gb.addPerParticleParameter('atom_type')
    for each in atom_types:
        gb.addParticle([each])
    
    # set exclusions
    for _, row in df_exclusions.iterrows():
        gb.addExclusion(int(row['a1']), int(row['a2']))
    
    # set PBC, cutoff, and force group
    if use_pbc:
        gb.setNonbondedMethod(gb.CutoffPeriodic)
    else:
        gb.setNonbondedMethod(gb.CutoffNonPeriodic)
    cutoff = max(np.max(4 * sigma_ah), cutoff_rho)
    gb.setCutoffDistance(cutoff)
    gb.setForceGroup(force_group)
    return gb


def density_spline_term(atom_types, use_pbc, spl_values, eta=10.0, r0=0.7, 
                        rho_min=0.0, rho_max=15.0, force_group=4):
    """
    Local density spline term.
    
    Parameters
    ----------
    atom_types : 1d array-like, shape = (n_atoms,)
        Atom types.
    
    use_pbc : bool
        Whether to use periodic boundary conditions.
    
    spl_values : 2d array-like, shape = (20, n_knots)
        Spline values.
    
    eta : float or int
        Parameter eta in unit 1 / nm.
    
    r0 : float or int
        Parameter r0 in unit nm.
    
    rho_min : float or int
        Parameter rho_min.
    
    rho_max : float or int
        Parameter rho_max.
    
    force_group : int
        Force group.
    
    """
    # set CustomGBForce
    gb = mm.CustomGBForce()
    
    # compute local density rho
    cutoff = r0 + 10 / eta # cutoff distance for computing rho
    offset_switch_rho = 0.5 * (1 + math.tanh(eta * (r0 - cutoff)))
    gb.addComputedValue('rho', 
                        f'''(switch_rho-{offset_switch_rho})*step({cutoff}-r);
                        switch_rho=0.5*(1+tanh({eta}*({r0}-r))); 
                        ''', 
                        mm.CustomGBForce.ParticlePairNoExclusions)
    
    # set the spline
    assert spl_values.ndim == 2
    assert spl_values.shape[0] == 20
    spl = mm.Continuous2DFunction(20, spl_values.shape[1], 
                                  spl_values.reshape(-1, order='F'), 0, 19, 
                                  rho_min, rho_max)
    gb.addTabulatedFunction('spl', spl)
    gb.addEnergyTerm('''energy;
                     energy=spl(atom_type, rho)''', 
                     mm.CustomGBForce.SingleParticle)
    
    # set particles
    gb.addPerParticleParameter('atom_type')
    for each in atom_types:
        gb.addParticle([each])
    
    # set PBC, cutoff, and force group
    if use_pbc:
        gb.setNonbondedMethod(gb.CutoffPeriodic)
    else:
        gb.setNonbondedMethod(gb.CutoffNonPeriodic)
    gb.setCutoffDistance(cutoff)
    gb.setForceGroup(force_group)
    return gb

def density_spline_term_group_i(atom_types, group_i_mask, use_pbc, spl_values, eta=10.0, r0=0.7, 
                        rho_min=0.0, rho_max=15.0, force_group=4):
    """
    Local density spline term. Only include atoms pairs that atom_group2 is in group i.
    
    Parameters
    ----------
    atom_types : 1d array-like, shape = (n_atoms,)
        Atom types.

    group_i_mask : 1d array-like, shape = (n_atoms,)
        Mask for group i, used to define the group of each atom.
    
    use_pbc : bool
        Whether to use periodic boundary conditions.
    
    spl_values : 2d array-like, shape = (20, n_knots)
        Spline values.
    
    eta : float or int
        Parameter eta in unit 1 / nm.
    
    r0 : float or int
        Parameter r0 in unit nm.
    
    rho_min : float or int
        Parameter rho_min.
    
    rho_max : float or int
        Parameter rho_max.
    
    force_group : int
        Force group.
    
    """
    # set CustomGBForce
    gb = mm.CustomGBForce()

    # set particles
    gb.addPerParticleParameter('atom_type')
    gb.addPerParticleParameter('is_group_i')
    for atom_type_i, is_group_i in zip(atom_types, group_i_mask):
        gb.addParticle([atom_type_i, is_group_i])
    
    # compute local density rho
    cutoff = r0 + 10 / eta # cutoff distance for computing rho
    offset_switch_rho = 0.5 * (1 + math.tanh(eta * (r0 - cutoff)))
    gb.addComputedValue('rho', 
                        f'''(switch_rho-{offset_switch_rho})*step({cutoff}-r)*mask_i;
                        switch_rho=0.5*(1+tanh({eta}*({r0}-r))); 
                        mask_i=delta(is_group_i2-1);
                        ''', 
                        mm.CustomGBForce.ParticlePairNoExclusions)
    
    # set the spline
    assert spl_values.ndim == 2
    assert spl_values.shape[0] == 20
    spl = mm.Continuous2DFunction(20, spl_values.shape[1], 
                                  spl_values.reshape(-1, order='F'), 0, 19, 
                                  rho_min, rho_max)
    gb.addTabulatedFunction('spl', spl)
    gb.addEnergyTerm('''energy;
                     energy=spl(atom_type, rho)''', 
                     mm.CustomGBForce.SingleParticle)
    
    
    # set PBC, cutoff, and force group
    if use_pbc:
        gb.setNonbondedMethod(gb.CutoffPeriodic)
    else:
        gb.setNonbondedMethod(gb.CutoffNonPeriodic)
    gb.setCutoffDistance(cutoff)
    gb.setForceGroup(force_group)
    return gb



# def multi_group_density_spline_term(atom_types, atom_groups, n_groups, use_pbc, spl_values, eta=10.0, r0=0.7, 
#                         rho_min=0.0, rho_max=15.0, force_group=4):
#     """
#     Local density spline term.
    
#     Parameters
#     ----------
#     atom_types : 1d array-like, shape = (n_atoms,)
#         Atom types.

#     atom_groups : 1d array-like, shape = (n_atoms,)
#         Atom groups, used to define the group of each atom.
#         Values should be in range [0, n_groups)

#     n_groups : int
#         Number of groups.
    
#     use_pbc : bool
#         Whether to use periodic boundary conditions.
    
#     spl_values : 3d array-like, shape = (20, n_groups, n_knots)
#         Spline values.
    
#     eta : float or int
#         Parameter eta in unit 1 / nm.
    
#     r0 : float or int
#         Parameter r0 in unit nm.
    
#     rho_min : float or int
#         Parameter rho_min.
    
#     rho_max : float or int
#         Parameter rho_max.
    
#     force_group : int
#         Force group.
    
#     """
#     assert len(atom_types) == len(atom_groups)

#     # set CustomGBForce
#     gb = mm.CustomGBForce()

#     # set particles
#     gb.addPerParticleParameter('atom_type')
#     gb.addPerParticleParameter('atom_group')
#     for atom_type_i, atom_group_i in zip(atom_types, atom_groups):
#         gb.addParticle([atom_type_i, atom_group_i])
    
#     ## compute local density rho
#     #cutoff = r0 + 10 / eta # cutoff distance for computing rho
#     #offset_switch_rho = 0.5 * (1 + math.tanh(eta * (r0 - cutoff)))
#     #gb.addComputedValue('rho', 
#     #                    f'''(switch_rho-{offset_switch_rho})*step({cutoff}-r);
#     #                    switch_rho=0.5*(1+tanh({eta}*({r0}-r))); 
#     #                    ''', 
#     #                    mm.CustomGBForce.ParticlePairNoExclusions)

#     # compute group density rho for each group
#     cutoff = r0 + 10 / eta # cutoff distance for computing rho
#     offset_switch_rho = 0.5 * (1 + math.tanh(eta * (r0 - cutoff))) 
#     for i in range(n_groups):
#         gb.addComputedValue(f'rho_group_{i}', 
#                             f'''(switch_rho-{offset_switch_rho})*step({cutoff}-r)*mask_i;
#                             switch_rho=0.5*(1+tanh({eta}*({r0}-r))); 
#                             mask_i=delta(atom_group2-{i});
#                             ''', # delta(x) = 1 if x is 0, 0 otherwise
#                             mm.CustomGBForce.ParticlePairNoExclusions)

    
#     # set the spline
#     #assert spl_values.ndim == 3
#     #assert spl_values.shape[0] == 20
#     #
#     #spl = mm.Continuous2DFunction(20, spl_values.shape[1], 
#     #                              spl_values.reshape(-1, order='F'), 0, 19, 
#     #                              rho_min, rho_max, periodic=False)
#     #gb.addTabulatedFunction('spl', spl)
#     #gb.addEnergyTerm('''energy;
#     #                 energy=spl(atom_type, rho)''', 
#     #                 mm.CustomGBForce.SingleParticle)
    
#     assert spl_values.ndim == 3
#     assert spl_values.shape[0] == 20
#     assert spl_values.shape[1] == n_groups
    
#     spl = mm.Continuous3DFunction(20, spl_values.shape[1], spl_values.shape[2], 
#                                   spl_values.reshape(-1, order='F'), 0, 19, 0, n_groups - 1,
#                                   rho_min, rho_max, periodic=False)
#     gb.addTabulatedFunction('spl', spl)
#     parameters_string = "atom_type, " + ", ".join([f"rho_group_{i}" for i in range(n_groups)])
#     gb.addEnergyTerm(f'''energy;
#                      energy=spl(parameters_string)''', 
#                      mm.CustomGBForce.SingleParticle)

#     # set PBC, cutoff, and force group
#     if use_pbc:
#         gb.setNonbondedMethod(gb.CutoffPeriodic)
#     else:
#         gb.setNonbondedMethod(gb.CutoffNonPeriodic)
#     gb.setCutoffDistance(cutoff)
#     gb.setForceGroup(force_group)
#     return gb
