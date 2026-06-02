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


def excluded_volume_lj_term(atom_types, exclusions, use_pbc, epsilon, sigma_map, force_group):
    """
    The excluded volume potential. 
    This is just the LJ potential with cutoff at 2**(1/6)*sigma.
    
    Parameters
    ----------
    atom_types : 1d array-like, shape=(n_atoms,)
        Atom types.
    
    exclusions : None or 2d array_like or pd.DataFrame
        Nonbonded exclusions. 
        If None, no exclusions. 
        If 2d array_like, shape = (n_exclusions, 2). 
        If pd.DataFrame, excluded pairs are specificed by columns 'a1' and 'a2'.
    
    use_pbc : bool
        Whether to use periodic boundary conditions.
    
    epsilon : float or int
        LJ potential epsilon parameter.
    
    sigma_map : 2d array-like, shape = (20, 20)
        LJ potential sigma parameter.
    
    force_group : int
        The force group.
    
    Returns
    -------
    nb : openmm.CustomNonbondedForce
        The nonbonded excluded volume potential.
    
    """
    nb = mm.CustomNonbondedForce(f'''energy;
           energy=4*{epsilon}*((sigma/r)^12-(sigma/r)^6+1/4)*step(cutoff-r);
           cutoff=2^(1/6)*sigma;
           sigma=sigma_map(atom_type1, atom_type2);
           ''')
    n_atom_types = 20
    if not isinstance(sigma_map, np.ndarray):
        sigma_map = np.array(sigma_map)
    assert sigma_map.shape == (20, 20)
    discrete_2d_sigma_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, 
    sigma_map.flatten(order='F'))
    nb.addTabulatedFunction('sigma_map', discrete_2d_sigma_map)
    nb.addPerParticleParameter('atom_type')
    for each in atom_types:
        nb.addParticle([each])
    if exclusions is not None:
        if isinstance(exclusions, pd.DataFrame):
            exclusions = exclusions[['a1', 'a2']].to_numpy()
        for each in exclusions:
            nb.addExclusion(int(each[0]), int(each[1]))
    if use_pbc:
        nb.setNonbondedMethod(nb.CutoffPeriodic)
    else:
        nb.setNonbondedMethod(nb.CutoffNonPeriodic)
    cutoff = 2**(1/6) * sigma_map
    nb.setCutoffDistance(np.max(cutoff))
    nb.setForceGroup(force_group)
    return nb

def ashbaugh_hatch_term(atom_types, df_exclusions, use_pbc, epsilon, sigma_ah_map, lambda_ah_map, force_group=2):
    """
    Ashbaugh-Hatch potential. 
    The cutoff is 4*sigma_ah. 
    """
    lj_at_cutoff = 4*epsilon*((1/4)**12 - (1/4)**6)
    contacts = mm.CustomNonbondedForce(f'''energy;
               energy=(f1+f2-offset)*step(4*sigma_ah-r);
               offset=lambda_ah*{lj_at_cutoff};
               f1=(lj+(1-lambda_ah)*{epsilon})*step(2^(1/6)*sigma_ah-r);
               f2=lambda_ah*lj*step(r-2^(1/6)*sigma_ah);
               lj=4*{epsilon}*((sigma_ah/r)^12-(sigma_ah/r)^6);
               sigma_ah=sigma_ah_map(atom_type1, atom_type2);
               lambda_ah=lambda_ah_map(atom_type1, atom_type2);
               ''')
    n_atom_types = sigma_ah_map.shape[0]
    discrete_2d_sigma_ah_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, sigma_ah_map.ravel().tolist())
    discrete_2d_lambda_ah_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, lambda_ah_map.ravel().tolist())
    contacts.addTabulatedFunction('sigma_ah_map', discrete_2d_sigma_ah_map)
    contacts.addTabulatedFunction('lambda_ah_map', discrete_2d_lambda_ah_map)
    contacts.addPerParticleParameter('atom_type')
    for each in atom_types:
        contacts.addParticle([each])
    for _, row in df_exclusions.iterrows():
        contacts.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        contacts.setNonbondedMethod(contacts.CutoffPeriodic)
    else:
        contacts.setNonbondedMethod(contacts.CutoffNonPeriodic)
    contacts.setCutoffDistance(4*np.amax(sigma_ah_map))
    contacts.setForceGroup(force_group)
    return contacts

def ashbaugh_hatch_with_gaussian_term(
    atom_types,
    df_exclusions,
    use_pbc,
    epsilon,
    sigma_ah_map,
    lambda_ah_map,
    gauss_height_map,
    gauss_delta_mu,
    gauss_width,
    force_group=2,
):
    """
    Ashbaugh-Hatch + Gaussian potential. 
    The cutoff is 4*sigma_ah. 
    """
    lj_at_cutoff = 4*epsilon*((1/4)**12 - (1/4)**6)
    contacts = mm.CustomNonbondedForce(f'''energy;
               energy=(f1+f2-offset+g_term)*step(4*sigma_ah-r);
               offset=lambda_ah*{lj_at_cutoff};
               f1=(lj+(1-lambda_ah)*{epsilon})*step(2^(1/6)*sigma_ah-r);
               f2=lambda_ah*lj*step(r-2^(1/6)*sigma_ah);
               g_term=gauss_height*exp(-0.5*((r-(2^(1/6)*sigma_ah+{gauss_delta_mu}))/({gauss_width}))^2)*step(r-2^(1/6)*sigma_ah);
               lj=4*{epsilon}*((sigma_ah/r)^12-(sigma_ah/r)^6);
               sigma_ah=sigma_ah_map(atom_type1, atom_type2);
               lambda_ah=lambda_ah_map(atom_type1, atom_type2);
               gauss_height=gauss_height_map(atom_type1, atom_type2)
               ''')
    n_atom_types = sigma_ah_map.shape[0]
    discrete_2d_sigma_ah_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, sigma_ah_map.ravel().tolist())
    discrete_2d_lambda_ah_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, lambda_ah_map.ravel().tolist())
    discrete_2d_lambda_gau_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, gauss_height_map.ravel().tolist())
    contacts.addTabulatedFunction('sigma_ah_map', discrete_2d_sigma_ah_map)
    contacts.addTabulatedFunction('lambda_ah_map', discrete_2d_lambda_ah_map)
    contacts.addTabulatedFunction('gauss_height_map',discrete_2d_lambda_gau_map)

    contacts.addPerParticleParameter('atom_type')
    for each in atom_types:
        contacts.addParticle([each])
    for _, row in df_exclusions.iterrows():
        contacts.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        contacts.setNonbondedMethod(contacts.CutoffPeriodic)
    else:
        contacts.setNonbondedMethod(contacts.CutoffNonPeriodic)
    contacts.setCutoffDistance(4*np.amax(sigma_ah_map))
    contacts.setForceGroup(force_group)
    return contacts

 
#def ashbaugh_hatch_with_gaussian_term(
#    atom_types,
#    df_exclusions,
#    use_pbc,
#    epsilon,
#    sigma_ah_map,
#    lambda_ah_map,
#    gauss_height_map,
#    gauss_delta_mu,
#    gauss_width,
#    force_group=2,
#):
#    """
#    Ashbaugh-Hatch potential + Gaussian bump.
#    """
#    lj_at_cutoff = 4*epsilon*((1/4)**12 - (1/4)**6)
#
#    contacts = mm.CustomNonbondedForce(f'''energy;
#               energy=(f1+f2-offset+g_term)*step(4*sigma_ah-r);
#               offset=lambda_ah*{lj_at_cutoff};
#               f1=(lj+(1-lambda_ah)*{epsilon})*step(2^(1/6)*sigma_ah-r);
#               f2=lambda_ah*lj*step(r-2^(1/6)*sigma_ah);
#               lj=4*{epsilon}*((sigma_ah/r)^12-(sigma_ah/r)^6);
#               sigma_ah=sigma_ah_map(atom_type1, atom_type2);
#               lambda_ah=lambda_ah_map(atom_type1, atom_type2);
#               g_term=gauss_height_map(atom_type1, atom_type2)*exp(-0.5*((r-(2^(1/6)*sigma_ah+{gauss_delta_mu}))/({gauss_width}))^2)*step(r-2^(1/6)*sigma_ah);
#               ''')
#
#    n_atom_types = sigma_ah_map.shape[0]
#
#    contacts.addTabulatedFunction(
#        'sigma_ah_map',
#        mm.Discrete2DFunction(n_atom_types, n_atom_types, sigma_ah_map.ravel().tolist())
#    )
#    contacts.addTabulatedFunction(
#        'lambda_ah_map',
#        mm.Discrete2DFunction(n_atom_types, n_atom_types, lambda_ah_map.ravel().tolist())
#    )
#    contacts.addTabulatedFunction(
#        'gauss_map',
#        mm.Discrete2DFunction(n_atom_types, n_atom_types, gauss_height_map.ravel().tolist())
#    )
#
#    contacts.addPerParticleParameter('atom_type')
#    for each in atom_types:
#        contacts.addParticle([int(each)])
#
#    for _, row in df_exclusions.iterrows():
#        contacts.addExclusion(int(row['a1']), int(row['a2']))
#
#    if use_pbc:
#        contacts.setNonbondedMethod(contacts.CutoffPeriodic)
#    else:
#        contacts.setNonbondedMethod(contacts.CutoffNonPeriodic)
#
#    contacts.setCutoffDistance(4*np.amax(sigma_ah_map))
#    contacts.setForceGroup(force_group)
#
#    return contacts


def ashbaugh_hatch_gaussian_term_version_old(
    atom_types,
    exclusions,
    use_pbc,
    sigma_map,
    gauss_height_map,
    gauss_delta_mu,
    gauss_width,
    force_group,
):
    """
    Additive Gaussian term on top of existing AH contacts.

    Energy per pair:
      U_G(r) = A_ij * exp( -0.5 * ((r - (r_min_ij + Δμ))/σ_G)^2 )
               * step(r - r_min_ij) * step(4σ_ij - r)

    where A_ij is gauss_height_map[i,j], Δμ = gauss_delta_mu, σ_G = gauss_width.

    Parameters
    ----------
    atom_types : 1d array-like of int, (n_atoms,)
    exclusions : None or array-like or DataFrame
    use_pbc : bool
    sigma_map : 2d (20,20), LJ/AH sigma in nm
    gauss_height_map : 2d (20,20), Gaussian amplitudes A_ij (kJ/mol)
    gauss_delta_mu : float, nm
    gauss_width : float, nm
    force_group : int

    Returns
    -------
    nb : openmm.CustomNonbondedForce
    """
    n_types = 20
    sigma_map = np.array(sigma_map, dtype=float)
    gauss_height_map = np.array(gauss_height_map, dtype=float)
    assert sigma_map.shape == (20, 20)
    assert gauss_height_map.shape == (20, 20)

    nb = mm.CustomNonbondedForce(
        f"""
        energy;
        energy = g_amp * exp(-0.5 * ((r - (r1 + {gauss_delta_mu}))
                                     / ({gauss_width}))^2)
                          * step(r - r1) * step(ah_cutoff - r);
        r1 = 2^(1/6) * sigma;
        ah_cutoff = 4 * sigma;
        sigma = sigma_map(atom_type1, atom_type2);
        g_amp = gauss_map(atom_type1, atom_type2);
        """
    )

    # tabulated sigma and Gaussian amplitudes
    sigma_func = mm.Discrete2DFunction(
        n_types, n_types, sigma_map.flatten(order="F")
    )
    gauss_func = mm.Discrete2DFunction(
        n_types, n_types, gauss_height_map.flatten(order="F")
    )
    nb.addTabulatedFunction("sigma_map", sigma_func)
    nb.addTabulatedFunction("gauss_map", gauss_func)

    # per-particle atom type index
    nb.addPerParticleParameter("atom_type")
    for t in atom_types:
        nb.addParticle([int(t)])

    # exclusions
    if exclusions is not None:
        if isinstance(exclusions, pd.DataFrame):
            exclusions = exclusions[["a1", "a2"]].to_numpy()
        for a1, a2 in exclusions:
            nb.addExclusion(int(a1), int(a2))

    if use_pbc:
        nb.setNonbondedMethod(nb.CutoffPeriodic)
    else:
        nb.setNonbondedMethod(nb.CutoffNonPeriodic)

    # cut off at max(4*sigma)
    cutoff = 4.0 * sigma_map
    nb.setCutoffDistance(float(np.max(cutoff)))

    nb.setForceGroup(force_group)
    return nb


