import numpy as np
import pandas as pd
try:
    import openmm as mm
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.unit as unit
from openabc.lib import NA, kB, EC, VEP, _amino_acids, _dna_nucleotides, _kcal_to_kj
import math
import sys
import os

"""
Note addGlobalParameter can automatically convert the values to the correct unit. 
Be careful that addGlobalParameter sets global parameters that is used by all the forces in the system. 
Important: for Discrete2DFunction, if the 2d matrix is symmetric, then flatten in order 'F' or 'C' is equivalent.
Start from version 1.0.7, for consistency, the 2d matrix is flattened in order 'F'.
"""

_dna_3spn2_atom_names = ['P', 'S', 'A', 'T', 'C', 'G']

def moff_mrg_contact_term(atom_types, df_exclusions, use_pbc, alpha_map, epsilon_map, eta=0.7/unit.angstrom, 
                          r0=8.0*unit.angstrom, cutoff=2.0*unit.nanometer, force_group=5):
    """
    MOFF+MRG model nonbonded contact term.
    """
    eta_value = eta.value_in_unit(unit.nanometer**-1)
    r0_value = r0.value_in_unit(unit.nanometer)
    cutoff_value = cutoff.value_in_unit(unit.nanometer)
    contacts = mm.CustomNonbondedForce(f'''energy;
               energy=(energy1+energy2-offset1-offset2)*step({cutoff_value}-r);
               energy1=alpha_con/(r^12);
               energy2=-0.5*epsilon_con*(1+tanh({eta_value}*({r0_value}-r)));
               offset1=alpha_con/({cutoff_value}^12);
               offset2=-0.5*epsilon_con*(1+tanh({eta_value}*({r0_value}-{cutoff_value})));
               alpha_con=alpha_con_map(atom_type1, atom_type2);
               epsilon_con=epsilon_con_map(atom_type1, atom_type2);
               ''')
    n_atom_types = alpha_map.shape[0]
    # alpha_map and epsilon_map are symmetric
    discrete_2d_alpha_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                  alpha_map.flatten(order='F').tolist())
    discrete_2d_epsilon_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                    epsilon_map.flatten(order='F').tolist())
    contacts.addTabulatedFunction('alpha_con_map', discrete_2d_alpha_map)
    contacts.addTabulatedFunction('epsilon_con_map', discrete_2d_epsilon_map)
    contacts.addPerParticleParameter('atom_type')
    for each in atom_types:
        contacts.addParticle([each])
    for _, row in df_exclusions.iterrows():
        contacts.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        contacts.setNonbondedMethod(contacts.CutoffPeriodic)
    else:
        contacts.setNonbondedMethod(contacts.CutoffNonPeriodic)
    contacts.setCutoffDistance(cutoff)
    contacts.setForceGroup(force_group)
    return contacts


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
    # sigma_ah_map and lambda_ah_map are symmetric
    discrete_2d_sigma_ah_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                     sigma_ah_map.flatten(order='F').tolist())
    discrete_2d_lambda_ah_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                      lambda_ah_map.flatten(order='F').tolist())
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


def ddd_dh_elec_term(charges, df_exclusions, use_pbc, salt_conc=150.0*unit.millimolar, 
                     temperature=300.0*unit.kelvin, cutoff=4.0*unit.nanometer, force_group=6):
    """
    Debye-Huckel potential with a distance-dependent dielectric.
    """
    alpha = NA*EC**2/(4*np.pi*VEP)
    gamma = VEP*kB*temperature/(2.0*NA*salt_conc*EC**2)
    # use a distance-dependent relative permittivity (dielectric)
    dielectric_water = 78.4
    A = -8.5525
    kappa = 7.7839
    B = dielectric_water - A
    zeta = 0.03627
    cutoff_value = cutoff.value_in_unit(unit.nanometer)
    alpha_value = alpha.value_in_unit(unit.kilojoule_per_mole*unit.nanometer)
    gamma_value = gamma.value_in_unit(unit.nanometer**2)
    dielectric_at_cutoff = A + B/(1 + kappa*math.exp(-zeta*B*cutoff_value))
    ldby_at_cutoff = (dielectric_at_cutoff*gamma_value)**0.5
    elec = mm.CustomNonbondedForce(f'''energy;
           energy=q1*q2*{alpha_value}*((exp(-r/ldby)/r)-offset)*step({cutoff_value}-r)/dielectric;
           offset={math.exp(-cutoff_value/ldby_at_cutoff)/cutoff_value};
           ldby=(dielectric*{gamma_value})^0.5;
           dielectric={A}+{B}/(1+{kappa}*exp(-{zeta}*{B}*r));
           ''')
    elec.addPerParticleParameter('q')
    for q in charges:
        elec.addParticle([q])
    for _, row in df_exclusions.iterrows():
        elec.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        elec.setNonbondedMethod(elec.CutoffPeriodic)
    else:
        elec.setNonbondedMethod(elec.CutoffNonPeriodic)
    elec.setCutoffDistance(cutoff)
    elec.setForceGroup(force_group)
    return elec
    

def ddd_dh_elec_switch_term(charges, df_exclusions, use_pbc, salt_conc=150.0*unit.millimolar, 
                            temperature=300.0*unit.kelvin, cutoff1=1.2*unit.nanometer, cutoff2=1.5*unit.nanometer, 
                            switch_coeff=[1, 0, 0, -10, 15, -6], force_group=6):
    """
    Debye-Huckel potential with a distance-dependent dielectric and a switch function. 
    The switch function value changes from 1 to 0 smoothly as distance r changes from cutoff1 to cutoff2. 
    To make sure the switch function works properly, the zeroth order coefficient has to be 1, and the sum of all the coefficients in switch_coeff has to be 0. 
    """
    alpha = NA*EC**2/(4*np.pi*VEP)
    gamma = VEP*kB*temperature/(2.0*NA*salt_conc*EC**2)
    # use a distance-dependent relative permittivity (dielectric)
    dielectric_water = 78.4
    A = -8.5525
    kappa = 7.7839
    B = dielectric_water - A
    zeta = 0.03627
    alpha_value = alpha.value_in_unit(unit.kilojoule_per_mole*unit.nanometer)
    cutoff1_value = cutoff1.value_in_unit(unit.nanometer)
    cutoff2_value = cutoff2.value_in_unit(unit.nanometer)
    gamma_value = gamma.value_in_unit(unit.nanometer**2)
    assert switch_coeff[0] == 1
    assert np.sum(np.array(switch_coeff)) == 0
    switch_term_list = []
    for i in range(len(switch_coeff)):
        if i == 0:
            switch_term_list.append(f'{switch_coeff[i]}')
        else:
            switch_term_list.append(f'({switch_coeff[i]}*((r-{cutoff1_value})/({cutoff2_value}-{cutoff1_value}))^{i})')
    switch_term_string = '+'.join(switch_term_list)
    elec = mm.CustomNonbondedForce(f'''energy;
           energy=q1*q2*{alpha_value}*exp(-r/ldby)*switch/(dielectric*r);
           switch=({switch_term_string})*step(r-{cutoff1_value})*step({cutoff2_value}-r)+step({cutoff1_value}-r);
           ldby=(dielectric*{gamma_value})^0.5;
           dielectric={A}+{B}/(1+{kappa}*exp(-{zeta}*{B}*r));
           ''')
    elec.addPerParticleParameter('q')
    for q in charges:
        elec.addParticle([q])
    for _, row in df_exclusions.iterrows():
        elec.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        elec.setNonbondedMethod(elec.CutoffPeriodic)
    else:
        elec.setNonbondedMethod(elec.CutoffNonPeriodic)
    elec.setCutoffDistance(cutoff2)
    elec.setForceGroup(force_group)
    return elec


def dh_elec_term(charges, df_exclusions, use_pbc, ldby=1*unit.nanometer, dielectric_water=80.0, 
                 cutoff=3.5*unit.nanometer, force_group=3):
    """
    Debye-Huckel potential with a constant dielectric. 
    
    Parameters
    ----------
    charges : sequence-like
        Atom charges. 
    
    df_exclusions : pd.DataFrame
        Nonbonded exclusions. 
    
    use_pbc : bool
        Whether to use PBC. 
    
    ldby : Quantity
        Debye length. 
    
    dielectric_water : float or int
        Water dielectric constant. 
    
    cutoff : Quantity
        Cutoff distance. 
    
    force_group : int
        Force group. 
    
    returns
    -------
    elec : Force
        Electrostatic interaction force. 
    
    """
    alpha = NA*EC**2/(4*np.pi*VEP)
    ldby_value = ldby.value_in_unit(unit.nanometer)
    alpha_value = alpha.value_in_unit(unit.kilojoule_per_mole*unit.nanometer)
    cutoff_value = cutoff.value_in_unit(unit.nanometer)
    elec = mm.CustomNonbondedForce(f'''energy;
           energy=q1*q2*{alpha_value}*((exp(-r/{ldby_value})/r)-offset)*step({cutoff_value}-r)/{dielectric_water};
           offset={math.exp(-cutoff_value/ldby_value)/cutoff_value};
           ''')
    elec.addPerParticleParameter('q')
    for q in charges:
        elec.addParticle([q])
    for _, row in df_exclusions.iterrows():
        elec.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        elec.setNonbondedMethod(elec.CutoffPeriodic)
    else:
        elec.setNonbondedMethod(elec.CutoffNonPeriodic)
    elec.setCutoffDistance(cutoff)
    elec.setForceGroup(force_group)
    return elec


def wang_frenkel_term(atom_types, df_exclusions, use_pbc, epsilon_wf_map, sigma_wf_map, mu_wf_map, nu_wf_map=1, 
                      cutoff_to_sigma_ratio=3, force_group=3):
    """
    Wang-Fenkel potential term. 
    
    Parameters
    ----------
    atom_types : sequence-like
        Atom types. 
    
    df_exclusions : pd.DataFrame
        Nonbonded exclusions. 
    
    use_pbc : bool
        Whether to use PBC. 
    
    epsilon_wf_map : 2d sequence-like
        Matrix of epsilon parameter. 
    
    sigma_wf_map : 2d sequence-like
        Matrix of sigma parameter. 
    
    mu_wf_map : 2d sequence-like
        Matrix of mu parameter. 
    
    nu_wf_map : float or int or 2d sequence-like
        Matrix of nu parameter. 
        If this variable is float or int, then it means all the pairs have the same nu value. 
    
    cutoff_to_sigma_ratio : float or int
        The ratio of cutoff to sigma. 
    
    """
    contacts = mm.CustomNonbondedForce(f'''energy;
               energy=epsilon_wf*alpha_wf*g1*g2*step({cutoff_to_sigma_ratio}*sigma_wf-r);
               g1=(sigma_wf/r)^(2*mu_wf)-1;
               g2=(({cutoff_to_sigma_ratio}*sigma_wf/r)^(2*mu_wf)-1)^(2*nu_wf);
               alpha_wf=2*nu_wf*f1*((f2/(2*nu_wf*(f1-1)))^f2);
               f1={cutoff_to_sigma_ratio}^(2*mu_wf);
               f2=2*nu_wf+1;
               epsilon_wf=epsilon_wf_map(atom_type1, atom_type2);
               sigma_wf=sigma_wf_map(atom_type1, atom_type2);
               mu_wf=mu_wf_map(atom_type1, atom_type2);
               nu_wf=nu_wf_map(atom_type1, atom_type2);
               ''')
    n_atom_types = epsilon_wf_map.shape[0]
    # epsilon_wf_map, sigma_wf_map, mu_wf_map, and nu_wf_map are symmetric
    discrete_2d_epsilon_wf_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                       epsilon_wf_map.flatten(order='F').tolist())
    discrete_2d_sigma_wf_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                     sigma_wf_map.flatten(order='F').tolist())
    discrete_2d_mu_wf_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                  mu_wf_map.flatten(order='F').tolist())
    if (isinstance(nu_wf_map, int)) or (isinstance(nu_wf_map, float)):
        nu_wf_map = np.full((n_atom_types, n_atom_types), nu_wf_map)
    discrete_2d_nu_wf_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                  nu_wf_map.flatten(order='F').tolist())
    contacts.addTabulatedFunction('epsilon_wf_map', discrete_2d_epsilon_wf_map)
    contacts.addTabulatedFunction('sigma_wf_map', discrete_2d_sigma_wf_map)
    contacts.addTabulatedFunction('mu_wf_map', discrete_2d_mu_wf_map)
    contacts.addTabulatedFunction('nu_wf_map', discrete_2d_nu_wf_map)
    contacts.addPerParticleParameter('atom_type')
    for each in atom_types:
        contacts.addParticle([each])
    for _, row in df_exclusions.iterrows():
        contacts.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        contacts.setNonbondedMethod(contacts.CutoffPeriodic)
    else:
        contacts.setNonbondedMethod(contacts.CutoffNonPeriodic)
    contacts.setCutoffDistance(cutoff_to_sigma_ratio*np.amax(sigma_wf_map))
    contacts.setForceGroup(force_group)
    return contacts
    

def all_smog_MJ_3spn2_term(mol, param_PP_MJ, cutoff_PD=1.425*unit.nanometer, force_group=11):
    """
    Combine all the SMOG (MJ potential for protein-protein nonbonded pairs) and 3SPN2 nonbonded interactions into one force.
    CG atom type 0-19 for amino acids.
    CG atom type 20-25 for DNA atoms. 
    Many parameters are saved as attributes of mol. 
        
    """
    vdwl = mm.CustomNonbondedForce('''energy;
           energy=4*epsilon*((sigma/r)^12-(sigma/r)^6-offset)*step(cutoff-r);
           offset=(sigma/cutoff)^12-(sigma/cutoff)^6;
           epsilon=epsilon_map(atom_type1, atom_type2);
           sigma=sigma_map(atom_type1, atom_type2);
           cutoff=cutoff_map(atom_type1, atom_type2)''')
    n_atom_types = len(_amino_acids) + len(_dna_3spn2_atom_names)
    epsilon_map = np.zeros((n_atom_types, n_atom_types))
    sigma_map = np.zeros((n_atom_types, n_atom_types))
    cutoff_map = np.zeros((n_atom_types, n_atom_types))
    # set protein-protein interactions
    for _, row in param_PP_MJ.iterrows():
        atom_type1, atom_type2 = row['atom_type1'], row['atom_type2']
        i = _amino_acids.index(atom_type1)
        j = _amino_acids.index(atom_type2)
        epsilon_map[i, j] = row['epsilon (kj/mol)']
        epsilon_map[j, i] = epsilon_map[i, j]
        sigma_map[i, j] = row['sigma (nm)']
        sigma_map[j, i] = sigma_map[i, j]
        cutoff_map[i, j] = row['cutoff_LJ (nm)']
        cutoff_map[j, i] = cutoff_map[i, j]
    # set DNA-DNA interactions
    # be careful with the definition of sigma_map and cutoff_map
    # in the original 3SPN2, potential is defined as epsilon*((sigma/r)^12-2*(sigma/r)^6) with cutoff as sigma
    # note for epsilon*((sigma/r)^12-2*(sigma/r)^6), the minimum is at r = sigma
    # here we use 4*epsilon*((sigma'/r)^12-(sigma'/r)^6) instead, with sigma' = 2^(-1/6)*sigma
    # note epsilon*((sigma/r)^12-2*(sigma/r)^6) = 4*epsilon*((sigma'/r)^12-(sigma'/r)^6)
    param_DD = mol.particle_definition[mol.particle_definition['DNA'] == mol.dna_type].copy()
    param_DD.index = param_DD['name'] # rearrange to make sure the row order is based on dna_atom_names
    param_DD = param_DD.loc[_dna_3spn2_atom_names]
    param_DD.index = list(range(len(param_DD.index)))
    for i1 in range(len(_dna_3spn2_atom_names)):
        for j1 in range(i1, len(_dna_3spn2_atom_names)):
            i = i1 + len(_amino_acids)
            j = j1 + len(_amino_acids)
            epsilon_i = param_DD.loc[i1, 'epsilon']
            epsilon_j = param_DD.loc[j1, 'epsilon']
            epsilon_map[i, j] = (epsilon_i * epsilon_j)**0.5
            epsilon_map[j, i] = epsilon_map[i, j]
            sigma_i = param_DD.loc[i1, 'sigma']
            sigma_j = param_DD.loc[j1, 'sigma']
            # sigma_i and sigma_j are the original sigma values, while sigam_map saves sigma'=2^(-1/6)*sigma.
            sigma_map[i, j] = 0.5 * (sigma_i + sigma_j) * (2**(-1/6)) # be careful with the definition of sigma_map
            sigma_map[j, i] = sigma_map[i, j]
            cutoff_map[i, j] = 0.5 * (sigma_i + sigma_j) # be careful with the definition of cutoff_map
            cutoff_map[j, i] = cutoff_map[i, j]
    # set protein-DNA interactions
    all_param_PD = mol.protein_dna_particle_definition
    param_dna_PD = all_param_PD[(all_param_PD['molecule'] == 'DNA') & (all_param_PD['DNA'] == mol.dna_type)].copy()
    param_dna_PD.index = param_dna_PD['name']
    param_dna_PD = param_dna_PD.loc[_dna_3spn2_atom_names].copy()
    param_dna_PD.index = list(range(len(param_dna_PD.index)))
    param_protein_PD =  all_param_PD[(all_param_PD['molecule'] == 'Protein')].copy()
    param_protein_PD.index = param_protein_PD['name']
    param_protein_PD = param_protein_PD.loc[['CA']].copy() # protein only has CA type CG atom
    param_protein_PD.index = list(range(len(param_protein_PD.index)))
    param_PD = pd.concat([param_dna_PD, param_protein_PD], ignore_index=True)
    for i1 in range(len(_dna_3spn2_atom_names)):
        i = i1 + len(_amino_acids)
        epsilon_i = param_PD.loc[i1, 'epsilon']
        epsilon_j = param_PD.loc[len(_dna_3spn2_atom_names), 'epsilon']
        epsilon_map[i, :len(_amino_acids)] = (epsilon_i * epsilon_j)**0.5
        epsilon_map[:len(_amino_acids), i] = epsilon_map[i, :len(_amino_acids)]
        sigma_i = param_PD.loc[i1, 'sigma']
        sigma_j = param_PD.loc[len(_dna_3spn2_atom_names), 'sigma']
        sigma_map[i, :len(_amino_acids)] = 0.5 * (sigma_i + sigma_j)
        sigma_map[:len(_amino_acids), i] = sigma_map[i, :len(_amino_acids)]
        cutoff_map[i, :len(_amino_acids)] = cutoff_PD.value_in_unit(unit.nanometer)
        cutoff_map[:len(_amino_acids), i] = cutoff_map[i, :len(_amino_acids)]
    max_cutoff = np.amax(cutoff_map)
    # epsilon_map, sigma_map, and cutoff_map are symmetric
    vdwl.addTabulatedFunction('epsilon_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                                   epsilon_map.flatten(order='F').tolist()))
    vdwl.addTabulatedFunction('sigma_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                                 sigma_map.flatten(order='F').tolist()))
    vdwl.addTabulatedFunction('cutoff_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                                  cutoff_map.flatten(order='F').tolist()))
    vdwl.addPerParticleParameter('atom_type')
    # set atom types
    for _, row in mol.atoms.iterrows():
        resname = row['resname']
        name = row['name']
        if (resname in _amino_acids) and (name == 'CA'):
            vdwl.addParticle([_amino_acids.index(resname)])
        elif (resname in _dna_nucleotides) and (name in _dna_3spn2_atom_names):
            vdwl.addParticle([len(_amino_acids) + _dna_3spn2_atom_names.index(name)])
        else:
            sys.exit(f'Cannot recognize atom with resname {resname} and name {name}.')
    # set exclusions
    for _, row in mol.exclusions.iterrows():
        vdwl.addExclusion(int(row['a1']), int(row['a2']))
    # set PBC, cutoff, and force group
    if mol.use_pbc:
        vdwl.setNonbondedMethod(vdwl.CutoffPeriodic)
    else:
        vdwl.setNonbondedMethod(vdwl.CutoffNonPeriodic)
    vdwl.setCutoffDistance(max_cutoff)
    vdwl.setForceGroup(force_group)
    return vdwl


def all_smog_3spn2_elec_term(mol, salt_conc=150*unit.millimolar, temperature=300*unit.kelvin, 
                             elec_DD_charge_scale=0.6, cutoff_DD=5*unit.nanometer, 
                             cutoff_PP_PD=3.141504539*unit.nanometer, dielectric_PP_PD=78, force_group=12):
    """
    Combine all the SMOG and 3SPN2 electrostatic interactions into one force. 
    
    CG atom types: 
    Type 0 for zero-charge CG atoms. 
    Type 1 for ARG and LYS CA atoms. 
    Type 2 for ASP and GLU CA atoms. 
    Type 3 for phosphate CG atoms. 

    """
    C = salt_conc.value_in_unit(unit.molar)
    T = temperature.value_in_unit(unit.kelvin)
    print(f'For electrostatic interactions, set monovalent salt concentration as {1000 * C} mM.')
    print(f'For electrostatic interactions, set temperature as {T} K.')
    e = 249.4 - 0.788 * T + 7.2E-4 * T**2
    a = 1 - 0.2551 * C + 5.151E-2 * C**2 - 6.889E-3 * C**3
    dielectric_DD = e * a
    print(f'DNA-DNA dielectric constant is {dielectric_DD}')
    print(f'Protein-protein and protein-DNA dielectric constant is {dielectric_PP_PD}.')
    elec = mm.CustomNonbondedForce('''energy;
           energy=alpha*exp(-r/ldby)*step(cutoff-r)/r;
           alpha=alpha_map(atom_type1, atom_type2);
           ldby=ldby_map(atom_type1, atom_type2);
           cutoff=cutoff_map(atom_type1, atom_type2)''')
    n_atom_types = 4
    charge_list = [0, 1, -1, -1]
    # use Discrete2DFunction to define mappings for alpha, sigma, and cutoff
    alpha_map = np.zeros((n_atom_types, n_atom_types))
    ldby_map = np.zeros((n_atom_types, n_atom_types))
    cutoff_map = np.zeros((n_atom_types, n_atom_types))
    # set mappings
    for i in range(n_atom_types):
        for j in range(i, n_atom_types):
            q_i = charge_list[i]
            q_j = charge_list[j]
            if (i == 3) and (j == 3):
                # phosphate-phosphate electrostatic interactions
                q_i *= elec_DD_charge_scale
                q_j *= elec_DD_charge_scale
                cutoff_ij = cutoff_DD
                denominator = 4 * np.pi * VEP * dielectric_DD / (NA * (EC**2))
                denominator = denominator.value_in_unit(unit.kilojoule_per_mole**-1 * unit.nanometer**-1)
                ldby_ij = (dielectric_DD * VEP * kB * temperature / (2.0 * NA * (EC**2) * salt_conc))**0.5
            else:
                cutoff_ij = cutoff_PP_PD
                denominator = 4 * np.pi * VEP * dielectric_PP_PD / (NA * (EC**2))
                denominator = denominator.value_in_unit(unit.kilojoule_per_mole**-1 * unit.nanometer**-1)
                ldby_ij = (dielectric_PP_PD * VEP * kB * temperature / (2.0 * NA * (EC**2) * salt_conc))**0.5
            alpha_map[i, j] = q_i * q_j / denominator
            alpha_map[j, i] = alpha_map[i, j]
            cutoff_map[i, j] = cutoff_ij.value_in_unit(unit.nanometer)
            cutoff_map[j, i] = cutoff_map[i, j]
            ldby_map[i, j] = ldby_ij.value_in_unit(unit.nanometer)
            ldby_map[j, i] = ldby_map[i, j]
    max_cutoff = np.amax(cutoff_map)
    # alpha_map, ldby_map, and cutoff_map are symmetric
    elec.addTabulatedFunction('alpha_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                                 alpha_map.flatten(order='F').tolist()))
    elec.addTabulatedFunction('ldby_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                                ldby_map.flatten(order='F').tolist()))
    elec.addTabulatedFunction('cutoff_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, 
                                                                  cutoff_map.flatten(order='F').tolist()))
    elec.addPerParticleParameter('atom_type')
    # add atom type
    for _, row in mol.atoms.iterrows():
        resname = row['resname']
        name = row['name']
        if (resname in ['ARG', 'LYS']) and (name == 'CA'):
            elec.addParticle([1])
        elif (resname in ['ASP', 'GLU']) and (name == 'CA'):
            elec.addParticle([2])
        elif (resname in _dna_nucleotides) and (name == 'P'):
            elec.addParticle([3])
        else:
            elec.addParticle([0])
    # add exclusions
    for _, row in mol.exclusions.iterrows():
        elec.addExclusion(int(row['a1']), int(row['a2']))
    # set PBC, cutoff, and force group
    if mol.use_pbc:
        elec.setNonbondedMethod(elec.CutoffPeriodic)
    else:
        elec.setNonbondedMethod(elec.CutoffNonPeriodic)
    elec.setCutoffDistance(max_cutoff)
    elec.setForceGroup(force_group)
    return elec


def all_smog_MJ_3spn2_explicit_ion_hydr_vdwl_term(mol, param_PP_MJ, force_group=11):
    """
    Combine all the SMOG-MJ, 3SPN2, and explicit ion hydration and vdwl interactions into one force.
    CG atom type 0-19 for amino acids.
    CG atom type 20-25 for DNA atoms. 
    CG atom types 26-28 for ions.
    Many parameters are saved as attributes of mol. 
    
    """
    hydr_vdwl = mm.CustomNonbondedForce('''energy;
                energy=hydr1+hydr2+vdwl;
                hydr2=gamma2*(exp(-(r-mu2)^2/(2*eta2^2))-offset_hydr2)*step(cutoff_hydr2-r);
                offset_hydr2=exp(-(cutoff_hydr2-mu2)^2/(2*eta2^2));
                hydr1=gamma1*(exp(-(r-mu1)^2/(2*eta1^2))-offset_hydr1)*step(cutoff_hydr1-r);
                offset_hydr1=exp(-(cutoff_hydr1-mu1)^2/(2*eta1^2));
                gamma1=gamma1_map(atom_type1, atom_type2);
                gamma2=gamma2_map(atom_type1, atom_type2);
                mu1=mu1_map(atom_type1, atom_type2);
                mu2=mu2_map(atom_type1, atom_type2);
                eta1=eta1_map(atom_type1, atom_type2);
                eta2=eta2_map(atom_type1, atom_type2);
                cutoff_hydr1=cutoff_hydr1_map(atom_type1, atom_type2);
                cutoff_hydr2=cutoff_hydr2_map(atom_type1, atom_type2);
                vdwl=4*epsilon*((sigma/r)^12-(sigma/r)^6-offset_vdwl)*step(cutoff_vdwl-r);
                offset_vdwl=(sigma/cutoff_vdwl)^12-(sigma/cutoff_vdwl)^6;
                epsilon=epsilon_map(atom_type1, atom_type2);
                sigma=sigma_map(atom_type1, atom_type2);
                cutoff_vdwl=cutoff_vdwl_map(atom_type1, atom_type2);''')
    _ions = ['NA', 'MG', 'CL']
    all_atom_types = _amino_acids + _dna_3spn2_atom_names + _ions
    n_atom_types = len(all_atom_types)
    
    # initialize parameter maps
    # note cutoff is shared by hydration and vdwl
    gamma1_map = np.zeros((n_atom_types, n_atom_types))
    gamma2_map = np.zeros((n_atom_types, n_atom_types))
    mu1_map = np.zeros((n_atom_types, n_atom_types))
    mu2_map = np.zeros((n_atom_types, n_atom_types))
    eta1_map = 0.01 * np.ones((n_atom_types, n_atom_types)) # initialize as 0.01 to avoid division by zero
    eta2_map = 0.01 * np.ones((n_atom_types, n_atom_types)) # initialize as 0.01 to avoid division by zero
    epsilon_map = np.zeros((n_atom_types, n_atom_types))
    sigma_map = np.zeros((n_atom_types, n_atom_types))
    cutoff_vdwl_map = np.zeros((n_atom_types, n_atom_types))
    
    # set atom type indices and classify atoms
    atom_type_index_dict = dict(zip(all_atom_types, list(range(n_atom_types))))
    positive_amino_acids = ['ARG', 'LYS'] # HIS is recognized as neutral amino acid
    negative_amino_acids = ['ASP', 'GLU']
    charged_amino_acids = positive_amino_acids + negative_amino_acids
    neutral_amino_acids = [i for i in _amino_acids if i not in charged_amino_acids]
    neutral_dna_atoms = [i for i in _dna_3spn2_atom_names if i != 'P']
    neutral_amino_acid_atom_type_indices = [atom_type_index_dict[i] for i in neutral_amino_acids]
    neutral_dna_atom_type_indices = [atom_type_index_dict[i] for i in neutral_dna_atoms]
    neutral_atom_type_indices = neutral_amino_acid_atom_type_indices + neutral_dna_atom_type_indices
    positive_amino_acid_type_indices = [atom_type_index_dict[i] for i in positive_amino_acids]
    negative_amino_acid_type_indices = [atom_type_index_dict[i] for i in negative_amino_acids]
    ion_type_indices = [atom_type_index_dict[i] for i in _ions]
    
    # set all the hydration parameters
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
    for k in H1_dict.keys():
        assert k[0] in _ions # k[0] is always ion
        i = atom_type_index_dict[k[0]]
        if k[1] == 'AA+':
            j = positive_amino_acid_type_indices
        elif k[1] == 'AA-':
            j = negative_amino_acid_type_indices
        else:
            j = atom_type_index_dict[k[1]]
        gamma1_map[i, j] = H1_dict[k] / (eta1_dict[k] * (2 * np.pi)**0.5)
        gamma1_map[j, i] = gamma1_map[i, j]
        mu1_map[i, j] = mu1_dict[k]
        mu1_map[j, i] = mu1_dict[k]
        eta1_map[i, j] = eta1_dict[k]
        eta1_map[j, i] = eta1_dict[k]
    for k in H2_dict.keys():
        assert k[0] in _ions # k[0] is always ion
        i = atom_type_index_dict[k[0]]
        if k[1] == 'AA+':
            j = positive_amino_acid_type_indices
        elif k[1] == 'AA-':
            j = negative_amino_acid_type_indices
        else:
            j = atom_type_index_dict[k[1]]
        gamma2_map[i, j] = H2_dict[k] / (eta2_dict[k] * (2 * np.pi)**0.5)
        gamma2_map[j, i] = gamma2_map[i, j]
        mu2_map[i, j] = mu2_dict[k]
        mu2_map[j, i] = mu2_dict[k]
        eta2_map[i, j] = eta2_dict[k]
        eta2_map[j, i] = eta2_dict[k]
    cutoff_hydr1_map = mu1_map + 10 * eta1_map
    cutoff_hydr2_map = mu2_map + 10 * eta2_map
    gamma1_map = gamma1_map.flatten(order='F').tolist()
    gamma2_map = gamma2_map.flatten(order='F').tolist()
    mu1_map = mu1_map.flatten(order='F').tolist()
    mu2_map = mu2_map.flatten(order='F').tolist()
    eta1_map = eta1_map.flatten(order='F').tolist()
    eta2_map = eta2_map.flatten(order='F').tolist()
    cutoff_hydr1_map = cutoff_hydr1_map.flatten(order='F').tolist()
    cutoff_hydr2_map = cutoff_hydr2_map.flatten(order='F').tolist()
    hydr_vdwl.addTabulatedFunction('gamma1_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, gamma1_map))
    hydr_vdwl.addTabulatedFunction('gamma2_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, gamma2_map))
    hydr_vdwl.addTabulatedFunction('mu1_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, mu1_map))
    hydr_vdwl.addTabulatedFunction('mu2_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, mu2_map))
    hydr_vdwl.addTabulatedFunction('eta1_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, eta1_map))
    hydr_vdwl.addTabulatedFunction('eta2_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, eta2_map))
    hydr_vdwl.addTabulatedFunction('cutoff_hydr1_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, cutoff_hydr1_map))
    hydr_vdwl.addTabulatedFunction('cutoff_hydr2_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, cutoff_hydr2_map))
    
    # set vdwl
    # set protein-protein interactions
    # protein-protein vdwl interactions are all MJ potentials
    for _, row in param_PP_MJ.iterrows():
        atom_type1, atom_type2 = row['atom_type1'], row['atom_type2']
        i = atom_type_index_dict[atom_type1]
        j = atom_type_index_dict[atom_type2]
        epsilon_map[i, j] = row['epsilon (kj/mol)']
        epsilon_map[j, i] = epsilon_map[i, j]
        sigma_map[i, j] = row['sigma (nm)']
        sigma_map[j, i] = sigma_map[i, j]
        cutoff_vdwl_map[i, j] = row['cutoff_LJ (nm)']
        cutoff_vdwl_map[j, i] = cutoff_vdwl_map[i, j]
    
    # set DNA-DNA interactions except phosphate-phosphate interactions
    # be careful with the definition of sigma_map and cutoff_vdwl_map
    # in the original 3SPN2, potential is defined as epsilon*((sigma/r)^12-2*(sigma/r)^6) with cutoff as sigma
    # note for epsilon*((sigma/r)^12-2*(sigma/r)^6), the minimum is at r = sigma
    # here we use 4*epsilon*((sigma'/r)^12-(sigma'/r)^6) instead, with sigma' = 2^(-1/6)*sigma
    # note epsilon*((sigma/r)^12-2*(sigma/r)^6) = 4*epsilon*((sigma'/r)^12-(sigma'/r)^6)
    param_DD = mol.particle_definition[mol.particle_definition['DNA'] == mol.dna_type].copy()
    param_DD.index = param_DD['name'] # rearrange to make sure the row order is based on dna_atom_names
    param_DD = param_DD.loc[_dna_3spn2_atom_names]
    param_DD.index = list(range(len(param_DD.index)))
    for i1 in range(len(_dna_3spn2_atom_names)):
        for j1 in range(i1, len(_dna_3spn2_atom_names)):
            a_i = _dna_3spn2_atom_names[i1]
            a_j = _dna_3spn2_atom_names[j1]
            if (a_i == 'P') and (a_j == 'P'):
                # do not set phosphate-phosphate vdwl parameters here
                # phosphate-phosphate vdwl parameters are set elsewhere
                continue
            i = atom_type_index_dict[a_i]
            j = atom_type_index_dict[a_j]
            epsilon_i = param_DD.loc[i1, 'epsilon']
            epsilon_j = param_DD.loc[j1, 'epsilon']
            epsilon_map[i, j] = (epsilon_i * epsilon_j)**0.5
            epsilon_map[j, i] = epsilon_map[i, j]
            sigma_i = param_DD.loc[i1, 'sigma']
            sigma_j = param_DD.loc[j1, 'sigma']
            # sigma_i and sigma_j are the original sigma values, while sigam_map saves sigma'=2^(-1/6)*sigma.
            sigma_map[i, j] = 0.5 * (sigma_i + sigma_j) * (2**(-1/6)) # be careful with the definition of sigma_map
            sigma_map[j, i] = sigma_map[i, j]
            cutoff_vdwl_map[i, j] = 0.5 * (sigma_i + sigma_j) # be careful with the definition of cutoff_vdwl_map
            cutoff_vdwl_map[j, i] = cutoff_vdwl_map[i, j]
    
    # set protein-DNA interactions
    # we directly assign protein-DNA interaction parameters, which is convenient
    # note cutoff is different from the value in implicit solvent SMOG+3SPN2 model
    amino_acid_atom_type_indices = np.array([atom_type_index_dict[i] for i in _amino_acids])
    dna_atom_type_indices = np.array([atom_type_index_dict[i] for i in _dna_3spn2_atom_names])
    v1, v2 = np.meshgrid(amino_acid_atom_type_indices, dna_atom_type_indices, indexing='ij')
    v1 = v1.flatten()
    v2 = v2.flatten()
    epsilon_map[v1, v2] = 0.02987572 * _kcal_to_kj
    epsilon_map[v2, v1] = 0.02987572 * _kcal_to_kj
    sigma_map[v1, v2] = 0.57
    sigma_map[v2, v1] = 0.57
    cutoff_vdwl_map[v1, v2] = 2**(1/6) * 0.57
    cutoff_vdwl_map[v2, v1] = 2**(1/6) * 0.57
    
    # set ion-neutral particle interactions
    v1, v2 = np.meshgrid(np.array(neutral_atom_type_indices), np.array(ion_type_indices), 
                         indexing='ij')
    v1 = v1.flatten()
    v2 = v2.flatten()
    epsilon_map[v1, v2] = 0.239 * _kcal_to_kj
    epsilon_map[v2, v1] = 0.239 * _kcal_to_kj
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
    for k, v in ion_neutral_atom_sigma_dict.items():
        assert k[0] in _ions # k[0] is always ion
        i = atom_type_index_dict[k[0]] # the first atom type is always ion
        if k[1] == 'AA':
            j = neutral_amino_acid_atom_type_indices
        else:
            j = atom_type_index_dict[k[1]]
        sigma_map[i, j] = v
        sigma_map[j, i] = v
        cutoff_vdwl_map[i, j] = 2**(1/6) * v
        cutoff_vdwl_map[j, i] = 2**(1/6) * v
    
    # set charged particle interactions (except both CG atoms are charged amino acids)
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
    for k in ion_charged_atom_epsilon_dict.keys():
        i = atom_type_index_dict[k[0]] # the first atom type is always ion or phosphate
        if k[1] == 'AA+':
            j = positive_amino_acid_type_indices
        elif k[1] == 'AA-':
            j = negative_amino_acid_type_indices
        else:
            j = atom_type_index_dict[k[1]]
        epsilon_map[i, j] = ion_charged_atom_epsilon_dict[k]
        epsilon_map[j, i] = ion_charged_atom_epsilon_dict[k]
        sigma_map[i, j] = ion_charged_atom_sigma_dict[k]
        sigma_map[j, i] = ion_charged_atom_sigma_dict[k]
        cutoff_vdwl_map[i, j] = 1.2 # all the charged atom contact cutoffs are 1.2 nm
        cutoff_vdwl_map[j, i] = 1.2
    
    # assert no missing parameters
    assert np.all(epsilon_map != 0) # there should be a non-zero epsilon for each type of pair
    assert np.all(sigma_map > 0) # there should be a positive sigma for each type of pair
    assert np.all(cutoff_vdwl_map > 0) # there should be a positive cutoff for each type of pair
    epsilon_map = epsilon_map.flatten(order='F').tolist()
    sigma_map = sigma_map.flatten(order='F').tolist()
    cutoff_vdwl_map = cutoff_vdwl_map.flatten(order='F').tolist()
    hydr_vdwl.addTabulatedFunction('epsilon_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, epsilon_map))
    hydr_vdwl.addTabulatedFunction('sigma_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, sigma_map))
    hydr_vdwl.addTabulatedFunction('cutoff_vdwl_map', 
                                   mm.Discrete2DFunction(n_atom_types, n_atom_types, cutoff_vdwl_map))
    hydr_vdwl.addPerParticleParameter('atom_type')
    
    # set atom types
    for _, row in mol.atoms.iterrows():
        resname = row['resname']
        name = row['name']
        if (resname in _amino_acids) and (name == 'CA'):
            hydr_vdwl.addParticle([atom_type_index_dict[resname]])
        elif (resname in _dna_nucleotides) and (name in _dna_3spn2_atom_names):
            hydr_vdwl.addParticle([atom_type_index_dict[name]])
        elif (resname in _ions) and (name in _ions):
            hydr_vdwl.addParticle([atom_type_index_dict[resname]])
        else:
            sys.exit(f'Cannot recognize atom with resname {resname} and name {name}.')
    
    # set exclusions
    assert hasattr(mol, 'exclusions'), 'Please set exclusions'
    for _, row in mol.exclusions.iterrows():
        hydr_vdwl.addExclusion(int(row['a1']), int(row['a2']))
    
    # set PBC, cutoff, and force group
    if mol.use_pbc:
        hydr_vdwl.setNonbondedMethod(hydr_vdwl.CutoffPeriodic)
    else:
        hydr_vdwl.setNonbondedMethod(hydr_vdwl.CutoffNonPeriodic)
    global_cutoff = max(np.max(cutoff_vdwl_map), np.max(cutoff_hydr1_map), np.max(cutoff_hydr2_map))
    hydr_vdwl.setCutoffDistance(global_cutoff)
    hydr_vdwl.setForceGroup(force_group)
    return hydr_vdwl


