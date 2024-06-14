import numpy as np
import pandas as pd
try:
    import openmm as mm
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.unit as unit
from openabc.lib import NA, kB, EC, VEP, _amino_acids, _dna_nucleotides
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
    # add protein-protein interactions
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
    # add DNA-DNA interactions
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
            epsilon_map[i, j] = (epsilon_i*epsilon_j)**0.5
            epsilon_map[j, i] = epsilon_map[i, j]
            sigma_i = param_DD.loc[i1, 'sigma']
            sigma_j = param_DD.loc[j1, 'sigma']
            sigma_map[i, j] = 0.5*(sigma_i + sigma_j)*(2**(-1/6))
            sigma_map[j, i] = sigma_map[i, j]
            cutoff_map[i, j] = 0.5*(sigma_i + sigma_j)
            cutoff_map[j, i] = cutoff_map[i, j]
    # add protein-DNA interactions
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
        epsilon_map[i, :len(_amino_acids)] = (epsilon_i*epsilon_j)**0.5
        epsilon_map[:len(_amino_acids), i] = epsilon_map[i, :len(_amino_acids)]
        sigma_i = param_PD.loc[i1, 'sigma']
        sigma_j = param_PD.loc[len(_dna_3spn2_atom_names), 'sigma']
        sigma_map[i, :len(_amino_acids)] = 0.5*(sigma_i + sigma_j)
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
    # add atom type
    for _, row in mol.atoms.iterrows():
        resname = row['resname']
        name = row['name']
        if (resname in _amino_acids) and (name == 'CA'):
            vdwl.addParticle([_amino_acids.index(resname)])
        elif (resname in _dna_nucleotides) and (name in _dna_3spn2_atom_names):
            vdwl.addParticle([len(_amino_acids) + _dna_3spn2_atom_names.index(name)])
        else:
            sys.exit(f'Cannot recognize atom with resname {resname} and name {name}.')
    # add exclusions
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
    print(f'For electrostatic interactions, set monovalent salt concentration as {1000*C} mM.')
    print(f'For electrostatic interactions, set temperature as {T} K.')
    e = 249.4 - 0.788*T + 7.2E-4*T**2
    a = 1 - 0.2551*C + 5.151E-2*C**2 - 6.889E-3*C**3
    dielectric_DD = e*a
    print(f'DNA-DNA dielectric constant is {dielectric_DD}')
    print(f'Protein-protein and protein-DNA dielectric constant is {dielectric_PP_PD}.')
    elec = mm.CustomNonbondedForce('''energy;
           energy=alpha*exp(-r/ldby)*step(cutoff-r)/r;
           alpha=alpha_map(cg_atom_type1, cg_atom_type2);
           ldby=ldby_map(cg_atom_type1, cg_atom_type2);
           cutoff=cutoff_map(cg_atom_type1, cg_atom_type2)''')
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
                denominator = 4*np.pi*VEP*dielectric_DD/(NA*(EC**2))
                denominator = denominator.value_in_unit(unit.kilojoule_per_mole**-1*unit.nanometer**-1)
                ldby_ij = (dielectric_DD*VEP*kB*temperature/(2.0*NA*(EC**2)*salt_conc))**0.5
            else:
                cutoff_ij = cutoff_PP_PD
                denominator = 4*np.pi*VEP*dielectric_PP_PD/(NA*(EC**2))
                denominator = denominator.value_in_unit(unit.kilojoule_per_mole**-1*unit.nanometer**-1)
                ldby_ij = (dielectric_PP_PD*VEP*kB*temperature/(2.0*NA*(EC**2)*salt_conc))**0.5
            alpha_map[i, j] = q_i*q_j/denominator
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
    elec.addPerParticleParameter('cg_atom_type')
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



