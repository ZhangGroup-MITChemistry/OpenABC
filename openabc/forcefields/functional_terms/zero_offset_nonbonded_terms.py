import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import math
import sys
import os

'''
Define some nonbonded interactions with zero offset.
These forces are used for comparing with outputs from other softwares. 
'''

# define some constants based on CODATA
NA = unit.AVOGADRO_CONSTANT_NA # Avogadro constant
kB = unit.BOLTZMANN_CONSTANT_kB  # Boltzmann constant
EC = 1.602176634e-19*unit.coulomb # elementary charge
VEP = 8.8541878128e-12*unit.farad/unit.meter # vacuum electric permittivity

def hps_ah_zero_offset_term(atom_types, df_exclusions, use_pbc, epsilon, sigma_ah_map, lambda_ah_map, force_group=2):
    '''
    HPS model nonbonded contact term (form proposed by Ashbaugh and Hatch). 
    The cutoff is 4*sigma_ah. 
    '''
    contacts = mm.CustomNonbondedForce(f'''energy;
               energy=(f1+f2-offset)*step(4*sigma_ah-r);
               offset=0;
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
    for i, row in df_exclusions.iterrows():
        contacts.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        contacts.setNonbondedMethod(contacts.CutoffPeriodic)
    else:
        contacts.setNonbondedMethod(contacts.CutoffNonPeriodic)
    contacts.setCutoffDistance(4*np.amax(sigma_ah_map))
    contacts.setForceGroup(force_group)
    return contacts


def dh_elec_zero_offset_term(charges, df_exclusions, use_pbc, ldby=1*unit.nanometer, dielectric_water=80.0, 
                             cutoff=3.5*unit.nanometer, force_group=3):
    '''
    Debye-Huckel potential with a constant dielectric.
    '''
    alpha = NA*EC**2/(4*np.pi*VEP)
    ldby_value = ldby.value_in_unit(unit.nanometer)
    alpha_value = alpha.value_in_unit(unit.kilojoule_per_mole*unit.nanometer)
    cutoff_value = cutoff.value_in_unit(unit.nanometer)
    elec = mm.CustomNonbondedForce(f'''energy;
           energy=q1*q2*{alpha_value}*((exp(-r/{ldby_value})/r)-offset)*step({cutoff_value}-r)/{dielectric_water};
           offset=0;
           ''')
    elec.addPerParticleParameter('q')
    for q in charges:
        elec.addParticle([q])
    for i, row in df_exclusions.iterrows():
        elec.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        elec.setNonbondedMethod(elec.CutoffPeriodic)
    else:
        elec.setNonbondedMethod(elec.CutoffNonPeriodic)
    elec.setCutoffDistance(cutoff)
    elec.setForceGroup(force_group)
    return elec



