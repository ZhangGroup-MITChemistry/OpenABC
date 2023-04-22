import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
from openabc.forcefields.hps_model import HPSModel
from openabc.forcefields.functional_terms.zero_offset_nonbonded_terms import hps_ah_zero_offset_term
from openabc.forcefields.functional_terms.zero_offset_nonbonded_terms import dh_elec_zero_offset_term
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL']

_kcal_to_kj = 4.184

class HPSZeroOffsetModel(HPSModel):
    """
    An HPS model with zero offset for nonbonded interactions. 
    This model is only used for comparisons, as HOOMD-Blue HPS model does not shift nonbonded potential to zero at cutoff. 
    """
    def add_contacts(self, hydropathy_scale='Urry', epsilon=0.2*_kcal_to_kj, mu=1, delta=0.08, force_group=2):
        print('Add nonbonded contacts.')
        resname_list = self.atoms['resname'].tolist()
        atom_types = [_amino_acids.index(x) for x in resname_list]
        if hydropathy_scale == 'KR':
            print('Use KR hydropathy scale.')
            df_contact_parameters = pd.read_csv(f'{__location__}/parameters/HPS_KR_parameters.csv')
        elif hydropathy_scale == 'Urry':
            print('Use Urry hydropathy scale.')
            df_contact_parameters = pd.read_csv(f'{__location__}/parameters/HPS_Urry_parameters.csv')
        else:
            sys.exit(f'Error: hydropathy scale {hydropathy_scale} cannot be recognized!')
        sigma_ah_map, lambda_ah_map = np.zeros((20, 20)), np.zeros((20, 20))
        for i, row in df_contact_parameters.iterrows():
            atom_type1 = _amino_acids.index(row['atom_type1'])
            atom_type2 = _amino_acids.index(row['atom_type2'])
            sigma_ah_map[atom_type1, atom_type2] = row['sigma']
            sigma_ah_map[atom_type2, atom_type1] = row['sigma']
            lambda_ah_map[atom_type1, atom_type2] = row['lambda']
            lambda_ah_map[atom_type2, atom_type1] = row['lambda']
        print(f'Scale factor mu = {mu} and shift delta = {delta}.')
        lambda_ah_map = mu*lambda_ah_map - delta
        force = hps_ah_zero_offset_term(atom_types, self.exclusions, self.use_pbc, epsilon, sigma_ah_map, 
                                        lambda_ah_map, force_group)
        self.system.addForce(force)
    
    def add_dh_elec(self, ldby=1*unit.nanometer, dielectric_water=80.0, cutoff=3.5*unit.nanometer, force_group=3):
        print('Add Debye-Huckel electrostatic interactions.')
        print(f'Set Debye length as {ldby.value_in_unit(unit.nanometer)} nm.')
        print(f'Set water dielectric as {dielectric_water}.')
        charges = self.atoms['charge'].tolist()
        force = dh_elec_zero_offset_term(charges, self.exclusions, self.use_pbc, ldby, dielectric_water, cutoff, force_group)
        self.system.addForce(force)
    
