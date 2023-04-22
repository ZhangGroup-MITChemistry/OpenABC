import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
from openabc.forcefields.cg_model import CGModel
from openabc.forcefields.functional_terms import bond_terms
from openabc.forcefields.functional_terms import nonbonded_terms
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL']

_kcal_to_kj = 4.184

class HPSModel(CGModel):
    """
    A class for HPS model that represents a mixture of HPS model proteins. 
    
    This class inherits CGModel class. 
    """
    def __init__(self):
        """
        Initialize. 
        """
        self.atoms = None
        self.bonded_attr_names = ['protein_bonds', 'exclusions']
        
    def add_protein_bonds(self, force_group=1):
        """
        Add protein bonds. 
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        print('Add protein bonds.')
        force = bond_terms.harmonic_bond_term(self.protein_bonds, self.use_pbc, force_group)
        self.system.addForce(force)
    
    def add_contacts(self, hydropathy_scale='Urry', epsilon=0.2*_kcal_to_kj, mu=1, delta=0.08, force_group=2):
        """
        Add nonbonded contacts. 
        
        The raw hydropathy scale is scaled and shifted by: mu*lambda - delta
        
        Parameters
        ----------
        hydropathy_scale : str
            Hydropathy scale, can be KR or Urry. 
        
        epsilon : float or int
            Contact strength. 
        
        mu : float or int
            Hydropathy scale factor. 
        
        delta : float or int
            Hydropathy shift factor. 
        
        force_group : int
            Force group. 
            
        """
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
        force = nonbonded_terms.hps_ah_term(atom_types, self.exclusions, self.use_pbc, epsilon, sigma_ah_map, 
                                            lambda_ah_map, force_group)
        self.system.addForce(force)
    
    def add_dh_elec(self, ldby=1*unit.nanometer, dielectric_water=80.0, cutoff=3.5*unit.nanometer, force_group=3):
        """
        Add Debye-Huckel electrostatic interactions. 
        
        Parameters
        ----------
        ldby : Quantity
            Debye length. 
        
        dielectric_water : float or int
            Dielectric constant of water. 
        
        cutoff : Quantity
            Cutoff distance. 
        
        force_group : int
            Force group. 
        
        """
        print('Add Debye-Huckel electrostatic interactions.')
        print(f'Set Debye length as {ldby.value_in_unit(unit.nanometer)} nm.')
        print(f'Set water dielectric as {dielectric_water}.')
        charges = self.atoms['charge'].tolist()
        force = nonbonded_terms.dh_elec_term(charges, self.exclusions, self.use_pbc, ldby, dielectric_water, cutoff, 
                                             force_group)
        self.system.addForce(force)

    def add_all_default_forces(self):
        """
        Add all the forces with default settings. 
        """
        print('Add all the forces with default settings.')
        self.add_protein_bonds(force_group=1)
        self.add_contacts(force_group=2)
        self.add_dh_elec(force_group=3)
        

