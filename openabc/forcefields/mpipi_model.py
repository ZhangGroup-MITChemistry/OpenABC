import numpy as np
import pandas as pd
try:
    import openmm.unit as unit
except ImportError:
    import simtk.unit as unit
from openabc.forcefields.cg_model import CGModel
from openabc.forcefields import functional_terms
from openabc.lib import _amino_acids, _rna_nucleotides
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

class MpipiModel(CGModel):
    """
    The class for Mpipi protein and RNA models. 
    This class inherits CGModel class. 
    We follow the parameters provided in the Mpipi LAMMPS input file. 
    """
    def __init__(self):
        """
        Initialize. 
        
        """
        self.atoms = None
        self.bonded_attr_names = ['protein_bonds', 'rna_bonds', 'exclusions']
    
    def add_protein_bonds(self, force_group=1):
        """
        Add protein bonds. 
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        print('Add protein bonds.')
        force = functional_terms.harmonic_bond_term(self.protein_bonds, self.use_pbc, force_group)
        self.system.addForce(force)
    
    def add_rna_bonds(self, force_group=2):
        """
        Add RNA bonds. 
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        print('Add RNA bonds.')
        force = functional_terms.harmonic_bond_term(self.rna_bonds, self.use_pbc, force_group)
        self.system.addForce(force)
    
    def add_contacts(self, cutoff_to_sigma_ratio=3, force_group=3):
        """
        Add nonbonded contacts, which is of Wang-Frenkel functional form. 
        
        Parameters
        ----------
        
        """
        print('Add nonbonded contacts.')
        atom_types = []
        for i, row in self.atoms.iterrows():
            if (row['resname'] in (_amino_acids + _rna_nucleotides)) and (row['name'] in ['CA', 'RN']):
                atom_types.append((_amino_acids + _rna_nucleotides).index(row['resname']))
            else:
                sys.exit('Error: atom type cannot recognize.')
        epsilon_wf_map = np.zeros((len(_amino_acids + _rna_nucleotides), len(_amino_acids + _rna_nucleotides)))
        sigma_wf_map = np.zeros((len(_amino_acids + _rna_nucleotides), len(_amino_acids + _rna_nucleotides)))
        mu_wf_map = np.zeros((len(_amino_acids + _rna_nucleotides), len(_amino_acids + _rna_nucleotides)))
        nu_wf_map = np.zeros((len(_amino_acids + _rna_nucleotides), len(_amino_acids + _rna_nucleotides)))
        # in df_Mpipi_parameters, RNA nucleotide names are RNA_A, RNA_C, RNA_G, RNA_U
        df_Mpipi_parameters = pd.read_csv(f'{__location__}/parameters/Mpipi_parameters.csv')
        _ext_rna_nucleotides = [f'RNA_{x}' for x in _rna_nucleotides]
        for i, row in df_Mpipi_parameters.iterrows():
            a1 = (_amino_acids + _ext_rna_nucleotides).index(row['atom_type1'])
            a2 = (_amino_acids + _ext_rna_nucleotides).index(row['atom_type2'])
            epsilon_wf_map[a1, a2] = float(row['epsilon'])
            epsilon_wf_map[a2, a1] = float(row['epsilon'])
            sigma_wf_map[a1, a2] = float(row['sigma'])
            sigma_wf_map[a2, a1] = float(row['sigma'])
            mu_wf_map[a1, a2] = float(row['mu'])
            mu_wf_map[a2, a1] = float(row['mu'])
            nu_wf_map[a1, a2] = float(row['nu'])
            nu_wf_map[a2, a1] = float(row['nu'])
        force = functional_terms.wang_frenkel_term(atom_types, self.exclusions, self.use_pbc, epsilon_wf_map, 
                                                  sigma_wf_map, mu_wf_map, nu_wf_map, cutoff_to_sigma_ratio, 
                                                  force_group)
        self.system.addForce(force)
    
    def add_dh_elec(self, ldby=(1/1.26)*unit.nanometer, dielectric_water=80.0, cutoff=3.5*unit.nanometer, 
                    force_group=4):
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
        force = functional_terms.dh_elec_term(charges, self.exclusions, self.use_pbc, ldby, dielectric_water, 
                                              cutoff, force_group)
        self.system.addForce(force)


