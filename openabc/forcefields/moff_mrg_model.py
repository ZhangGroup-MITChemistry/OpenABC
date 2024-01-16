import numpy as np
import pandas as pd
try:
    import openmm.unit as unit
except ImportError:
    import simtk.unit as unit
from openabc.forcefields.cg_model import CGModel
from openabc.forcefields import functional_terms
from openabc.lib import _amino_acids, _dna_nucleotides
import warnings
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

class MOFFMRGModel(CGModel):
    """
    A class for MOFF+MRG model that represents a mixture of MOFF proteins and MRG DNA. 
    """
    def __init__(self):
        """
        Initialize. 
        """
        self.atoms = None
        self.bonded_attr_names = ['protein_bonds', 'protein_angles', 'protein_dihedrals', 'native_pairs', 'dna_bonds', 
                                  'dna_angles', 'dna_fan_bonds', 'exclusions']

    def add_protein_bonds(self, force_group=1):
        """
        Add protein bonds.
        
        Parameters
        ----------
        
        force_group : int
            Force group. 
        
        """
        if hasattr(self, 'protein_bonds'):
            print('Add protein bonds.')
            force = functional_terms.harmonic_bond_term(self.protein_bonds, self.use_pbc, force_group)
            self.system.addForce(force)

    def add_protein_angles(self, threshold=130*np.pi/180, clip=False, force_group=2, verbose=True):
        """
        Add protein angles.
        
        Note that the angle potential is a harmonic angle potential, which may lead to unstable simulation if harmonic potential center theta0 is too large.
        
        Based on some tests, theta0 <= 130 degrees (130*np.pi/180 radians) can facilitate 10 fs timestep. 
        
        Parameters
        ----------
        threshold : float or int
            The suggested upper limit value of theta0.
        
        clip : bool
            Whether to change theta0 values larger than threshold to threshold.
            If True, all the theta0 values larger than threshold will be changed to threshold. 
            If False, do not change theta0 values.
        
        force_group : int
            Force group. 
        
        verbose: bool
            Whether to show warnings if theta0 is larger than the threshold. 
        
        """
        if hasattr(self, 'protein_angles'):
            any_theta0_beyond_threshold = False
            for i, row in self.protein_angles.iterrows():
                a1 = int(row['a1'])
                a2 = int(row['a2'])
                a3 = int(row['a3'])
                theta0 = float(row['theta0'])
                if (theta0 > threshold) and verbose:
                    warnings.warn(f'Warning: angle composed of atom ({a1}, {a2}, {a3}) has theta0 equal to {theta0}, which is larger than the threshold value equal to {threshold}!')
                    any_theta0_beyond_threshold = True
            if clip and any_theta0_beyond_threshold:
                print(f'Decrease all the theta0 values larger than {threshold} to {threshold}.')
                # set threshold as np.float32(threshold) to avoid warnings
                threshold = np.float32(threshold)
                self.protein_angles.loc[self.protein_angles['theta0'] > threshold, 'theta0'] = threshold
            force = functional_terms.harmonic_angle_term(self.protein_angles, self.use_pbc, force_group)
            self.system.addForce(force)
    
    def add_protein_dihedrals(self, force_group=3):
        """
        Add protein dihedrals. 
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        if hasattr(self, 'protein_dihedrals'):
            print('Add protein dihedrals.')
            force = functional_terms.periodic_dihedral_term(self.protein_dihedrals, self.use_pbc, force_group)
            self.system.addForce(force)
    
    def add_native_pairs(self, force_group=4):
        """
        Add native pairs. 
        
        Parameters
        ----------
        force_group : int
            Force group.
        
        """
        if hasattr(self, 'native_pairs'):
            print('Add native pairs.')
            force = functional_terms.native_pair_12_10_term(self.native_pairs, self.use_pbc, force_group)
            self.system.addForce(force)

    def add_dna_bonds(self, force_group=5):
        """
        Add DNA bonds. 
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        if hasattr(self, 'dna_bonds'):
            print('Add DNA bonds.')
            force = functional_terms.class2_bond_term(self.dna_bonds, self.use_pbc, force_group)
            self.system.addForce(force)
    
    def add_dna_angles(self, force_group=6):
        """
        Add DNA angles. 
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        if hasattr(self, 'dna_angles'):
            print('Add DNA angles.')
            force = functional_terms.class2_angle_term(self.dna_angles, self.use_pbc, force_group)
            self.system.addForce(force)
        
    def add_dna_fan_bonds(self, force_group=7):
        """
        Add DNA fan bonds. 
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        if hasattr(self, 'dna_fan_bonds'):
            print('Add DNA fan bonds.')
            force = functional_terms.class2_bond_term(self.dna_fan_bonds, self.use_pbc, force_group)
            self.system.addForce(force)
    
    def add_contacts(self, eta=0.7/unit.angstrom, r0=8*unit.angstrom, cutoff=2.0*unit.nanometer, 
                     alpha_protein_dna=1.6264e-3, alpha_dna_dna=1.678e-5, epsilon_protein_dna=0, epsilon_dna_dna=0, 
                     force_group=8):
        """
        Add nonbonded contacts for MOFF protein and MRG DNA. 
        
        For amino acids, the CA atom type indices are 0-19, and CG nucleotide atom type index is 20. 
        
        The potential expression is: alpha/r^12 - 0.5*epsilon*(1 + tanh(eta*(r0 - r)))
        
        Parameters
        ----------
        eta : Quantity
            Parameter eta. 
        
        r0 : Quantity
            Parameter r0. 
        
        cutoff : Quantity
            Cutoff distance
        
        alpha_protein_dna : float or int
            Protein-DNA interaction parameter alpha. 
        
        alpha_dna_dna : float or int
            DNA-DNA interaction parameter alpha. 
        
        epsilon_protein_dna : float or int
            Protein-DNA interaction parameter epsilon. 
        
        epsilon_dna_dna : float or int
            DNA-DNA interaction parameter epsilon. 
        
        force_group : int
            Force group.  
        
        """
        print('Add protein and DNA nonbonded contacts.')
        atom_types = []
        for i, row in self.atoms.iterrows():
            if (row['resname'] in _amino_acids) and (row['name'] == 'CA'):
                atom_types.append(_amino_acids.index(row['resname']))
            elif (row['resname'] in _dna_nucleotides) and (row['name'] == 'DN'):
                atom_types.append(20)
            else:
                sys.exit('Error: atom type cannot recognize.')
        df_MOFF_contact_parameters = pd.read_csv(f'{__location__}/parameters/MOFF_contact_parameters.csv')
        alpha_map, epsilon_map = np.zeros((21, 21)), np.zeros((21, 21))
        for i, row in df_MOFF_contact_parameters.iterrows():
            a1 = _amino_acids.index(row['atom_type1'])
            a2 = _amino_acids.index(row['atom_type2'])
            alpha_map[a1, a2] = row['alpha']
            alpha_map[a2, a1] = row['alpha']
            epsilon_map[a1, a2] = row['epsilon']
            epsilon_map[a2, a1] = row['epsilon']
        alpha_map[:20, 20] = alpha_protein_dna
        alpha_map[20, :20] = alpha_protein_dna
        alpha_map[20, 20] = alpha_dna_dna
        epsilon_map[:20, 20] = epsilon_protein_dna
        epsilon_map[20, :20] = epsilon_protein_dna
        epsilon_map[20, 20] = epsilon_dna_dna
        force = functional_terms.moff_mrg_contact_term(atom_types, self.exclusions, self.use_pbc, alpha_map, 
                                                       epsilon_map, eta, r0, cutoff, force_group)
        self.system.addForce(force)
    
    def add_elec_switch(self, salt_conc=150.0*unit.millimolar, temperature=300.0*unit.kelvin, 
                        cutoff1=1.2*unit.nanometer, cutoff2=1.5*unit.nanometer, switch_coeff=[1, 0, 0, -10, 15, -6], 
                        add_native_pair_elec=True, force_group=9):
        """
        Add electrostatic interaction with switch function. 
        
        The switch function switches potential to zero within range cutoff1 < r <= cutoff2. 
        
        Parameters
        ----------
        salt_conc : Quantity
            Monovalent salt concentration. 
        
        temperature : Quantity
            Temperature. 
        
        cutoff1 : Quantity
            Cutoff distance 1. 
        
        cutoff2 : Quantity
            Cutoff distance 2. 
        
        switch_coeff : sequence-like
            Switch function coefficients. 
        
        add_native_pair_elec : bool
            Whether to add electrostatic interactions between native pairs. 
        
        force_group : int
            Force group. 
        
        """
        print('Add protein and DNA electrostatic interactions with distance-dependent dielectric and switch.')
        charges = self.atoms['charge'].tolist()
        force1 = functional_terms.ddd_dh_elec_switch_term(charges, self.exclusions, self.use_pbc, salt_conc, 
                                                          temperature, cutoff1, cutoff2, switch_coeff, force_group)
        self.system.addForce(force1)
        if add_native_pair_elec and hasattr(self, 'native_pairs'):
            print('Add electrostatic interactions between native pair atoms.')
            df_charge_bonds = pd.DataFrame(columns=['a1', 'a2', 'q1', 'q2'])
            for i, row in self.native_pairs.iterrows():
                a1, a2 = int(row['a1']), int(row['a2'])
                q1, q2 = float(charges[a1]), float(charges[a2])
                if (q1 != 0) and (q2 != 0):
                    df_charge_bonds.loc[len(df_charge_bonds.index)] = [a1, a2, q1, q2]
            force2 = functional_terms.ddd_dh_elec_switch_bond_term(df_charge_bonds, self.use_pbc, salt_conc, 
                                                                   temperature, cutoff1, cutoff2, switch_coeff, 
                                                                   force_group)
            self.system.addForce(force2)
        else:
            print('Do not add electrostatic interactions between native pair atoms.')
        
    def add_all_default_forces(self):
        """
        Add all the forces with default settings. 
        """
        print('Add all the forces with default settings.')
        self.add_protein_bonds()
        self.add_protein_angles()
        self.add_protein_dihedrals()
        self.add_native_pairs()
        self.add_dna_bonds()
        self.add_dna_angles()
        self.add_dna_fan_bonds()
        self.add_contacts()
        self.add_elec_switch()


