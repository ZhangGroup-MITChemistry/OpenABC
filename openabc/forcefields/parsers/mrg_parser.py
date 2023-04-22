import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
from openabc.utils import helper_functions
import sys
import os

_nucleotides = ['DA', 'DC', 'DG', 'DT']

_kcal_to_kj = 4.184
_deg_to_rad = np.pi/180

_k_mrg_dna_bonds = _kcal_to_kj*np.array([262.5, -226, 149])
_r0_mrg_dna_bonds = 0.496
_k_mrg_dna_angles = _kcal_to_kj*np.array([9.22, 4.16, 1.078])
_theta0_mrg_dna_angles = 156*_deg_to_rad
_delta_mrg_dna_fan_bonds = np.arange(-5, 6)
_k_mrg_dna_fan_bonds = _kcal_to_kj*np.array([[4.67, 2.1, 1.46], 
                                             [1.324e-4, -12.2, 18.5], 
                                             [8.5, -44.4, 50], 
                                             [12.3, -40, 37], 
                                             [4, -10, 8], 
                                             [292, 410, 720], 
                                             [11.5, -41, 58], 
                                             [9.55, -45.9, 50.2], 
                                             [13.78, -52.7, 50], 
                                             [13.86, -56.8, 50], 
                                             [36.26, -77, 50]])
_r0_mrg_dna_fan_bonds = np.array([1.71, 1.635, 1.47, 1.345, 1.23, 1.13, 0.99, 0.92, 1.02, 1.25, 1.69])

class MRGdsDNAParser(object):
    """
    MRG dsDNA parser. 
    Note this parser only works on one dsDNA!
    """
    def __init__(self, cg_pdb, default_parse=True):
        """
        Initialize a dsDNA with MRG model. 
        
        Parameters
        ----------
        cg_pdb : str
            Path for the CG dsDNA pdb. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        """
        self.pdb = cg_pdb
        self.atoms = helper_functions.parse_pdb(self.pdb)
        # check if all the atoms are CG nucleotide atoms
        atom_names = self.atoms['name']
        assert (self.atoms['resname'].isin(_nucleotides).all() and atom_names.eq('NU').all())
        # check if there are only 2 chains
        unique_chainID = list(set(self.atoms['chainID'].tolist()))
        assert len(unique_chainID) == 2
        # check if the first half is the first ssDNA, and the second half is the second ssDNA
        n_atoms = len(self.atoms.index)
        n_bp = int(n_atoms/2)
        first_half_atoms = self.atoms.loc[0:n_bp - 1]
        second_half_atoms = self.atoms.loc[n_bp:n_atoms - 1]
        assert len(first_half_atoms.index) == len(second_half_atoms.index)
        assert first_half_atoms['chainID'].eq(first_half_atoms['chainID'].iloc[0]).all()
        assert second_half_atoms['chainID'].eq(second_half_atoms['chainID'].iloc[0]).all()
        if default_parse:
            print('Parse molecule with default settings.')
            self.parse_mol()
    
    @classmethod
    def from_atomistic_pdb(cls, atomistic_pdb, cg_pdb, write_TER=False, default_parse=True):
        """
        Initialize an MRG model dsDNA from atomistic pdb. 
        
        Parameters
        ----------
        atomistic_pdb : str
            Path for the atomistic dsDNA pdb file. 
        
        cg_pdb : str
            Output path for the CG dsDNA pdb file. 
        
        write_TER : bool
            Whether to write TER between two chains. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        """
        helper_functions.atomistic_pdb_to_nucleotide_pdb(atomistic_pdb, cg_pdb, write_TER)
        return cls(cg_pdb, default_parse)
    
    def parse_exclusions(self, exclude12=True, exclude13=True):
        """
        Parse nonbonded exclusions based on bonds and angles. 
        Note nonbonded interactions are not excluded for atom pairs with fan bonds. 
        
        Parameters
        ----------
        exclude12 : bool
            Whether to exclude nonbonded interactions between 1-2 atom pairs. 
        
        exclude13 : bool
            Whether to exclude nonbonded interactions between 1-3 atom pairs. 
        
        """
        exclusions = []
        if exclude12 and hasattr(self, 'dna_bonds'):
            for i, row in self.dna_bonds.iterrows():
                exclusions.append((int(row['a1']), int(row['a2'])))
        if exclude13 and hasattr(self, 'dna_angles'):
            for i, row in self.dna_angles.iterrows():
                exclusions.append((int(row['a1']), int(row['a3'])))
        exclusions = np.array(sorted(exclusions))
        self.exclusions = pd.DataFrame(exclusions, columns=['a1', 'a2']).drop_duplicates(ignore_index=True)
    
    def parse_mol(self, exclude12=True, exclude13=True, bonded_energy_scale=0.9, mass=325, charge=-1):
        """
        Parse molecule.
        
        Parameters
        ----------
        exclude12 : bool
            Whether to exclude nonbonded interactions between 1-2 atom pairs. 
        
        exclude13 : bool
            Whether to exclude nonbonded interactions between 1-3 atom pairs. 
        
        bonded_energy_scale : float or int
            Scale factor for all the bonded energies. 
        
        mass : float or int
            Mass of CG nucleotide bead. 
        
        charge : float or int
            Charge of each CG nucleotide bead. 
        
        """
        # set bonds and angles
        bonds, angles = [], []
        n_atoms = len(self.atoms.index)
        for atom1 in range(n_atoms):
            chain1 = self.atoms.loc[atom1, 'chainID']
            if atom1 < n_atoms - 1:
                atom2 = atom1 + 1
                chain2 = self.atoms.loc[atom2, 'chainID']
                if chain1 == chain2:
                    bonds.append([atom1, atom2] + [_r0_mrg_dna_bonds] + _k_mrg_dna_bonds.tolist())
            if atom1 < n_atoms - 2:
                atom3 = atom1 + 2
                chain3 = self.atoms.loc[atom3, 'chainID']
                if (chain1 == chain2) and (chain1 == chain3):
                    angles.append([atom1, atom2, atom3] + [_theta0_mrg_dna_angles] + _k_mrg_dna_angles.tolist())
        bonds, angles = np.array(bonds), np.array(angles)
        self.dna_bonds = pd.DataFrame(bonds, columns=['a1', 'a2', 'r0', 'k_bond_2', 'k_bond_3', 'k_bond_4'])
        self.dna_angles = pd.DataFrame(angles, columns=['a1', 'a2', 'a3', 'theta0', 'k_angle_2', 'k_angle_3', 
                                                        'k_angle_4'])
        # set fan bonds
        self.dna_fan_bonds = pd.DataFrame(columns=['a1', 'a2', 'r0', 'k_bond_2', 'k_bond_3', 'k_bond_4'])
        n_bp = int(n_atoms/2)
        for atom1 in range(n_bp):
            for j in range(11):
                atom2 = n_atoms - 1 - atom1 + _delta_mrg_dna_fan_bonds[j]
                if (atom2 >= n_bp) and (atom2 <= n_atoms - 1):
                    row = [atom1, atom2] + [_r0_mrg_dna_fan_bonds[j]] + _k_mrg_dna_fan_bonds[j].tolist()
                    self.dna_fan_bonds.loc[len(self.dna_fan_bonds.index)] = row
        # scale bonded interactions
        self.dna_bonds[['k_bond_2', 'k_bond_3', 'k_bond_4']] *= bonded_energy_scale
        self.dna_angles[['k_angle_2', 'k_angle_3', 'k_angle_4']] *= bonded_energy_scale
        self.dna_fan_bonds[['k_bond_2', 'k_bond_3', 'k_bond_4']] *= bonded_energy_scale
        # set exclusions
        self.parse_exclusions(exclude12, exclude13)
        # set mass and charge
        self.atoms.loc[:, 'mass'] = mass
        self.atoms.loc[:, 'charge'] = charge 
                
        

