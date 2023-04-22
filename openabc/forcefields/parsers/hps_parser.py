import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
from openabc.utils import helper_functions
import sys
import os

_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL']

_hps_amino_acid_mass_dict = dict(ALA=71.08, ARG=156.20, ASN=114.10, ASP=115.10, CYS=103.10, 
                                GLN=128.10, GLU=129.10, GLY=57.05, HIS=137.10, ILE=113.20, 
                                LEU=113.20, LYS=128.20, MET=131.20, PHE=147.20, PRO=97.12, 
                                SER=87.08, THR=101.10, TRP=186.20, TYR=163.20, VAL=99.07)

_hps_amino_acid_charge_dict = dict(ALA=0.0, ARG=1.0, ASN=0.0, ASP=-1.0, CYS=0.0, 
                                  GLN=0.0, GLU=-1.0, GLY=0.0, HIS=0.5, ILE=0.0,
                                  LEU=0.0, LYS=1.0, MET=0.0, PHE=0.0, PRO=0.0,
                                  SER=0.0, THR=0.0, TRP=0.0, TYR=0.0, VAL=0.0)

_kcal_to_kj = 4.184

class HPSParser(object):
    """
    HPS protein parser.
    """
    def __init__(self, ca_pdb, default_parse=True):
        """
        Initialize a protein with HPS model.
        
        Parameters
        ----------
        ca_pdb : str
            Path for the CA pdb file. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        """
        self.pdb = ca_pdb
        self.atoms = helper_functions.parse_pdb(ca_pdb)
        # check if all the atoms are protein CA atoms
        assert ((self.atoms['resname'].isin(_amino_acids)).all() and self.atoms['name'].eq('CA').all())
        if default_parse:
            print('Parse molecule with default settings.')
            self.parse_mol()
    
    @classmethod
    def from_atomistic_pdb(cls, atomistic_pdb, ca_pdb, write_TER=False, default_parse=True):
        """
        Initialize an HPS model protein from atomistic pdb. 
        
        Parameters
        ----------
        atomistic_pdb : str
            Path for the atomistic pdb file. 
        
        ca_pdb : str
            Output path for the CA pdb file. 
        
        write_TER : bool
            Whether to write TER between two chains. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        """
        helper_functions.atomistic_pdb_to_ca_pdb(atomistic_pdb, ca_pdb, write_TER)
        return cls(ca_pdb, default_parse)
    
    def parse_mol(self, exclude12=True, mass_dict=_hps_amino_acid_mass_dict, 
                     charge_dict=_hps_amino_acid_charge_dict):
        """
        Parse molecule. 
        
        Parameters
        ----------
        exclude12 : bool
            Whether to exclude nonbonded interactions between 1-2 atoms. 
        
        mass_dict : dict
            Mass dictionary. 
        
        charge_dict : dict
            Charge dictionary. 
        
        """
        bonds = []
        n_atoms = len(self.atoms.index)
        for atom1 in range(n_atoms):
            chain1 = self.atoms.loc[atom1, 'chainID']
            if atom1 < n_atoms - 1:
                atom2 = atom1 + 1
                chain2 = self.atoms.loc[atom2, 'chainID']
                if chain1 == chain2:
                    bonds.append([atom1, atom2])
        bonds = np.array(bonds)
        self.protein_bonds = pd.DataFrame(bonds, columns=['a1', 'a2'])
        self.protein_bonds.loc[:, 'r0'] = 0.38
        self.protein_bonds.loc[:, 'k_bond'] = 2000*_kcal_to_kj
        if exclude12:
            self.exclusions = self.protein_bonds[['a1', 'a2']].copy()
        else:
            self.exclusions = pd.DataFrame(columns=['a1', 'a2'])
        # set mass and charge
        for i, row in self.atoms.iterrows():
            self.atoms.loc[i, 'mass'] = mass_dict[row['resname']]
            self.atoms.loc[i, 'charge'] = charge_dict[row['resname']]
        
     

