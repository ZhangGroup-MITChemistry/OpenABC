import numpy as np
import pandas as pd
from openabc.utils import parse_pdb, write_pdb, atomistic_pdb_to_ca_pdb, build_straight_CA_chain
from openabc.lib import _amino_acids, _rna_nucleotides, _kcal_to_kj, _angstrom_to_nm
import sys
import os

# use A, C, G, and U to represent RNA nucleotides

_mpipi_mass_dict = dict(ALA=71.08, ARG=156.20, ASN=114.10, ASP=115.10, CYS=103.10, 
                        GLN=128.10, GLU=129.10, GLY=57.05, HIS=137.10, ILE=113.20, 
                        LEU=113.20, LYS=128.20, MET=131.20, PHE=147.20, PRO=97.12, 
                        SER=87.08, THR=101.10, TRP=186.20, TYR=163.20, VAL=99.07, 
                        A=329.2, C=305.2, G=345.2, U=306.2)

_mpipi_charge_dict = dict(ALA=0.0, ARG=0.75, ASN=0.0, ASP=-0.75, CYS=0.0, 
                          GLN=0.0, GLU=-0.75, GLY=0.0, HIS=0.375, ILE=0.0,
                          LEU=0.0, LYS=0.75, MET=0.0, PHE=0.0, PRO=0.0,
                          SER=0.0, THR=0.0, TRP=0.0, TYR=0.0, VAL=0.0, 
                          A=-0.75, C=-0.75, G=-0.75, U=-0.75)

class MpipiProteinParser(object):
    """
    Mpipi protein parser.
    """
    def __init__(self, ca_pdb, default_parse=True):
        """
        Initialize a protein with Mpipi model.
        
        Parameters
        ----------
        ca_pdb : str
            CA pdb file path. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        """
        self.pdb = ca_pdb
        self.atoms = parse_pdb(ca_pdb)
        # check if all the atoms are protein CA atoms
        assert ((self.atoms['resname'].isin(_amino_acids)).all() and self.atoms['name'].eq('CA').all())
        if default_parse:
            print('Parse molecule with default settings.')
            self.parse_mol()
    
    @classmethod
    def from_atomistic_pdb(cls, atomistic_pdb, ca_pdb, write_TER=False, default_parse=True):
        """
        Initialize an Mpipi model protein from atomistic pdb. 
        
        Parameters
        ----------
        atomistic_pdb : str
            Atomistic pdb file path. 
        
        ca_pdb : str
            Output CA pdb file path. 
        
        write_TER : bool
            Whether to write TER between two chains. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        """
        atomistic_pdb_to_ca_pdb(atomistic_pdb, ca_pdb, write_TER)
        return cls(ca_pdb, default_parse)
    
    def parse_mol(self, exclude12=True, mass_dict=_mpipi_mass_dict, charge_dict=_mpipi_charge_dict):
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
        self.protein_bonds.loc[:, 'r0'] = 0.381
        self.protein_bonds.loc[:, 'k_bond'] = 2*9.6*_kcal_to_kj/(_angstrom_to_nm**2)
        if exclude12:
            self.exclusions = self.protein_bonds[['a1', 'a2']].copy()
        else:
            self.exclusions = pd.DataFrame(columns=['a1', 'a2'])
        # set mass and charge
        for i, row in self.atoms.iterrows():
            self.atoms.loc[i, 'mass'] = mass_dict[row['resname']]
            self.atoms.loc[i, 'charge'] = charge_dict[row['resname']]
        

class MpipiRNAParser(object):
    """
    Mpipi RNA parser.
    """
    def __init__(self, cg_pdb, default_parse=True):
        """
        Initialize a single strand RNA with Mpipi model. 
        
        Parameters
        ----------
        cg_pdb : str
            CG RNA pdb file path. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        """
        self.pdb = cg_pdb
        self.atoms = parse_pdb(self.pdb)
        # check if all the atoms are CG nucleotide atoms
        atom_names = self.atoms['name']
        assert (self.atoms['resname'].isin(_rna_nucleotides).all() and atom_names.eq('RN').all())
        if default_parse:
            print('Parse molecule with default settings.')
            self.parse_mol()
    
    @classmethod
    def from_sequence(cls, seq, cg_pdb, write_TER=False, default_parse=True):
        """
        Initialize an Mpipi model ssRNA from sequence. 
        The initial structure is a straight chain. 
        
        Parameters
        ----------
        seq : str or sequence-like
            Input RNA sequence. 
        
        cg_pdb : str
            Output CG RNA pdb file path. 
        
        write_TER : bool
            Whether to write TER between two chains. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        Returns
        ------- 
        result : class instance
            A class instance. 

        """
        n_atoms = len(seq)
        atoms = build_straight_CA_chain(n_atoms, chainID='A', r0=0.5)
        atoms.loc[:, 'name'] = 'RN'
        # RNA resnames should be from A, C, G, and U
        for i in range(n_atoms):
            atoms.loc[i, 'resname'] = seq[i]
        write_pdb(atoms, cg_pdb, write_TER)
        result = cls(cg_pdb, default_parse)
        return result
    
    def parse_mol(self, exclude12=True, mass_dict=_mpipi_mass_dict, charge_dict=_mpipi_charge_dict):
        """
        Parse molecule. 
        
        Parameters
        ----------
        exclude12 : bool
            Whether to exclude nonbonded interactions between 1-2 atom pairs. 
        
        mass_dict : dict
            Mass dictionary. 
        
        charge : float or int
            Nucleotide charge. 
        
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
        self.rna_bonds = pd.DataFrame(bonds, columns=['a1', 'a2'])
        self.rna_bonds.loc[:, 'r0'] = 0.5
        self.rna_bonds.loc[:, 'k_bond'] = 2*9.6*_kcal_to_kj/(_angstrom_to_nm**2)
        if exclude12:
            self.exclusions = self.rna_bonds[['a1', 'a2']].copy()
        else:
            self.exclusions = pd.DataFrame(columns=['a1', 'a2'])
        # set mass and charge
        for i, row in self.atoms.iterrows():
            self.atoms.loc[i, 'mass'] = mass_dict[row['resname']]
            self.atoms.loc[:, 'charge'] = charge_dict[row['resname']]


