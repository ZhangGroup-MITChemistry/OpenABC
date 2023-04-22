import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import mdtraj
from openabc.utils import helper_functions
from openabc.utils.shadow_map import find_ca_pairs_from_atomistic_pdb
import sys
import os

_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL']

_moff_amino_acid_mass_dict = dict(ALA=71.08, ARG=156.20, ASN=114.10, ASP=115.10, CYS=103.10, 
                                  GLN=128.10, GLU=129.10, GLY=57.05, HIS=137.10, ILE=113.20, 
                                  LEU=113.20, LYS=128.20, MET=131.20, PHE=147.20, PRO=97.12, 
                                  SER=87.08, THR=101.10, TRP=186.20, TYR=163.20, VAL=99.07)

_moff_amino_acid_charge_dict = dict(ALA=0.0, ARG=1.0, ASN=0.0, ASP=-1.0, CYS=0.0, 
                                    GLN=0.0, GLU=-1.0, GLY=0.0, HIS=0.25, ILE=0.0,
                                    LEU=0.0, LYS=1.0, MET=0.0, PHE=0.0, PRO=0.0,
                                    SER=0.0, THR=0.0, TRP=0.0, TYR=0.0, VAL=0.0)

_kcal_to_kj = 4.184

class MOFFParser(object):
    """
    MOFF protein parser.
    """
    def __init__(self, atomistic_pdb, ca_pdb, default_parse=True):
        """
        Initialize a protein with MOFF model. 
        
        Parameters
        ----------
        atomistic_pdb : str
            Path for the atomistic pdb file. 
        
        ca_pdb : str
            path for the CA pdb file. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        """
        self.atomistic_pdb = atomistic_pdb
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
            path for the CA pdb file. 
            
        write_TER : bool
            Whether to write TER between two chains. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        """
        helper_functions.atomistic_pdb_to_ca_pdb(atomistic_pdb, ca_pdb, write_TER)
        return cls(atomistic_pdb, ca_pdb, default_parse)
    
    def parse_exclusions(self, exclude12=True, exclude13=True, exclude14=True, exclude_native_pairs=True):
        """
        Parse nonbonded exclusions based on bonds, angles, dihedrals, and native pairs.
        
        Parameters
        ----------
        exclude12 : bool
            Whether to exclude nonbonded interactions between 1-2 atom pairs. 
        
        exclude13 : bool
            Whether to exclude nonbonded interactions between 1-3 atom pairs. 
        
        exclude14 : bool
            Whether to exclude nonbonded interactions between 1-4 atom pairs. 
        
        exclude_native_pairs : bool
            Whether to exclude nonbonded interactions between native pairs. 
        
        """
        exclusions = []
        if exclude12 and hasattr(self, 'protein_bonds'):
            for i, row in self.protein_bonds.iterrows():
                exclusions.append((int(row['a1']), int(row['a2'])))
        if exclude13 and hasattr(self, 'protein_angles'):
            for i, row in self.protein_angles.iterrows():
                exclusions.append((int(row['a1']), int(row['a3'])))
        if exclude14 and hasattr(self, 'protein_dihedrals'):
            for i, row in self.protein_dihedrals.iterrows():
                exclusions.append((int(row['a1']), int(row['a4'])))
        if exclude_native_pairs and hasattr(self, 'native_pairs'):
            for i, row in self.native_pairs.iterrows():
                exclusions.append((int(row['a1']), int(row['a2'])))
        exclusions = np.array(sorted(exclusions))
        self.exclusions = pd.DataFrame(exclusions, columns=['a1', 'a2']).drop_duplicates(ignore_index=True)
        
    def parse_mol(self, get_native_pairs=True, frame=0, radius=0.1, bonded_radius=0.05, cutoff=0.6, box=None, 
                  use_pbc=False, exclude12=True, exclude13=True, exclude14=True, exclude_native_pairs=True, 
                  mass_dict=_moff_amino_acid_mass_dict, charge_dict=_moff_amino_acid_charge_dict):
        """
        Parse molecule.
        
        Parameters
        ----------
        get_native_pairs : bool
            Whether to get native pairs from atomistic pdb with shadow algorithm. 
        
        frame : int
            Frame index to compute native structure parameters for both CG and atomistic models. 
            
        radius : float or int
            Shadow algorithm radius. 
        
        bonded_radius : float or int
            Shadow algorithm bonded radius. 
        
        cutoff : float or int
            Shadow algorithm cutoff. 
        
        box : None or np.ndarray
        Specify box shape. 
        Note `use_pbc` = False is only compatible with orthogonal box. 
        If `use_pbc` = False, then set `box` as None. 
        If `use_pbc` = True, then `box` = np.array([lx, ly, lz, alpha1, alpha2, alpha3]). 
        
        use_pbc : bool
            Whether to use periodic boundary condition (PBC) for searching native pairs. 
        
        exclude12 : bool
            Whether to exclude nonbonded interactions between 1-2 atom pairs. 
        
        exclude13 : bool
            Whether to exclude nonbonded interactions between 1-3 atom pairs. 
        
        exclude14 : bool
            Whether to exclude nonbonded interactions between 1-4 atom pairs. 
        
        exclude_native_pairs : bool
            Whether to exclude nonbonded interactions between native pairs. 
        
        mass_dict : dict
            Mass dictionary. 
        
        charge_dict : dict
            Charge dictionary. 
        
        """
        # set bonds, angles, and dihedrals
        bonds, angles, dihedrals = [], [], []
        n_atoms = len(self.atoms.index)
        for atom1 in range(n_atoms):
            chain1 = self.atoms.loc[atom1, 'chainID']
            if atom1 < n_atoms - 1:
                atom2 = atom1 + 1
                chain2 = self.atoms.loc[atom2, 'chainID']
                if chain1 == chain2:
                    bonds.append([atom1, atom2])
            if atom1 < n_atoms - 2:
                atom3 = atom1 + 2
                chain3 = self.atoms.loc[atom3, 'chainID']
                if (chain1 == chain2) and (chain1 == chain3):
                    angles.append([atom1, atom2, atom3])
            if atom1 < n_atoms - 3:
                atom4 = atom1 + 3
                chain4 = self.atoms.loc[atom4, 'chainID']
                if (chain1 == chain2) and (chain1 == chain3) and (chain1 == chain4):
                    dihedrals.append([atom1, atom2, atom3, atom4])
        bonds, angles, dihedrals = np.array(bonds), np.array(angles), np.array(dihedrals)
        traj = mdtraj.load_pdb(self.pdb)
        self.protein_bonds = pd.DataFrame(bonds, columns=['a1', 'a2'])
        self.protein_bonds.loc[:, 'r0'] = 0.38
        self.protein_bonds.loc[:, 'k_bond'] = 1000
        self.protein_angles = pd.DataFrame(angles, columns=['a1', 'a2', 'a3'])
        self.protein_angles['theta0'] = mdtraj.compute_angles(traj, angles, periodic=use_pbc)[frame]
        self.protein_angles.loc[:, 'k_angle'] = 120
        self.protein_dihedrals = pd.DataFrame(columns=['a1', 'a2', 'a3', 'a4', 'periodicity', 'phi0', 'k_dihedral'])
        phi = mdtraj.compute_dihedrals(traj, dihedrals, periodic=use_pbc)[frame]
        for i in range(dihedrals.shape[0]):
            row = dihedrals[i].tolist() + [1, phi[i] + np.pi, 3.0]
            self.protein_dihedrals.loc[len(self.protein_dihedrals.index)] = row
            row = dihedrals[i].tolist() + [3, 3*(phi[i] + np.pi), 1.5]
            self.protein_dihedrals.loc[len(self.protein_dihedrals.index)] = row
        # set native pairs
        if get_native_pairs:
            print('Get native pairs with shadow algorithm.')
            self.native_pairs = find_ca_pairs_from_atomistic_pdb(self.atomistic_pdb, frame, radius, bonded_radius, 
                                                                 cutoff, box, use_pbc)
            self.native_pairs.loc[:, 'epsilon'] = 3.0
        # set exclusions
        self.parse_exclusions(exclude12, exclude13, exclude14, exclude_native_pairs) 
        # set mass and charge
        for i, row in self.atoms.iterrows():
            self.atoms.loc[i, 'mass'] = mass_dict[row['resname']]
            self.atoms.loc[i, 'charge'] = charge_dict[row['resname']]


