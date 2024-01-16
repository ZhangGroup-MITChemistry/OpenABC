import numpy as np
import pandas as pd
from openabc.utils import get_WC_paired_sequence, fix_pdb, write_pdb
from openabc.forcefields.parameters import Mixin3SPN2ConfigParser
from openabc.lib import _dna_nucleotides, _dna_WC_pair_dict, _angstrom_to_nm
import subprocess
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

_atom_masses = {"H": 1.00794, "C": 12.0107, "N": 14.0067, "O": 15.9994, "P": 30.973762}

_CG_map = {"O5\'": 'P', "C5\'": 'S', "C4\'": 'S', "O4\'": 'S', "C3\'": 'S', "O3\'": 'P',
           "C2\'": 'S', "C1\'": 'S', "O5*": 'P', "C5*": 'S', "C4*": 'S', "O4*": 'S',
           "C3*": 'S', "O3*": 'P', "C2*": 'S', "C1*": 'S', "N1": 'B', "C2": 'B', "O2": 'B',
           "N2": 'B', "N3": 'B', "C4": 'B', "N4": 'B', "C5": 'B', "C6": 'B', "N9": 'B',
           "C8": 'B', "O6": 'B', "N7": 'B', "N6": 'B', "O4": 'B', "C7": 'B', "P": 'P',
           "OP1": 'P', "OP2": 'P', "O1P": 'P', "O2P": 'P', "OP3": 'P', "HO5'": 'P',
           "H5'": 'S', "H5''": 'S', "H4'": 'S', "H3'": 'S', "H2'": 'S', "H2''": 'S',
           "H1'": 'S', "H8": 'B', "H61": 'B', "H62": 'B', 'H2': 'B', 'H1': 'B', 'H21': 'B',
           'H22': 'B', 'H3': 'B', 'H71': 'B', 'H72': 'B', 'H73': 'B', 'H6': 'B', 'H41': 'B',
           'H42': 'B', 'H5': 'B', "HO3'": 'P'}

_cg_atom_name_to_element_map = {'P': 'P', 'S': 'H', 'A': 'N', 'T': 'S', 'G': 'C', 'C': 'O'}

"""
Open3SPN2 was originally developed by Carlos Bueno. 
Most code is adapted from the original Open3SPN2. 
"""


class DNA3SPN2Parser(Mixin3SPN2ConfigParser):
    """
    DNA 3SPN2 parser. 
    For B-curved DNA, the parser works best for a single strand ssDNA or WC-paired dsDNA. 
    Please ensure the parsed DNA has unique chainID for each chain. 
    """
    def __init__(self, dna_type='B_curved'):
        """
        Initialize. 
        """
        self.dna_type = dna_type
    
    def build_x3dna_template(self, temp_name='dna'):
        """
        Build template DNA structure with x3dna. 
        We do not need to specify whether to use PSB order, since template atom order is aligned with self.atoms.
        If self.atoms is one dsDNA molecule and the target sequence is W-C paired, x3dna input sequence is the sequence of the first ssDNA. Else, use the full sequence of the DNA as x3dna input sequence. 
        
        Parameters
        ----------
        temp_name : str
            Name for built template files. 
        
        Returns
        -------
        temp_atoms : pd.DataFrame
            X3DNA built CG DNA template structure. 
        
        """
        # get sequence
        sequence_list = self.get_sequence_list()
        n_chains = len(sequence_list)
        
        # if self.atoms is one dsDNA molecule with W-C paired sequence, then use the first ssDNA sequence to build the DNA template
        # else, use full sequence to build the DNA template
        flag = False
        if n_chains == 2:
            sequence1, sequence2 = sequence_list[0], sequence_list[1]
            if len(sequence1) == len(sequence2):
                if get_WC_paired_sequence(sequence1) == sequence2:
                    # one dsDNA molecule with W-C paired sequence
                    flag = True
        if flag:
            print('The input CG molecule is one dsDNA with W-C paired sequence.')
            print('Use the sequence of the first ssDNA to build the DNA template.')
            x3dna_input_sequence = sequence1
        else:
            print('Use the sequence including all ssDNA sequences to build the DNA template.')
            x3dna_input_sequence = ''.join(sequence_list)
        
        # set parameters
        pair = self.base_pair_geometry.copy()
        step = self.base_step_geometry.copy()
        pair.index = pair['stea']
        step.index = step['stea'] + step['steb']
        data = []
        _s = None
        for s in x3dna_input_sequence:
            pair_s = pair.loc[s, ['shear', 'stretch', 'stagger', 'buckle', 'propeller', 'opening']]
            # use "if _s is None" for clarity
            if _s is None:
                step_s = (step.loc['AA', ['shift', 'slide', 'rise', 'tilt', 'roll', 'twist']] + 100)*0
            else:
                step_s = step.loc[_s + s, ['shift', 'slide', 'rise', 'tilt', 'roll', 'twist']]
            data += [pd.concat([pd.Series([f'{s}-{_dna_WC_pair_dict[s]}'], index=['Sequence']), pair_s, step_s])]
            _s = s
        data = pd.concat(data, axis=1).T
        
        try:
            location_x3dna = os.environ['X3DNA']
        except KeyError:
            sys.exit('Cannot find X3DNA variable from the environment. ')
        
        with open(f'{temp_name}_template_parameters.par', 'w') as par:
            par.write(f' {len(data)} # Number of base pairs\n')
            par.write(f' 0 # local base-pair & step parameters\n')
            par.write('#')
            par.write(data.to_csv(sep=' ', index=False))
        
        # Attempting to call rebuild multiple times
        # This function fails sometimes when called by multiple because a file with the same name is created
        # This function call 3DNA to rebuild the {temp_name}_template.pdb based on given the given CG PDB template
        attempt = 0
        max_attempts = 10
        while attempt < max_attempts:
            try:
                subprocess.check_output([f'{location_x3dna}/bin/x3dna_utils',
                                         'cp_std', 'BDNA'])
                subprocess.check_output([f'{location_x3dna}/bin/rebuild',
                                         '-atomic', f'{temp_name}_template_parameters.par',
                                         f'{temp_name}_template.pdb'])
                break
            except subprocess.CalledProcessError as e:
                attempt += 1
                if attempt == max_attempts:
                    print(f"subprocess.CalledProcessError failed {max_attempts} times {e.args[0]}: {e.args[1]}")
    
        # update {temp_name}_template.pdb file so that atoms with serial number larger than 99999 is converted to "*****" (formatting issue)
        # rewrite the code for this part
        with open(f'{temp_name}_template.pdb', 'r') as f:
            lines = f.readlines()
            
        with open(f'{temp_name}_template.pdb', 'w') as f:
            atom_serial = 1
            for each_line in lines:
                if each_line[:4] == 'ATOM':
                    assert len(each_line) >= 12
                    if atom_serial <= 99999:
                        new_line = each_line[:6] + f'{atom_serial:>5}' + each_line[11:]
                        atom_serial += 1
                    else:
                        new_line = each_line[:6] + '*****' + each_line[11:]
                else:
                    new_line = each_line
                f.write(new_line)
        
        # parse x3dna template
        # from_atomistic_pdb is a class method, so dna_temp is a class object
        dna_temp = self.from_atomistic_pdb(f'{temp_name}_template.pdb', f'cg_{temp_name}_template.pdb', 
                                           default_parse=False)
        # check sequence
        # the target sequence is equal to the template sequence or the first half of template sequence
        target_sequence = self.get_sequence()
        template_sequence = dna_temp.get_sequence()
        n1 = len(target_sequence)
        n2 = len(template_sequence)
        assert ((n1 == n2) or (2*n1 == n2))
        assert template_sequence[:n1] == target_sequence
        
        # merge x3dna template structure with the original structure
        # as both CG structures use unique resSeq, we can merge by matching resSeq and name
        # the merging ensures the atom orders in dna_temp_atoms are consistent with the ones in original_atoms
        original_atoms = self.atoms.copy()
        dna_temp_atoms = dna_temp.atoms.copy()
        merged_atoms = pd.merge(original_atoms, dna_temp_atoms, on=['resSeq', 'name'], how='left', 
                                suffixes=['_old', ''])
        dna_temp_atoms = original_atoms.copy()
        dna_temp_atoms[['x', 'y', 'z']] = merged_atoms[['x', 'y', 'z']]
        return dna_temp_atoms
    
    def parse_mol(self, temp_from_x3dna=True, temp_name='dna', input_temp=None):
        """
        Parse molecule with given template. 
        
        Parameters
        ----------
        temp_from_x3dna : bool
            Whether to use x3dna template. 
        
        temp_name : str
            Name for x3dna built template files. 
        
        input_temp : None or pd.DataFrame
            Additional input template. Use input_temp only `if (not ((self.dna_type == 'B_curved') and temp_from_x3dna)) and (input_temp is not None)`. 
        
        """
        # parse configuration file if not yet
        if not hasattr(self, 'particle_definition'):
            self.parse_config_file()
        
        # reset index
        self.atoms.index = list(range(len(self.atoms.index)))
        
        # set mass
        particle_types = self.particle_definition[self.particle_definition['DNA'] == self.dna_type]
        for i, row in particle_types.iterrows():
            self.atoms.loc[self.atoms['name'] == row['name'], 'mass'] = row['mass']
        
        # set template
        if (self.dna_type == 'B_curved') and temp_from_x3dna:
            self.temp_atoms = self.build_x3dna_template(temp_name=temp_name)
        else:
            if input_temp is None:
                self.temp_atoms = self.atoms
            else:
                self.temp_atoms = input_temp
        
        # build atoms dataframe with multi-index
        atoms = self.atoms.copy()
        unique_cr = atoms[['chainID', 'resSeq']].drop_duplicates(ignore_index=True).copy()
        atoms['index'] = list(range(len(atoms.index)))
        atoms = atoms.set_index(['chainID', 'resSeq', 'name'])
        
        # set interaction types
        bond_types = self.bond_definition[self.bond_definition['DNA'] == self.dna_type]
        angle_types = self.angle_definition[self.angle_definition['DNA'] == self.dna_type]
        stacking_types = self.stacking_definition[self.stacking_definition['DNA'] == self.dna_type]
        dihedral_types = self.dihedral_definition[self.dihedral_definition['DNA'] == self.dna_type]
        
        # set bonds
        bond_data = []
        for i, ftype in bond_types.iterrows():
            ai = ftype['i']
            aj = ftype['j']
            s1 = ftype['s1']
            for _j, row in unique_cr.iterrows():
                c, r = row['chainID'], row['resSeq']
                k1 = (c, r, ai)
                k2 = (c, r + s1, aj)
                if (k1 in atoms.index) and (k2 in atoms.index):
                    a1 = int(atoms.loc[k1, 'index'])
                    a2 = int(atoms.loc[k2, 'index'])
                    bond_data += [[i, a1, a2]]
        bond_data = pd.DataFrame(bond_data, columns=['name', 'a1', 'a2'])
        self.dna_bonds = pd.merge(bond_data, bond_types, left_on='name', right_index=True)
        
        if self.dna_type == 'B_curved':
            # read r0 from template and save r0 in unit nm
            x1 = self.temp_atoms.loc[self.dna_bonds['a1'], ['x', 'y', 'z']].to_numpy().astype(np.float64)
            x2 = self.temp_atoms.loc[self.dna_bonds['a2'], ['x', 'y', 'z']].to_numpy().astype(np.float64)
            self.dna_bonds['r0'] = np.linalg.norm(x1 - x2, axis=1)*_angstrom_to_nm
        
        # set angles
        angle_data = []
        base = self.atoms['resname'].str[1:2]
        for i, ftype in angle_types.iterrows():
            ai = ftype['i']
            aj = ftype['j']
            ak = ftype['k']
            s1 = ftype['s1']
            s2 = ftype['s2']
            b1 = ftype['Base1']
            b2 = ftype['Base2']
            sb = ftype['sB']
            for _j, row in unique_cr.iterrows():
                c, r = row['chainID'], row['resSeq']
                k1 = (c, r, ai)
                k2 = (c, r + s1, aj)
                k3 = (c, r + s2, ak)
                k4 = (c, r + sb, 'S')
                if ((k1 in atoms.index) 
                    and (k2 in atoms.index) 
                    and (k3 in atoms.index) 
                    and (k4 in atoms.index)
                    and (b1 == '*' or base[int(atoms.loc[k1, 'index'])] == b1)
                    and (b2 == '*' or base[int(atoms.loc[k4, 'index'])] == b2)):
                    a1 = int(atoms.loc[k1, 'index'])
                    a2 = int(atoms.loc[k2, 'index'])
                    a3 = int(atoms.loc[k3, 'index'])
                    ax = int(atoms.loc[k4, 'index'])
                    angle_data += [[i, a1, a2, a3, ax, sb]]
        angle_data = pd.DataFrame(angle_data, columns=['name', 'a1', 'a2', 'a3', 'ax', 'sB'])
        self.dna_angles = pd.merge(angle_data, angle_types, left_on='name', right_index=True)
        
        if self.dna_type == 'B_curved':
            # read native angle theta0 from template and save theta0 in unit rad
            # originally open3SPN2 uses name t0, and here we use name theta0
            # originally in open3SPN2 t0 is saved in degree, here for consistency we save it in rad
            # here we use x1, x2, and x3 for coordinates, and we use v1 and v2 for vectors between points
            x1 = self.temp_atoms.loc[self.dna_angles['a1'], ['x', 'y', 'z']].to_numpy().astype(np.float64)
            x2 = self.temp_atoms.loc[self.dna_angles['a2'], ['x', 'y', 'z']].to_numpy().astype(np.float64)
            x3 = self.temp_atoms.loc[self.dna_angles['a3'], ['x', 'y', 'z']].to_numpy().astype(np.float64)
            v1 = (x1 - x2)/np.linalg.norm(x1 - x2, axis=1, keepdims=True)
            v2 = (x3 - x2)/np.linalg.norm(x3 - x2, axis=1, keepdims=True)
            self.dna_angles['theta0'] = np.arccos(np.sum(v1*v2, axis=1))
        
        # set stackings
        stacking_data = []
        for i, ftype in stacking_types.iterrows():
            ai = ftype['i']
            aj = ftype['j']
            ak = ftype['k']
            s1 = ftype['s1']
            s2 = ftype['s2']
            for _j, row in unique_cr.iterrows():
                c, r = row['chainID'], row['resSeq']
                k1 = (c, r, ai)
                k2 = (c, r + s1, aj)
                k3 = (c, r + s2, ak)
                if (k1 in atoms.index) and (k2 in atoms.index) and (k3 in atoms.index):
                    a1 = int(atoms.loc[k1, 'index'])
                    a2 = int(atoms.loc[k2, 'index'])
                    a3 = int(atoms.loc[k3, 'index'])
                    stacking_data += [[i, a1, a2, a3]]
        stacking_data = pd.DataFrame(stacking_data, columns=['name', 'a1', 'a2', 'a3'])
        self.dna_stackings = pd.merge(stacking_data, stacking_types, left_on='name', right_index=True)
        
        # set dihedrals
        dihedral_data = []
        for i, ftype in dihedral_types.iterrows():
            ai = ftype['i']
            aj = ftype['j']
            ak = ftype['k']
            al = ftype['l']
            s1 = ftype['s1']
            s2 = ftype['s2']
            s3 = ftype['s3']
            for _j, row in unique_cr.iterrows():
                c, r = row['chainID'], row['resSeq']
                k1 = (c, r, ai)
                k2 = (c, r + s1, aj)
                k3 = (c, r + s2, ak)
                k4 = (c, r + s3, al)
                if (k1 in atoms.index) and (k2 in atoms.index) and (k3 in atoms.index) and (k4 in atoms.index):
                    a1 = int(atoms.loc[k1, 'index'])
                    a2 = int(atoms.loc[k2, 'index'])
                    a3 = int(atoms.loc[k3, 'index'])
                    a4 = int(atoms.loc[k4, 'index'])
                    dihedral_data += [[i, a1, a2, a3, a4]]
        dihedral_data = pd.DataFrame(dihedral_data, columns=['name', 'a1', 'a2', 'a3', 'a4'])
        self.dna_dihedrals = pd.merge(dihedral_data, dihedral_types, left_on='name', right_index=True)
        
        if self.dna_type == 'B_curved':
            # native dihedral value theta is read from the template
            # be careful with how the dihedral is computed
            # parameter t0 is -1*theta - np.pi (unit is rad)
            x1 = self.temp_atoms.loc[self.dna_dihedrals['a1'], ['x', 'y', 'z']].to_numpy().astype(np.float64)
            x2 = self.temp_atoms.loc[self.dna_dihedrals['a2'], ['x', 'y', 'z']].to_numpy().astype(np.float64)
            x3 = self.temp_atoms.loc[self.dna_dihedrals['a3'], ['x', 'y', 'z']].to_numpy().astype(np.float64)
            x4 = self.temp_atoms.loc[self.dna_dihedrals['a4'], ['x', 'y', 'z']].to_numpy().astype(np.float64)
            v1 = x2 - x1
            v2 = x3 - x2
            v3 = x4 - x3
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v3)
            n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
            n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
            v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
            m1 = np.cross(n1, v2) # n1 is orthogonal to v2, n1 and v2 are normalized, so m1 is normalized
            theta = np.arctan2(np.sum(m1*n2, axis=1), np.sum(n1*n2, axis=1)) # dihedral
            self.dna_dihedrals['theta0'] = -1*theta
    
    @staticmethod
    def aa_to_cg(aa_atoms, PSB_order=True):
        """
        Convert DNA all-atom structure to CG structure. 
        Both input and output structures are saved as pandas dataframe. 
        Notably, this function gives each residue in the output CG model a unique resSeq (index starts from 0). 
        Using unique resSeq can be helpful when comparing with x3dna built template as chainID may be different. 
        This code is long and specific to 3SPN2 model, so we put it here instead of in helper_functions.py. 
        
        Parameters
        ----------
        aa_atoms : pd.DataFrame
            Input all-atom structure. 
        
        PSB_order : bool
            Whether to ensure CG atom order in each nucleotide is P-S-B. 
        
        Returns
        -------
        cg_atoms : pd.DataFrame
            Output CG structure. 
        
        """
        columns = ['recname', 'serial', 'name', 'altLoc',
                   'resname', 'chainID', 'resSeq', 'iCode',
                   'x', 'y', 'z', 'occupancy', 'tempFactor',
                   'element', 'charge', 'type']
        mol = aa_atoms.copy()
        mol = mol[mol['resname'].isin(_dna_nucleotides)] # select DNA residues
        mol['group'] = mol['name'].replace(_CG_map) # assign each atom to phosphate, sugar, or base
        mol = mol[mol['group'].isin(['P', 'S', 'B'])].copy() # select DNA atoms based on group
        
        # Move the O3' to the next residue
        # also remove the O3' atom at the final residue of the chain
        for c in mol['chainID'].unique():
            sel = mol.loc[(mol['name'] == "O3\'") & (mol['chainID'] == c), "resSeq"]
            mol.loc[(mol['name'] == "O3\'") & (mol['chainID'] == c), "resSeq"] = list(sel)[1:] + [-1]
            sel = mol.loc[(mol['name'] == "O3\'") & (mol['chainID'] == c), "resname"]
            mol.loc[(mol['name'] == "O3\'") & (mol['chainID'] == c), "resname"] = list(sel)[1:] + ["remove"]
        mol = mol[mol['resname'] != 'remove'].copy()

        # Calculate center of mass
        # perform multiplication with numpy array
        # view atom mass as weights for computing mean
        mol['element'] = mol['element'].str.strip() # remove white spaces on both ends
        mol['mass'] = mol.element.replace(_atom_masses).astype(float)
        coord = mol[['x', 'y', 'z']].to_numpy()
        weight = mol['mass'].to_numpy()
        mol[['x', 'y', 'z']] = (coord.T*weight).T
        mol = mol[mol['element'] != 'H'].copy()  # Exclude hydrogens
        cg_atoms = mol.groupby(['chainID', 'resSeq', 'resname', 'group']).sum(numeric_only=True).reset_index()
        coord = cg_atoms[['x', 'y', 'z']].to_numpy()
        weight = cg_atoms['mass'].to_numpy()
        cg_atoms[['x', 'y', 'z']] = (coord.T/weight).T
        
        # Set pdb columns
        cg_atoms.loc[:, 'recname'] = 'ATOM'
        cg_atoms['name'] = cg_atoms['group']
        cg_atoms.loc[:, ['altLoc', 'iCode', 'charge']] = ''
        cg_atoms.loc[:, ['occupancy', 'tempFactor']] = 0.0
        # Change name of base to real base
        mask = (cg_atoms['name'] == 'B')
        cg_atoms.loc[mask, 'name'] = cg_atoms[mask].resname.str[-1] # takes last letter from the residue name
        cg_atoms['type'] = cg_atoms['name']
        # Set element (depends on base)
        cg_atoms['element'] = cg_atoms['name'].replace(_cg_atom_name_to_element_map)
        # Remove P from the beggining
        drop_list = []
        for c in cg_atoms.chainID.unique():
            sel = cg_atoms[cg_atoms.chainID == c]
            drop_list += list(sel[(sel.resSeq == sel.resSeq.min()) & sel['name'].isin(['P'])].index)
        cg_atoms = cg_atoms.drop(drop_list)
        unique_cr = cg_atoms[['chainID', 'resSeq']].drop_duplicates(ignore_index=True).copy()
        cg_atoms = cg_atoms.set_index(['chainID', 'resSeq', 'group'])
        if PSB_order:
            # rearrange CG atom order to P-S-B in each nucleotide
            new_index = []
            for _i, row in unique_cr.iterrows():
                c, r = row['chainID'], row['resSeq']
                j = (c, r, 'P')
                if j in cg_atoms.index:
                    new_index.append(j)
                new_index += [(c, r, 'S'), (c, r, 'B')]
            assert len(new_index) == len(cg_atoms.index)
            cg_atoms = cg_atoms.reindex(new_index, copy=True)
        
        # give each residue a unique resSeq
        # this is useful as when we compare original structure with template built by x3dna, chainID may be different
        new_r = 0
        for _i, row in unique_cr.iterrows():
            c, r = row['chainID'], row['resSeq']
            cg_atoms.loc[pd.IndexSlice[c, r, :], 'new_resSeq'] = new_r
            new_r += 1
        cg_atoms = cg_atoms.reset_index()
        cg_atoms['resSeq'] = cg_atoms['new_resSeq']
        cg_atoms['serial'] = list(range(len(cg_atoms.index)))
        cg_atoms = cg_atoms[columns].copy()
        return cg_atoms
    
    @staticmethod
    def get_sequence_list_from_cg_atoms(cg_atoms):
        """
        Get all ssDNA sequence from CG atoms as a list. 
        Since we use `groupby` to group by chainID, each chain has to possess a unique chainID. 
        
        Parameters
        ----------
        cg_atoms : pd.DataFrame
            CG DNA atoms. 
        
        Returns
        -------
        sequence_list : list
            A list including multiple strings. Each string is the sequence of one chain. 
        
        """
        cg_atoms = cg_atoms[cg_atoms['resname'].isin(_dna_nucleotides)].copy()
        sequence_list = []
        for c, chain in cg_atoms.groupby('chainID'):
            sequence_c = ''
            for i, r in chain.groupby('resSeq'):
                sequence_c += r.iloc[0]['resname'][1]
            sequence_list.append(sequence_c)
        return sequence_list

    def get_sequence_list(self):
        """
        Get all ssDNA sequence from self.atoms as a list. 
        Note each chain has to possess a unique chainID. 
        
        Returns
        -------
        sequence_list : list
            A list including multiple strings. Each string is the sequence of one chain. 
        
        """
        sequence_list = self.get_sequence_list_from_cg_atoms(self.atoms)
        return sequence_list
    
    def get_sequence(self):
        """
        Get all ssDNA sequence from CG atoms as a string. 
        Note each chain has to possess a unique chainID. 
        
        Returns
        -------
        sequence : str
            DNA sequence. 
            
        """
        sequence_list = self.get_sequence_list()
        sequence = ''.join(sequence_list)
        return sequence
    
    @staticmethod
    def change_sequence(cg_atoms, sequence):
        """
        Change DNA sequence. 
        Note sequence includes the sequence of all the ssDNA chains. 
        
        Parameters
        ----------
        cg_atoms : pd.DataFrame
            DNA CG structure. 
        
        sequence : str
            New DNA sequence. 
        
        Returns
        -------
        new_cg_atoms : pd.DataFrame
            DNA CG structure with the new sequence. 
        
        """
        new_cg_atoms = cg_atoms[cg_atoms['resname'].isin(_dna_nucleotides)].copy()
        # ensure the new sequence has correct length
        assert len(new_cg_atoms[new_cg_atoms['name'].isin(['A', 'T', 'C', 'G'])]) == len(sequence)
        new_cg_atoms.loc[new_cg_atoms['name'].isin(['A', 'T', 'C', 'G']), 'name'] = [x for x in sequence]
        new_cg_atoms['element'] = new_cg_atoms['name'].replace(_cg_atom_name_to_element_map)
        bases = new_cg_atoms[new_cg_atoms['name'].isin(['A', 'T', 'C', 'G'])]
        for i, row in bases.iterrows():
            chainID = row['chainID']
            resSeq = row['resSeq']
            flag1 = (new_cg_atoms['chainID'] == chainID)
            flag2 = (new_cg_atoms['resSeq'] == resSeq)
            new_cg_atoms.loc[flag1 & flag2, 'resname'] = 'D' + row['name']
        return new_cg_atoms
    
    @classmethod
    def from_atomistic_pdb(cls, atomistic_pdb, cg_pdb, PSB_order=True, new_sequence=None, dna_type='B_curved', 
                           default_parse=True, temp_from_x3dna=True, temp_name='dna', input_temp=None):
        """
        Create object from atomistic pdb file. 
        Ensure each chain in input pdb_file has unique chainID. 
        If a new sequence is provided, then the CG DNA topology is from input pdb file, but using the new sequence. 
        
        Parameters
        ----------
        atomistic_pdb : str
            Path for the input atomistic pdb file. 
        
        cg_pdb : str
            Path for the output CG pdb file. 
        
        PSB_order : bool
            Whether to ensure CG atom order in each nucleotide is P-S-B.
        
        new_sequence : None or str
            The new full DNA sequence. 
            If None, keep the original sequence. 
        
        dna_type : str
            DNA type.  
        
        default_parse : bool
            Whether to parse molecule with default settings. 
        
        temp_from_x3dna : bool
            Whether to use template built by x3dna. 
            If False, use the input pdb file as the template. 
        
        temp_name : str
            Template file names.
        
        input_temp : None or pd.DataFrame
            Additional input template.
        
        """
        try:
            atomistic_atoms = fix_pdb(atomistic_pdb) # fix pdb
        except Exception as e:
            print('Do not fix pdb file.')
        cg_atoms = cls.aa_to_cg(atomistic_atoms, PSB_order=PSB_order) # do coarse-graining
        if new_sequence is not None:
            print(f'Change to new sequence: {new_sequence}')
            cg_atoms = cls.change_sequence(cg_atoms, new_sequence)
        write_pdb(cg_atoms, cg_pdb)
        self = cls(dna_type)
        # directly set self.atoms from cg_atoms instead of loading cg_pdb
        # numerical accuracy is decreased when saving coordinates to pdb
        self.atoms = cg_atoms
        if default_parse:
            self.parse_config_file()
            self.parse_mol(temp_from_x3dna=temp_from_x3dna, temp_name=temp_name, input_temp=input_temp)
        return self
        
        

