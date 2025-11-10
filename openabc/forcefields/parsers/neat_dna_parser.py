import os
import numpy as np
import pandas as pd
from openabc.utils.helper_functions import parse_pdb,write_pdb
from openabc.utils.openabc_mmCIF_parser import parse_mmCIF, write_mmCIF
from openabc.lib import _dna_nucleotides,_rna_nucleotides
import warnings
_terminal_dna_nucleotides = [dnt+'5' for dnt in _dna_nucleotides] + [dnt+'3' for dnt in _dna_nucleotides]
_og_dna_nuc = _dna_nucleotides
_dna_nucleotides = _dna_nucleotides + _terminal_dna_nucleotides
_nucleotides = _dna_nucleotides + _rna_nucleotides

__location__ = os.path.dirname(os.path.abspath(__file__))

class NEATDNAParser(object):
    def __init__(self, cg_pdb, atoms,default_parse=True):
        '''
            Calling NEATDNAParser default constructor is not recommended!
            Please call NEATDNAParser.from_pdb() or NEATDNAParser.from_mmCIF() instead.
            If loading from atomistic pdb or mmCIF file, call NEATDNAParser.from_atomistic_pdb() or NEATDNAParser.from_atomistic_mmCIF()"
        '''
        self.pdb = cg_pdb
        self.atoms = atoms
        if default_parse:
            self.parse_mol()

    @classmethod
    def from_mmCIF(cls, cg_mmCIF_file, default_parse=True):
        """
        Initialize a NEAT-DNA model from coarse-grain mmCIF. 
        
        Parameters
        ----------
        cg_mmCIF_file : str
            Path for the coarse-grain dsDNA mmCIF file. 
        
        default_parse : bool
            Whether to parse with default settings. 
        
        Returns
        ------- 
        result : class instance
            A class instance. 
        
        """
        atoms = parse_mmCIF(cg_mmCIF_file)
        cls.validate_cg_atoms(atoms)
        return cls(cg_mmCIF_file,atoms,default_parse)

    @classmethod
    def from_atomistic_mmCIF(cls, aa_mmCIF_file, cg_out_file, default_parse=True, output_fmt='mmCIF'):
        """
        Initialize a NEAT-DNA model from an atomistic mmCIF. 
        
        Parameters
        ----------
        cg_mmCIF_file : str
            Path for the atomistic dsDNA mmCIF file. 

        cg_out_file : str
            Path to write coarse-grain dsDNA file. Can be either PDB or mmCIF as specificed by output_fmt
        
        default_parse : bool
            Whether to parse with default settings. 

        output_fmt : str
            Output format of CG structure. Options 'PDB' or 'mmCIF'
        
        Returns
        ------- 
        result : class instance
            A class instance. 
        
        """
        aa_atoms = parse_mmCIF(aa_mmCIF_file)
        atoms = cls.aa_to_cg(aa_atoms)
        if output_fmt == 'mmCIF':
            write_mmCIF(atoms,cg_out_file)
        elif output_fmt == 'PDB':
            write_pdb(atoms, cg_out_file)
        cls.validate_cg_atoms(atoms)
        return cls(cg_out_file,atoms,default_parse)

    @classmethod
    def from_pdb(cls, cg_pdb, default_parse=True):
        """
        Initialize a NEAT-DNA model from a coarse-grain PDB. 
        
        Parameters
        ----------
        cg_pdb : str
            Path for the coarse-grain dsDNA PDB file. 

        default_parse : bool
            Whether to parse with default settings. 
        
        Returns
        ------- 
        result : class instance
            A class instance. 
        
        """
        atoms = parse_pdb(cg_pdb)
        cls.validate_cg_atoms(atoms)
        return cls(cg_pdb,atoms,default_parse)

    @classmethod
    def from_atomistic_pdb(cls, aa_pdb, cg_pdb, write_TER=False, default_parse=True):
        """
        Initialize a NEAT-DNA model from an atomistic PDB. 
        
        Parameters
        ----------
        aa_pdb : str
            Path for the atomistic dsDNA PDB file. 

        cg_pdb : str
            Path to write coarse-grain dsDNA PDB file. 

        write_TER : bool
            Whether to write TER between two chains. 

        default_parse : bool
            Whether to parse with default settings. 
        
        Returns
        ------- 
        result : class instance
            A class instance. 
        
        """
        aa_atoms = parse_pdb(aa_pdb)
        atoms = cls.aa_to_cg(aa_atoms)
        write_pdb(atoms,cg_pdb,write_TER)
        cls.validate_cg_atoms(atoms)
        return cls(cg_pdb,atoms,default_parse)

    @staticmethod
    def validate_cg_atoms(atoms: pd.DataFrame):
        # check if all the atoms are CG nucleotide atoms
        atom_names = atoms['name']
        assert (atoms['resname'].isin(_dna_nucleotides).all() and atom_names.eq('DN').all())
        # check if there are only 2 chains
        unique_chainID = list(set(atoms['chainID'].tolist()))
        assert len(unique_chainID) == 2
        # check if the first half is the first ssDNA, and the second half is the second ssDNA
        n_atoms = len(atoms.index)
        n_bp = int(n_atoms/2)
        first_half_atoms = atoms.loc[0:n_bp - 1]
        second_half_atoms = atoms.loc[n_bp:n_atoms - 1]
        assert len(first_half_atoms.index) == len(second_half_atoms.index)
        assert first_half_atoms['chainID'].eq(first_half_atoms['chainID'].iloc[0]).all()
        assert second_half_atoms['chainID'].eq(second_half_atoms['chainID'].iloc[0]).all()

    @staticmethod 
    def aa_to_cg(atomistic_pdb_atoms: pd.DataFrame):
        '''
        Convert DNA all-atom structure to CG structure. 
        Both input and output structures are saved as pandas dataframe. 
        This is functionally the same as openabc.utils.helper_functions.atomistic_pdb_to_nucleotide_pdb(...),
        but written in a much more vectorized fashion, allowing for efficient processing of very large structures.
        
        Parameters
        ----------
        atomistic_pdb_atoms: pd.DataFrame
            Input all-atom structure. 
        
        Returns
        -------
        cg_nucleotide_pdb_atoms: pd.DataFrame
            Output CG structure. 
        
        '''
        atomistic_pdb_atoms = atomistic_pdb_atoms.loc[atomistic_pdb_atoms['resname'].isin(_nucleotides)].copy()
        atomistic_pdb_atoms.index = list(range(len(atomistic_pdb_atoms.index)))
        cg_nucleotide_pdb_atoms = atomistic_pdb_atoms.groupby(['chainID','resSeq'],sort=False).agg({
            'recname': 'first',
            'resname': 'first',
            'altLoc': 'first',
            'iCode': 'first',
            'x': 'mean',
            'y': 'mean', 
            'z': 'mean' }).reset_index()
        N = len(cg_nucleotide_pdb_atoms)
        cg_nucleotide_pdb_atoms['name'] = cg_nucleotide_pdb_atoms['resname'].apply(lambda name: 'RN' if name in _rna_nucleotides else 'DN')
        def correct_term_bp(resname):
            if resname not in _og_dna_nuc:
                for dnt in _og_dna_nuc:
                    if resname == dnt+'5' or resname == dnt+'3':
                        return dnt
            else:
                return resname
        cg_nucleotide_pdb_atoms['resname'] = cg_nucleotide_pdb_atoms['resname'].apply(correct_term_bp)
        cg_nucleotide_pdb_atoms['occupancy'] = np.ones(N)
        cg_nucleotide_pdb_atoms['tempFactor'] = np.ones(N)
        cg_nucleotide_pdb_atoms['element'] = ['P'] * N
        cg_nucleotide_pdb_atoms['charge'] = [''] * N
        cg_nucleotide_pdb_atoms['serial'] = np.arange(1,N+1)
        return cg_nucleotide_pdb_atoms


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
            for _, row in self.dna_bonds.iterrows():
                exclusions.append((int(row['a1']), int(row['a2'])))
        if exclude13 and hasattr(self, 'dna_angles'):
            for _, row in self.dna_angles.iterrows():
                exclusions.append((int(row['a1']), int(row['a3'])))
        exclusions = np.array(sorted(exclusions))
        self.exclusions = pd.DataFrame(exclusions, columns=['a1', 'a2']).drop_duplicates(ignore_index=True)

    def parse_mol(self, paramdf_file=f"{__location__}/../parameters/NEAT_DNA_parameters.csv", exclude12=True, exclude13=True, bonded_energy_scale=1.0, charge=-1.0, uhb_scalar=1.0, bp_ang_scalar=1.0):
            n_atoms = len(self.atoms.index)
            n_bp = int(n_atoms/2)
            idx_max = int(n_atoms) - 1

            map = {'DA':'A', 'DC':'C','DG':'G','DT':'T'}
            mass_map = {'DA': 331.2, 'DC': 307.2,'DG': 347.2,'DT': 322.2}
            mass = [mass_map[r] for r in self.atoms.resname.tolist()]

            sequence = ''.join([map[r] for r in self.atoms.resname.tolist()][:n_atoms//2])       
            sequence = 'G'+sequence+'C'

            trimers = []
            for j in range(2, len(sequence)):
                trimers.append(''.join([sequence[j-2],sequence[j-1],sequence[j]]))
            trimers = np.array(trimers)

            paramdf = pd.read_csv(paramdf_file,header=0,index_col=0)

            ''' SS Bonds '''
            bond_params5 = paramdf.loc[trimers[:-1]][[f'ssbond5.k{i}' for i in range(2,5)] + ['ssbond5.x0']].to_numpy()
            bond_params3 = paramdf.loc[trimers[:-1]][[f'ssbond3.k{i}' for i in range(2,5)] + ['ssbond3.x0']].to_numpy()
            atom5_1 = np.arange(0, n_bp-1)
            atom5_2 = atom5_1 + 1
            atom3_1 = idx_max - atom5_1
            atom3_2 = atom3_1 - 1

            bonds5 = np.stack((atom5_1,atom5_2), axis=-1)
            bonds3 = np.stack((atom3_1,atom3_2), axis=-1)
            bonds = np.concatenate((bonds5,bonds3),axis=0)
            params = np.concatenate((bond_params5,bond_params3),axis=0)
            bonds = np.append(bonds, params, axis=-1)
            self.dna_bonds = pd.DataFrame(bonds, columns=['a1', 'a2', 'k2', 'k3', 'k4', 'r0'])

            ''' SS Angles '''
            angle5_params = paramdf.loc[trimers[1:-1]][[f'ssangle5.k{i}' for i in range(2,5)] + ['ssangle5.x0']].to_numpy()
            angle3_params = paramdf.loc[trimers[1:-1]][[f'ssangle3.k{i}' for i in range(2,5)] + ['ssangle3.x0']].to_numpy()
            atom5_1 = np.arange(0, n_bp - 2)
            atom5_2 = atom5_1 + 1
            atom5_3 = atom5_1 + 2

            atom3_1 = idx_max - atom5_1
            atom3_2 = atom3_1 - 1
            atom3_3 = atom3_1 - 2

            angles5 = np.stack((atom5_1,atom5_2,atom5_3), axis=-1)
            angles3 = np.stack((atom3_1,atom3_2,atom3_3), axis=-1)
            angles = np.concatenate((angles5,angles3),axis=0)
            params = np.concatenate((angle5_params,angle3_params),axis=0)
            angles = np.append(angles, params, axis=-1)
            self.dna_angles = pd.DataFrame(angles, columns=['a1', 'a2', 'a3', 'k2', 'k3', 'k4', 'theta0'])

            ''' Base-Pairing Bonds - Distance '''
            bp_params = paramdf.loc[trimers][[f'cs_bond.k{i}' for i in range(2,5)] + ['cs_bond.x0']].to_numpy()
            lb = 0
            ub = n_bp
            atom1 = np.array(range(lb,ub))
            atom2 = idx_max - atom1 
            f_bonds = np.stack((atom1,atom2), axis=-1)
            f_bonds = np.append(f_bonds, bp_params, axis=-1)
            self.dna_bp_distance_bonds = pd.DataFrame(f_bonds, columns=['a1', 'a2', 'k2', 'k3', 'k4', 'r0'])

            ''' Base-Pairing Bonds - Angle '''
            bp_anglep_params = paramdf.loc[trimers[:-1]][['cs_anglep.k', 'cs_anglep.x0']].to_numpy()
            bp_anglep_params = np.concatenate((bp_anglep_params, paramdf.loc[trimers[1:]][['cs_anglep.k', 'cs_anglep.x0']].to_numpy()),axis=0)
            bp_anglem_params = paramdf.loc[trimers[1:]][['cs_anglep.k', 'cs_anglem.x0']].to_numpy() #I am intentionally using anglep.k
            bp_anglem_params = np.concatenate((bp_anglem_params, paramdf.loc[trimers[:-1]][['cs_anglep.k', 'cs_anglem.x0']].to_numpy()),axis=0)
            #First add "plus" angles
            a5_2 = np.arange(0,n_bp-1)
            a5_1 = idx_max - a5_2
            a5_3 = a5_2 + 1

            a3_2 = np.flip(np.arange(n_bp,idx_max))
            a3_1 = idx_max - a3_2
            a3_3 = a3_2 + 1

            f_bonds_p = np.concatenate((np.stack((a5_1,a5_2,a5_3), axis=-1),np.stack((a3_1,a3_2,a3_3), axis=-1)),axis=0)
            f_bonds_p = np.append(f_bonds_p, bp_anglep_params, axis=-1)
            #f_bonds = np.append(f_bonds_p, bp_anglep_params, axis=-1)
            #f_bonds = np.append(np.stack((a5_1,a5_2,a5_3),axis=-1), bp_anglep_params, axis=-1)

            #Now "minus" angles
            a2_5 = np.arange(1,n_bp)
            a1_5 = idx_max - a2_5
            a3_5 = a2_5 - 1

            a2_3 = np.flip(np.arange(n_bp+1,idx_max+1))
            a1_3 = idx_max - a2_3
            a3_3 = a2_3 - 1

            f_bonds_m = np.concatenate((np.stack((a1_5,a2_5,a3_5), axis=-1),np.stack((a1_3,a2_3,a3_3), axis=-1)),axis=0)
            f_bonds_m = np.append(f_bonds_m, bp_anglem_params, axis=-1)

            f_bonds = np.concatenate((f_bonds_p,f_bonds_m),axis=0)
            self.dna_bp_angles = pd.DataFrame(f_bonds, columns=['a1', 'a2', 'a3', 'kt', 'theta0'])

            ''' Base-Pairing Bonds - Dihedrals '''
            bp_dihed_params_p = paramdf.loc[trimers[1:-1]][['cs_dihedp.k', 'cs_dihedp.x0']].to_numpy()
            bp_dihed_params_m = paramdf.loc[trimers[1:-1]][['cs_dihedp.k', 'cs_dihedm.x0']].to_numpy() #I am intentionally using dihedp.k
            bp_dihed_params = np.concatenate((bp_dihed_params_p, bp_dihed_params_m),axis=0)

            p2 = np.arange(1,n_bp-1)
            p1 = p2 - 1
            p3 = p2 + 1

            p5 = idx_max - p2
            p4 = p5 - 1
            p6 = p5 + 1

            '''kp*(1. + cos(dihedral(p6, p5, p2, p3) + phi2))'''
            f_bondsp = np.stack((p6,p5,p2,p3), axis=-1)

            '''kp*(1. + cos(dihedral(p4, p5, p2, p1) + phi1))'''
            f_bondsm = np.stack((p4,p5,p2,p1), axis=-1)

            f_bonds = np.concatenate((f_bondsp,f_bondsm),axis=0)
            f_bonds = np.append(f_bonds, bp_dihed_params, axis=-1)
            self.dna_bp_dihedrals = pd.DataFrame(f_bonds, columns=['a1', 'a2', 'a3', 'a4', 'kp', 'phi0'])

            ''' Fan Bonds '''
            f_bonds = np.empty((1,15))
            hb_idx = np.append(np.arange(-5 ,0, dtype=int), np.arange(1 ,6, dtype=int))
            for i,fb_i in enumerate(hb_idx):
                lb = max(1,fb_i + 1)
                ub = min(n_bp - 1, n_bp - 1 + fb_i)
                p2 = np.arange(lb,ub)
                p1 = p2 - 1
                p3 = p2 + 1

                p5 = idx_max - p2 + fb_i
                p4 = p5 - 1
                p6 = p5 + 1

                fb = np.stack((p1,p2,p3,p4,p5,p6), axis=-1)
                fb = np.append(fb, np.tile(paramdf.loc[f'cs.{fb_i}'][['ug','cs_bondg.k','cs_anglep.k','cs_dihedp.k','cs_bondg.x0','cs_anglem.x0','cs_anglep.x0','cs_dihedm.x0','cs_dihedp.x0']].to_numpy(), (fb.shape[0],1)), axis=-1)
                f_bonds = np.append(f_bonds,fb, axis=0)
            f_bonds = f_bonds[1:] #discard "empty" row
            self.dna_fan_bonds = pd.DataFrame(f_bonds, columns=[['a1','a2','a3','a4','a5','a6','uhb', 'kr', 'kt', 'kp', 'r0', 'theta1', 'theta2', 'phi1', 'phi2']])

            # scale bonded interactions
            self.dna_bonds[['k2', 'k3', 'k4']] *= bonded_energy_scale
            self.dna_angles[['k2', 'k3', 'k4']] *= bonded_energy_scale
            self.dna_bp_distance_bonds[['k2', 'k3', 'k4']] *= bonded_energy_scale
            self.dna_fan_bonds['uhb'] *= uhb_scalar
            self.dna_bp_angles['kt'] *= bp_ang_scalar
            self.dna_bp_dihedrals['kp'] *= bp_ang_scalar
        
            # set exclusions
            self.parse_exclusions(exclude12, exclude13)
            # set mass and charge
            self.atoms.loc[:, 'mass'] = mass
            self.atoms.loc[:, 'charge'] = charge 
