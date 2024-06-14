import numpy as np
import pandas as pd
import configparser
from openabc.lib import _kcal_to_kj, _angstrom_to_nm, _deg_to_rad
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))
dna_3SPN2_conf = f'{__location__}/3SPN2.conf'

class Mixin3SPN2ConfigParser(object):
    """
    Define a class for 3SPN2 configuration file parser. 
    
    Make this an individual class is convenient for other classes to inherit. 
    
    This class can load parameters from 3SPN2.conf and automatically convert the units. 
    
    Also this class loads all the parameters as pandas dataframes, and it is easy for user to modify the parameters. 
    """
    def parse_config_file(self, config_file=dna_3SPN2_conf):
        """
        Parse configuration file. The parameters are loaded as pandas dataframes. 
        The method automatically loads parameters from configuration file and convert the units. 
        The converted units are consistent with the workflow. 
        """
        
        def parse_row(row):
            """
            Parse one row from the configuration. 
            """
            values = row.split('#')[0].split() # remove comments
            for i in range(len(values)):
                values[i] = values[i].strip()
                try: 
                    x = int(values[i])
                except ValueError:
                    try:
                        x = float(values[i])
                    except ValueError:
                        x = values[i]
                values[i] = x
            return values
        
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        config_3spn = {}
        for i in config.sections():
            data = []
            for j in config[i]:
                if j == 'name':
                    columns = parse_row(config[i][j])
                elif len(j) > 3 and j[:3] == 'row':
                    data += [parse_row(config[i][j])]
            config_3spn[i] = pd.DataFrame(data, columns=columns)
        
        # the loaded parameters are saved in pandas dataframe as attributes
        self.particle_definition = config_3spn['Particles'].copy()
        self.bond_definition = config_3spn['Bonds'].copy()
        self.angle_definition = config_3spn['Harmonic Angles'].copy()
        self.dihedral_definition = config_3spn['Dihedrals'].copy()
        self.stacking_definition = config_3spn['Base Stackings'].copy()
        self.pair_definition = config_3spn['Base Pairs'].copy()
        self.cross_definition = config_3spn['Cross Stackings'].copy()
        self.protein_dna_particle_definition = config_3spn['Protein-DNA particles'].copy()
        self.base_pair_geometry = config_3spn['Base Pair Geometry'].copy()
        self.base_step_geometry = config_3spn['Base Step Geometry'].copy()
        
        # fix units and item names to follow the convention
        # units in 3SPN2.conf are not consistent, and we convert them to consistent units
        # particle definition
        self.particle_definition['epsilon'] *= _kcal_to_kj
        self.particle_definition = self.particle_definition.rename(columns={'radius': 'sigma'})
        self.particle_definition['sigma'] *= _angstrom_to_nm
        # bond definition
        self.bond_definition = self.bond_definition.rename(columns={'Kb2': 'k_bond_2', 
                                                                    'Kb3': 'k_bond_3', 
                                                                    'Kb4': 'k_bond_4'})
        flag = (self.bond_definition['DNA'] == 'B_curved')
        self.bond_definition.loc[flag, 'r0'] = np.nan # will be set based on the reference structure
        # angle definition
        self.angle_definition['k_angle'] = self.angle_definition['epsilon']*2
        self.angle_definition = self.angle_definition.rename(columns={'t0': 'theta0'})
        flag1 = (self.angle_definition['DNA'].isin(['A', 'B']))
        flag2 = (self.angle_definition['DNA'] == 'B_curved')
        self.angle_definition.loc[flag1, 'theta0'] *= _deg_to_rad
        self.angle_definition.loc[flag2, 'theta0'] = np.nan # will be set based on the reference structure
        # stacking definition
        self.stacking_definition = self.stacking_definition.rename(columns={'t0': 'theta0'})
        self.stacking_definition['theta0'] *= _deg_to_rad
        # dihedral definition
        self.dihedral_definition = self.dihedral_definition.rename(columns={'t0': 'theta0'})
        flag1 = (self.dihedral_definition['DNA'].isin(['A', 'B']))
        flag2 = (self.dihedral_definition['DNA'] == 'B_curved')
        x = self.dihedral_definition.loc[flag1, 'theta0']
        self.dihedral_definition.loc[flag1, 'theta0'] = _deg_to_rad*(x + 180)
        self.dihedral_definition.loc[flag2, 'theta0'] = np.nan # will be set based on the reference structure
        # base pair definition
        self.pair_definition['torsion'] *= _deg_to_rad
        self.pair_definition['sigma'] *= _angstrom_to_nm
        self.pair_definition[['t1', 't2']] *= _deg_to_rad
        self.pair_definition['epsilon'] *= _kcal_to_kj
        self.pair_definition['alpha'] /= _angstrom_to_nm
        # cross stacking definition
        self.cross_definition[['t03', 'T0CS_1', 'T0CS_2']] *= _deg_to_rad
        self.cross_definition[['eps_cs1', 'eps_cs2']] *= _kcal_to_kj
        self.cross_definition[['alpha_cs1', 'alpha_cs2']] /= _angstrom_to_nm
        self.cross_definition[['Sigma_1', 'Sigma_2']] *= _angstrom_to_nm
        # protein-DNA particle definition (parameters for protein-DNA nonbonded interactions)
        self.protein_dna_particle_definition['epsilon'] *= _kcal_to_kj
        self.protein_dna_particle_definition = self.protein_dna_particle_definition.rename(columns={'radius': 'sigma'})
        self.protein_dna_particle_definition[['sigma', 'cutoff']] *= _angstrom_to_nm
        


