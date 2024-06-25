import numpy as np
import pandas as pd
try:
    import openmm.unit as unit
except ImportError:
    import simtk.unit as unit
from openabc.forcefields.cg_model import CGModel
from openabc.forcefields import SMOG3SPN2Model
from openabc.forcefields import functional_terms
from openabc.lib import df_sodium_ion, df_magnesium_ion, df_chloride_ion
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

class SMOG3SPN2ExplicitIonModel(SMOG3SPN2Model):
    """
    The class for SMOG+3SPN2+explicit ion model.
    The class inherits from class SMOG3SPN2Model.
    Importantly, since this is explicit ion model, electrostatic interactions are unscreened Coulombic potential.
    """
    def append_ions(self, ion_type, n_ions, coord=None):
        """
        Append ions to the system by appending ion information at the end of self.atoms.
        
        Parameters
        ----------
        ion_type : str
            The type of ion. It should be 'NA' or 'CL' or 'MG'.
        
        n_ions : int
            The number of ions to be appended.
        
        coord : None or array-like
            The coordinates of the ions in unit nm. 
            If None, the coordinates will be randomly generated. 
            The coordinates will be updated for ions in self.atoms. 
            Note normally the coordinates in self.atoms are not used as simulation initial coordinates, so we can set them as random numbers.
        
        Returns
        -------
        self.atoms : pd.DataFrame
            The updated self.atoms.
        
        """
        assert ion_type in ['NA', 'MG', 'CL']
        df_ion_dict = {'NA': df_sodium_ion, 'MG': df_magnesium_ion, 'CL': df_chloride_ion}
        df_ions = pd.DataFrame(columns=df_ion_dict[ion_type].columns)
        row = df_ion_dict[ion_type].loc[0].copy()
        for i in range(n_ions):
            df_ions.loc[i] = row.copy()
        if coord is None:
            coord = np.random.normal(size=(n_ions, 3))
        df_ions.loc[:, ['x', 'y', 'z']] = coord
        self.atoms = pd.concat([self.atoms, df_ions], ignore_index=True)
        return self.atoms
    
    
    def add_all_vdwl(self, param_PP_MJ_path=f'{__location__}/parameters/pp_MJ.csv', force_group=11):
        """
        Add all the nonbonded Van der Waals interactions. 
        
        CG atom type 0-19 for amino acids. CG atom type 20-25 for DNA atoms. CG atom type 26-28 for Na, Mg, and Cl.
        
        Parameters
        ----------
        param_PP_MJ_path : str
            The path of the MJ parameter file. 
        
        force_group : int
            Force group. 
        
        """
        print('Add all the nonbonded contact interactions.')
        param_PP_MJ = pd.read_csv(param_PP_MJ_path)
        force = functional_terms.all_smog_MJ_3spn2_explicit_ion_hydr_vdwl_term(self, param_PP_MJ, force_group=force_group)
        self.system.addForce(force)
        
    
    def add_all_elec(self, force_group=12):
        """
        Add all the electrostatic interactions as the Coulombic interactions.
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        print('Add all the electrostatic interactions.')
        force = functional_terms.all_smog_MJ_3spn2_explicit_ion_elec_term(self, force_group=force_group)
        self.system.addForce(force)
        
    
