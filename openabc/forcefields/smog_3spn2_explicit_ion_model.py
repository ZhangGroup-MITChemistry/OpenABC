import numpy as np
import pandas as pd
try:
    import openmm.unit as unit
except ImportError:
    import simtk.unit as unit
from openabc.forcefields.cg_model import CGModel
from openabc.forcefields import SMOG3SPN2Model
from openabc.forcefields import functional_terms
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

class SMOG3SPN2ExplicitIonModel(SMOG3SPN2Model):
    """
    The class for SMOG+3SPN2+explicit ion model.
    The class inherits from class SMOG3SPN2Model.
    Importantly, since this is explicit ion model, electrostatic interactions are Coulombic.
    """
    def append_ions(self, ion_type, n_ions, coordinates=None):
        """
        Append ions to the system by appending ion information at the end of self.atoms.
        
        Parameters
        ----------
        ion_type : str
            The type of ion. It should be 'NA' or 'CL' or 'MG'.
        
        n_ions : int
            The number of ions to be appended.
        
        coordinates : None or array-like
            The coordinates of the ions in unit Angstrom, following pdb format. 
            If None, the coordinates will be randomly generated. 
            Note normally the coordinates in self.atoms are not used as simulation initial coordinates, so we can set them as random numbers.
        
        Returns
        -------
        self.atoms : pd.DataFrame
            The updated self.atoms.
        
        """
        assert ion_type in ['NA', 'CL', 'MG']
        df_ions = pd.DataFrame({'serial': np.arange(n_ions) + 1})
        df_ions.loc[:, 'recname'] = 'ATOM'
        df_ions.loc[:, ['name', 'resname', 'element']] = ion_type
        df_ions.loc[:, ['altloc', 'iCode']] = ''
        df_ions.loc[:, 'chainID'] = 'X'
        df_ions.loc[:, 'resSeq'] = np.arange(n_ions) + 1
        if coordinates is None:
            coordinates = np.random.normal(size=(n_ions, 3))
        df_ions.loc[:, ['x', 'y', 'z']] = coordinates
        df_ions.loc[:, 'occupancy'] = 0.0
        df_ions.loc[:, 'tempFactor'] = 0.0
        df_ions.loc[:, 'charge'] = ''
        self.atoms = pd.concat([self.atoms, df_ions], ignore_index=True)
        return self.atoms
    
    def add_all_vdwl(self, param_PP_MJ_path=f'{__location__}/parameters/pp_MJ.csv', force_group=11):
        """
        Add all the nonbonded Van der Waals interactions. 
        
        CG atom type 0-19 for amino acids. CG atom type 20-25 for DNA atoms. CG atom type 26-28 for Na, Mg, and Cl.
        
        Parameters
        ----------
        """
        print('Add all the nonbonded contact interactions.')
        param_PP_MJ = pd.read_csv(param_PP_MJ_path)
        # to be continued
        
    
    def add_all_elec(self, force_group=12):
        """
        Add all the electrostatic interactions as the Coulombic interactions.
        """
        # to be continued
        return None
        
    
