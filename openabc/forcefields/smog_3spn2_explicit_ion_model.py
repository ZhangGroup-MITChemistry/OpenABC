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
    def add_all_vdwl(self, param_PP_MJ_path=f'{__location__}/parameters/pp_MJ.csv', force_group=11):
        """
        Add all the nonbonded Van der Waals interactions. 
        
        CG atom type 0-19 for amino acids. CG atom type 20-25 for DNA atoms. CG atom type 26-28 for Na, Mg, and Cl.
        
        Parameters
        ----------
        """
        print('Add all the nonbonded contact interactions.')
        param_PP_MJ = pd.read_csv(param_PP_MJ_path)
        
    
    def add_all_elec(self, force_group=12):
        """
        Add all the electrostatic interactions as the Coulombic interactions.
        """
        # to be continued
        return None
        
    
