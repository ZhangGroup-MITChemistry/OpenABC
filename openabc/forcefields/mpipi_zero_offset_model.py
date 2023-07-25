import numpy as np
import pandas as pd
try:
    import openmm.unit as unit
except ImportError:
    import simtk.unit as unit
from openabc.forcefields.mpipi_model import MpipiModel
from openabc.forcefields.functional_terms.zero_offset_nonbonded_terms import dh_elec_zero_offset_term
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

class MpipiZeroOffsetModel(MpipiModel):
    """
    The Mpipi model with zero offset for the electrostatic interactions. 
    This model is only used for comparisons, as LAMMPS pair_style coul/debye does not shift the electrostatic interaction to zero at cutoff. 
    """
    def add_dh_elec(self, ldby=(1/1.26)*unit.nanometer, dielectric_water=80.0, cutoff=3.5*unit.nanometer, 
                    force_group=4):
        """
        Add Debye-Huckel electrostatic interactions. 
        
        Parameters
        ----------
        ldby : Quantity
            Debye length. 
        
        dielectric_water : float or int
            Dielectric constant of water. 
        
        cutoff : Quantity
            Cutoff distance. 
        
        force_group : int
            Force group. 
        
        """
        print('Add Debye-Huckel electrostatic interactions.')
        print(f'Set Debye length as {ldby.value_in_unit(unit.nanometer)} nm.')
        print(f'Set water dielectric as {dielectric_water}.')
        charges = self.atoms['charge'].tolist()
        force = dh_elec_zero_offset_term(charges, self.exclusions, self.use_pbc, ldby, dielectric_water, cutoff, 
                                         force_group)
        self.system.addForce(force)


