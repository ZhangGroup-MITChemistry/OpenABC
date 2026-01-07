import numpy as np
import pandas as pd
try:
    import openmm.unit as unit
except ImportError:
    import simtk.unit as unit
from openabc.forcefields import HPSModel
from openabc.forcefields.functional_terms import ashbaugh_hatch_cutoff_term, dh_elec_term
from openabc.lib import _amino_acids, _kcal_to_kj
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

df_calvados_params = pd.read_csv(f'{__location__}/parameters/calvados_parameters.csv')
df_calvados_params = df_calvados_params.set_index('three')

class CALVADOSModel(HPSModel):
    def add_contacts(self, hydropathy_scale='CALVADOS2', epsilon=0.2 * _kcal_to_kj, 
                     cutoff=2.0 * unit.nanometer, force_group=2):
        print('Add nonbonded contacts.')
        print(f'Use {hydropathy_scale} hydropathy scale.')
        resname_list = self.atoms['resname'].tolist()
        atom_types = [_amino_acids.index(x) for x in resname_list]
        assert hydropathy_scale in ['CALVADOS1', 'CALVADOS2']
        sigma_ah = np.array([df_calvados_params.loc[x, 'sigmas'] for x in _amino_acids])
        lambda_ah = np.array([df_calvados_params.loc[x, hydropathy_scale] for x in _amino_acids])
        sigma_ah_map = 0.5 * (sigma_ah + sigma_ah[:, None])
        lambda_ah_map = 0.5 * (lambda_ah + lambda_ah[:, None])
        force = ashbaugh_hatch_cutoff_term(
            atom_types=atom_types,
            df_exclusions=self.exclusions,
            use_pbc=self.use_pbc,
            epsilon=epsilon, 
            sigma_ah_map=sigma_ah_map,
            lambda_ah_map=lambda_ah_map,
            cutoff=cutoff,
            force_group=force_group,
        )
        self.system.addForce(force)
    
    def add_dh_elec(self, ldby=1 * unit.nanometer, dielectric_water=80.0, 
                    cutoff=4.0 * unit.nanometer, force_group=3):
        print('Add Debye-Huckel electrostatic interactions.')
        print(f'Set Debye length as {ldby.value_in_unit(unit.nanometer)} nm.')
        print(f'Set water dielectric as {dielectric_water}.')
        charges = self.atoms['charge'].tolist()
        force = dh_elec_term(
            charges=charges,
            df_exclusions=self.exclusions,
            use_pbc=self.use_pbc,
            ldby=ldby,
            dielectric_water=dielectric_water,
            cutoff=cutoff,
            force_group=force_group,
        )
        self.system.addForce(force)

