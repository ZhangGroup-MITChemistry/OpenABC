import numpy as np
import pandas as pd
try:
    import openmm as mm
except ImportError:
    import simtk.openmm as mm
import sys
import os

def periodic_dihedral_term(df_dihedrals, use_pbc, force_group=3):
    dihedrals = mm.PeriodicTorsionForce()
    for i, row in df_dihedrals.iterrows():
        a1 = int(row['a1'])
        a2 = int(row['a2'])
        a3 = int(row['a3'])
        a4 = int(row['a4'])
        periodicity = int(row['periodicity'])
        phi0 = row['phi0']
        k_dihedral = row['k_dihedral']
        dihedrals.addTorsion(a1, a2, a3, a4, periodicity, phi0, k_dihedral)
    dihedrals.setUsesPeriodicBoundaryConditions(use_pbc)
    dihedrals.setForceGroup(force_group)
    return dihedrals


