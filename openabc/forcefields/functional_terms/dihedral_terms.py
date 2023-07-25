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
    for _, row in df_dihedrals.iterrows():
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


def dna_3spn2_dihedral_term(df_dihedrals, use_pbc, force_group=8):
    dihedrals = mm.CustomTorsionForce(f"""energy;
                energy = K_periodic*(1-cs)-K_gaussian*exp(-dt_periodic^2/2/sigma^2);
                cs = cos(dt);
                dt_periodic = dt-floor((dt+{np.pi})/(2*{np.pi}))*(2*{np.pi});
                dt = theta-theta0""")
    dihedrals.addPerTorsionParameter('K_periodic')
    dihedrals.addPerTorsionParameter('K_gaussian')
    dihedrals.addPerTorsionParameter('sigma')
    dihedrals.addPerTorsionParameter('theta0')
    # add parameters
    for _, row in df_dihedrals.iterrows():
        parameters = [row['K_dihedral'], row['K_gaussian'], row['sigma'], row['theta0']]
        a1, a2, a3, a4 = int(row['a1']), int(row['a2']), int(row['a3']), int(row['a4'])
        particles = [a1, a2, a3, a4]
        dihedrals.addTorsion(*particles, parameters)
    dihedrals.setUsesPeriodicBoundaryConditions(use_pbc)
    dihedrals.setForceGroup(force_group)
    return dihedrals

