import numpy as np
import pandas as pd
try:
    import openmm as mm
except ImportError:
    import simtk.openmm as mm
import sys
import os

def dna_3spn2_stacking_term(df_stackings, use_pbc, force_group=7):
    """
    DNA 3SPN2 stacking potential. 
    """
    stackings = mm.CustomCompoundBondForce(3, f"""energy;
                energy=rep+f2*attr;
                rep=epsilon*(1-exp(-alpha*(dr)))^2*step(-dr);
                attr=epsilon*(1-exp(-alpha*(dr)))^2*step(dr)-epsilon;
                dr=distance(p2,p3)-sigma;
                f2=max(f*pair2,pair1);
                pair1=step(dt+{np.pi}/2)*step({np.pi}/2-dt);
                pair2=step(dt+{np.pi})*step({np.pi}-dt);
                f=1-cos(dt)^2;
                dt=rng*(angle(p1,p2,p3)-theta0);""")
    stackings.addPerBondParameter('epsilon')
    stackings.addPerBondParameter('sigma')
    stackings.addPerBondParameter('theta0')
    stackings.addPerBondParameter('alpha')
    stackings.addPerBondParameter('rng')
    # add parameters
    for _, row in df_stackings.iterrows():
        parameters = [row['epsilon'], row['sigma'], row['theta0'], row['alpha'], row['rng']]
        a1, a2, a3 = int(row['a1']), int(row['a2']), int(row['a3'])
        stackings.addBond([a1, a2, a3], parameters)
    stackings.setUsesPeriodicBoundaryConditions(use_pbc)
    stackings.setForceGroup(force_group)
    return stackings

