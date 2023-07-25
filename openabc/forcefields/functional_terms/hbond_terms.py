import numpy as np
import pandas as pd
try:
    import openmm as mm
except ImportError:
    import simtk.openmm as mm
import sys
import os

def dna_3spn2_base_pair_term(use_pbc, cutoff=1.8, force_group=9):
    """
    Need to add donors, acceptors, and set exclusions after running this function. 
    """
    base_pairs = mm.CustomHbondForce(f'''energy;
                 energy=rep+1/2*(1+cos(dphi))*fdt1*fdt2*attr;
                 rep  = epsilon*(1-exp(-alpha*dr))^2*(1-step(dr));
                 attr = epsilon*(1-exp(-alpha*dr))^2*step(dr)-epsilon;
                 fdt1 = max(f1*pair0t1,pair1t1);
                 fdt2 = max(f2*pair0t2,pair1t2);
                 pair1t1 = step({np.pi}/2+dt1)*step({np.pi}/2-dt1);
                 pair1t2 = step({np.pi}/2+dt2)*step({np.pi}/2-dt2);
                 pair0t1 = step({np.pi}+dt1)*step({np.pi}-dt1);
                 pair0t2 = step({np.pi}+dt2)*step({np.pi}-dt2);
                 f1 = 1-cos(dt1)^2;
                 f2 = 1-cos(dt2)^2;
                 dphi = dihedral(d2,d1,a1,a2)-phi0;
                 dr = distance(d1,a1)-sigma;
                 dt1 = rng*(angle(d2,d1,a1)-t01);
                 dt2 = rng*(angle(a2,a1,d1)-t02);''')
    parameters = ['phi0', 'sigma', 't01', 't02', 'rng', 'epsilon', 'alpha']
    for p in parameters:
        base_pairs.addPerDonorParameter(p)
    if use_pbc:
        base_pairs.setNonbondedMethod(base_pairs.CutoffPeriodic)
    else:
        base_pairs.setNonbondedMethod(base_pairs.CutoffNonPeriodic)
    base_pairs.setCutoffDistance(cutoff)
    base_pairs.setForceGroup(force_group)
    return base_pairs


def dna_3spn2_cross_stacking_term(use_pbc, cutoff=1.8, force_group=10):
    """
    Need to add donors, acceptors, and set exclusions after running this function. 
    
    Some notes about the expression:
        t3 is the angle composed of vector d1-d2 and a1-a2. 
        In this version, each (atom1, atom2, atom3) group only act as either donor or acceptor. 
        
    """
    cross_stackings = mm.CustomHbondForce(f'''energy;
                                          energy=fdt3*(fdtCS1*attr1+fdtCS2*attr2)/2;
                                          attr1=epsilon1*(1-exp(-alpha1*dr1))^2*step(dr1)-epsilon1;
                                          attr2=epsilon2*(1-exp(-alpha2*dr2))^2*step(dr2)-epsilon2;
                                          fdt3=max(f3*pair0t3,pair1t3);
                                          fdtCS1=max(f1*pair0tCS1,pair1tCS1);
                                          fdtCS2=max(f2*pair0tCS2,pair1tCS2);
                                          pair0t3=step({np.pi}+dt3)*step({np.pi}-dt3);
                                          pair0tCS1=step({np.pi}+dtCS1)*step({np.pi}-dtCS1);
                                          pair0tCS2=step({np.pi}+dtCS2)*step({np.pi}-dtCS2);
                                          pair1t3=step({np.pi}/2+dt3)*step({np.pi}/2-dt3);
                                          pair1tCS1=step({np.pi}/2+dtCS1)*step({np.pi}/2-dtCS1);
                                          pair1tCS2=step({np.pi}/2+dtCS2)*step({np.pi}/2-dtCS2);
                                          f1=1-cos(dtCS1)^2;
                                          f2=1-cos(dtCS2)^2;
                                          f3=1-cos(dt3)^2;
                                          dr1=distance(d1,a3)-sigma1;
                                          dr2=distance(a1,d3)-sigma2;
                                          dt3=rng_BP*(t3-t03);
                                          dtCS1=rng_CS1*(tCS1-t0CS1);
                                          dtCS2=rng_CS2*(tCS2-t0CS2);
                                          tCS1=angle(d2,d1,a3);
                                          tCS2=angle(a2,a1,d3);
                                          t3=acos(cost3lim);
                                          cost3lim=min(max(cost3,-0.99),0.99);
                                          cost3=sin(t1)*sin(t2)*cos(phi)-cos(t1)*cos(t2);
                                          t1=angle(d2,d1,a1);
                                          t2=angle(d1,a1,a2);
                                          phi=dihedral(d2,d1,a1,a2);
                                          ''')
    donor_parameters = ['t0CS2', 'rng_CS2', 'epsilon2', 'alpha2', 'sigma2']
    acceptor_parameters = ['t03', 't0CS1', 'rng_CS1', 'rng_BP', 'epsilon1', 'alpha1', 'sigma1']
    for p in donor_parameters:
        cross_stackings.addPerDonorParameter(p)
    for p in acceptor_parameters:
        cross_stackings.addPerAcceptorParameter(p)
    if use_pbc:
        cross_stackings.setNonbondedMethod(cross_stackings.CutoffPeriodic)
    else:
        cross_stackings.setNonbondedMethod(cross_stackings.CutoffNonPeriodic)
    cross_stackings.setCutoffDistance(cutoff)
    cross_stackings.setForceGroup(force_group)
    return cross_stackings


def legacy_dna_3spn2_cross_stacking_term(use_pbc, cutoff=1.8, force_group=10):
    """
    Need to add donors, acceptors, and set exclusions after running this function.  
    """
    cross_stackings = mm.CustomHbondForce(f'''energy;
                      energy   = fdt3*fdtCS*attr/2;
                      attr     = epsilon*(1-exp(-alpha*dr))^2*step(dr)-epsilon;
                      fdt3     = max(f1*pair0t3,pair1t3);
                      fdtCS    = max(f2*pair0tCS,pair1tCS);
                      pair0t3  = step({np.pi}+dt3)*step({np.pi}-dt3);
                      pair0tCS = step({np.pi}+dtCS)*step({np.pi}-dtCS);
                      pair1t3  = step({np.pi}/2+dt3)*step({np.pi}/2-dt3);
                      pair1tCS = step({np.pi}/2+dtCS)*step({np.pi}/2-dtCS);
                      f1       = 1-cos(dt3)^2;
                      f2       = 1-cos(dtCS)^2;
                      dr       = distance(d1,a3)-sigma;
                      dt3      = rng_BP*(t3-t03);
                      dtCS     = rng_CS*(tCS-t0CS);
                      tCS      = angle(d2,d1,a3);
                      t3       = acos(cost3lim);
                      cost3lim = min(max(cost3,-0.99),0.99);
                      cost3    = sin(t1)*sin(t2)*cos(phi)-cos(t1)*cos(t2);
                      t1       = angle(d2,d1,a1);
                      t2       = angle(d1,a1,a2);
                      phi      = dihedral(d2,d1,a1,a2);''')
    parameters = ['t03', 't0CS', 'rng_CS', 'rng_BP', 'epsilon', 'alpha', 'sigma']
    for p in parameters:
        cross_stackings.addPerAcceptorParameter(p)
    if use_pbc:
        cross_stackings.setNonbondedMethod(cross_stackings.CutoffPeriodic)
    else:
        cross_stackings.setNonbondedMethod(cross_stackings.CutoffNonPeriodic)
    cross_stackings.setCutoffDistance(cutoff)
    cross_stackings.setForceGroup(force_group)
    return cross_stackings
    
    
