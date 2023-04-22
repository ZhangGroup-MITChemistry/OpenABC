import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import sys
import os

def harmonic_angle_term(df_angles, use_pbc, force_group=2):
    angles = mm.HarmonicAngleForce()
    for i, row in df_angles.iterrows():
        a1 = int(row['a1'])
        a2 = int(row['a2'])
        a3 = int(row['a3'])
        theta0 = row['theta0']
        k_angle = row['k_angle']
        angles.addAngle(a1, a2, a3, theta0, k_angle)
    angles.setUsesPeriodicBoundaryConditions(use_pbc)
    angles.setForceGroup(force_group)
    return angles


def class2_angle_term(df_angles, use_pbc, force_group=2):
    angles = mm.CustomAngleForce('k_angle_2*(theta-theta0)^2+k_angle_3*(theta-theta0)^3+k_angle_4*(theta-theta0)^4')
    angles.addPerAngleParameter('theta0')
    angles.addPerAngleParameter('k_angle_2')
    angles.addPerAngleParameter('k_angle_3')
    angles.addPerAngleParameter('k_angle_4')
    for i, row in df_angles.iterrows():
        a1 = int(row['a1'])
        a2 = int(row['a2'])
        a3 = int(row['a3'])
        parameters = row[['theta0', 'k_angle_2', 'k_angle_3', 'k_angle_4']].tolist()
        angles.addAngle(a1, a2, a3, parameters)
    angles.setUsesPeriodicBoundaryConditions(use_pbc)
    angles.setForceGroup(force_group)
    return angles



