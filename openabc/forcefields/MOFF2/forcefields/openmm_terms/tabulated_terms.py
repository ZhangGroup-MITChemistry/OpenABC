import numpy as np
import pandas as pd
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import warnings
from openabc.forcefields.MOFF2.lib import GAS_CONST


def tabulated_angle_term(angle_energy_table, df_angles, use_pbc, jacobian=False, temperature=None, force_group=2):
    """
    Angle potential with tabulated function.
    
    Parameters
    ----------
    angle_energy_table : 2d array-like, shape = (n_angle_types, n_angle_grids)
        Tabulated angle potential.
    
    df_angles : pd.DataFrame
        DataFrame with angle information.
    
    use_pbc : bool
        Whether to use periodic boundary conditions.
    
    jacobian : bool
        Whether to use the Jacobian term.
    
    temperature : None or Quantity
        Temperature.
        This is only useful when jac=True. If jac is False, T can be None.
    
    force_group : int
        Force group.
    
    Returns
    -------
    angle_force : Force
        OpenMM force.
    
    """
    if not isinstance(angle_energy_table, np.ndarray):
        angle_energy_table = np.array(angle_energy_table)
    assert isinstance(df_angles, pd.DataFrame)
    angle_energy_2D_function = mm.Continuous2DFunction(xsize=angle_energy_table.shape[0], ysize=angle_energy_table.shape[1], 
                                                       values=angle_energy_table.flatten(order='F'), 
                                                       xmin=0, xmax=angle_energy_table.shape[0] - 1, 
                                                       ymin=0, ymax=np.pi, periodic=False)
    if jacobian:
        RT = (GAS_CONST * temperature).value_in_unit(unit.kilojoule_per_mole)
        angle_force = mm.CustomCompoundBondForce(3, f'U_angle(angle_type,angle(p1,p2,p3))+{RT}*log(sin(angle(p1,p2,p3)))')
    else:
        angle_force = mm.CustomCompoundBondForce(3, f'U_angle(angle_type,angle(p1,p2,p3))')
    angle_force.addTabulatedFunction('U_angle', angle_energy_2D_function)
    angle_force.addPerBondParameter('angle_type')
    angle_force.setUsesPeriodicBoundaryConditions(use_pbc)
    for _, row in df_angles.iterrows():
        a1, a2, a3 = int(row['a1']), int(row['a2']), int(row['a3'])
        angle_type = int(row['angle_type'])
        assert angle_type <= angle_energy_table.shape[0] - 1
        angle_force.addBond([a1, a2, a3], [angle_type])
    angle_force.setForceGroup(force_group)
    return angle_force


def tabulated_dihedral_term(dihedral_energy_table, df_dihedrals, use_pbc, force_group=3):
    """
    Dihedral potential with tabulated function.
    
    Parameters
    ----------
    dihedral_energy_table : 2d array-like, shape = (n_dihedral_types, n_dihedral_grids)
        Tabulated dihedral potential.
    
    df_dihedrals : pd.DataFrame
        DataFrame with dihedral information.
    
    use_pbc : bool
        Whether to use periodic boundary conditions.
    
    force_group : int
        Force group.
    
    Returns
    -------
    dihedral_force : Force
        OpenMM force.
    
    """
    if not isinstance(dihedral_energy_table, np.ndarray):
        dihedral_energy_table = np.array(dihedral_energy_table)
    dihedral_energy_table = np.concatenate((dihedral_energy_table, dihedral_energy_table[[0]]), axis=0) # ensure periodicity along axis 0
    boundary_values = np.mean(dihedral_energy_table[:, [0, -1]], axis=1)
    dihedral_energy_table[:, 0] = boundary_values # ensure periodicity along axis 1
    dihedral_energy_table[:, -1] = boundary_values # ensure periodicity along axis 1
    assert isinstance(df_dihedrals, pd.DataFrame)
    dihedral_energy_2D_function = mm.Continuous2DFunction(xsize=dihedral_energy_table.shape[0], 
                                                          ysize=dihedral_energy_table.shape[1], 
                                                          values=dihedral_energy_table.flatten(order='F'), 
                                                          xmin=0, xmax=dihedral_energy_table.shape[0] - 1, 
                                                          ymin=-np.pi, ymax=np.pi, periodic=True)
    dihedral_force = mm.CustomCompoundBondForce(4, f'U_dihedral(dihedral_type,dihedral(p1,p2,p3,p4))')
    dihedral_force.addTabulatedFunction('U_dihedral', dihedral_energy_2D_function)
    dihedral_force.addPerBondParameter('dihedral_type')
    dihedral_force.setUsesPeriodicBoundaryConditions(use_pbc)
    for _, row in df_dihedrals.iterrows():
        a1, a2, a3, a4 = int(row['a1']), int(row['a2']), int(row['a3']), int(row['a4'])
        dihedral_type = int(row['dihedral_type'])
        assert dihedral_type <= dihedral_energy_table.shape[0] - 2
        dihedral_force.addBond([a1, a2, a3, a4], [dihedral_type])
    dihedral_force.setForceGroup(force_group)
    return dihedral_force


def tabulated_pair_term(pair_energy_table, r_min, r_max, atom_types, exclusions, use_pbc, symmetric=True, force_group=4):
    """
    Nonbonded pair interaction with tabulated function.
    
    Parameters
    ----------
    pair_energy_table : 3d array-like, shape = (n_atom_types, n_atom_types, n_pair_distance_grids)
        Tabulated nonbonded pair potential.
    
    r_min : float or int
        Minimal pair distance that the tabulated potential covers.
    
    r_max : float or int
        Maximal pair distance that the tabulated potential covers.
        Note the potential will be viewed as zero when r > r_max.
    
    atom_types : 1d array-like, shape = (n_atoms,)
        Atom types.
    
    exclusions : pd.DataFrame or 2d array-like
        The nonbonded exclusions. 
        If pd.DataFrame, the columns are ['a1', 'a2'], and each row specify an excluded pair.
        If array-like, each row specify an excluded pair.
    
    use_pbc : bool
        Whether to use periodic boundary conditions.
    
    symmetric : bool
        Whether to ensure the pair_energy_table is symmetric.
    
    force_group : int
        Force group.
    
    Returns
    -------
    nonbonded_force : Force
        OpenMM force.
        
    """
    if not isinstance(pair_energy_table, np.ndarray):
        pair_energy_table = np.array(pair_energy_table)
    assert pair_energy_table.ndim == 3
    n_atom_types = pair_energy_table.shape[0]
    assert pair_energy_table.shape[1] == n_atom_types
    pair_energy_3D_function = mm.Continuous3DFunction(xsize=pair_energy_table.shape[0], 
                                                      ysize=pair_energy_table.shape[1], 
                                                      zsize=pair_energy_table.shape[2], 
                                                      values=pair_energy_table.flatten(order='F'), 
                                                      xmin=0, xmax=n_atom_types - 1, 
                                                      ymin=0, ymax=n_atom_types - 1, zmin=r_min, zmax=r_max, 
                                                      periodic=False)
    nonbonded_force = mm.CustomNonbondedForce(f'U_pair(atom_type1,atom_type2,r)*step({r_max}-r)')
    nonbonded_force.addTabulatedFunction('U_pair', pair_energy_3D_function)
    nonbonded_force.addPerParticleParameter('atom_type')
    for each in atom_types:
        nonbonded_force.addParticle([each])
    if isinstance(exclusions, pd.DataFrame):
        exclusions = exclusions[['a1', 'a2']].to_numpy()
    elif not isinstance(exclusions, np.ndarray):
        exclusions = np.array(exclusions)
    assert exclusions.ndim == 2
    assert exclusions.shape[1] == 2
    for each in exclusions:
        a1, a2 = int(each[0]), int(each[1])
        nonbonded_force.addExclusion(a1, a2)
    if use_pbc:
        nonbonded_force.setNonbondedMethod(nonbonded_force.CutoffPeriodic)
    else:
        nonbonded_force.setNonbondedMethod(nonbonded_force.CutoffNonPeriodic)
    nonbonded_force.setCutoffDistance(r_max)
    nonbonded_force.setForceGroup(force_group)
    return nonbonded_force






