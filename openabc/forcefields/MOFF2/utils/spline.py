import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import quad
import warnings

"""
Define some useful B-spline basis functions. 
When naming functions, we put the number of dimensions at the end of the function name.
"""

def clamped_bspline_basis_1d(x, x_min, x_max, n_internal_knots, degree=3, intercept=False, omega=False):
    """
    Clamped 1d B-spline basis.
    Clamped B-spline basis can be uniquely determined by x_min, x_max, n_internal_knots, degree, and intercept.
    Clamp means the boundary knots are repeated degree + 1 times.
    
    Parameters
    ----------
    x : np.ndarray
        The input values. 
        Values in x should be within range [x_min, x_max].
        If not within range [x_min, x_max], a warning will be raised and x will be clipped to [x_min, x_max].
    
    x_min : float
        The left boundary of the B-spline.
    
    x_max : float
        The right boundary of the B-spline.
    
    n_internal_knots : int
        The number of internal knots (i.e. knots within range (x_min, x_max)).
    
    degree : int
        The degree of the B-spline.
    
    intercept : bool
        Whether to include the intercept term.
    
    omega : bool
        Whether to compute the omega values.
    
    Returns
    -------
    output_dict : dict
        The output dictionary.
    
    Here are the details of output_dict:
        
    output_dict['design_matrix'] : np.ndarray, shape = (*x.shape, d)
        The design matrix of the B-spline basis.
        The B-spline basis will be the last dimension of the design matrix.
    
    output_dict['degree'] : int
        The degree of the B-spline.    
    
    output_dict['augmented_knots'] : np.ndarray, shape = (n_internal_knots + 2 * degree + 2,)
        The augmented knots.
    
    output_dict['basis_coeffs'] : np.ndarray, shape = (d, degree + n_internal_knots + 1)
        The coefficients of the B-spline basis. 
    
    output_dict['omega'] : None or np.ndarray
        The omega values. 
        If omega is False, this will be None.
        If omega is True, this will be a numpy array of shape (d, d). output_dict['omega'][i, j] is the integral of the product of the 2nd order derivatives of the i-th and j-th bases within range [x_min, x_max].
    
    """
    assert x_min < x_max
    if (np.min(x) < x_min) or (np.max(x) > x_max):
        warnings.warn(f'Input values are clipped to [{x_min}, {x_max}].')
        x = np.clip(x, x_min, x_max)
    M = degree + 1
    left_boundary_knots = np.array([float(x_min)] * M)
    right_boundary_knots = np.array([float(x_max)] * M)
    internal_knots = np.linspace(x_min, x_max, num=n_internal_knots + 2)[1:-1]
    augmented_knots = np.concatenate((left_boundary_knots, internal_knots, right_boundary_knots), axis=0)
    basis_coeffs = []
    design_matrix = []
    bspl_list = []
    for i in range(M + n_internal_knots):
        c = np.zeros(M + n_internal_knots)
        c[i] = 1.0
        basis_coeffs.append(c)
        f = BSpline(augmented_knots, c, degree, extrapolate=False)
        design_matrix.append(f(x))
        bspl_list.append(f)
    basis_coeffs = np.array(basis_coeffs)
    design_matrix = np.array(design_matrix)
    if not intercept:
        # drop a basis to remove intercept
        basis_coeffs = basis_coeffs[1:]
        design_matrix = design_matrix[1:]
        bspl_list = bspl_list[1:]
    design_matrix = np.moveaxis(design_matrix, 0, -1) # move BSpline basis to the last dimension
    if omega:
        _omega = np.zeros((len(bspl_list), len(bspl_list)))
        for i in range(len(bspl_list)):
            for j in range(i, len(bspl_list)):
                d2_bspline_i = bspl_list[i].derivative(2)
                d2_bspline_j = bspl_list[j].derivative(2)
                _omega[i, j] = quad(lambda a: d2_bspline_i(a) * d2_bspline_j(a), x_min, x_max, limit=10000)[0]
                _omega[j, i] = _omega[i, j] # symmetric
        assert not np.any(np.isnan(_omega))
    else:
        _omega = None
    output_dict = {'design_matrix': design_matrix, 'degree': degree, 'augmented_knots': augmented_knots, 
                   'basis_coeffs': basis_coeffs, 'omega': _omega}
    return output_dict


def periodic_bspline_basis_1d(x, x_min, x_max, n_internal_knots, degree=3, intercept=False, omega=False):
    """
    Periodic 1D B-spline basis.
    Periodic B-spline basis can be uniquely determined by x_min, x_max, n_internal_knots, degree, and intercept.
    To achieve continuity at two boundaries, some extra knots are added beyond the boundaries.
    Though extra knots extend beyonod boundaries, the basis is supposed to be only effective within the boundaries. 
    The function follows `pccg.utils.spline.pbs` with some modifications.
    
    Parameters
    ----------
    x : np.ndarray
        The input values. 
        Values in x should be within range [x_min, x_max].
        If not within range [x_min, x_max], a warning will be raised and x will be clipped to [x_min, x_max].
    
    x_min : float
        The left boundary.
    
    x_max : float
        The right boundary.
    
    n_internal_knots : int
        The number of internal knots (i.e. knots within range (x_min, x_max)).
    
    degree : int
        The degree of the B-spline.
        Note degree should satisfy degree >= 1 and degree <= n_internal_knots + 1.
    
    intercept : bool
        Whether to include the intercept term.
    
    omega : bool
        Whether to compute the omega values.
    
    Returns
    -------
    output_dict : dict
        The output dictionary.
    
    Here are the details of output_dict:
        
    output_dict['design_matrix'] : np.ndarray, shape = (*x.shape, d)
        The design matrix of the B-spline basis.
        The B-spline basis will be the last dimension of the design matrix.
    
    output_dict['degree'] : int
        The degree of the B-spline.    
    
    output_dict['augmented_knots'] : np.ndarray, shape = (n_internal_knots + 2 * degree + 2,)
        The augmented knots.
    
    output_dict['basis_coeffs'] : np.ndarray, shape = (d, degree + n_internal_knots + 1)
        The coefficients of the B-spline basis. 
    
    output_dict['omega'] : None or np.ndarray
        The omega values. 
        If omega is False, this will be None.
        If omega is True, this will be a numpy array of shape (d, d). output_dict['omega'][i, j] is the integral of the product of the 2nd order derivatives of the i-th and j-th bases within range [x_min, x_max].
    
    """
    assert x_min < x_max
    if (np.min(x) < x_min) or (np.max(x) > x_max):
        warnings.warn(f'Input values are clamped to [{x_min}, {x_max}].')
        x = np.clip(x, x_min, x_max)
    assert degree >= 1 # the method here cannot ensure periodicity for degree equal to 0
    assert degree <= n_internal_knots + 1
    M = degree + 1
    knots = np.linspace(x_min, x_max, num=n_internal_knots + 2)
    augmented_knots = np.concatenate((knots[0] - (knots[-1] - knots[-M:-1]), 
                                      knots, 
                                      knots[-1] + knots[1:M] - knots[0]), axis=0)
    basis_coeffs = []
    design_matrix = []
    bspl_list = []
    for i in range(degree, M + n_internal_knots - degree):
        # M + n_internal_knots - degree >= degree since degree <= n_internal_knots + 1
        c = np.zeros(M + n_internal_knots)
        c[i] = 1.0
        basis_coeffs.append(c)
        f = BSpline(augmented_knots, c, degree, extrapolate=False)
        design_matrix.append(f(x))
        bspl_list.append(f)
    for i in range(degree):
        c = np.zeros(M + n_internal_knots)
        c[[i, -degree + i]] = 1.0
        basis_coeffs.append(c)
        f = BSpline(augmented_knots, c, degree, extrapolate=False)
        design_matrix.append(f(x))
        bspl_list.append(f)
    basis_coeffs = np.array(basis_coeffs)
    design_matrix = np.array(design_matrix)
    if not intercept:
        # bases directly related to periodicity are all kept
        # drop a basis that does not directly contribute to periodicity to remove intercept
        basis_coeffs = basis_coeffs[1:]
        design_matrix = design_matrix[1:]
        bspl_list = bspl_list[1:]
    design_matrix = np.moveaxis(design_matrix, 0, -1) # move BSpline basis to the last dimension
    if omega:
        _omega = np.zeros((len(bspl_list), len(bspl_list)))
        for i in range(len(bspl_list)):
            for j in range(i, len(bspl_list)):
                d2_bspline_i = bspl_list[i].derivative(2)
                d2_bspline_j = bspl_list[j].derivative(2)
                _omega[i, j] = quad(lambda a: d2_bspline_i(a) * d2_bspline_j(a), x_min, x_max, limit=10000)[0]
                _omega[j, i] = _omega[i, j] # symmetric
        assert not np.any(np.isnan(_omega))
    else:
        _omega = None
    output_dict = {'design_matrix': design_matrix, 'degree': degree, 'augmented_knots': augmented_knots, 
                   'basis_coeffs': basis_coeffs, 'omega': _omega}
    return output_dict


def pair_bspline_basis_1d(x, x_min, x_max, n_internal_knots, degree=3, omega=False):
    """
    Nonbonded pair B-spline basis.
    The 0th basis will be extrapolated so that the 0th basis gives repulsion at x < x_min.
    All the basis except the 0th basis will give 0 at x <= x_min.
    All the basis will give 0 at x >= x_max.
    The function follows `pccg.utils.spline.bs_lj` with some modifications.
    This function can also be applied to RMSD potential.
    
    Parameters
    ----------
    x : np.ndarray
        The input values.
    
    x_min : float
        The left boundary of the B-spline.
    
    x_max : float
        The right boundary of the B-spline. This is the cutoff distance.
    
    n_internal_knots : int
        The number of internal knots (i.e. knots within range (x_min, x_max)).
        We require n_internal_knots >= degree so that there is at least 1 basis that covers degree + 1 intervals.
    
    degree : int
        The degree of the B-spline.
    
    omega : bool
        Whether to return the omega values.
    
    Returns
    -------
    output_dict : dict
        The output dictionary including many keys and values.
    
    Here are the details of the output dictionary:
        
    output_dict['design_matrix'] : np.ndarray, shape = (*x.shape, d)
        The design matrix of the B-spline basis.
        The B-spline basis will be the last dimension of the design matrix.
    
    output_dict['degree'] : int
        The degree of the B-spline.    
    
    output_dict['augmented_knots'] : np.ndarray, shape = (n_internal_knots + 2 * degree + 2,)
        The augmented knots.
    
    output_dict['basis_coeffs'] : np.ndarray, shape = (d, degree + n_internal_knots + 2)
        The coefficients of the B-spline basis. 
    
    output_dict['omega'] : None or np.ndarray
        The omega values. 
        If omega is False, this will be None.
        If omega is True, this will be a numpy array of shape (d, d). output_dict['omega'][i, j] is the integral of the product of the 2nd order derivatives of the i-th and j-th bases within range [x_min, x_max].
    
    """
    assert x_min < x_max
    assert n_internal_knots >= degree
    M = degree + 1
    # add one additional unique knot as the right boundary to ensure the last effective basis gives 0 at x >= x_max
    ext_internal_knots = np.linspace(x_min, x_max, num=n_internal_knots + 2)[1:]
    augmented_knots = np.concatenate((np.array([float(x_min)] * M),
                                      ext_internal_knots, 
                                      np.array([float(x_max + ext_internal_knots[0] - x_min)] * M)), axis=0)
    basis_coeffs = []
    design_matrix = []
    bspl_list = []
    for i in range(n_internal_knots + 1):
        # only keep basis that gives 0 at x >= x_max
        c = np.zeros(M + len(ext_internal_knots))
        c[i] = 1.0
        basis_coeffs.append(c)
        f = BSpline(augmented_knots, c, degree, extrapolate=True)
        y = f(x)
        assert np.all(y[x >= x_max] == 0.0) # ensure zero at x >= x_max
        assert np.all(f(np.array([x_max]), nu=1) == 0.0) # ensure first order derivative is continuous at x_max
        assert np.all(f(np.array([x_max]), nu=2) == 0.0) # ensure second order derivative is continuous at x_max
        if i >= 1:
            y[x <= x_min] = 0.0 # ensure zero at x <= x_min for all bases except the 0th basis
        design_matrix.append(y)
        bspl_list.append(f)
    basis_coeffs = np.array(basis_coeffs)
    design_matrix = np.array(design_matrix)
    design_matrix = np.moveaxis(design_matrix, 0, -1) # move BSpline basis to the last dimension
    assert not np.any(np.isnan(design_matrix))
    if omega:
        _omega = np.zeros((len(bspl_list), len(bspl_list)))
        for i in range(len(bspl_list)):
            for j in range(i, len(bspl_list)):
                d2_bspline_i = bspl_list[i].derivative(2)
                d2_bspline_j = bspl_list[j].derivative(2)
                _omega[i, j] = quad(lambda a: d2_bspline_i(a) * d2_bspline_j(a), x_min, x_max, limit=10000)[0]
                _omega[j, i] = _omega[i, j] # symmetric
        assert not np.any(np.isnan(_omega))
    else:
        _omega = None
    output_dict = {'design_matrix': design_matrix, 'degree': degree, 'augmented_knots': augmented_knots, 
                   'basis_coeffs': basis_coeffs, 'omega': _omega}
    return output_dict


