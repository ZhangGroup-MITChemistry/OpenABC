import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from scipy.optimize import minimize
from scipy.interpolate import BSpline
from openabc.forcefields.MOFF2.utils import clamped_bspline_basis_1d, periodic_bspline_basis_1d
import warnings
import math

def learn_marginal_u_angle(x1, n_internal_knots=10, degree=3, dtype=torch.float64, 
                           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), reduction='mean', 
                           method='L-BFGS-B', options={'disp': False, 'gtol': 1e-5, 'maxiter': 1000}):
    """
    Learn the marginal distribution of angles.
    
    Parameters
    ----------
    x1 : array-like
        The observed angles. It will be viewed as flattened array. We use label 1 to represent data.
    
    n_internal_knots : int
        The number of internal knots.
    
    degree : int
        The degree of the B-spline.
    
    dtype : torch.dtype
        The data type of the parameters when computing loss and gradient with torch.
    
    device : torch.device
        The device of the parameters when computing loss and gradient with torch.
    
    reduction : str
        The reduction method when computing loss with torch. See torch.nn.functional.binary_cross_entropy_with_logits for details.
    
    method : str
        The optimization method. See scipy.optimize.minimize for details.
    
    options : dict
        The options for the optimization method. See scipy.optimize.minimize for details.
    
    Returns
    -------
    u_angle : BSpline object
        The fitted reduced energy of angles. 
        See scipy.interpolate.BSpline.__call__ for details.
    
    coeffs : np.ndarray
        The coefficients of the fitted B-spline in reduced energy unit. 
        Note the number of coefficients is consistent with the last dimension of design_matrix given by clamped_1d_bspline_basis.
        
    """
    if not isinstance(x1, np.ndarray):
        x1 = np.array(x1)
    if np.min(x1) < 0 or np.max(x1) > np.pi:
        warnings.warn('Input angles are clipped to [0, pi].')
        x1 = np.clip(x1, 0, np.pi)
    x1 = x1.flatten()
    x0 = np.random.uniform(0, np.pi, len(x1))
    n0 = len(x0)
    n1 = len(x1)
    nu = torch.tensor(n0 / n1, dtype=dtype, device=device)
    x = np.concatenate((x0, x1))
    labels = torch.tensor(np.concatenate((np.zeros(len(x0)), np.ones(len(x1)))), dtype=dtype, device=device)
    u0 = -torch.tensor([math.log(1 / np.pi)] * len(x), dtype=dtype, device=device)
    basis_info_dict = clamped_bspline_basis_1d(x, 0, np.pi, n_internal_knots, degree, intercept=False, omega=False)
    A = basis_info_dict['design_matrix']
    A = torch.tensor(A, dtype=dtype, device=device)
    def _compute_loss_and_grad(theta):
        assert len(theta) == A.shape[-1] + 1
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=dtype, device=device)
        theta.requires_grad = True
        coeffs = theta[:-1]
        delta_f = theta[-1]
        u1 = torch.matmul(A, coeffs)
        logit = -torch.log(nu) + u0 - u1 + delta_f
        loss = binary_cross_entropy_with_logits(logit, labels, reduction=reduction)
        loss.backward()
        grad = theta.grad.detach().cpu().numpy().copy() # make a copy to avoid issues when training on cpu
        theta.grad.zero_() # clear gradient
        loss = loss.item()
        return loss, grad
    theta = np.zeros(A.shape[-1] + 1)
    results = minimize(_compute_loss_and_grad, theta, jac=True, method=method, options=options)
    augmented_knots = basis_info_dict['augmented_knots']
    basis_coeffs = basis_info_dict['basis_coeffs']
    coeffs = results.x[:-1]
    _coeffs = np.sum(coeffs[:, None] * basis_coeffs, axis=0)
    u_angle = BSpline(augmented_knots, _coeffs, degree, extrapolate=False)
    return u_angle, coeffs


def learn_marginal_u_dihedral(x1, n_internal_knots=10, degree=3, dtype=torch.float64, 
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), reduction='mean', 
                              method='L-BFGS-B', options={'disp': False, 'gtol': 1e-5, 'maxiter': 1000}):
    """
    Learn the marginal distribution of dihedrals.
    
    Parameters
    ----------
    x1 : array-like
        The observed dihedrals. It will be viewed as flattened array. We use label 1 to represent data.
    
    n_internal_knots : int
        The number of internal knots.
    
    degree : int
        The degree of the B-spline.
        Note degree should satisfy degree >= 1 and degree <= n_internal_knots + 1.
    
    dtype : torch.dtype
        The data type of the parameters when computing loss and gradient with torch.
    
    device : torch.device
        The device of the parameters when computing loss and gradient with torch.
    
    reduction : str
        The reduction method when computing loss with torch. See torch.nn.functional.binary_cross_entropy_with_logits for details.
    
    method : str
        The optimization method. See scipy.optimize.minimize for details.
    
    options : dict
        The options for the optimization method. See scipy.optimize.minimize for details.
    
    Returns
    -------
    u_dihedral : BSpline object
        The fitted reduced energy of dihedrals. 
        See scipy.interpolate.BSpline.__call__ for details.
    
    coeffs : np.ndarray
        The coefficients of the fitted B-spline in reduced energy unit. 
        Note the number of coefficients is consistent with the last dimension of design_matrix given by periodic_1d_bspline_basis.
        
    """
    if not isinstance(x1, np.ndarray):
        x1 = np.array(x1)
    if np.min(x1) < -np.pi or np.max(x1) > np.pi:
        warnings.warn('Input dihedrals are clipped to [-pi, pi].')
        x1 = np.clip(x1, -np.pi, np.pi)
    assert degree >= 1 # required by the method to implement periodic B-spline
    assert degree <= n_internal_knots + 1 # required by the method to implement periodic B-spline
    x1 = x1.flatten()
    x0 = np.random.uniform(-np.pi, np.pi, len(x1))
    n0 = len(x0)
    n1 = len(x1)
    nu = torch.tensor(n0 / n1, dtype=dtype, device=device)
    x = np.concatenate((x0, x1))
    labels = torch.tensor(np.concatenate((np.zeros(len(x0)), np.ones(len(x1)))), dtype=dtype, device=device)
    u0 = -torch.tensor([math.log(1 / (2 * np.pi))] * len(x), dtype=dtype, device=device)
    basis_info_dict = periodic_bspline_basis_1d(x, -np.pi, np.pi, n_internal_knots, degree, intercept=False, omega=False)
    A = basis_info_dict['design_matrix']
    A = torch.tensor(A, dtype=dtype, device=device)
    def _compute_loss_and_grad(theta):
        assert len(theta) == A.shape[-1] + 1
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=dtype, device=device)
        theta.requires_grad = True
        coeffs = theta[:-1]
        delta_f = theta[-1]
        u1 = torch.matmul(A, coeffs)
        logit = -torch.log(nu) + u0 - u1 + delta_f
        loss = binary_cross_entropy_with_logits(logit, labels, reduction=reduction)
        loss.backward()
        grad = theta.grad.detach().cpu().numpy().copy() # make a copy to avoid issues when training on cpu
        theta.grad.zero_() # clear gradient
        loss = loss.item()
        return loss, grad
    theta = np.zeros(A.shape[-1] + 1)
    results = minimize(_compute_loss_and_grad, theta, jac=True, method=method, options=options)
    augmented_knots = basis_info_dict['augmented_knots']
    basis_coeffs = basis_info_dict['basis_coeffs']
    coeffs = results.x[:-1]
    _coeffs = np.sum(coeffs[:, None] * basis_coeffs, axis=0)
    u_dihedral = BSpline(augmented_knots, _coeffs, degree, extrapolate=False)
    return u_dihedral, coeffs


