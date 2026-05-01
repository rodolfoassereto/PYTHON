import numpy as np
import cmath
from library_general import safe_divide
from differential_operators import partialplus, partialminus, nabla_y, div_y
from algorithms_general import CP

### BENAMOU BRENIER prova 1 ###

def cubic_real_roots(b, c, d):
    """Compute the roots (and returns a real one) of a family of cubic equation x^3 + b_i x^2 + c_i x + d_i = 0 using the general cubic formula."""
    
    delta_0 = b**2 - 3*c
    delta_1 = 2*b**3 - 9*b*c + 27*d
    discriminant = delta_1**2 - 4*delta_0**3
    C = ((delta_1 + np.emath.sqrt(discriminant)) / 2 +0j) ** (1/3) # rmk: (-1)**(1/3) is complex
    
    omega = complex(-0.5, cmath.sqrt(3)/2) # Cube roots of unity

    # Compute the three roots using the general formula
    root1 = (-1/(3)) * ( b + C + safe_divide( delta_0 , C ) )
    root2 = (-1/(3)) * ( b + omega*C + safe_divide( delta_0 , (omega*C) ) )
    root3 = (-1/(3)) * ( b + omega**2*C + safe_divide( delta_0 , (omega**2*C) ) )

    return select_reals(np.stack([root1, root2, root3], axis=0))

def select_reals(X, tol = 1e-14):
    real_mask = np.abs(X.imag) < tol # Mask where the imaginary part is close to zero
    real_values = np.where(real_mask, X.real, -np.inf) # Extract real parts of those elements
    return np.max(real_values, axis=0) # Choose the first nonzero real value along axis 0

def proj_K(a,b): # a~(n,n), b~(n,n,2)
    mask = a + 0.5 * np.sum(b**2, axis=-1)**2 > 0
    a1 = a[mask]
    b1 = b[mask]
    coeff_b, coeff_c, coeff_d = 2*(1+a1), (1+a1)**2, -np.sum(b1**2, axis=-1) / 2
    t_stars = cubic_real_roots(coeff_b, coeff_c, coeff_d)
    a_out, b_out = a.copy(), b.copy()
    a_out[mask] = - t_stars
    b_out[mask] = b1 / (1 + a1 + t_stars).reshape(a1.shape + (1,))
    return a_out, b_out

def W22_2(f, g, M_grid=5, maxit=5): # it uses the Benamou-Brenier dualization ( (a,b) in K )

### BENAMOU BRENIER prova 2 ###
   
    maxval = max( np.max(f), np.max(g) )
    def proj_C(rho):
        temp = rho.clip(min=0).clip(max=maxval)
        temp[0], temp[-1] = f, g
        return temp
    
    prox_f = lambda X, tau: { 'rho': proj_C(X['rho']) ,
                              'm': X['m'] }
    prox_gstar = lambda Y, sig: ( res := proj_K(Y['a'], Y['b']) ) and { 'p': Y['p'] ,
                                                                        'a': res[0] ,
                                                                        'b': res[1] }
    L = lambda X: { 'p': partialplus(X['rho'], 0) + div_y(X['m'], axis=-1) ,
                    'a': X['rho'] ,
                    'b': X['m'] }
    Lstar = lambda Y: { 'rho': Y['a'] - partialminus(Y['p'], 0) ,
                        'm': Y['b'] - np.moveaxis( nabla_y(Y['p'], dim=2), 0, -1) }
    
    tau, sig = 1/np.sqrt(14), 1/np.sqrt(14)
    
    shape_rho = (M_grid,) + f.shape
    rho0 = np.zeros( shape_rho )
    rho0[0], rho0[-1] = f, g
    m0 = np.moveaxis(nabla_y(rho0, dim=2), 0, -1)
    X0 = { 'rho': rho0 ,
           'm': m0 }
    
    p0 = partialminus(rho0, 0)
    a0 = rho0.copy()
    b0 = m0.copy()
    Y0 = { 'p': p0 ,
           'a': a0 ,
           'b': b0 }
    
    X, Y = CP(X0, Y0, tau, sig, prox_f, prox_gstar, L, Lstar, maxit, printprogress=True)
    
    return X, Y

### BENAMOU BRENIER prova 2 ###

from prox_and_proj import prox_square_over_linear

def W22_3(f, g, M_grid=20, maxit=5000):
    
    maxval = max( np.max(f), np.max(g) )
    def proj_C_rho(rho):
        rho1 = rho.clip(min=0).clip(max=maxval)
        rho1[0], rho1[-1] = f, g
        return rho1
    def proj_C_m(m):
        m1 = m.copy()
        m1[0], m1[-1] = 0, 0
        return m1
    
    def prox_f(X, tau):
        m_tmp, rho_tmp = prox_square_over_linear( X['m'].reshape(-1, X['m'].shape[-1]), X['rho'].ravel(), tau )        
        return { 'rho': proj_C_rho( rho_tmp.reshape(X['rho'].shape) ) ,
                 'm':   proj_C_m( m_tmp.reshape(X['m'].shape) ) }
    
    prox_gstar = lambda Y, sig: Y
    
    L = lambda X: { 'p': partialplus(X['rho'], 0) + div_y(X['m'], axis=-1) }
    Lstar = lambda Y: { 'rho': - partialminus(Y['p'], 0) ,
                        'm': - np.moveaxis( nabla_y(Y['p'], dim=2), 0, -1) }
    
    tau, sig = 2e-1/np.sqrt(12), 1e1/(2*np.sqrt(12))
    
    shape_rho = (M_grid,) + f.shape
    
    rho0 = np.random.rand( *shape_rho )
    rho0[0], rho0[-1] = f, g
    m0 = np.moveaxis(nabla_y(rho0, dim=2), 0, -1)
    X0 = { 'rho': rho0 ,
           'm': m0 }
    
    p0 = partialminus(rho0, 0)
    Y0 = { 'p': p0 }
    
    X, Y = CP(X0, Y0, tau, sig, prox_f, prox_gstar, L, Lstar, maxit, printprogress=True)
    
    return X, Y


f = np.zeros((30,30))
g = f.copy()

f[-2,-2] = 1
f[-2,1] = 1
g[1,1] = 2

'''X1, Y1 = W22_3(f, g, M_grid=20, maxit=5000)'''

# 0. The new formulation is a bit more numerically stable: more parameters are allowed without
# returnin NaN. However, I still observe:
#     - "Dispersion"

#     - Teleporting, ghosts and lack of convergence on bigger grids
# 1. "Dispersion" is unavoidable due to discrete differential operators?
# 2. C.E. is not met; it looks like it's due to the dualization of the C.E. constraint
#     np.sum( np.abs( partialplus(X1['rho'],0) + div_y(X1['m'], axis=-1) ) )
#         Out[]: 0.25753788521848314
#     np.sum( partialplus(X1['rho'],0) + div_y(X1['m'], axis=-1) )
#         Out[]: 6.938893903907228e-18
# 3. I would expect the algorithm to work with "partialminus", but it doesn't


######################################################################################

""" Given rho of shape (M,N,N) and m of shape (M,N,N,2) representing density and momentum (with m corresponding to a 2D vector field), project (rho, m) onto the constraint ∂ₜ rho + divₓ m = 0.
This implementation uses FFTs to solve the Poisson equation:
   -∂ₜ φ - Δₓ φ = r,
where r = ∂ₜ rho + divₓ m is the residual.

The projected variables are then:
   rho_proj = rho + ∂ₜ φ,
   m_proj   = m + ∇ₓ φ.

Assumes time and space are uniformly discretized on [0,1] with 
periodic boundary conditions.
"""

def solve_poisson(rhs):
    """ Solve the Poisson equation Δφ = rhs on a 3D grid (time and 2 spatial dims) using FFT with periodic boundary conditions. """
    M, N1, N2 = rhs.shape
    rhs_hat = np.fft.fftn(rhs, axes=(0, 1, 2))
    # Build the frequency grids for each dimension.
    k0 = np.fft.fftfreq(M)
    k1 = np.fft.fftfreq(N1)
    k2 = np.fft.fftfreq(N2)
    K0, K1, K2 = np.meshgrid(k0, k1, k2, indexing='ij')
    
    # The eigenvalues of the finite-difference Laplacian (using the forward/backward scheme)
    # are  λ = [2-2cos(2πK0)] + [2-2cos(2πK1)] + [2-2cos(2πK2)].
    eigen = (2 - 2*np.cos(2*np.pi*K0)) + (2 - 2*np.cos(2*np.pi*K1)) + (2 - 2*np.cos(2*np.pi*K2))
    
    # Avoid division by zero for the zero-frequency (set the mean of φ to zero).
    eigen[0,0,0] = 1
    phi_hat = rhs_hat / eigen
    phi_hat[0,0,0] = 0  # enforce zero-mean for φ
    phi = np.fft.ifftn(phi_hat, axes=(0, 1, 2)).real
    return phi

def project_continuity(rho, m):
    """ Given: rho: array of shape (M, N, N) representing a 2-D probability evolving in time (M time steps), m: array of shape (M, N, N, 2) representing a 2-D vector field over time.
    This function computes (rho_proj, m_proj) as the orthogonal projection of (rho, m)
    onto the set of fields satisfying the discrete continuity equation
         ∂ₜρ + div_y m = 0.
    
    The method is based on the formula (see paragraph 4.3 in [Peyré, Papadakis, Oudet :contentReference[oaicite:0]{index=0}])
    that, for the linear operator A defined by
         A(ρ, m) = ∂ₜρ + div_y m,
    the projection is given by
         (ρ, m)_proj = (ρ, m) - A* (Δ^{-1}(A(ρ, m))).
    Here we approximate A by using the forward differences (partialplus) and A* by the corresponding
    backward differences (partialminus).
    """
    # Compute the time derivative (forward difference) of rho.
    drho_dt = partialplus(rho, 0)  # axis 0 is time
    
    # Compute the spatial divergence of m.
    # For m[..., 0] (first spatial component) use forward difference along axis=1,
    # for m[..., 1] (second spatial component) use forward difference along axis=2.
    dm_dx = partialplus(m[..., 0], 1)
    dm_dy = partialplus(m[..., 1], 2)
    div_m = dm_dx + dm_dy
    
    # The residual error of the continuity equation.
    error = drho_dt + div_m  # shape: (M, N, N)
    
    # Solve the Poisson equation: Δφ = error.
    phi = solve_poisson(error)
    
    # Compute the correction terms using the backward differences.
    # These correspond to the adjoint of the forward operators.
    corr_rho = partialminus(phi, 0)   # correction for rho (time component)
    corr_m0 = partialminus(phi, 1)      # correction for m[...,0] (first spatial component)
    corr_m1 = partialminus(phi, 2)      # correction for m[...,1] (second spatial component)
    
    # Apply the correction.
    rho_proj = rho + corr_rho
    m_proj = np.empty_like(m)
    m_proj[..., 0] = m[..., 0] + corr_m0
    m_proj[..., 1] = m[..., 1] + corr_m1
    
    return rho_proj, m_proj