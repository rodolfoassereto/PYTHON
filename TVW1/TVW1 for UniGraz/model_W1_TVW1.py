# %%

import numpy as np
import matplotlib.pyplot as plt
import sys
# Add Libraries folder to sys.path
libraries_path = r"C:\Users\rodol\My Drive\PYHTON\Libraries"
if libraries_path not in sys.path:
    sys.path.insert(0, libraries_path)
from plottings import plot_u, plot_u_light  # noqa: E402

# %%

def model_W1_TVW1(u_bar, shape_x, shape_y, alpha, iterations, printprogress=True, stepsize_ratio=0.2):
    assert type(shape_x) is tuple and type(shape_y) is tuple # se sono array la somma shape_x + shape_y è la somma entrywise
    
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from prox_and_proj import prox_norm21, proj_simplex_array
    from numpy.random import rand as rd

    assert  np.allclose( proj_simplex_array( u_bar, ndim_y ), u_bar )
    
    ndim_x, ndim_y = len(shape_x), len(shape_y)
    shape_u = shape_x + shape_y
    u0 = rd( *shape_u )
    shape_v = nabla_y( u0, dim=ndim_x ).shape
    v0 = rd( *shape_v )
    shape_lam = shape_u
    lam0 = rd( *shape_lam )
    shape_eta = nabla_y( u0, dim=ndim_y ).shape
    eta0 = rd( *shape_eta )
    shape_w = nabla_y( eta0, dim=ndim_y ).shape
    w0 = rd( *shape_w )
    
    X0 = { 'u': u0, 'v': v0, 'w': w0 }
    U0 = { 'lam': lam0, 'eta': eta0}
    
    prox_f =     lambda X, tau:   { 'u': proj_simplex_array( X['u'], ndim_y),  # noqa: E731
                                    'v': prox_norm21( X['v'], tau, 0 ),
                                    'w': prox_norm21( X['w'], tau*alpha, 0 ) }
    prox_gstar = lambda U, sigma: { 'lam': U['lam'] + sigma*u_bar,  # noqa: E731
                                    'eta': U['eta'] }
    L =     lambda X: { 'lam': -X['u'] + div_y( X['v'] ) ,  # noqa: E731
                        'eta': nabla_x( X['u'], dim=ndim_x ) + div_y( X['w'] ) }
    Lstar = lambda U: { 'u': -U['lam'] - div_x( U['eta'] ) ,  # noqa: E731
                        'v': -nabla_y( U['lam'], dim=ndim_y ),
                        'w': -nabla_y( U['eta'], dim=ndim_y ) }
    
    L_norm_sq = 1 + 4*ndim_x + 8*ndim_y
    sigma = 1 / np.sqrt( stepsize_ratio * L_norm_sq )
    tau = stepsize_ratio * sigma

    print( f"sigma={sigma}, tau={tau}" )
    
    from algorithms_general import CP
    
    def aux_funct(x, i):
        # if i%50 == 1:
        #     print(np.max(x['P']))
        return
    
    X, U = CP( X0, U0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=printprogress, aux_funct=aux_funct)
    
    return X['u']
