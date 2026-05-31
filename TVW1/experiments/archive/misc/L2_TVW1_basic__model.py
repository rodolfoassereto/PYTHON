# %% #################### IMPORTS
import numpy as np
import sys
# Add Libraries folder to sys.path
libraries_path = r"C:\Users\rodol\My Drive\PYHTON\Libraries"
if libraries_path not in sys.path:
    sys.path.insert(0, libraries_path)

# %% #################### MODEL

def KK(P, mask): # mask is mask_rfft
    assert np.isrealobj(P) # !! rfftn only if P is real
    return np.fft.rfftn(P, norm='ortho')[mask]
def KKstar(l, shape_xy, mask): # mask is mask_rfft  # noqa: E741
    full_array = np.zeros_like( mask, dtype=l.dtype )
    full_array[mask] = l
    return np.fft.irfftn( full_array, s=shape_xy, axes=list(range(len(shape_xy))), norm='ortho' )


def model_basic(E, mask_rfft, shape_x, shape_y, alpha, beta, stepsize_ratio, iterations, printprogress=True):
    assert stepsize_ratio < 1
    assert type(shape_x) is tuple and type(shape_y) is tuple # se sono array la somma shape_x + shape_y è la somma entrywise
    
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from numpy.random import rand as rd
    
    shape_xy = shape_x + shape_y
    ndim_x, ndim_y = len(shape_x), len(shape_y)
    shape_P = shape_x + shape_y
    P0 = rd( *shape_P )
    shape_f = nabla_x( P0, dim=ndim_x ).shape
    f0 = rd( *shape_f )
    shape_Q = nabla_y( f0, dim=ndim_y  ).shape
    Q0 = rd( *shape_Q )
    shape_l = E.shape
    l0 = rd( *shape_l) + 1j* rd(*shape_l)
    shape_h = nabla_y( P0, dim=ndim_y ).shape
    h0 = rd( *shape_h )
    
    x0 = { 'P': P0, 'Q': Q0 }
    u0 = { 'f': f0, 'l': l0, 'h': h0 }

    from prox_and_proj import prox_norm21
    
    prox_f =     lambda x, tau:   { 'P': x['P'].clip(min=0), # !! do we need .real or .clip(min=0)?  # noqa: E731
                                    'Q': prox_norm21( x['Q'], tau*alpha, 0 ) }
    prox_gstar = lambda u, sigma: { 'f': u['f'],  # noqa: E731
                                    'l': ( u['l'] - sigma*E ) / (1+sigma), # !! mistake: why sigma**2? Should be sigma (not squared!)
                                    'h': beta*u['h'] / ( sigma + beta ) }
    L =     lambda x: { 'f': nabla_x( x['P'], dim=ndim_x ) + div_y( x['Q'] ),  # noqa: E731
                        'l': KK( x['P'], mask_rfft ),
                        'h': nabla_y( x['P'], dim=ndim_y ) }
    Lstar = lambda u: { 'P': -div_x( u['f'] ) + KKstar( u['l'], shape_xy, mask_rfft ) - div_y( u['h'] ) ,  # noqa: E731
                        'Q': -nabla_y( u['f'], dim=ndim_y ) }
    
    L_norm_sq = 4*ndim_x + 4*ndim_y + 1
    sigma = 1 / np.sqrt( stepsize_ratio * L_norm_sq )
    tau = stepsize_ratio * sigma
    
    from algorithms_general import CP
    
    def aux_funct(x, i):
        # if i%50 == 1:
        #     print(np.max(x['P']))
        return
    
    x, u = CP( x0, u0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=printprogress, aux_funct=aux_funct)
    
    return x['P']
