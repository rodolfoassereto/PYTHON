# %% imports
import numpy as np
import sys
# Add Libraries folder to sys.path
libraries_path = r"C:\Users\rodol\My Drive\PYHTON\Libraries"
if libraries_path not in sys.path:
    sys.path.insert(0, libraries_path)

# %% Preliminary definitions (PP, II, JJ, KK_factory)

def II(P, shape_x, shape_y):
    assert P.shape == shape_x + shape_y
    axes_to_sum = tuple( [ -(i+1) for i in range(len(shape_y)) ] )
    return np.sum( P, axis=axes_to_sum )
# note: the spectral norm of II is exactly size_y = np.prod(shape_y)

def IIstar(x, shape_y):
    ndim_y = len(shape_y)
    x1 = x.reshape( x.shape + ndim_y * (1,) )
    return np.broadcast_to(x1, x.shape + shape_y)

def PP(P, shape_x, shape_y):
    size_y = np.prod(shape_y)
    return IIstar( II(P, shape_x, shape_y), shape_y ) / size_y

def JJ(P, shape_x, shape_y):
    N = np.prod(shape_y)
    return P - IIstar( II(P, shape_x, shape_y)/N, shape_y )

JJstar = JJ # JJ is self adjoint

def KK_factory(mask_rfft):
    def KK(P):
        assert np.all( np.isreal(P) )  # rfftn only if P is real
        return np.fft.rfftn(P, norm='ortho')[mask_rfft]
    return KK

def KKstar_factory(mask_rfft):
    def KKstar(E):
        full_array = np.zeros_like( mask_rfft, dtype=E.dtype )
        full_array[mask_rfft] = E
        return np.fft.irfftn( full_array, norm='ortho' )
    return KKstar

# %% Model naif

def model_naif(E, mask_rfft, shape_x, shape_y, alpha1, alpha2, stepsize_ratio, iterations, printprogress=True):

    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from numpy.random import rand as rd
    
    ndim_x, ndim_y = len(shape_x), len(shape_y)
    shape_xy = shape_x + shape_y
    P0 = rd( *shape_xy )
    shape_f = nabla_x( P0, dim=ndim_x ).shape
    f0 = rd( *shape_f )
    shape_Q = nabla_y( f0, dim=ndim_y  ).shape
    Q0 = rd( *shape_Q )
    shape_g = nabla_x(II(P0, shape_x, shape_y), dim=ndim_x).shape
    g0 = rd( *shape_g )
    shape_l = E.shape
    l0 = rd( *shape_l) + 1j* rd(*shape_l)

    assert np.fft.rfftn( P0 ).shape == mask_rfft.shape and np.sum( mask_rfft ) == E.size # check compatibility of (shape_x, shape_y), mask_rfft and E
    
    x0 = { 'P': P0, 'Q': Q0 }
    u0 = { 'f': f0, 'g': g0, 'l': l0 }
    
    KK = KK_factory(mask_rfft)
    KKstar = KKstar_factory(mask_rfft)

    from prox_and_proj import prox_norm21, proj_infty_ball
    
    prox_f =     lambda x, tau:   { 'P': x['P'].clip(min=0), # !! do we need .real or .clip(min=0)?
                                    'Q': prox_norm21( x['Q'], tau*alpha1, 0 ) }
    prox_gstar = lambda u, sigma: { 'f': u['f'],
                                    'g': proj_infty_ball( u['g'] / alpha2 ),
                                    'l': ( u['l'] - sigma*E ) / (1+sigma) }
    L =     lambda x: { 'f': nabla_x( JJ( x['P'], shape_x, shape_y ), dim=ndim_x ) + div_y( x['Q'] ),
                    'g': nabla_x( II( x['P'], shape_x, shape_y ), dim=ndim_x ),
                    'l': KK( x['P'] ) }
    Lstar = lambda u: { 'P': -JJstar( div_x( u['f'] ), shape_x, shape_y ) - IIstar( div_x( u['g'] ), shape_y) + KKstar( u['l'] ) ,
                        'Q': -nabla_y( u['f'], dim=ndim_y ) }
    
    size_y = np.prod(shape_y)
    L_norm_sq = 4*ndim_x + np.sqrt(size_y)*4*ndim_x + 1 + 4*ndim_y
    sigma = 1 / np.sqrt( stepsize_ratio * L_norm_sq )
    tau = stepsize_ratio * sigma
    
    from algorithms_general import CP
    
    x, u = CP( x0, u0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=printprogress)
    
    return x['P']

# %% Model_naif_graph
'''
The function model_naif employs Graph-DR
'''
def model_naif_graph(E, mask_rfft, shape_x, shape_y, alpha1, alpha2, graph_DR_parameters, sigma, iterations, printprogress=True, return_history=False, extra_metrics_fn=None, record_every=1):

    Z, parent_node, d = graph_DR_parameters

    # Nota: le stepsizes sono già implementate in graph_DR come sigma / d

    from differential_operators import nabla_x, nabla_y, div_x, div_y, laplacian_x
    from numpy.random import rand as rd
    
    ndim_x, ndim_y = len(shape_x), len(shape_y)
    shape_xy = shape_x + shape_y
    P0 = rd( *shape_xy )
    shape_f = nabla_x( P0, dim=ndim_x ).shape
    f0 = rd( *shape_f )
    shape_Q = nabla_y( f0, dim=ndim_y  ).shape
    # Q0 = rd( *shape_Q )
    shape_g = nabla_x(II(P0, shape_x, shape_y), dim=ndim_x).shape
    # g0 = rd( *shape_g )

    assert np.fft.rfftn( P0 ).shape == mask_rfft.shape and np.sum( mask_rfft ) == E.size # check compatibility of (shape_x, shape_y), mask_rfft and E
    
    # KK = KK_factory(mask_rfft)
    KKstar = KKstar_factory(mask_rfft)

    from solve_linear_systems import laplacian_eigenvalues, resolvent_with_laplacian, resolvent_with_II, resolvent_with_JJ, resolvent_with_undersampling_withmask
    from prox_and_proj import prox_norm21, proj_infty_ball

    def prox_F1(P, lam):
        b = np.fft.rfftn( P + lam*KKstar(E), norm='ortho' )
        solution_linear_system = resolvent_with_undersampling_withmask( b, lam, mask_rfft )
        return np.fft.irfftn( solution_linear_system, norm='ortho' )
    
    def prox_F2(Q, lam):
        return prox_norm21(Q, lam*alpha1, 0)
    
    # prox_G1 is the identity, prox_G2 is just a projection
    
    def J_0(z, lam):
        P = prox_F1(z['P'], lam)
        Q = prox_F2(z['Q'], lam)
        f = z['f']
        g = proj_infty_ball(z['g'], alpha2)
        return { 'P':P, 'Q':Q, 'f':f, 'g':g }
    
    ndim_x, ndim_y = len(shape_x), len(shape_y)
    axes_x = tuple( [ i for i in range(ndim_x)] )
    axes_y = tuple( [ 1 + ndim_x + i for i in range(ndim_y)] ) # axes_y is needed for z['f'] which is a gradient, so it has an extra dimension at the beginning (hence the "1+...")
    axes_y_for_shape_xy = tuple([ndim_x + i for i in range(ndim_y)])   # (2, 3)

    eigenvalues_J1 = laplacian_eigenvalues(shape_xy, axes_x)
    eigenvalues_J2 = laplacian_eigenvalues(shape_x, axes_x )
    eigenvalues_J3 = laplacian_eigenvalues(shape_xy, axes_y_for_shape_xy)

    def J_1(z, lam):
        b = z['P'] + lam * JJ( div_x ( z['f'] ), shape_x, shape_y )
        pointwise_division_J1 = 1 - lam**2 * eigenvalues_J1
        P = resolvent_with_JJ(b, PP, JJ, shape_x, shape_y, axes_x, pointwise_division_J1, sigma=None)
        Q = z['Q']
        f = z['f'] + lam * nabla_x( JJ(P, shape_x, shape_y ), dim=ndim_x )
        g = z['g']
        return { 'P':P.clip(min=0), 'Q':Q, 'f':f, 'g':g }

    def J_2(z, lam):
        b = z['P'] + lam * IIstar( div_x( z['g'] ), shape_y )
        size_y = np.prod(shape_y)
        pointwise_division_J2 = 1 - lam**2 * size_y * eigenvalues_J2
        P = resolvent_with_II(b, II, JJ, shape_x, shape_y, pointwise_division_J2, sigma_times_size_y=None)
        Q = z['Q']
        f = z['f']
        g = z['g'] + lam * nabla_x( II(P, shape_x, shape_y) , dim=ndim_x)
        return { 'P':P, 'Q':Q, 'f':f, 'g':g }
    
    def J_3(z, lam):
        P = z['P']
        b = z['f'] + lam * div_y( z['Q'] )
        pointwise_division_J3 = 1 - lam**2 * eigenvalues_J3
        f = resolvent_with_laplacian(b, pointwise_division_J3, axes_y, sigma=None)
        Q = z['Q'] + lam * nabla_y( f, dim=ndim_y )
        g = z['g']
        return { 'P':P, 'Q':Q, 'f':f, 'g':g }
    
    resolvents = [J_0, J_1, J_2, J_3]
    N = len(resolvents)
    
    w0 = [{'P': rd(*shape_xy), 'Q': rd(*shape_Q), 'f': rd(*shape_f), 'g': rd(*shape_g)} for _ in range(N-1)]

    from algorithms_general import graph_DR
    
    x, w, history = graph_DR( sigma, Z, parent_node, d, w0, iterations, resolvents, printprogress=printprogress,
    return_history=return_history, extra_metrics_fn=extra_metrics_fn, record_every=record_every )

    return x, w, history

    # z_list = graph_DR(sigma, Z, parent_node, d, w0, iterations, resolvents, printprogress=printprogress)
    # return sum(z['P'] for z in z_list) / len(z_list)