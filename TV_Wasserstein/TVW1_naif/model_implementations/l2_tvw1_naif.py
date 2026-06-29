"""L2 fidelity + naif unbalanced TV-Wasserstein regularizer.

Reconstructs an EAP from undersampled (k,q) Fourier data E by solving the
saddle-point form of
        1/2 ||K P - E||^2 + alpha1 TV_W1(J P) + alpha2 TV(I P)

  model_naif        : Chambolle-Pock (CP)
  model_naif_graph  : graph Douglas-Rachford (Bredies, Chenchene & Naldi 2022)

Shared operators come from core/ (operators.II/JJ/PP, forward.KK_*); generic
solvers and discrete operators from Libraries/....
Callers must `import paths` first so those directories are on sys.path.
"""
import numpy as np

def central_Y_indices(shape_f, shape_x, shape_y): # returns the indices of the Y-centers; f is only needed for its shape actually
    ndim_x, ndim_y = len(shape_x), len(shape_y)
    ndim_0 = len(shape_f) - ndim_x - ndim_y
    assert ndim_0 >= 0
    return (slice(None),)*(ndim_0+ndim_x) + tuple( np.array(shape_y) // 2 )

def model_CP(E, mask_rfft, shape_x, shape_y, alpha1, alpha2, stepsize_ratio, iterations, printprogress=True):

    '''This should be renamed model_naif_CP and it does not solve the same problem as model_naif_graph actually (for example, no P>=0 constraint)'''

    from operators import II, IIstar, JJ, JJstar
    from forward import KK_factory, KKstar_factory
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from prox_and_proj import prox_norm21, proj_L_infty_ball
    from proximal_algorithms.CP import CP
    from numpy.random import rand as rd

    ndim_x, ndim_y = len(shape_x), len(shape_y)
    shape_xy = shape_x + shape_y
    P0 = rd(*shape_xy)
    shape_f = nabla_x(P0, dim=ndim_x).shape
    f0 = rd(*shape_f)
    shape_Q = nabla_y(f0, dim=ndim_y).shape
    Q0 = rd(*shape_Q)
    shape_g = nabla_x(II(P0, shape_x, shape_y), dim=ndim_x).shape
    g0 = rd(*shape_g)
    shape_l = E.shape
    l0 = rd(*shape_l) + 1j * rd(*shape_l)

    assert np.fft.rfftn(P0).shape == mask_rfft.shape and np.sum(mask_rfft) == E.size

    x0 = {'P': P0, 'Q': Q0}
    u0 = {'f': f0, 'g': g0, 'l': l0}

    KK = KK_factory(mask_rfft)
    KKstar = KKstar_factory(mask_rfft)

    def prox_f(x, tau):
        return {'P': x['P'].clip(min=0),
                'Q': prox_norm21(x['Q'], tau * alpha1, 0)}

    ind = central_Y_indices(f0.shape, shape_x, shape_y)

    def prox_gstar(u, sigma):
        f = u['f'].copy()
        f[ind] = 0
        return {'f': f,
                'g': proj_L_infty_ball(u['g'], alpha2),
                'l': (u['l'] - sigma * E) / (1 + sigma)}

    def L(x):
        return {'f': nabla_x(JJ(x['P'], shape_x, shape_y), dim=ndim_x) + div_y(x['Q']),
                'g': nabla_x(II(x['P'], shape_x, shape_y), dim=ndim_x),
                'l': KK(x['P'])}

    def Lstar(u):
        return {'P': -JJstar(div_x(u['f']), shape_x, shape_y) - IIstar(div_x(u['g']), shape_y) + KKstar(u['l']),
                'Q': -nabla_y(u['f'], dim=ndim_y)}

    size_y = np.prod(shape_y)
    L_norm_sq = 4 * ndim_x + np.sqrt(size_y) * 4 * ndim_x + 1 + 4 * ndim_y
    sigma = 1 / np.sqrt(stepsize_ratio * L_norm_sq)
    tau = stepsize_ratio * sigma

    x, u = CP(x0, u0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=printprogress)

    return x['P']


def model_graphDR(E, mask_rfft, shape_x, shape_y, alpha1, alpha2, C_bounds, graph_DR_parameters, sigma, iterations,
                     printprogress=True, extra_metrics_fn=None, record_every=1):
    """Same model, solved by graph Douglas-Rachford.

    graph_DR_parameters = (Z, parent_node, d); build it with core/graph_DR_auxiliary_functions.py.
    """
    C_P, C_Q, C_f = C_bounds
    Z, parent_node, d = graph_DR_parameters

    from prox_and_proj import prox_norm21
    from operators import II, IIstar, JJ, JJstar, PP
    from forward import KK_factory, KKstar_factory
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from solve_linear_systems import (laplacian_eigenvalues, resolvent_with_laplacian,
                                      resolvent_with_II, resolvent_with_JJ,
                                      resolvent_with_undersampling_withmask)
    from proximal_algorithms.graph_DR import graph_DR
    from numpy.random import rand as rd

    ndim_x, ndim_y = len(shape_x), len(shape_y)
    shape_xy = shape_x + shape_y
    P0 = rd(*shape_xy)
    shape_f = nabla_x(P0, dim=ndim_x).shape
    f0 = rd(*shape_f)
    shape_Q = nabla_y(f0, dim=ndim_y).shape
    shape_g = nabla_x(II(P0, shape_x, shape_y), dim=ndim_x).shape

    assert np.fft.rfftn(P0).shape == mask_rfft.shape and np.sum(mask_rfft) == E.size

    KK, KKstar = KK_factory(mask_rfft), KKstar_factory(mask_rfft)

    def prox_F1(P, lam):
        b = np.fft.rfftn(P + lam * KKstar(E), norm='ortho')
        solution_linear_system = resolvent_with_undersampling_withmask(b, lam, mask_rfft)
        return np.fft.irfftn(solution_linear_system, norm='ortho')

    def prox_F2(Q, lam):
        return prox_norm21(Q, lam * alpha1, 0)

    ind = central_Y_indices(f0.shape, shape_x, shape_y)

    def J_0(z, lam):
        P = prox_F1(z['P'], lam)
        Q = prox_F2(z['Q'], lam)
        f = z['f'].copy()
        f[ind] = 0
        # f = f.clip(max=C_f)
        g = z['g'].clip(-alpha2, alpha2)
        return {'P': P, 'Q': Q, 'f': f, 'g': g, 'h':KK(z['P'])-E}

    axes_x = tuple(i for i in range(ndim_x))
    axes_y = tuple(1 + ndim_x + i for i in range(ndim_y))  # z['f'] is a gradient (extra leading axis)
    axes_y_for_shape_xy = tuple(ndim_x + i for i in range(ndim_y))

    eigenvalues_J1 = laplacian_eigenvalues(shape_xy, axes_x)
    eigenvalues_J2 = laplacian_eigenvalues(shape_x, axes_x)
    eigenvalues_J3 = laplacian_eigenvalues(shape_xy, axes_y_for_shape_xy)

    def J_1(z, lam):
        b = z['P'] + lam * JJstar(div_x(z['f']), shape_x, shape_y)
        pointwise_division_J1 = 1 - lam ** 2 * eigenvalues_J1
        P = resolvent_with_JJ(b, PP, JJ, shape_x, shape_y, axes_x, pointwise_division_J1, sigma=None)
        Q = z['Q']
        f = z['f'] + lam * nabla_x(JJ(P, shape_x, shape_y), dim=ndim_x)
        g = z['g']
        return {'P': P, 'Q': Q, 'f': f, 'g': g, 'h':KK(z['P'])-E}

    def J_2(z, lam):
        b = z['P'] + lam * IIstar(div_x(z['g']), shape_y)
        size_y = np.prod(shape_y)
        pointwise_division_J2 = 1 - lam ** 2 * size_y * eigenvalues_J2
        P = resolvent_with_II(b, II, JJ, shape_x, shape_y, pointwise_division_J2, sigma_times_size_y=None)
        Q = z['Q']
        f = z['f']
        g = z['g'] + lam * nabla_x(II(P, shape_x, shape_y), dim=ndim_x)
        return {'P': P, 'Q': Q, 'f': f, 'g': g, 'h':KK(z['P'])-E}

    def J_3(z, lam):
        # P = z['P'].clip(min=0, max=C_P)
        P = z['P'].clip(min=0)
        b = z['f'] + lam * div_y(z['Q'])
        pointwise_division_J3 = 1 - lam ** 2 * eigenvalues_J3
        f = resolvent_with_laplacian(b, pointwise_division_J3, axes_y, sigma=None)
        Q = z['Q'] + lam * nabla_y(f, dim=ndim_y)
        g = z['g']
        return {'P': P.clip(min=0), 'Q': Q, 'f': f, 'g': g, 'h':KK(z['P'])-E}

    resolvents = [J_0, J_1, J_2, J_3]
    N = len(resolvents)

    w0 = [{'P': np.zeros(shape_xy), 'Q': np.zeros(shape_Q), 'f': np.zeros(shape_f), 'g': np.zeros(shape_g), 'h':E} for _ in range(N - 1)]

    result = graph_DR(sigma, Z, parent_node, d, w0, iterations, resolvents, printprogress=printprogress,
                      extra_metrics_fn=extra_metrics_fn, record_every=record_every)
    if extra_metrics_fn is not None:
        x, w, history = result
    else:
        x, w = result
        history = None

    return x, w, history


def phi_surrogate(t, M):
    if np.abs(t) <= M:
        return np.abs(t)
    else:
        return M/2 + t**2 / (2*M)

def phi_surrogate_conjugate(s, M): # convex conjugate of phi_surrogate: (1/2) M (s^2 - 1)_+
    return 0.5 * M * max(s**2 - 1, 0.0)

def primal(P, Q, E, KK, C_bounds, alpha_1, alpha_2, shape_x, shape_y):
    from differential_operators import nabla_x, div_y
    from operators import II, JJ
    C_P, C_Q, C_f = C_bounds
    ndim_x = len(shape_x)

    argument = nabla_x( JJ(P, shape_x, shape_y), dim=ndim_x ) + div_y(Q)
    ind = central_Y_indices( argument.shape, shape_x, shape_y )
    argument[ind] = 0 # summing the not-Y-central indices is equivalent to summing all but setting the centers to zero

    add_1 = 0.5 * np.sum( np.abs( KK(P)-E )**2 )
    add_2 = phi_surrogate( alpha_1 * np.sum(np.linalg.norm(Q, axis=0)) , alpha_1*C_Q )
    add_3 = C_f * np.sum( np.abs( argument ) )
    add_4 = alpha_2 * np.sum( np.abs( nabla_x( II(P, shape_x, shape_y), dim=ndim_x ) ) )

    return add_1 + add_2 + add_3 + add_4

def dual(f, g, h, E, C_bounds, alpha_1, shape_x, shape_y, KKstar):
    from differential_operators import nabla_y, div_x
    from operators import JJ, IIstar
    C_P, C_Q, C_f = C_bounds
    ndim_y = len(shape_y)

    xi = JJ(div_x(f), shape_x, shape_y) + IIstar(div_x(g), shape_y) - KKstar(h)
    s = np.max( np.linalg.norm( nabla_y(f, dim=ndim_y), axis=0 ) ) / alpha_1

    add_1 = 0.5 * np.sum( np.abs(h)**2 ) + np.real( np.vdot(E, h) )
    add_2 = C_P * np.sum( xi.clip(min=0) )
    add_3 = phi_surrogate_conjugate(s, alpha_1 * C_Q)

    return add_1 + add_2 + add_3

def primal_dual_gap(P, Q, f, g, h, E, KK, KKstar, C_bounds, alpha_1, alpha_2, shape_x, shape_y):
    # project P and f onto the surrogate's feasible boxes so the dropped indicator
    # terms vanish (g already lies in its box from the solver).
    # P must be projected onto the *box* {0 <= P <= C_P} -- the set whose support
    # function C_P ||(.)_+||_1 is exactly the dual term in dual(); the solver's J_3
    # also clips P to this same box. (A simplex projection, forcing sum(P)=C_P, is
    # inconsistent with both and makes the gap plateau far from zero.)
    C_P, C_Q, C_f = C_bounds
    P = P.clip(0, C_P)
    f = f.clip(min=-C_f, max=C_f)
    ind = central_Y_indices(f.shape, shape_x, shape_y)
    f[ind] = 0
    h = KK(P) - E

    p = primal(P, Q, E, KK, C_bounds, alpha_1, alpha_2, shape_x, shape_y)
    d = dual(f, g, h, E, C_bounds, alpha_1, shape_x, shape_y, KKstar)
    return p + d

def compute_C_bounds(shape_y, alpha_1, alpha_2, E):
    E_norm = np.linalg.norm(E)
    C_P = 2 * E_norm         # bound on ||P*||_inf
    C_Q = E_norm ** 2 / (2 * alpha_1)           # bound on ||Q*||_{2,1}
    C_f = alpha_1 * np.floor( max(shape_y)+1 )/2 * (1 + np.sqrt(2) / 2)   # bound on ||f*||_inf
    return (C_P, C_Q, C_f)

def objective(P, shape_x, shape_y, alpha_1, alpha_2, KK, E):
    from operators import JJ, II
    from my_optimal_transport import TVW1
    from standard_TV_denoising import TVL1
    return 0.5 * np.sum(np.abs( KK(P) - E )**2) + alpha_1 * TVW1(JJ(P, shape_x, shape_y), shape_x, shape_y) + alpha_2 * TVL1( II(P, shape_x, shape_y))
