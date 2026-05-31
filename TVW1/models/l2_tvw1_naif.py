"""L2 fidelity + naif unbalanced TV-Wasserstein regularizer.

Reconstructs an EAP from undersampled (k,q) Fourier data E by solving the
saddle-point form of
        1/2 ||K P - E||^2 + alpha1 TV_W1(J P) + alpha2 TV(I P)

  model_naif        : Chambolle-Pock (CP)
  model_naif_graph  : graph Douglas-Rachford (Bredies, Chenchene & Naldi 2022)

Shared operators come from core/ (operators.II/JJ/PP, forward.KK_*); generic
solvers and discrete operators from Libraries/ (algorithms_general, ...).
Callers must `import paths` first so those directories are on sys.path.
"""
import numpy as np


def model_naif(E, mask_rfft, shape_x, shape_y, alpha1, alpha2, stepsize_ratio, iterations, printprogress=True):

    from operators import II, IIstar, JJ, JJstar
    from forward import KK_factory, KKstar_factory
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from prox_and_proj import prox_norm21, proj_infty_ball
    from algorithms_general import CP
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

    prox_f = lambda x, tau: {'P': x['P'].clip(min=0),
                             'Q': prox_norm21(x['Q'], tau * alpha1, 0)}
    prox_gstar = lambda u, sigma: {'f': u['f'],
                                   'g': proj_infty_ball(u['g'] / alpha2),
                                   'l': (u['l'] - sigma * E) / (1 + sigma)}
    L = lambda x: {'f': nabla_x(JJ(x['P'], shape_x, shape_y), dim=ndim_x) + div_y(x['Q']),
                   'g': nabla_x(II(x['P'], shape_x, shape_y), dim=ndim_x),
                   'l': KK(x['P'])}
    Lstar = lambda u: {'P': -JJstar(div_x(u['f']), shape_x, shape_y) - IIstar(div_x(u['g']), shape_y) + KKstar(u['l']),
                       'Q': -nabla_y(u['f'], dim=ndim_y)}

    size_y = np.prod(shape_y)
    L_norm_sq = 4 * ndim_x + np.sqrt(size_y) * 4 * ndim_x + 1 + 4 * ndim_y
    sigma = 1 / np.sqrt(stepsize_ratio * L_norm_sq)
    tau = stepsize_ratio * sigma

    x, u = CP(x0, u0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=printprogress)

    return x['P']


def model_naif_graph(E, mask_rfft, shape_x, shape_y, alpha1, alpha2, graph_DR_parameters, sigma, iterations,
                     printprogress=True, return_history=False, extra_metrics_fn=None, record_every=1):
    """Same model, solved by graph Douglas-Rachford.

    graph_DR_parameters = (Z, parent_node, d); build it with core/graphs.py.
    """
    Z, parent_node, d = graph_DR_parameters

    from operators import II, IIstar, JJ, PP
    from forward import KKstar_factory
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from solve_linear_systems import (laplacian_eigenvalues, resolvent_with_laplacian,
                                      resolvent_with_II, resolvent_with_JJ,
                                      resolvent_with_undersampling_withmask)
    from prox_and_proj import prox_norm21, proj_infty_ball
    from algorithms_general import graph_DR
    from numpy.random import rand as rd

    ndim_x, ndim_y = len(shape_x), len(shape_y)
    shape_xy = shape_x + shape_y
    P0 = rd(*shape_xy)
    shape_f = nabla_x(P0, dim=ndim_x).shape
    f0 = rd(*shape_f)
    shape_Q = nabla_y(f0, dim=ndim_y).shape
    shape_g = nabla_x(II(P0, shape_x, shape_y), dim=ndim_x).shape

    assert np.fft.rfftn(P0).shape == mask_rfft.shape and np.sum(mask_rfft) == E.size

    KKstar = KKstar_factory(mask_rfft)

    def prox_F1(P, lam):
        b = np.fft.rfftn(P + lam * KKstar(E), norm='ortho')
        solution_linear_system = resolvent_with_undersampling_withmask(b, lam, mask_rfft)
        return np.fft.irfftn(solution_linear_system, norm='ortho')

    def prox_F2(Q, lam):
        return prox_norm21(Q, lam * alpha1, 0)

    def J_0(z, lam):
        P = prox_F1(z['P'], lam)
        Q = prox_F2(z['Q'], lam)
        f = z['f']
        g = proj_infty_ball(z['g'], alpha2)
        return {'P': P, 'Q': Q, 'f': f, 'g': g}

    axes_x = tuple(i for i in range(ndim_x))
    axes_y = tuple(1 + ndim_x + i for i in range(ndim_y))  # z['f'] is a gradient (extra leading axis)
    axes_y_for_shape_xy = tuple(ndim_x + i for i in range(ndim_y))

    eigenvalues_J1 = laplacian_eigenvalues(shape_xy, axes_x)
    eigenvalues_J2 = laplacian_eigenvalues(shape_x, axes_x)
    eigenvalues_J3 = laplacian_eigenvalues(shape_xy, axes_y_for_shape_xy)

    def J_1(z, lam):
        b = z['P'] + lam * JJ(div_x(z['f']), shape_x, shape_y)
        pointwise_division_J1 = 1 - lam ** 2 * eigenvalues_J1
        P = resolvent_with_JJ(b, PP, JJ, shape_x, shape_y, axes_x, pointwise_division_J1, sigma=None)
        Q = z['Q']
        f = z['f'] + lam * nabla_x(JJ(P, shape_x, shape_y), dim=ndim_x)
        g = z['g']
        return {'P': P.clip(min=0), 'Q': Q, 'f': f, 'g': g}

    def J_2(z, lam):
        b = z['P'] + lam * IIstar(div_x(z['g']), shape_y)
        size_y = np.prod(shape_y)
        pointwise_division_J2 = 1 - lam ** 2 * size_y * eigenvalues_J2
        P = resolvent_with_II(b, II, JJ, shape_x, shape_y, pointwise_division_J2, sigma_times_size_y=None)
        Q = z['Q']
        f = z['f']
        g = z['g'] + lam * nabla_x(II(P, shape_x, shape_y), dim=ndim_x)
        return {'P': P, 'Q': Q, 'f': f, 'g': g}

    def J_3(z, lam):
        P = z['P']
        b = z['f'] + lam * div_y(z['Q'])
        pointwise_division_J3 = 1 - lam ** 2 * eigenvalues_J3
        f = resolvent_with_laplacian(b, pointwise_division_J3, axes_y, sigma=None)
        Q = z['Q'] + lam * nabla_y(f, dim=ndim_y)
        g = z['g']
        return {'P': P, 'Q': Q, 'f': f, 'g': g}

    resolvents = [J_0, J_1, J_2, J_3]
    N = len(resolvents)

    w0 = [{'P': rd(*shape_xy), 'Q': rd(*shape_Q), 'f': rd(*shape_f), 'g': rd(*shape_g)} for _ in range(N - 1)]

    result = graph_DR(sigma, Z, parent_node, d, w0, iterations, resolvents, printprogress=printprogress,
                      return_history=return_history, extra_metrics_fn=extra_metrics_fn, record_every=record_every)
    if return_history:
        x, w, history = result
    else:
        x, w = result
        history = None

    return x, w, history
