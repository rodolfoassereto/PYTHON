import numpy as np
from core.auxiliary_functions import central_Y_indices


def phi_surrogate(t, M):
    if np.abs(t) <= M:
        return np.abs(t)
    else:
        return M/2 + t**2 / (2*M)

def phi_surrogate_conjugate(s, M): # convex conjugate of phi_surrogate: (1/2) M (s^2 - 1)_+
    return 0.5 * M * max(s**2 - 1, 0.0)

def primal(P, Q, E, KK, C_bounds, alpha_1, alpha_2, shape_x, shape_y):
    from differential_operators import nabla_x, div_y
    from core.tvw1_naif.operators import II, JJ
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
    from core.tvw1_naif.operators import JJ, IIstar
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

def compute_C_bounds(shape_x, shape_y, alpha_1, alpha_2, E):
    E_norm = np.linalg.norm(E)
    N = np.prod(shape_x) * np.prod(shape_y)
    C_P = 2 * E_norm / np.sqrt(N)        # bound on ||P*||_inf
    C_Q = E_norm ** 2 / (2 * alpha_1)           # bound on ||Q*||_{2,1}
    C_f = alpha_1 * np.floor( max(shape_y)+1 )/2 * (1 + np.sqrt(2) / 2)   # bound on ||f*||_inf
    return (C_P, C_Q, C_f)
