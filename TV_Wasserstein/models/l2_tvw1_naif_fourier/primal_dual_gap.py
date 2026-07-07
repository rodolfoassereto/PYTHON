import numpy as np
from core.auxiliary_functions import central_Y_indices


def phi_surrogate(t, M):
    if np.abs(t) <= M:
        return np.abs(t)
    else:
        return M/2 + t**2 / (2*M)

def phi_surrogate_conjugate(s, M): # convex conjugate of phi_surrogate: (1/2) M (s^2 - 1)_+
    return 0.5 * M * max(s**2 - 1, 0.0)

def primal_terms(P, Q, E, KK, C_bounds, alpha_1, alpha_2, shape_x, shape_y):
    """The four summands of the surrogate primal, as a dict (primal == their sum).
    All four are >= 0."""
    from differential_operators import nabla_x, div_y
    from core.tvw1_naif.operators import II, JJ
    C_P, C_Q, C_f = C_bounds
    ndim_x = len(shape_x)

    argument = nabla_x( JJ(P, shape_x, shape_y), dim=ndim_x ) + div_y(Q)
    ind = central_Y_indices( argument.shape, shape_x, shape_y )
    argument[ind] = 0 # summing the not-Y-central indices is equivalent to summing all but setting the centers to zero

    return {
        'primal_fidelity': 0.5 * np.sum( np.abs( KK(P)-E )**2 ),                                    # 1/2 ||KP-E||^2
        'primal_flux':     phi_surrogate( alpha_1 * np.sum(np.linalg.norm(Q, axis=0)) , alpha_1*C_Q ),  # phi(alpha_1 ||Q||_{2,1})
        'primal_coupling': C_f * np.sum( np.abs( argument ) ),                                      # C_f ||deltabar(grad_x JP + div_y Q)||_1
        'primal_mass_TV':  alpha_2 * np.sum( np.abs( nabla_x( II(P, shape_x, shape_y), dim=ndim_x ) ) ),  # alpha_2 ||grad I P||_1
    }

def primal(P, Q, E, KK, C_bounds, alpha_1, alpha_2, shape_x, shape_y):
    C_P, C_Q, C_f = C_bounds

    if np.max(P) > C_P:
        print(f"primal at iterate found to be +infty: np.max(P)={np.max(P)} > C_P")
        return np.inf

    return sum(primal_terms(P, Q, E, KK, C_bounds, alpha_1, alpha_2, shape_x, shape_y).values())

def dual_terms(f, g, h, E, C_bounds, alpha_1, shape_x, shape_y, KKstar):
    """The three summands of the surrogate dual, as a dict (dual == their sum).
    'dual_fidelity' contains <E,h> and can be negative; the other two are >= 0."""
    from differential_operators import nabla_y, div_x
    from core.tvw1_naif.operators import JJ, IIstar
    C_P, C_Q, C_f = C_bounds
    ndim_y = len(shape_y)

    xi = JJ(div_x(f), shape_x, shape_y) + IIstar(div_x(g), shape_y) - KKstar(h)
    s = np.max( np.linalg.norm( nabla_y(f, dim=ndim_y), axis=0 ) ) / alpha_1

    return {
        'dual_fidelity':   0.5 * np.sum( np.abs(h)**2 ) + np.real( np.vdot(E, h) ),  # 1/2 ||h||^2 + <E,h>
        'dual_positivity': C_P * np.sum( xi.clip(min=0) ),                            # C_P ||(JJ div f + II* div g - K*h)_+||_1
        'dual_transport':  phi_surrogate_conjugate(s, alpha_1 * C_Q),                 # phi*( ||grad_y f||_{2,inf} / alpha_1 )
    }

def dual(f, g, h, E, C_bounds, alpha_1, shape_x, shape_y, KKstar):
    C_P, C_Q, C_f = C_bounds

    ind = central_Y_indices(f.shape, shape_x, shape_y)
    norm_of_delta_f = np.sum(np.abs(f[ind]))
    norm_of_f = np.max(np.abs(f))
    if norm_of_delta_f > 0:
        print( f"||delta(f)||_1 is {norm_of_delta_f} (instead of = 0)" )
    if norm_of_f > C_f:
        print( f"||f / C_f ||_infty = {norm_of_f/C_f} (instead of <= 1)" )

    return sum(dual_terms(f, g, h, E, C_bounds, alpha_1, shape_x, shape_y, KKstar).values())

def primal_dual_gap(P, Q, f, g, E, KK, KKstar, C_bounds, alpha_1, alpha_2, shape_x, shape_y, return_terms=False):
    # Project P and f onto the surrogate's feasible sets, so that the indicator
    # terms dropped from primal() and dual() vanish:
    #   P onto the box {0 <= P <= C_P}  (the set whose support function
    #   C_P ||(.)_+||_1 is exactly the dual term add_2; a simplex projection,
    #   forcing sum(P)=C_P, is inconsistent with it and makes the gap plateau),
    #   f onto {delta f = 0, ||f||_inf <= C_f}.
    # Note: g is NOT projected onto alpha_2*B_inf for now (the node-averaged g
    # can violate its box), and J_3's box clip on P is disabled in the solver.
    C_P, C_Q, C_f = C_bounds
    P = P.clip(0, C_P)
    f = f.clip(min=-C_f, max=C_f)
    ind = central_Y_indices(f.shape, shape_x, shape_y)
    f[ind] = 0
    h = KK(P) - E

    if return_terms:
        # gap == sum of the seven terms; all are >= 0 except 'dual_fidelity'
        terms = {**primal_terms(P, Q, E, KK, C_bounds, alpha_1, alpha_2, shape_x, shape_y),
                 **dual_terms(f, g, h, E, C_bounds, alpha_1, shape_x, shape_y, KKstar)}
        return sum(terms.values()), terms

    p = primal(P, Q, E, KK, C_bounds, alpha_1, alpha_2, shape_x, shape_y)
    d = dual(f, g, h, E, C_bounds, alpha_1, shape_x, shape_y, KKstar)
    return p + d

def compute_C_bounds(shape_x, shape_y, alpha_1, alpha_2, E):
    E_norm = np.linalg.norm(E)
    N = np.prod(shape_x) * np.prod(shape_y)
    C_P = 2 * E_norm * np.sqrt(N)        # bound on ||P*||_inf
    C_Q = E_norm ** 2 / (2 * alpha_1)           # bound on ||Q*||_{2,1}
    C_f = alpha_1 * np.floor( max(shape_y)+1 )/2 * (1 + np.sqrt(2) / 2)   # bound on ||f*||_inf
    return (C_P, C_Q, C_f)
