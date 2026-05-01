'''Here we define a general version of Chambolle-Pock, using dictionary variables'''
'''I refer to Condat (https://doi.org/10.1137/20M1379344) for the notation. Here, the saddle problem has F and G^*'''

import numpy as np
import time

def graph_DR_old(sigma, Z, parent_node, d, w0, iterations, resolvents, theta=1.0, printprogress=False):

    N = len(resolvents) # number of nodes

    assert Z.shape == (N, N-1), "Z must be shape (N, N-1)"
    assert len(w0) == N-1, "w must have length N-1"

    stepsizes = sigma / d # list of resolvents stepsizes
    # the initialization J=J, lam=lam avoids python's so-called closure binding behavior (or else only the last J and lam from the loop are used)
    resolvents = [ lambda _, J=J, lam=lam : J( _ , lam ) for J, lam in zip(resolvents, stepsizes) ]

    x = N * [None] # x needs not to be initialized: I just create an empty list
    w = w0.copy()
    
    start_time = time.time()
    next_update = start_time  # Update every 1 second
    
    for k in range(1, iterations+1):

        current_time = time.time()
        if (current_time >= next_update) & printprogress:
            print(f"Progress: ~{int(100 * k / iterations)}% ({k}/{iterations}),  Time elapsed:\
~{int((current_time - start_time) / 60)} min,  Time left:~{int((current_time - start_time) / 60 * (iterations - k) / k)} min")
            next_update = current_time + 60  # Schedule next update
    
        for i in range(N):

            J_lam = resolvents[i] # J_lam is the resolvent with the already correct stepsize, hence "lam"

            x_sum_i = add_dicts( *[ x[h] for h in parent_node[i] ] )
            w_sum_i = add_dicts( *w, weights=Z[i,:] )

            left_term = scalar_multiply_dict( x_sum_i, 2/d[i] )
            right_term = scalar_multiply_dict( w_sum_i, 1/d[i] )

            x[i] = J_lam( add_dicts( left_term, right_term ) )

        for j in range(N-1):

            x_sum_j = add_dicts( *x, weights=theta*Z[:,j] )

            w[j] = subtract_dicts( w[j], x_sum_j )

    return x

# Implementation to check consensus and other things

def graph_DR(sigma, Z, parent_node, d, w0, iterations, resolvents,
             theta=1.0, printprogress=False, return_history=False,
             extra_metrics_fn=None, record_every=1):
    """
    Graph-based Douglas-Rachford method (Algorithm 1 from Bredies, Chenchene, Naldi 2022).

    Parameters
    ----------
    extra_metrics_fn : callable, optional
        Function x_list -> dict of scalars, called every `record_every` iterations.
        Each key creates an entry in history; values are appended as lists.
    record_every : int
        How often to record history (default: every iteration).
    """

    N = len(resolvents)

    assert Z.shape == (N, N-1), "Z must be shape (N, N-1)"
    assert len(w0) == N-1, "w must have length N-1"

    stepsizes = sigma / d
    resolvents = [lambda z, J=J, lam=lam: J(z, lam) for J, lam in zip(resolvents, stepsizes)]

    x = N * [None]
    w = w0.copy()

    history = {
        'iterations_recorded': [],
        'state_variance': [],
        'P_variance': [],
        'edge_residual_norm_sq': [],
        'P_edge_residual_norm_sq': [],
    } if return_history else None

    # Initialize extra metric keys
    if return_history and extra_metrics_fn is not None:
        _dummy_keys = None  # will be populated on first call

    start_time = time.time()
    next_update = start_time

    for k in range(1, iterations + 1):

        current_time = time.time()
        if (current_time >= next_update) and printprogress:
            print(f"Progress: ~{int(100 * k / iterations)}% ({k}/{iterations}),  Time elapsed: "
                  f"~{int((current_time - start_time) / 60)} min")
            next_update = current_time + 60

        for i in range(N):
            J_lam = resolvents[i]

            x_sum_i = add_dicts(*[x[h] for h in parent_node[i]])
            w_sum_i = add_dicts(*w, weights=Z[i, :])

            left_term = scalar_multiply_dict(x_sum_i, 2 / d[i])
            right_term = scalar_multiply_dict(w_sum_i, 1 / d[i])

            x[i] = J_lam(add_dicts(left_term, right_term))

        for j in range(N - 1):
            x_sum_j = add_dicts(*x, weights=theta * Z[:, j])
            w[j] = subtract_dicts(w[j], x_sum_j)

        if return_history and (k % record_every == 0 or k == iterations):
            history['iterations_recorded'].append(k)

            x_bar = scalar_multiply_dict(add_dicts(*x), 1 / N)

            # full state variance
            var_full = 0.0
            var_P = 0.0
            for xi in x:
                diff = subtract_dicts(xi, x_bar)
                var_full += inner_product_dicts(diff, diff)

                diffP = xi['P'] - x_bar['P']
                var_P += np.sum(diffP * diffP)

            var_full /= N
            var_P /= N

            # edge residuals r_j = sum_i Z_{ij} x_i
            edge_res_sq = 0.0
            P_edge_res_sq = 0.0
            for j in range(N - 1):
                rj = add_dicts(*x, weights=Z[:, j])
                edge_res_sq += inner_product_dicts(rj, rj)
                P_edge_res_sq += np.sum(rj['P'] * rj['P'])

            history['state_variance'].append(var_full)
            history['P_variance'].append(var_P)
            history['edge_residual_norm_sq'].append(edge_res_sq)
            history['P_edge_residual_norm_sq'].append(P_edge_res_sq)

            # extra metrics (e.g., L2 error vs ground truth)
            if extra_metrics_fn is not None:
                extra = extra_metrics_fn(x)
                for key, val in extra.items():
                    if key not in history:
                        history[key] = []
                    history[key].append(val)

    if return_history:
        return x, w, history
    return x, w
# '''

def CP(x0, u0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=False, aux_funct = lambda x, i: None): # Condat Notation (f+g)
    """
    Chambolle-Pock (CP) algorithm as seen in Condat's article, with in-memory checkpointing.
    Saves state in the `current_state` variable on KeyboardInterrupt.
    """
    x = x0.copy()
    u = u0.copy()
    start_time = time.time()
    next_update = start_time  # Update every 1 second

    for i in range(1, iterations + 1):
        aux_funct(x, i)
        x_prev = x.copy()        
        # x_{i+1} = prox_{tau f}( x_i - tau * Lstar(u_i) )
        x_minus = subtract_dicts(x, scalar_multiply_dict(Lstar(u), tau))
        x = prox_f(x_minus, tau)
        # u_{i+1} = prox_{sigma g^*}( u_i + sigma * L( 2 * x_{i+1} - x_i ) )
        x_tilde = subtract_dicts(scalar_multiply_dict(x, 2), x_prev)
        u_plus = add_dicts(u, scalar_multiply_dict(L(x_tilde), sigma))
        u = prox_gstar(u_plus, sigma)
        
        current_time = time.time()
        if (current_time >= next_update) & printprogress:
            print(f"Progress: ~{int(100 * i / iterations)}% ({i}/{iterations}),  Time elapsed:\
~{int((current_time - start_time) / 60)} min,  Time left:\
~{int((current_time - start_time) / 60 * (iterations - i) / i)} min")
            next_update = current_time + 60  # Schedule next update
    return x, u

def pDR_Richardson(x0, sig, prox_f, prox_g, K, Kstar, norm_K_sq, iterations=10000, printprogress=False):
    # Bredies notation (f+g*)
    """
    Preconditioned Douglas-Rachford (PDR) algorithm.
    
    Inputs:
        x0, xbar0, y0, ybar0: Initial primal and dual variables (dicts of arrays).
        sigma: Step-size parameter.
        prox_f, prox_g: Proximal operators for F and G (functions taking (dict, sigma)).
        K, Kstar: Forward and adjoint linear operators (functions taking dicts).
        M_inv: Inverse of preconditioner M (function taking a dict).
        T: Function computing T(x) = x + sigma^2 * K^T K x (returns dict).
        iterations: Number of iterations.
        printprogress: If True, prints progress every ~30 seconds.
    """
    x = x0.copy()
    xbar = x.copy()
    y = K(x)
    ybar = y.copy()

    
    T = lambda x: add_dicts( x, scalar_multiply_dict( Kstar( K(x) ), sig**2 ) )
    lam = 1 + sig**2 * norm_K_sq
    M_inv = lambda x: scalar_multiply_dict(x, 1/lam)
    
    start_time = time.time()
    next_update = start_time
    
    # M_inv is initialized at lam=1 as in the Richardson preconditioner

    for i in range(1, iterations + 1):
        # b^k = x̄^k - σ K^T ȳ^k
        b = subtract_dicts(xbar, scalar_multiply_dict(Kstar(ybar), sig))
        # x^{k+1} = x^k + M^{-1}( b^k - T(x^k) )
        diff = subtract_dicts(b, T(x))
        x = add_dicts(x, M_inv(diff))
        # y^{k+1} = ȳ^k + σ K x^{k+1}
        y = add_dicts(ybar, scalar_multiply_dict(K(x), sig))
        # x̄^{k+1} = x̄^k + prox_{σF}( 2x^{k+1} - x̄^k ) - x^{k+1}
        two_x = scalar_multiply_dict(x, 2)
        prox_arg = subtract_dicts(two_x, xbar)
        prox_x = prox_f(prox_arg, sig)
        xbar = add_dicts(xbar, subtract_dicts(prox_x, x))
        # ȳ^{k+1} = ȳ^k + prox_{σG}( 2y^{k+1} - ȳ^k ) - y^{k+1}
        two_y = scalar_multiply_dict(y, 2)
        prox_arg = subtract_dicts(two_y, ybar)
        prox_y = prox_g(prox_arg, sig)
        ybar = add_dicts(ybar, subtract_dicts(prox_y, y))
        current_time = time.time()
        if (current_time >= next_update) and printprogress:
            elapsed = int((current_time - start_time) / 60)
            remaining = int(elapsed * (iterations - i) / i) if i > 0 else '?'
            print(f"Progress: ~{int(100 * i / iterations)}% ({i}/{iterations}),  "
                  f"Time elapsed: ~{elapsed} min,  Time left: ~{remaining} min")
            next_update = current_time + 30

    return x, y

def DR(s0, prox_f, prox_g, iterations, tau=1, rho=lambda i: 1, printprogress=False): # Condat notation (f+g)
    """
    Douglas-Rachford (DR) splitting algorithm using dictionaries.
    
    Parameters:
    - s0: Initial variable (dictionary)
    - tau: Step size
    - prox_f: Proximal operator of f
    - prox_g: Proximal operator of g
    - iterations: Number of iterations
    - rho: Relaxation parameter (default=1). Note: it's a function!
    - printprogress: If True, prints progress updates
    
    Returns:
    - Final solution s
    """
    s = s0.copy()
    start_time = time.time()
    next_update = start_time  # Time update interval
    
    for i in range(1, iterations + 1):
        s_prev = s.copy()

        # First proximal step: x_half = prox_f(s)
        x_half = prox_f(s, tau)

        # Reflection step: compute x_tilde = 2x_half - s
        x_tilde = subtract_dicts(scalar_multiply_dict(x_half, 2), s)

        # Second proximal step: prox_g(x_tilde)
        prox_x_tilde = prox_g(x_tilde, tau)

        # Final update step
        s = add_dicts(s_prev, scalar_multiply_dict(subtract_dicts(prox_x_tilde, x_half), rho(i)))
        
        # Progress logging
        current_time = time.time()
        if (current_time >= next_update) & printprogress:
            print(f"Progress: ~{int(100 * i / iterations)}% ({i}/{iterations}), Time elapsed:\
~{int((current_time - start_time) / 60)} min, Time left:\
~{int((current_time - start_time) / 60 * (iterations - i) / i)} min")
            next_update = current_time + 30  # Schedule next update

    return x_half, s



from collections.abc import Mapping
from numbers import Number

def add_dicts(*terms, weights=1):
    """
    Sum dictionaries pointwise. The number 0 is treated as the neutral element:
      - add_dicts(d1, 0, d2) == add_dicts(d1, d2)
      - add_dicts() == 0

    If weights is not 1, it must be a sequence with the same length as `terms`.
    Weights corresponding to zero-terms are ignored together with those terms.
    """
    if not terms:
        return 0

    # Build per-term weights
    if isinstance(weights, Number) and weights == 1:
        ws = (1,) * len(terms)
    else:
        if len(weights) != len(terms):
            raise ValueError("Number of weights doesn't match number of terms")
        ws = tuple(weights)

    # Filter out numeric zeros (and their weights)
    filtered = [
        (w, d) for w, d in zip(ws, terms)
        if not (isinstance(d, Number) and d == 0)
    ]

    if not filtered:
        return 0

    dicts = [d for _, d in filtered]
    keys = dicts[0].keys()
    if any(d.keys() != keys for d in dicts[1:]):
        raise ValueError("The keys don't match")

    return {k: sum(w * d[k] for w, d in filtered) for k in keys}

def subtract_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        raise Exception('The keys don''t match')
    return {key: dict1[key] - dict2[key] for key in dict1}

def scalar_multiply_dict(dict0, scalar):
    # Treat numeric zero as the zero element
    if isinstance(dict0, Number) and not isinstance(dict0, bool) and dict0 == 0:
        return 0
    if not isinstance(dict0, Mapping):
        raise TypeError(f"Expected a dict-like mapping or 0; got {type(dict0).__name__}")
    return {key: scalar * value for key, value in dict0.items()}

def inner_product_dicts(dict1, dict2): # useful to verify that the adjoint operator is correctly defined
    return np.sum( [ np.dot( dict1[key].flatten(), dict2[key].flatten() ) for key in dict1 ] )