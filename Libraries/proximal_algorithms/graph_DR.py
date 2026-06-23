import numpy as np
import time
from .utils_for_dict import add_dicts, scalar_multiply_dict, subtract_dicts, inner_product_dicts

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