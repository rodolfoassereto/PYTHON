import time
from .utils_for_dict import add_dicts, scalar_multiply_dict, subtract_dicts


def graph_DR(sigma, Z, parent_node, d, w0, iterations, resolvents,
             theta=1.0, printprogress=False, extra_metrics_fn=None, record_every=1):
    """
    Graph-based Douglas-Rachford method (Algorithm 1 from Bredies, Chenchene, Naldi 2022).

    The iteration itself is just the resolvent sweep followed by the w-update; it
    computes no diagnostics. Passing `extra_metrics_fn` turns on history recording:
    every `record_every` iterations the iteration index is stored together with
    whatever `extra_metrics_fn(x)` returns. Convergence diagnostics (consensus
    variance, fixed-point residual, ...) are therefore opt-in and live in the
    caller; see `graph_DR_auxiliary_functions.graph_DR_diagnostics`.

    Returns
    -------
    (x, w)            when no `extra_metrics_fn` is given;
    (x, w, history)   when one is, where history is the recorded dict.

    Parameters
    ----------
    extra_metrics_fn : callable, optional
        Function x_list -> dict of scalars, called every `record_every` iterations.
        Each key creates an entry in history; values are appended as lists.
        When None, no history is recorded.
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

    record = extra_metrics_fn is not None
    history = {'iterations_recorded': []} if record else None

    start_time = time.time()
    next_update = start_time

    for k in range(1, iterations + 1):

        current_time = time.time()
        if (current_time >= next_update) and printprogress:
            print(f"Progress: ~{int(100 * k / iterations)}% ({k}/{iterations}),  Time elapsed:\
~{int((current_time - start_time) / 60)} min,  Time left:\
~{int((current_time - start_time) / 60 * (iterations - k) / k)} min")
            next_update = current_time + 60  # Schedule next update

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

        if record and (k % record_every == 0 or k == iterations):
            history['iterations_recorded'].append(k)
            for key, val in extra_metrics_fn(x).items():
                history.setdefault(key, []).append(val)

    if record:
        return x, w, history
    return x, w
