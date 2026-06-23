import numpy as np
import time
from .utils_for_dict import add_dicts, scalar_multiply_dict, subtract_dicts, inner_product_dicts

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