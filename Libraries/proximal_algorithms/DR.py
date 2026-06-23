import numpy as np
import time
from .utils_for_dict import add_dicts, scalar_multiply_dict, subtract_dicts, inner_product_dicts

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