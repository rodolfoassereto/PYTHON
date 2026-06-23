'''Here we define a general version of Chambolle-Pock, using dictionary variables'''
'''I refer to Condat (https://doi.org/10.1137/20M1379344) for the notation. Here, the saddle problem has F and G^*'''

import numpy as np
import time
from .utils_for_dict import add_dicts, scalar_multiply_dict, subtract_dicts, inner_product_dicts


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
