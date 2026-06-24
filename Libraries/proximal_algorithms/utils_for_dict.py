import numpy as np
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
    # np.vdot conjugates its first argument, so this is a genuine (Hermitian) inner
    # product for complex entries too
    return np.sum( [ np.vdot( dict1[key].flatten(), dict2[key].flatten() ) for key in dict1 ] )