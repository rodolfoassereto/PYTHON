import numpy as np

def build_mass_imbalance_factor( shape_x, shape_y ):
    assert len(shape_x)==2
    yy = np.linspace(-1, 1, shape_x[0])
    xx = np.linspace(-1, 1.25, shape_x[1])
    Y, X = np.meshgrid(yy, xx, indexing='ij')
    W = 1 - 0.1*X**2 - 0.2*Y**2
    W2 = W[(...,) + (None,) * len(shape_y)]
    return W2.clip(min=0)