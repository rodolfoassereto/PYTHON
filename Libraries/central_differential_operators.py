import numpy as np

def partialcent(x, i):
    x = np.moveaxis(x, i, 0)
    y = np.zeros(x.shape, dtype=x.dtype)
    y[1:-1] = (x[2:] - x[:-2]) / 2
    return np.moveaxis(y, 0, i)

def partialcent_stag(x, i):
    x = np.moveaxis(x, i, 0)
    y = np.zeros(x.shape, dtype=x.dtype)
    y[1:-1] = (x[2:] - x[:-2]) / 2
    return np.moveaxis(y, 0, i)
