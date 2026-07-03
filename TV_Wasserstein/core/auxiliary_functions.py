import numpy as np

def central_Y_indices(shape_f, shape_x, shape_y): # returns the indices of the Y-centers; f is only needed for its shape actually
    ndim_x, ndim_y = len(shape_x), len(shape_y)
    ndim_0 = len(shape_f) - ndim_x - ndim_y
    assert ndim_0 >= 0
    return (slice(None),)*(ndim_0+ndim_x) + tuple( np.array(shape_y) // 2 )