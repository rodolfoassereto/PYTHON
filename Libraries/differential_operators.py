''' Here we define the discrete differential operators '''

import numpy as np

# %% partialplus, partialminus

def partialplus_old(x, i):
    x = np.moveaxis(x, i, 0)
    y = np.zeros(x.shape, dtype=x.dtype)
    y[0:-1] = x[1:] - x[:-1]
    return np.moveaxis(y, 0, i)

def partialplus(x, i): # slightly faster than the old version; scipy's convolve2d is even slower
    slices = [slice(None)] * x.ndim # A slice object is (start,stop,step) and is meant to be used as an index. The command slice(None) creates a slice object (None,None,None) that works just as the slicing ":"
    slices[i] = slice(-1, None) # at entry "i", we build the slice (-1,None,None), which selects the last slice along axis i, but keeps intact the number of dimensions ndim (collapsing axis i to dim=1)
    return np.diff(x, axis=i, append=x[tuple(slices)])

def partialplus_1(x, i): # Outputs finite differences, with 0 at the end. Doesn't perform better than partialplus.
    padding = [(0,0)] * x.ndim
    padding[i] = (0,1)
    return np.diff( np.pad(x, padding, mode='edge'), axis=i)

def partialplus_2(x,i): # a different definition of partialplus: last row/column stays the same (I append zeros)
    return np.diff(x, axis=i, append=0)

''' Note: partialminus is *minus* the adjoint of partialplus'''

def partialminus(x, i):
    if x.shape[i] == 1: return np.zeros(x.shape)
    x = np.moveaxis(x, i, 0)
    y = x.copy() # the first line is already copied
    y[1:] = x[1:] - x[:-1]
    y[-1] = -x[-2]
    return np.moveaxis(y, 0, i)

def partialminus_2(x,i):
    return np.diff(x, axis=i, prepend=0)


# %% Gradients, Divergences, Laplacians

def nabla(u, start=0, stop=2, stack_to_axis=0):
    return np.stack( [ partialplus(u, k) for k in range(start,stop) ], axis=stack_to_axis )

def nabla_x(u, dim): # from (n1...nd,m1,...,mD) and dim=d, returns (d,n1...nd,m1,...,mD)
    return np.stack( [ partialplus(u, k) for k in range(dim) ] )

def nabla_y(u, dim): # from (n1...nd,m1,...,mD) and dim=D, returns (D,n1...nd,m1,...,mD)
    return np.stack( [partialplus(u, u.ndim-dim+k) for k in range(dim)] )

''' Note: the divergence is *minus* the adjoint of the gradient'''

''' A divergence has two arguments:
     1) The axis to collapse ("axis");
     2) which derivatives you are taking (div_x for the first ones, div_y for the last ones).
    Note: the number of derivatives to take is fixed by the length v.shape[axis]
'''

def div_x(v, axis=0): # computes the divergence (if axis=0, derivatives start from axis 1 of v) by collapsing the axis "axis"
    if v.shape[axis] > v.ndim-1: raise Exception('You want to take too many derivatives')
    v = np.moveaxis(v, axis, 0)
    return np.sum( [partialminus(x, k) for k, x in enumerate(v)],  axis=0)

def div_y(v, axis=0): # computes the divergence (if axis=0, derivatives start from axis -v.shape[0]) by collapsing the axis "axis"
    if v.shape[axis] > v.ndim-1: raise Exception('You want to take too many derivatives')
    v = np.moveaxis(v, axis, 0)
    start = len(v)
    return np.sum( [partialminus(x, -start+k) for k, x in enumerate(v)],  axis=0)


def laplacian_x(v, dim): # checked with its sparse counterpart (which defaults partialplus to 'noboundary'), produces the same output
    return div_x( nabla_x(v, dim) )


# %% TV and Frobenius product

tv_1d = lambda x: np.linalg.norm( partialplus(x, 0), ord=1 )
tv_2d_iso = lambda x: np.sum( np.linalg.norm( nabla_x(x), axis=0 ) )
tv_2d_aniso = lambda x: np.sum(np.abs(nabla_x(x)))

def Frobenius(A,B):
    return np.dot(A.flatten(),B.flatten())


# %% Central differential operators

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
