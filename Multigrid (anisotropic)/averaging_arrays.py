import numpy as np
import scipy.sparse as sp

def average_along_axis(x, axes): # averages along all axes contained in the tuple "axes" (axes can also be one single int)
    if isinstance(axes, int):
        axes = (axes,)
    
    for axis in sorted(axes):  # Sort to avoid shifting axes during iteration
        slices_1 = [slice(None)] * x.ndim
        slices_2 = [slice(None)] * x.ndim

        slices_1[axis] = slice(1, None)
        slices_2[axis] = slice(None, -1)

        x = (x[tuple(slices_1)] + x[tuple(slices_2)]) / 2
    return x

def averaging_matrix_1d(n):
    """Returns a sparse matrix of shape (n-1, n) for 1D averaging."""
    data = np.array([0.5, 0.5])
    offsets = np.array([0, 1])
    return sp.diags(data, offsets, shape=(n-1, n), format='csr')

def create_avg_sparse_1d_zeroboundary(n):
    D = np.zeros((n+1, n))  # Initialize matrix with zeros
    np.fill_diagonal(D, 0.5)   # Set main diagonal to -1
    np.fill_diagonal(D[1:, :], 0.5)    # Set upper diagonal to 1
    return sp.csr_matrix( D )

def create_avg_sparse_zeroboundary(shape, i):
    
    if not (-len(shape) <= i < len(shape)):  # Valid range is [-ndim, ndim - 1]
        raise ValueError(f"Invalid axis {i} for shape {shape} with {len(shape)} dimensions")
    i = i % len(shape) # allows for negative indices i (as in numpy's convention)
    
    if len(shape) == 1:
        return create_avg_sparse_1d_zeroboundary( shape[0] )
    elif i == len(shape)-1:
        result = sp.kron( sp.eye(shape[0], dtype=int) , create_avg_sparse_zeroboundary(shape[1:], -1) )
    else:
        result = sp.kron( create_avg_sparse_zeroboundary(shape[:-1], i) , sp.eye(shape[-1], dtype=int) )
    return result.tocsr()

def average_along_axis_as_matrix(shape, axes):
    """Returns a sparse matrix A such that A @ x.flatten() = average_along_axis(x, axes).flatten()"""
    if isinstance(axes, int):
        axes = (axes,)
    
    ndim = len(shape)
    for axis in axes:
        if not (0 <= axis < ndim): raise ValueError(f"Invalid axis {axis} for shape {shape}")
    
    A = sp.identity(np.prod(shape), format='csr')
    current_shape = shape

    for axis in sorted(axes):
        ops = []
        for i, dim in enumerate(current_shape):
            if i == axis:
                ops.append(averaging_matrix_1d(dim))
            else:
                ops.append(sp.identity(dim, format='csr'))
        
        kron_op = ops[0]
        for op in ops[1:]:
            kron_op = sp.kron(kron_op, op, format='csr')
        
        A = kron_op @ A
        # update shape after applying averaging along this axis
        current_shape = tuple(
            (d - 1 if i == axis else d) for i, d in enumerate(current_shape)
        )

    return A