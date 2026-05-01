import numpy as np
import scipy.sparse as sp


def create_partialplus_sparse_1d_zeroboundary(n):
    D = np.zeros((n, n), dtype=int)  # Initialize matrix with zeros
    np.fill_diagonal(D[:-1,:], -1)   # Set main diagonal to -1
    np.fill_diagonal(D[:, 1:], 1)    # Set upper diagonal to 1
    return sp.csr_matrix( D )

def create_partialplus_sparse_1d_symmboundary(n):
    D = np.zeros((n, n), dtype=int)  # Initialize matrix with zeros
    np.fill_diagonal(D[:,:], -1)   # Set main diagonal to -1
    np.fill_diagonal(D[:, 1:], 1)    # Set upper diagonal to 1
    return sp.csr_matrix( D )

def create_partialplus_sparse_1d_noboundary(n):
    D = np.zeros((n-1, n), dtype=int)  # Initialize matrix with zeros
    np.fill_diagonal(D[:,:], -1)   # Set main diagonal to -1
    np.fill_diagonal(D[:, 1:], 1)    # Set upper diagonal to 1
    return sp.csr_matrix( D )

def create_laplacian_sparse_1d(n):
    L = np.zeros((n, n), dtype=int)  # Initialize matrix with zeros
    np.fill_diagonal(L[:,:], -2)   # Set main diagonal to -1
    np.fill_diagonal(L[:, 1:], 1)    # Set upper diagonal to 1
    np.fill_diagonal(L[1:, :], 1)    # Set upper diagonal to 1
    L[0,0], L[-1, -1] = -1, -1
    return sp.csr_matrix( L )

def create_laplacian_sparse_1d_periodic(n):
    D = np.zeros((n, n), dtype=int)  # Initialize matrix with zeros
    np.fill_diagonal(D[:,:], -2)   # Set main diagonal to -1
    np.fill_diagonal(D[:, 1:], 1)    # Set upper diagonal to 1
    np.fill_diagonal(D[1:, :], 1)    # Set upper diagonal to 1
    D[-1,0], D[0, -1] = 1, 1
    return sp.csr_matrix( D )


def create_partialplus_sparse(shape, i, boundary='zero'):
    '''
    Creates a sparse matrix that computes the partial derivative along the i-th axis of an array with shape 'shape'
    '''
    
    if boundary == 'zero': create_partial_1d = create_partialplus_sparse_1d_zeroboundary
    if boundary == 'no': create_partial_1d = create_partialplus_sparse_1d_noboundary
    if boundary == 'symm': create_partial_1d = create_partialplus_sparse_1d_symmboundary
    if not (-len(shape) <= i < len(shape)):  # Valid range is [-ndim, ndim - 1]
        raise ValueError(f"Invalid axis {i} for shape {shape} with {len(shape)} dimensions")
    i = i % len(shape) # allows for negative indices i (as in numpy's convention)
    
    if len(shape) == 1:
        return create_partial_1d( shape[0] )
    elif i == len(shape)-1:
        result = sp.kron( sp.eye(shape[0], dtype=int) , create_partialplus_sparse(shape[1:], -1, boundary=boundary) )
    else:
        result = sp.kron( create_partialplus_sparse(shape[:-1], i, boundary=boundary) , sp.eye(shape[-1], dtype=int) )
    return result.tocsr()

# def nabla_sparse(x, *partials, stack_to_axis=0):
#     return np.stack( [ partial @ x.ravel()  for partial in partials ], axis=stack_to_axis )

def create_partials_sparse( shape, axes, boundary='no' ): # creates (sparse) partial derivatives to plug in create_nabla_sparse
    return [ create_partialplus_sparse(shape, i, boundary) for i in axes ]

def create_nabla_sparse(*partials):
    return sp.bmat( [ [ partial ] for partial in partials ], format='csr' )


def create_laplacian_sparse(*partials):
    return np.sum( [ -partial.T @ partial for partial in partials ] )

def create_colaplacian_sparse(*partials): #  similar definition as Δ but defined as -∇∇* instead of -∇*∇
    N = create_nabla_sparse(*partials)
    return - N @ N.T


# def div_sparse(v, indices, partials_tuple):
#     partial @ v[i] for i, partial in zip(indices, partials_tuple):
#     if v.shape[axis] > v.ndim-1: raise Exception('You want to take too many derivatives')
#     v = np.moveaxis(v, axis, 0)
#     return np.sum( [partialminus(x, k) for k, x in enumerate(v)],  axis=0)

def div(sig, M1, M2): # Computes the divergence of a vector field sig
    return M1 @ sig[:, 0] + M2 @ sig[:, 1]

# %% STAGGERED GRIDS

def create_staggered_partialplus_sparse_1d(n):
    D = np.zeros((n-1, n), dtype=int)  # Initialize matrix with zeros
    np.fill_diagonal(D[:,:-1], -1)   # Set main diagonal to -1
    np.fill_diagonal(D[:, 1:], 1)    # Set upper diagonal to 1
    return sp.csr_matrix( D )

def create_staggered_partialplus_sparse(shape, i):
    '''
    Creates a sparse matrix that computes the partial derivative along the i-th axis of an array with shape 'shape'
    '''
    if not (-len(shape) <= i < len(shape)):  # Valid range is [-ndim, ndim - 1]
        raise ValueError(f"Invalid axis {i} for shape {shape} with {len(shape)} dimensions")
    i = i % len(shape) # allows for negative indices i (as in numpy's convention)
    
    if len(shape) == 1:
        return create_staggered_partialplus_sparse_1d( shape[0] )
    elif i == len(shape)-1:
        result = sp.kron( sp.eye(shape[0], dtype=int) , create_staggered_partialplus_sparse(shape[1:], -1) )
    else:
        result = sp.kron( create_staggered_partialplus_sparse(shape[:-1], i) , sp.eye(shape[-1], dtype=int) )
    return result.tocsr()

# %%

class BlockLinearOperator:
    def __init__(self, blocks):
        """
        blocks: list of lists (N rows, M columns) of sparse matrices or LinearOperators
                blocks[i][j] maps from x_j to y_i
        """
        self.blocks = blocks
        self.N = len(blocks)          # Number of output blocks (rows)
        self.M = len(blocks[0])       # Number of input blocks (columns)

        # Check all rows have the same number of columns
        for row in blocks:
            assert len(row) == self.M, "All rows must have same number of columns"

    def matvec(self, x):
        """
        x: list or tuple of input vectors [x0, x1, ..., x_{M-1}]
        returns: list of output vectors [y0, y1, ..., y_{N-1}]
        """
        assert len(x) == self.M, "Number of inputs must match block columns"
        y = []
        for i in range(self.N):
            # Sum over j of A_{i,j} @ x_j
            y_i = sum(self.blocks[i][j] @ x[j] for j in range(self.M))
            y.append(y_i)
        return y

    def __call__(self, x):
        return self.matvec(x)

def example_BlockLinearOperator():
    # Example blocks: 3 rows (outputs), 2 columns (inputs)
    blocks = [
        [sp.csr_matrix([[1, 0], [0, 1]]), sp.csr_matrix([[2, 0], [0, 2]])],
        [sp.csr_matrix([[0, 1], [1, 0]]), sp.csr_matrix([[0, 0], [1, 1]])],
        [sp.csr_matrix([[3, 3], [0, 0]]), sp.csr_matrix([[1, 0], [0, 1]])],
    ]
    
    # Create the block operator
    block_op = BlockLinearOperator(blocks)
    
    # Inputs: one vector for each input column
    x0 = np.array([1, 1])
    x1 = np.array([2, 0])
    
    # Apply the block operator
    y = block_op([x0, x1])
    for i, yi in enumerate(y):
        print(f"y[{i}] =", yi)


