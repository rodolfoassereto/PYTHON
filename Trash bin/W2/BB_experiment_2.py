import numpy as np
import scipy.sparse as sp
from differential_operators_sparse import create_partialplus_sparse, create_nabla_sparse

# definisco le shape da usare

rho_shape = (7,6,5) # (tempo, x, y)
mx_shape = np.array(rho_shape) + np.array([0,1,0])
my_shape = np.array(rho_shape) + np.array([0,0,1])

f, g = np.zeros(rho_shape[1:]), np.zeros(rho_shape[1:])
f[0,0], g[-1,-1] = 1, 1

# definisco gli operatori differenziali

d_t = create_partialplus_sparse(rho_shape, 0)
print( 'd_t.shape: ', d_t.shape ) # (210, 210)

d_x = create_partialplus_sparse(mx_shape, 1)
d_y = create_partialplus_sparse(my_shape, 2)

# np.ravel_multi_index(multi_index, shape)
# multi_index = np.unravel_index(flat_index, shape)

# shape = (  )
# size = np.prod(shape)
# index_obj = (slice(1, -1), slice(None), slice(2, None))  # Equivalent to A[1:-1, :, 2:]
# # Create a reference index array
# indices = np.arange(size).reshape(shape)
# # Apply indexing to get selected flat indices
# flat_indices = indices[index_obj].ravel()

def build_div(P, N0, N1):
    # Total number of grid points
    n_points = P * N0 * N1
    n_momentum = 2 * P * N0 * N1  # 2 components (x,y) for momentum
    
    rows, cols, data = [], [], []
    
    # Create sparse matrix for divergence
    for t in range(P):
        for i in range(N0):
            for j in range(N1):
                idx = t * (N0 * N1) + i * N1 + j
                
                # x-component divergence (central difference)
                if 0 < i < N0-1:
                    # (m_x[i+1,j] - m_x[i-1,j])/2
                    rows.append(idx)
                    cols.append(t * (2 * N0 * N1) + (i+1) * N1 + j)  # m_x[i+1,j]
                    data.append(0.5)
                    
                    rows.append(idx)
                    cols.append(t * (2 * N0 * N1) + (i-1) * N1 + j)  # m_x[i-1,j]
                    data.append(-0.5)
                
                # y-component divergence (central difference)
                if 0 < j < N1-1:
                    # (m_y[i,j+1] - m_y[i,j-1])/2
                    rows.append(idx)
                    cols.append(t * (2 * N0 * N1) + N0 * N1 + i * N1 + (j+1))  # m_y[i,j+1]
                    data.append(0.5)
                    
                    rows.append(idx)
                    cols.append(t * (2 * N0 * N1) + N0 * N1 + i * N1 + (j-1))  # m_y[i,j-1]
                    data.append(-0.5)
    
    div = sp.csr_matrix((data, (rows, cols)), shape=(n_points, n_momentum))
    return div

div = build_div(*rho_shape)
print( 'div.shape: ', div.shape )

